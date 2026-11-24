"""
Amharic Text Preprocessing and Dataset Utilities.

This module implements a transparent, from-scratch preprocessing pipeline:
1. Unicode NFKC normalization
2. Noise, HTML, and non-printable character removal
3. Ethiopic/Amharic fidel and punctuation preservation
4. Whitespace and newline regularization
5. Train/validation data splitting
"""

import os
import re
import json
import unicodedata
import argparse
from typing import Tuple, Dict, List, Optional, Any
from collections import Counter
import numpy as np

from src.amharic_normalizer import (
    AmharicNormalizer,
    is_ethiopic_character,
    is_ethiopic_fidel,
    normalize_unicode,
    normalize_homophones,
    normalize_wordspace,
    normalize_question_mark,
    normalize_punctuation,
    calculate_ethiopic_ratio,
    filter_low_ethiopic_lines,
    deduplicate_lines,
    ETHIOPIC_PUNCTUATION,
    ETHIOPIC_NUMERALS,
    ALL_HOMOPHONES,
)

# Ethiopic Unicode script range:
# U+1200 - U+137F (Ethiopic fidel syllabary, digits, and punctuation)
# U+1380 - U+139F (Ethiopic Supplement)
# U+2D80 - U+2DDF (Ethiopic Extended)
ETHIOPIC_RANGE_REGEX = r"\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF"

# Standard punctuation allowed alongside Ethiopic punctuation:
# Includes standard spaces, newlines, English/Arabic numerals, common punctuation marks
ALLOWED_CHARS_REGEX = re.compile(
    rf"[^{ETHIOPIC_RANGE_REGEX}a-zA-Z0-9\s\.,;:!\?'\"()\[\]\-–—/]+"
)


def normalize_amharic_text(text: str) -> str:
    """
    Applies Unicode NFKC normalization to text.
    
    CRITICAL TECHNICAL NOTE ON NFKC AND ETHIOPIC:
    ----------------------------------------------
    What NFKC DOES:
    - Resolves Unicode compatibility formatting characters (full-width latin, ligatures,
      superscripts, decomposed combining marks).
      
    What NFKC DOES NOT DO for Ethiopic:
    - Ethiopic syllables (Fidel) are atomically precomposed in Unicode (e.g. U+1200 'ሀ').
    - Ethiopic syllables DO NOT have canonical decompositions in Unicode.
    - NFKC does NOT decompose syllables into consonant + vowel marks.
    - NFKC does NOT normalize homophones (e.g. 'ሐ' and 'ኀ' are never changed by NFKC).
    - NFKC does NOT convert Ethiopic wordspaces (U+1361 '፡') to spaces.
    - NFKC does NOT normalize Ethiopic punctuation or numerals.
    
    Therefore, domain-specific Amharic normalization logic must be implemented explicitly.
    """
    return unicodedata.normalize("NFKC", text)


def clean_amharic_text(
    text: str,
    normalize_homophones_flag: bool = False,
    convert_ascii_question: bool = False,
    min_ethiopic_ratio: float = 0.5,
) -> str:
    """
    Cleans and normalizes raw Amharic text:
    1. Unicode NFKC normalization.
    2. Ethiopic wordspace (U+1361 '፡') -> standard ASCII space (' ').
    3. Optional homophone harmonization (ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ, all 7 orders).
    4. Configurable ASCII '?' to Ethiopic '፧' in Amharic contexts (default False).
    5. Normalizing repeated punctuation while preserving Ethiopic punctuation and numerals.
    6. Stripping HTML tags, URLs, emails, zero-width spaces, and noisy symbols.
    7. Regularizing whitespace and newlines.
    """
    normalizer = AmharicNormalizer(
        normalize_homophones=normalize_homophones_flag,
        convert_ascii_question=convert_ascii_question,
        min_ethiopic_ratio=min_ethiopic_ratio,
    )
    return normalizer.normalize(text)


def split_train_val(text: str, val_ratio: float = 0.1) -> Tuple[str, str]:
    """
    Splits text into training and validation sets.
    Preserves text order to avoid leaking future tokens for simple single-stream tasks.
    """
    assert 0.0 < val_ratio < 1.0, f"val_ratio must be between 0 and 1, got {val_ratio}"
    split_idx = int(len(text) * (1.0 - val_ratio))
    train_text = text[:split_idx].rstrip()
    val_text = text[split_idx:].lstrip()
    return train_text, val_text


def split_corpus_documents(
    text: str,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    doc_separator: str = "\n<EOS>\n",
    min_ethiopic_ratio: float = 0.5,
    normalize_homophones_flag: bool = False,
    convert_ascii_question: bool = False,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Splits a corpus at DOCUMENT or PARAGRAPH level with a seeded shuffle.
    
    Why Document-Level Seeded Shuffling is Superior to Contiguous Tails:
    -------------------------------------------------------------------
    A simple contiguous tail split (e.g. taking the final 10% of characters) causes
    domain and topic bias if documents are ordered chronologically or grouped by topic.
    Document-level splitting with a deterministic pseudo-random shuffle ensures
    that train, validation, and test splits have identical topic distributions
    without slicing through sentences or paragraphs.
    
    Document Separator:
    -------------------
    Inserts a document separator token (e.g. '<EOS>') between documents so the model
    learns when to terminate generated Amharic text autoregressively.
    
    Returns:
        train_text (str): Training split with document separators.
        val_text (str): Validation split with document separators.
        test_text (str): Test split with document separators.
        report (dict): Detailed corpus metrics and split statistics.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), (
        f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio} = "
        f"{train_ratio + val_ratio + test_ratio})"
    )

    # 1. Parse raw text into documents / paragraphs
    raw_docs = [d.strip() for d in re.split(r"\n\s*\n+", text) if d.strip()]
    if len(raw_docs) <= 1:
        raw_docs = [l.strip() for l in text.splitlines() if l.strip()]
    if not raw_docs:
        raw_docs = [text.strip()]

    total_raw_docs = len(raw_docs)

    # 2. Normalize and clean each document
    normalizer = AmharicNormalizer(
        normalize_homophones=normalize_homophones_flag,
        convert_ascii_question=convert_ascii_question,
        min_ethiopic_ratio=min_ethiopic_ratio,
    )

    cleaned_docs: List[str] = []
    dropped_low_ratio = 0
    for doc in raw_docs:
        cleaned = normalizer.normalize(doc)
        if not cleaned:
            continue
        ratio = calculate_ethiopic_ratio(cleaned)
        if ratio >= min_ethiopic_ratio:
            cleaned_docs.append(cleaned)
        else:
            dropped_low_ratio += 1

    # 3. Exact-duplicate document removal
    unique_docs, duplicate_count = deduplicate_lines(cleaned_docs)

    n_docs = len(unique_docs)
    if n_docs == 0:
        raise ValueError("No documents remained after cleaning and filtering.")

    # 4. Seeded deterministic shuffle
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n_docs)
    shuffled_docs = [unique_docs[i] for i in shuffled_indices]

    # 5. Compute split partition boundaries
    if n_docs >= 3:
        n_train = max(1, int(round(n_docs * train_ratio)))
        n_val = max(1, int(round(n_docs * val_ratio)))
        # Ensure test gets at least 1 document if possible, and sum matches n_docs
        if n_train + n_val >= n_docs:
            n_train = n_docs - 2
            n_val = 1
        n_test = n_docs - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = n_docs - n_val - n_test
    elif n_docs == 2:
        n_train, n_val, n_test = 1, 1, 0
    else:
        n_train, n_val, n_test = 1, 0, 0

    train_docs = shuffled_docs[:n_train]
    val_docs = shuffled_docs[n_train : n_train + n_val]
    test_docs = shuffled_docs[n_train + n_val :]

    # 6. Join documents using doc_separator
    train_text = doc_separator.join(train_docs)
    val_text = doc_separator.join(val_docs)
    test_text = doc_separator.join(test_docs)

    # 7. Generate comprehensive corpus report
    report = generate_corpus_report(
        full_text="\n\n".join(unique_docs),
        train_text=train_text,
        val_text=val_text,
        test_text=test_text,
        total_raw_docs=total_raw_docs,
        duplicates_removed=duplicate_count,
        dropped_low_ratio=dropped_low_ratio,
        train_docs_count=len(train_docs),
        val_docs_count=len(val_docs),
        test_docs_count=len(test_docs),
    )

    return train_text, val_text, test_text, report


def generate_corpus_report(
    full_text: str,
    train_text: str = "",
    val_text: str = "",
    test_text: str = "",
    total_raw_docs: int = 0,
    duplicates_removed: int = 0,
    dropped_low_ratio: int = 0,
    train_docs_count: int = 0,
    val_docs_count: int = 0,
    test_docs_count: int = 0,
) -> Dict[str, Any]:
    """
    Computes rigorous statistical report on the corpus:
    - Characters, words, average word length
    - Unique fidel count, total unique characters
    - Top 20 most frequent characters with frequency and percentage
    - Split sizes (chars, words, documents, percentage)
    - Duplicate and dropped document counts
    """
    words = full_text.split()
    total_words = len(words)
    total_chars = len(full_text)
    avg_word_length = (sum(len(w) for w in words) / max(1, total_words)) if total_words > 0 else 0.0

    char_counts = Counter(full_text)
    top_20 = [
        {"char": char, "count": count, "percentage": round(count / max(1, total_chars) * 100, 2)}
        for char, count in char_counts.most_common(20)
    ]

    unique_fidel = sorted([c for c in char_counts.keys() if is_ethiopic_fidel(c)])
    unique_ethiopic_all = sorted([c for c in char_counts.keys() if is_ethiopic_character(c)])

    total_split_chars = len(train_text) + len(val_text) + len(test_text)
    train_pct = round(len(train_text) / max(1, total_split_chars) * 100, 2)
    val_pct = round(len(val_text) / max(1, total_split_chars) * 100, 2)
    test_pct = round(len(test_text) / max(1, total_split_chars) * 100, 2)

    return {
        "total_characters": total_chars,
        "total_words": total_words,
        "average_word_length": round(avg_word_length, 2),
        "unique_characters": len(char_counts),
        "unique_fidel_count": len(unique_fidel),
        "unique_ethiopic_all_count": len(unique_ethiopic_all),
        "top_20_characters": top_20,
        "raw_documents_count": total_raw_docs,
        "duplicates_removed": duplicates_removed,
        "dropped_low_ratio_documents": dropped_low_ratio,
        "splits": {
            "train": {
                "characters": len(train_text),
                "words": len(train_text.split()),
                "documents": train_docs_count,
                "percentage": train_pct,
            },
            "val": {
                "characters": len(val_text),
                "words": len(val_text.split()),
                "documents": val_docs_count,
                "percentage": val_pct,
            },
            "test": {
                "characters": len(test_text),
                "words": len(test_text.split()),
                "documents": test_docs_count,
                "percentage": test_pct,
            },
        },
    }


def get_corpus_statistics(text: str) -> Dict[str, Any]:
    """
    Calculates summary statistics of the Amharic corpus.
    """
    return generate_corpus_report(text)


def preprocess_file(
    input_path: str,
    output_dir: str = "data/processed",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    normalize_homophones_flag: bool = False,
    convert_ascii_question: bool = False,
    min_ethiopic_ratio: float = 0.5,
) -> Tuple[str, str, str]:
    """
    Reads a raw Amharic text file, preprocesses with document-level seeded split,
    and writes train.txt, val.txt, test.txt, and corpus_report.json.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Data Preprocessing] Reading raw text from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[Data Preprocessing] Raw character count: {len(raw_text)}")
    train_ratio = max(0.1, 1.0 - val_ratio - test_ratio)

    train_text, val_text, test_text, report = split_corpus_documents(
        text=raw_text,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        doc_separator="\n<EOS>\n",
        min_ethiopic_ratio=min_ethiopic_ratio,
        normalize_homophones_flag=normalize_homophones_flag,
        convert_ascii_question=convert_ascii_question,
    )

    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")
    test_path = os.path.join(output_dir, "test.txt")
    report_path = os.path.join(output_dir, "corpus_report.json")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_text)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Data Preprocessing] Saved train split ({len(train_text)} chars) -> {train_path}")
    print(f"[Data Preprocessing] Saved val split   ({len(val_text)} chars) -> {val_path}")
    print(f"[Data Preprocessing] Saved test split  ({len(test_text)} chars) -> {test_path}")
    print(f"[Data Preprocessing] Saved corpus report -> {report_path}")

    return train_path, val_path, test_path


# ============================================================================
# Next-Token Dataset & NumPy Batching Pipeline
# ============================================================================

class AmharicDataset:
    """
    NumPy-based Dataset for Next-Token Autoregressive Prediction.
    
    Next-Token Prediction Concept:
    -----------------------------
    Language models learn by predicting the next token in a sequence.
    Given an integer token sequence [x_0, x_1, x_2, ..., x_T]:
      - Input  X: [x_0, x_1, ..., x_{T-1}]
      - Target Y: [x_1, x_2, ..., x_T]
      
    Notice that Y is shifted to the right by exactly one token.
    At any position t, the causal language model uses only past tokens
    (x_0 through x_t) to predict target token Y[t] = x_{t+1}.
    
    Array Shapes:
      Input  X: shape (batch_size, block_size), dtype np.int64
      Target Y: shape (batch_size, block_size), dtype np.int64
    """

    def __init__(self, token_ids: List[int], block_size: int):
        """
        Args:
            token_ids (List[int]): Complete 1D sequence of token IDs.
            block_size (int): Context window length (T).
        """
        assert len(token_ids) > block_size, (
            f"Token sequence length ({len(token_ids)}) must be strictly greater "
            f"than block_size ({block_size}) to form at least one (input, target) pair."
        )
        self.data = np.array(token_ids, dtype=np.int64)
        self.block_size = block_size

    def __len__(self) -> int:
        """Total number of valid contiguous training chunks of length block_size."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves a single (input, target) pair shifted by one token.
        
        Args:
            idx (int): Starting index in token sequence.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                x: slice [idx : idx + block_size]
                y: slice [idx + 1 : idx + block_size + 1]
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def get_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a mini-batch of (X, Y) pairs randomly from the dataset.
        
        Args:
            batch_size (int): Number of parallel sequences (B).
            rng (Optional[np.random.RandomState]): Optional NumPy random state for reproducibility.
            
        Returns:
            X (np.ndarray): Shape (B, T) - token IDs for context.
            Y (np.ndarray): Shape (B, T) - shifted token IDs for next-token supervision.
        """
        max_idx = len(self)
        if rng is None:
            rng = np.random.RandomState(42)
        indices = rng.randint(0, max_idx, size=batch_size)

        # Batch inputs and targets
        X = np.empty((batch_size, self.block_size), dtype=np.int64)
        Y = np.empty((batch_size, self.block_size), dtype=np.int64)

        for b, idx in enumerate(indices):
            X[b] = self.data[idx : idx + self.block_size]
            Y[b] = self.data[idx + 1 : idx + self.block_size + 1]

        return X, Y


def create_datasets_from_text(
    train_text: str,
    val_text: str,
    tokenizer,
    block_size: int = 128
) -> Tuple[AmharicDataset, AmharicDataset]:
    """
    Encodes raw train and validation text into integer IDs and wraps them into AmharicDataset instances.
    """
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    
    train_ds = AmharicDataset(train_ids, block_size=block_size)
    val_ds = AmharicDataset(val_ids, block_size=block_size)
    
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Amharic Text Preprocessing")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/sample_amharic.txt",
        help="Path to raw Amharic text file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save train.txt and val.txt",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    args = parser.parse_args()

    preprocess_file(
        input_path=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
