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
import unicodedata
import argparse
from typing import Tuple, Dict, List, Optional
import numpy as np


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
    
    Why NFKC?
    NFKC (Normalization Form Compatibility Composition) resolves composite
    Unicode characters into their canonical precomposed forms. In Ethiopic
    script, this guarantees consistent character representations.
    """
    return unicodedata.normalize("NFKC", text)


def clean_amharic_text(text: str) -> str:
    """
    Cleans raw Amharic text by:
    1. Normalizing Unicode.
    2. Stripping HTML tags, URLs, and email addresses.
    3. Removing unprintable/zero-width control characters.
    4. Removing non-standard noise while preserving Amharic fidels and punctuation.
    5. Normalizing multiple spaces and redundant line breaks.
    """
    # 1. Unicode normalization
    text = normalize_amharic_text(text)

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 5. Remove zero-width characters (e.g. \u200b, \ufeff)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # 6. Filter out unwanted noisy symbols while keeping Ethiopic script & punctuation
    text = ALLOWED_CHARS_REGEX.sub(" ", text)

    # 7. Collapse multiple horizontal spaces to a single space
    text = re.sub(r"[ \t]+", " ", text)

    # 8. Collapse 3 or more consecutive newlines into 2 (paragraph break)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # 9. Strip leading and trailing whitespace
    return text.strip()


def split_train_val(text: str, val_ratio: float = 0.1) -> Tuple[str, str]:
    """
    Splits text into training and validation sets.
    Preserves text order to avoid leaking future tokens.
    """
    assert 0.0 < val_ratio < 1.0, f"val_ratio must be between 0 and 1, got {val_ratio}"
    split_idx = int(len(text) * (1.0 - val_ratio))
    train_text = text[:split_idx].rstrip()
    val_text = text[split_idx:].lstrip()
    return train_text, val_text


def get_corpus_statistics(text: str) -> Dict[str, int]:
    """
    Calculates summary statistics of the Amharic corpus.
    """
    lines = text.splitlines()
    words = text.split()
    unique_chars = sorted(list(set(text)))
    ethiopic_chars = [
        c for c in unique_chars 
        if 0x1200 <= ord(c) <= 0x137F or 0x1380 <= ord(c) <= 0x139F or 0x2D80 <= ord(c) <= 0x2DDF
    ]
    return {
        "total_characters": len(text),
        "total_words": len(words),
        "total_lines": len(lines),
        "unique_characters": len(unique_chars),
        "ethiopic_unique_characters": len(ethiopic_chars),
    }


def preprocess_file(
    input_path: str,
    output_dir: str = "data/processed",
    val_ratio: float = 0.1,
) -> Tuple[str, str]:
    """
    Reads a raw Amharic text file, preprocesses it, and writes train.txt and val.txt.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Data Preprocessing] Reading raw text from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[Data Preprocessing] Raw character count: {len(raw_text)}")
    cleaned_text = clean_amharic_text(raw_text)
    print(f"[Data Preprocessing] Cleaned character count: {len(cleaned_text)}")

    stats = get_corpus_statistics(cleaned_text)
    print(f"[Data Preprocessing] Corpus statistics: {stats}")

    train_text, val_text = split_train_val(cleaned_text, val_ratio=val_ratio)

    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)

    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)

    print(f"[Data Preprocessing] Saved train split ({len(train_text)} chars) -> {train_path}")
    print(f"[Data Preprocessing] Saved val split ({len(val_text)} chars) -> {val_path}")

    return train_path, val_path


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
            indices = np.random.randint(0, max_idx, size=batch_size)
        else:
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
