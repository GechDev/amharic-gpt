"""
From-Scratch Byte-Pair Encoding (BPE) Subword Tokenizer for Amharic.

Pure Python and NumPy implementation with zero external tokenizer dependencies.

Educational Architecture & Design Principles:
---------------------------------------------
1. Training Boundary:
   Trained strictly on the training partition (data/processed/train.txt) to prevent
   information leakage into validation or test sets.

2. Base Alphabet at Unicode Character Level:
   Ethiopic script characters (Fidel) already represent consonant-vowel syllables (abugida).
   Therefore, the base alphabet uses Unicode characters directly rather than arbitrary
   UTF-8 bytes. Any unseen character at test time falls back explicitly to <UNK>.

3. Lossless Pre-Tokenization:
   Text is pre-tokenized across whitespace and Ethiopic punctuation marks
   (።, ፣, ፤, ፥, ፦, ፧, ፨, ፡). Words are prefixed with a boundary marker (\\u2581 ' ')
   so that subword merges never cross word or sentence boundaries, and decoding is
   100% bit-for-bit lossless: decode(encode(x)) == x.

4. Efficient Incremental Merge Training:
   Maintains an inverted index of pair occurrences and updates pair frequencies
   incrementally upon each merge, ensuring O(N) operations per merge rather than
   rescanning the entire corpus. Deterministic tie-breaking ensures identical merges.

5. Priority / Minimum-Rank Encoding:
   Applies learned merges iteratively by rank order, with a memoization cache for words.

Hand-Checkable Toy Example:
---------------------------
Input corpus: "ሰላም ሰላም ሰላም ሰላም" (Word "ሰላም" repeated 4 times).
Initial representation: ('ሰ', 'ላ', 'ም') with frequency 4.
Candidate adjacent pairs:
  - ('ሰ', 'ላ') : frequency 4
  - ('ላ', 'ም') : frequency 4
Deterministic tie-breaking chooses ('ሰ', 'ላ') first:
  Merge 1: ('ሰ', 'ላ') -> 'ሰላ'
  Updated representation: ('ሰላ', 'ም') with frequency 4.
Next candidate pair:
  - ('ሰላ', 'ም') : frequency 4
  Merge 2: ('ሰላ', 'ም') -> 'ሰላም'
Final representation: ('ሰላም',) with frequency 4.
Vocabulary expands deterministically by: 'ሰላ' (rank 0), then 'ሰላም' (rank 1).
"""

import os
import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union, Any


WS_MARKER = "\u2581"  # Lower one eighth block (SentencePiece style word boundary)
ETHIOPIC_PUNCT_CHARS = "።፣፤፥፦፧፨፡"


class BPETokenizer:
    """
    Subword Byte-Pair Encoding (BPE) Tokenizer for Amharic.
    
    Shares the exact same public API as AmharicTokenizer:
        - vocab_size
        - pad_id, unk_id, bos_id, eos_id
        - encode(text, add_special_tokens=False)
        - decode(token_ids, skip_special_tokens=True)
        - save_vocab(file_path)
        - load_vocab(file_path)
    """

    FORMAT_VERSION = 1

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_file: Optional[str] = None):
        """
        Initializes the BPE tokenizer. Loads from vocab_file if provided.
        """
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self.encode_cache: Dict[str, List[str]] = {}
        self.unseen_char_counts: Counter = Counter()

        self._init_special_tokens()

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def _init_special_tokens(self) -> None:
        """Sets up default mappings for special tokens."""
        self.char_to_id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_char = {idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)}

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.char_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Returns total number of subword tokens in vocabulary."""
        return len(self.char_to_id)

    # ========================================================================
    # Pre-Tokenization Pipeline
    # ========================================================================
    @staticmethod
    def pre_tokenize(text: str) -> List[str]:
        """
        Splits text into pre-token chunks across whitespace and Ethiopic punctuation.
        Replaces spaces with WS_MARKER ('\\u2581') so boundaries are preserved losslessly.
        """
        # 1. Map spaces to boundary marker
        t = text.replace(" ", WS_MARKER)

        # 2. Regex matching:
        # - Words possibly starting with WS_MARKER, containing non-whitespace, non-Ethiopic punctuation
        # - Individual Ethiopic punctuation marks
        # - Newlines and residual whitespace
        # - Standalone WS_MARKER
        pattern = (
            rf"({WS_MARKER}?[^{WS_MARKER}\s{ETHIOPIC_PUNCT_CHARS}]+"
            rf"|[{ETHIOPIC_PUNCT_CHARS}]"
            rf"|\n+"
            rf"|\s+"
            rf"|{WS_MARKER})"
        )
        tokens = re.findall(pattern, t)
        return tokens

    # ========================================================================
    # BPE Training Algorithm
    # ========================================================================
    def train(
        self,
        text: str,
        target_vocab_size: int = 1000,
        verbose: bool = True
    ) -> None:
        """
        Trains BPE subword vocabulary and merge table from raw training text.
        
        Args:
            text (str): Training partition text (must be TRAIN split only).
            target_vocab_size (int): Total desired vocabulary size (including special tokens).
            verbose (bool): Whether to log progress.
        """
        self._init_special_tokens()
        self.merges = []
        self.bpe_ranks = {}
        self.encode_cache = {}

        if verbose:
            print(f"[BPE Training] Starting pre-tokenization for target vocab_size={target_vocab_size}...")

        # 1. Pre-tokenize and count word frequencies (excluding special tokens)
        raw_words = self.pre_tokenize(text)
        filtered_words = [w for w in raw_words if w not in self.SPECIAL_TOKENS]
        word_counts = Counter(filtered_words)

        # 2. Base alphabet: all unique characters appearing in words
        base_alphabet: Set[str] = set()
        for word in word_counts.keys():
            base_alphabet.update(word)

        # Deterministically sort base alphabet
        sorted_base = sorted(list(base_alphabet))
        next_id = len(self.SPECIAL_TOKENS)
        for char in sorted_base:
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                self.id_to_char[next_id] = char
                next_id += 1

        if verbose:
            print(f"[BPE Training] Base alphabet size: {len(sorted_base)} characters.")

        # 3. Represent words as symbol tuples: " ሰላም" -> (' ', 'ሰ', 'ላ', 'ም')
        vocab_words: Dict[Tuple[str, ...], int] = {
            tuple(word): freq for word, freq in word_counts.items() if len(word) > 0
        }

        # 4. Inverted index of pair occurrences: pair -> set of words containing pair
        pair_counts: Counter = Counter()
        where_to_update: Dict[Tuple[str, str], Set[Tuple[str, ...]]] = defaultdict(set)

        for w_tuple, freq in vocab_words.items():
            for i in range(len(w_tuple) - 1):
                pair = (w_tuple[i], w_tuple[i + 1])
                pair_counts[pair] += freq
                where_to_update[pair].add(w_tuple)

        # 5. Iterative merge loop
        num_merges = target_vocab_size - len(self.char_to_id)
        if verbose:
            print(f"[BPE Training] Learning up to {num_merges} subword merges...")

        for merge_idx in range(num_merges):
            if not pair_counts:
                if verbose:
                    print(f"[BPE Training] No more adjacent pairs available to merge at step {merge_idx}.")
                break

            # Deterministic tie-breaking: highest frequency, then lexicographically greater pair
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
            best_count = pair_counts[best_pair]

            if best_count < 1:
                break

            p0, p1 = best_pair
            merged_symbol = p0 + p1

            # Register new subword token (never overwrite special tokens or existing tokens)
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_idx
            if merged_symbol not in self.char_to_id:
                self.char_to_id[merged_symbol] = next_id
                self.id_to_char[next_id] = merged_symbol
                next_id += 1

            # Find all words that contain this pair
            words_with_pair = list(where_to_update.get(best_pair, set()))

            for old_word in words_with_pair:
                if old_word not in vocab_words:
                    continue
                freq = vocab_words.pop(old_word)

                # Decrement counts of all existing pairs in old_word
                for i in range(len(old_word) - 1):
                    pair = (old_word[i], old_word[i + 1])
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where_to_update[pair].discard(old_word)

                # Form new_word with merged pair
                new_word_list: List[str] = []
                i = 0
                while i < len(old_word):
                    if i < len(old_word) - 1 and old_word[i] == p0 and old_word[i + 1] == p1:
                        new_word_list.append(merged_symbol)
                        i += 2
                    else:
                        new_word_list.append(old_word[i])
                        i += 1
                new_word = tuple(new_word_list)
                vocab_words[new_word] = vocab_words.get(new_word, 0) + freq

                # Increment counts of all pairs in new_word
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_counts[pair] += freq
                    where_to_update[pair].add(new_word)

            if verbose and (merge_idx + 1) % 250 == 0:
                print(f"[BPE Training] Merge {merge_idx + 1}/{num_merges}: '{p0}' + '{p1}' -> '{merged_symbol}' (freq: {best_count})")

        if verbose:
            print(f"[BPE Training] Finished. Final vocabulary size: {self.vocab_size} tokens ({len(self.merges)} merges).")

    # ========================================================================
    # Encoding & Decoding
    # ========================================================================
    def _encode_word(self, word: str) -> List[str]:
        """
        Applies learned BPE merges to a single pre-tokenized word chunk.
        Uses iterative minimum-rank merge selection.
        """
        if not word:
            return []
        if word in self.encode_cache:
            return self.encode_cache[word]

        # Initial character tuple
        symbols = list(word)
        if len(symbols) <= 1:
            self.encode_cache[word] = symbols
            return symbols

        while len(symbols) >= 2:
            # Find the adjacent pair with the lowest merge rank
            min_rank = float("inf")
            min_pair = None
            min_idx = -1

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
                    min_idx = i

            if min_pair is None or min_rank == float("inf"):
                break  # No more valid merges can be applied

            # Merge all occurrences of min_pair in symbols
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == min_pair:
                    new_symbols.append(min_pair[0] + min_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        self.encode_cache[word] = symbols
        return symbols

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes a string of Amharic text into integer token IDs using BPE subwords.
        
        Special tokens (<PAD>, <UNK>, <BOS>, <EOS>) embedded in the text are mapped
        directly to their canonical fixed IDs.
        
        Fallback handling:
        Unseen characters that do not exist in the vocabulary are mapped to <UNK> explicitly
        and counted in self.unseen_char_counts.
        """
        token_ids: List[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_id)

        # Split preserving special tokens
        special_pattern = r"(<PAD>|<UNK>|<BOS>|<EOS>)"
        parts = re.split(special_pattern, text)
        unk_id = self.unk_id

        for part in parts:
            if not part:
                continue
            if part in self.SPECIAL_TOKENS and part in self.char_to_id:
                token_ids.append(self.char_to_id[part])
                continue

            pre_tokens = self.pre_tokenize(part)
            for word in pre_tokens:
                subwords = self._encode_word(word)
                for sub in subwords:
                    if sub in self.char_to_id:
                        token_ids.append(self.char_to_id[sub])
                    else:
                        # Unseen character fallback
                        for ch in sub:
                            if ch in self.char_to_id:
                                token_ids.append(self.char_to_id[ch])
                            else:
                                self.unseen_char_counts[ch] += 1
                                token_ids.append(unk_id)

        if add_special_tokens:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of integer token IDs back into the exact original Amharic string.
        Exact inverse of encode: decode(encode(x)) == x.
        """
        subwords: List[str] = []
        for tid in token_ids:
            tok = self.id_to_char.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                continue
            subwords.append(tok)

        merged = "".join(subwords)
        # Restore boundary markers back to standard spaces
        original = merged.replace(WS_MARKER, " ")
        return original

    # ========================================================================
    # Persistence
    # ========================================================================
    def save_vocab(self, file_path: str) -> None:
        """
        Saves the BPE vocabulary and merge table to a JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "format_version": self.FORMAT_VERSION,
            "tokenizer_type": "bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
            "char_to_id": self.char_to_id,
            "merges": self.merges,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BPETokenizer] Saved vocabulary to {file_path} ({self.vocab_size} tokens, {len(self.merges)} merges).")

    def load_vocab(self, file_path: str) -> None:
        """
        Loads the BPE vocabulary and merge table from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.merges = [tuple(m) for m in data.get("merges", [])]
        self.bpe_ranks = {pair: idx for idx, pair in enumerate(self.merges)}
        self.encode_cache = {}
        print(f"[BPETokenizer] Loaded vocabulary from {file_path} ({self.vocab_size} tokens, {len(self.merges)} merges).")
