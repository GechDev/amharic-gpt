"""
Custom Character-Level Unicode-Aware Tokenizer for Amharic.

Why Character-Level Tokenization for Amharic?
---------------------------------------------
Amharic is written in the Ethiopic syllabary (ግዕዝ ፊደላት), where each character
represents a consonant-vowel syllable (an "abugida"). For example, 'ለ' (lä),
'ሉ' (lu), 'ሊ' (li), 'ላ' (la), 'ሌ' (le), 'ል' (l), 'ሎ' (lo).

Advantages for an Educational Model:
1. Compact Vocabulary: The entire Amharic syllabary, numerals, and punctuation
   fit into a compact vocabulary (~200 to 400 unique characters).
2. No Out-of-Vocabulary (OOV) explosion: Every conceivable Amharic word can be
   represented as a sequence of character tokens.
3. Completely Transparent: No complex BPE merge tables or external tokenizers
   (such as tiktoken or SentencePiece) are needed. The mapping is a direct 1:1
   bijection between Unicode code points and integer IDs.

Limitations:
1. Sequence Length: Representing words character-by-character results in longer
   token sequences than subword tokenization (e.g. 5-10 tokens per word).
2. Context Window: With a fixed context length (e.g. block_size=128), fewer words
   fit inside the attention window.
"""

import json
import os
from typing import List, Dict, Union, Optional


class AmharicTokenizer:
    """
    A custom, from-scratch character-level tokenizer for Amharic text.
    
    Special Tokens:
        <PAD>: Index 0 - Padding token used to align sequence lengths
        <UNK>: Index 1 - Unknown token used for characters not in vocabulary
        <BOS>: Index 2 - Beginning-of-sequence token
        <EOS>: Index 3 - End-of-sequence token
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_file: Optional[str] = None):
        """
        Initializes the tokenizer. If vocab_file is provided, loads vocabulary from it.
        """
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Initialize with special tokens
        self._init_special_tokens()

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def _init_special_tokens(self) -> None:
        """Sets up default mappings for special tokens."""
        self.char_to_id = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_char = {idx: token for idx, token in enumerate(self.SPECIAL_TOKENS)}

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
        """Returns the total number of tokens in the vocabulary."""
        return len(self.char_to_id)

    @property
    def vocab(self) -> Dict[str, int]:
        """Provides consistent dictionary interface mapping tokens/characters to IDs."""
        return self.char_to_id

    def build_vocab(self, text: str) -> None:
        """
        Builds the character-level vocabulary from training text.
        
        Extracts all unique characters, sorts them deterministically,
        and assigns an integer token ID to each character after the special tokens.
        """
        self._init_special_tokens()

        # Find all unique characters in the input text
        unique_chars = sorted(list(set(text)))

        next_id = len(self.SPECIAL_TOKENS)
        for char in unique_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                self.id_to_char[next_id] = char
                next_id += 1

        print(f"[Tokenizer] Vocabulary built: {self.vocab_size} tokens "
              f"({len(self.SPECIAL_TOKENS)} special, {self.vocab_size - len(self.SPECIAL_TOKENS)} regular).")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes a string of Amharic text into a list of integer token IDs.
        
        Args:
            text (str): Input text to encode.
            add_special_tokens (bool): If True, prepends <BOS> and appends <EOS>.
            
        Returns:
            List[int]: List of integer token IDs.
        """
        unk_id = self.unk_id
        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.bos_id)

        for char in text:
            token_ids.append(self.char_to_id.get(char, unk_id))

        if add_special_tokens:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of integer token IDs back into an Amharic string.
        
        Args:
            token_ids (List[int]): List of token IDs.
            skip_special_tokens (bool): If True, skips <PAD>, <BOS>, <EOS>, <UNK>.
            
        Returns:
            str: Decoded string.
        """
        chars: List[str] = []
        for tid in token_ids:
            char = self.id_to_char.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and char in self.SPECIAL_TOKENS:
                continue
            chars.append(char)
        return "".join(chars)

    def save_vocab(self, file_path: str) -> None:
        """
        Saves the vocabulary mapping to a JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved vocabulary to {file_path}")

    def load_vocab(self, file_path: str) -> None:
        """
        Loads the vocabulary mapping from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        # Invert mapping to reconstruct id_to_char
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        print(f"[Tokenizer] Loaded vocabulary from {file_path} ({self.vocab_size} tokens).")


if __name__ == "__main__":
    sample_text = "ሰላም እንደምን አላችሁ? ይህ የአማርኛ ቋንቋ ሞዴል ነው።"
    tok = AmharicTokenizer()
    tok.build_vocab(sample_text)
    encoded = tok.encode(sample_text)
    decoded = tok.decode(encoded)
    print("Original :", sample_text)
    print("Encoded  :", encoded[:15], "...")
    print("Decoded  :", decoded)
    assert sample_text == decoded, "Decode did not match original!"
    print("Tokenizer self-test passed!")
