"""
Unit tests for the AmharicTokenizer.
"""

import os
import tempfile
import pytest
from src.tokenizer import AmharicTokenizer


def test_tokenizer_build_vocab_and_encode_decode():
    text = "ሰላም ዓለም! 123 ኢትዮጵያ።"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)

    # All characters plus special tokens should be present
    assert tokenizer.vocab_size == len(set(text)) + len(AmharicTokenizer.SPECIAL_TOKENS)

    encoded = tokenizer.encode(text)
    assert len(encoded) == len(text)
    assert all(isinstance(i, int) for i in encoded)

    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_special_tokens():
    text = "ሰላም"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)

    encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
    assert encoded_with_special[0] == tokenizer.bos_id
    assert encoded_with_special[-1] == tokenizer.eos_id

    # Decoding without skipping special tokens
    decoded_raw = tokenizer.decode(encoded_with_special, skip_special_tokens=False)
    assert decoded_raw == f"<BOS>{text}<EOS>"

    # Decoding with skipping special tokens
    decoded_clean = tokenizer.decode(encoded_with_special, skip_special_tokens=True)
    assert decoded_clean == text


def test_unknown_token():
    text = "አማርኛ"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)

    # 'Z' is not in the vocabulary
    encoded = tokenizer.encode("አZ")
    assert encoded[1] == tokenizer.unk_id


def test_save_and_load_vocab():
    text = "የኢትዮጵያ ታሪክ አስደናቂ ነው።"
    tok1 = AmharicTokenizer()
    tok1.build_vocab(text)

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.json")
        tok1.save_vocab(vocab_path)

        tok2 = AmharicTokenizer(vocab_file=vocab_path)
        assert tok2.vocab_size == tok1.vocab_size
        assert tok2.encode(text) == tok1.encode(text)
        assert tok2.decode(tok2.encode(text)) == text
