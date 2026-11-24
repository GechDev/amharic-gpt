"""
Unit Tests and Property-Based Verification for BPE Subword Tokenizer.

Verifies:
1. Hand-checkable toy example from docstring.
2. Determinism test (same corpus -> bit-identical merges).
3. Lossless roundtrip property: decode(encode(x)) == x across 1,000 sample lines.
4. Fallback for unseen characters (<UNK> behavior and counter).
5. Special tokens fixed IDs: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3.
6. Save and load vocabulary integrity test.
7. Interface parity with AmharicTokenizer.
"""

import os
import tempfile
import numpy as np
import pytest

from src.bpe_tokenizer import BPETokenizer, WS_MARKER
from src.tokenizer import AmharicTokenizer


def test_hand_checkable_toy_example():
    """
    Verifies the hand-checkable toy example documented in the module docstring:
    Input: "ሰላም ሰላም ሰላም ሰላም"
    Merges:
      1. ('ሰ', 'ላ') -> 'ሰላ'
      2. ('ሰላ', 'ም') -> 'ሰላም'
    """
    toy_corpus = "ሰላም ሰላም ሰላም ሰላም"
    tokenizer = BPETokenizer()
    # Base alphabet: <PAD>, <UNK>, <BOS>, <EOS>, ' ', 'ም', 'ላ', 'ሰ' (8 tokens)
    # Target vocab: 10 tokens (8 base + 2 merges)
    tokenizer.train(toy_corpus, target_vocab_size=10, verbose=False)

    assert len(tokenizer.merges) == 2
    # Verify exact merge sequence
    assert tokenizer.merges[0] == ("ሰ", "ላ")
    assert tokenizer.merges[1] == ("ሰላ", "ም")

    # Verify that "ሰላም" is encoded as a single subword token
    encoded = tokenizer.encode("ሰላም")
    assert len(encoded) == 1
    assert tokenizer.id_to_char[encoded[0]] == "ሰላም"
    assert tokenizer.decode(encoded) == "ሰላም"


def test_bpe_determinism():
    """Verify that training twice on the same corpus yields identical merges and vocab."""
    corpus = "ሰላም ለኢትዮጵያ ሀገራችን። ትምህርት ለሰው ልጅ እድገት ወሳኝ መሠረት ነው።"
    
    tok1 = BPETokenizer()
    tok1.train(corpus, target_vocab_size=40, verbose=False)

    tok2 = BPETokenizer()
    tok2.train(corpus, target_vocab_size=40, verbose=False)

    assert tok1.merges == tok2.merges
    assert tok1.char_to_id == tok2.char_to_id
    assert tok1.id_to_char == tok2.id_to_char


def test_lossless_roundtrip_property():
    """
    Property test: decode(encode(x)) == x across 1,000 generated lines from corpus phrases.
    """
    # Build training corpus containing all characters and punctuations
    phrases = [
        "ሰላም ለኢትዮጵያ ሀገራችን ይሁን።",
        "ትምህርት፣ ለሰው ልጅ እድገት ወሳኝ መሠረት ነው፤",
        "እውቀት ኃይል ነው፤ የተማረ ትውልድ ሀገሩን ይለውጣል!",
        "ሳይንስና ቴክኖሎጂ ለዘመናዊ እድገት አስፈላጊ ናቸው?",
        "የኮምፒውተር ሳይንስ የወደፊት ዓለምን የሚቀይር መስክ ነው።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
        "ኢትዮጵያ ጥንታዊ ታሪክ እና አስደናቂ ባህል ያላት ሀገር ናት።",
    ]
    train_corpus = "\n".join(phrases * 10)

    tokenizer = BPETokenizer()
    tokenizer.train(train_corpus, target_vocab_size=80, verbose=False)

    rng = np.random.RandomState(42)
    # Generate 1,000 test lines by combining phrases
    for i in range(1000):
        n_parts = rng.randint(1, 4)
        chosen = [rng.choice(phrases) for _ in range(n_parts)]
        line = " ".join(chosen)
        
        encoded = tokenizer.encode(line)
        decoded = tokenizer.decode(encoded)
        assert decoded == line, f"Lossless roundtrip failed on iteration {i}:\nOriginal: {line}\nDecoded:  {decoded}"


def test_unseen_character_fallback_and_reporting():
    """
    Verify that unseen characters at inference time fall back explicitly to <UNK>
    and are tracked in unseen_char_counts.
    """
    train_text = "ሰላም ለኢትዮጵያ"
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, target_vocab_size=30, verbose=False)

    # Character 'Z' and '9' are unseen in train_text
    test_text = "ሰላም Z 9"
    encoded = tokenizer.encode(test_text)
    
    assert tokenizer.unk_id in encoded
    assert tokenizer.unseen_char_counts["Z"] >= 1
    assert tokenizer.unseen_char_counts["9"] >= 1


def test_special_tokens_fixed_ids():
    """Verify special tokens have fixed canonical IDs across char and BPE tokenizers."""
    bpe = BPETokenizer()
    char_tok = AmharicTokenizer()

    assert bpe.pad_id == 0
    assert bpe.unk_id == 1
    assert bpe.bos_id == 2
    assert bpe.eos_id == 3

    assert bpe.pad_id == char_tok.pad_id
    assert bpe.unk_id == char_tok.unk_id
    assert bpe.bos_id == char_tok.bos_id
    assert bpe.eos_id == char_tok.eos_id


def test_save_and_load_vocab_bpe():
    """Verify JSON serialization and deserialization of BPE merges and vocabulary."""
    corpus = "ሰላም ለኢትዮጵያ ሀገራችን። ትምህርት ለሰው ልጅ እድገት ወሳኝ መሠረት ነው።"
    tok = BPETokenizer()
    tok.train(corpus, target_vocab_size=50, verbose=False)

    sample = "ትምህርት ለሰው ልጅ እድገት ወሳኝ ነው።"
    ref_encoded = tok.encode(sample)

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file = os.path.join(tmpdir, "bpe_vocab.json")
        tok.save_vocab(vocab_file)

        loaded_tok = BPETokenizer(vocab_file=vocab_file)
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.merges == tok.merges
        assert loaded_tok.char_to_id == tok.char_to_id

        loaded_encoded = loaded_tok.encode(sample)
        assert ref_encoded == loaded_encoded
        assert loaded_tok.decode(loaded_encoded) == sample
