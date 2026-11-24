"""
Determinism Tests:
Ensures reproducibility across model initialization, data sampling, and generation.
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.data import AmharicDataset
from src.tokenizer import AmharicTokenizer
from src.generate import generate


def test_seed_yields_identical_model_weights():
    """Identical seeds must yield bit-identical initial parameters."""
    config = GPTConfig.get_tiny_config(vocab_size=30)
    model1 = GPTLanguageModel(config, rng=np.random.RandomState(42))
    model2 = GPTLanguageModel(config, rng=np.random.RandomState(42))
    model3 = GPTLanguageModel(config, rng=np.random.RandomState(99))

    # model1 and model2 must be 100% bit-identical
    np.testing.assert_array_equal(model1.embeddings.token_emb.W, model2.embeddings.token_emb.W)
    np.testing.assert_array_equal(model1.blocks[0].attn.W_q, model2.blocks[0].attn.W_q)

    # model3 must differ
    assert not np.array_equal(model1.embeddings.token_emb.W, model3.embeddings.token_emb.W)


def test_seed_yields_identical_batch_sampling():
    """Identical seeds must yield identical batch sequences from AmharicDataset.get_batch."""
    data = list(range(100))
    ds = AmharicDataset(data, block_size=8)

    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    rng3 = np.random.RandomState(99)

    x1, y1 = ds.get_batch(batch_size=4, rng=rng1)
    x2, y2 = ds.get_batch(batch_size=4, rng=rng2)
    x3, y3 = ds.get_batch(batch_size=4, rng=rng3)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(x1, x3)


def test_seed_yields_identical_generation_samples():
    """Identical seeds must yield identical generated strings under sampling."""
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab("ሰላም ለኢትዮጵያ ሀገር።")
    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))

    sample1 = generate(model, tokenizer, prompt="ሰ", max_new_tokens=20, temperature=0.8, rng=np.random.RandomState(42))
    sample2 = generate(model, tokenizer, prompt="ሰ", max_new_tokens=20, temperature=0.8, rng=np.random.RandomState(42))
    sample3 = generate(model, tokenizer, prompt="ሰ", max_new_tokens=20, temperature=0.8, rng=np.random.RandomState(99))

    assert sample1 == sample2, "Generations with identical seed must match character-for-character"
    assert sample1 != sample3, "Generations with different seeds should differ"
