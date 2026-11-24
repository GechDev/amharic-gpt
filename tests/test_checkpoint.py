"""
Unit tests for model checkpoint saving and loading.
"""

import os
import tempfile
import numpy as np
import pytest
from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.model import GPTLanguageModel
from src.utils import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build tokenizer and config
        sample_text = "ሰላም ለኢትዮጵያ ሀገራችን።"
        tokenizer = AmharicTokenizer()
        tokenizer.build_vocab(sample_text)

        config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
        model = GPTLanguageModel(config, rng=np.random.RandomState(42))

        # Forward pass on sample input to get reference output
        sample_input = np.array([[2, 4, 6, 8]], dtype=np.int64)
        reference_logits = model.forward(sample_input).copy()

        # Save checkpoint
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            step=100,
            val_loss=1.234,
            name="test_model",
        )

        # Load checkpoint into a new instance
        loaded_model, loaded_config, loaded_tokenizer, meta = load_checkpoint(
            checkpoint_dir=tmpdir,
            name="test_model",
        )

        assert meta["step"] == 100
        assert np.isclose(meta["val_loss"], 1.234)
        assert loaded_config.n_embd == config.n_embd
        assert loaded_tokenizer is not None
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

        # Verify loaded model produces EXACT same logits
        loaded_logits = loaded_model.forward(sample_input)
        assert np.allclose(reference_logits, loaded_logits, atol=1e-6)
