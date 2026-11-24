"""
Unit tests for model checkpoint saving and loading.

Harden and verify:
- save -> load -> forward produces bit-identical logits
- missing parameter keys raise ValueError
- unexpected extra parameter keys raise ValueError
- parameter shape mismatches raise ValueError
- config hash tampering raises ValueError
- tokenizer vocabulary size mismatch raises ValueError
"""

import os
import json
import tempfile
import numpy as np
import pytest
from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.model import GPTLanguageModel
from src.utils import save_checkpoint, load_checkpoint


def _setup_sample_model(tmpdir):
    sample_text = "ሰላም ለኢትዮጵያ ሀገራችን።"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(sample_text)

    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))
    return tokenizer, config, model


def test_save_and_load_checkpoint_bit_identical():
    """Verify save -> load -> forward gives bit-identical logits (np.array_equal)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer, config, model = _setup_sample_model(tmpdir)

        sample_input = np.array([[2, 4, 6, 8]], dtype=np.int64)
        reference_logits = model.forward(sample_input).copy()

        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            step=100,
            val_loss=1.234,
            name="test_model",
        )

        loaded_model, loaded_config, loaded_tokenizer, meta = load_checkpoint(
            checkpoint_dir=tmpdir,
            name="test_model",
        )

        assert meta["step"] == 100
        assert np.isclose(meta["val_loss"], 1.234)
        assert meta["format_version"] == 1
        assert "config_hash" in meta
        assert loaded_config.n_embd == config.n_embd
        assert loaded_tokenizer is not None
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

        loaded_logits = loaded_model.forward(sample_input)
        # Bit-identical logits check
        assert np.array_equal(reference_logits, loaded_logits), (
            "Loaded model forward pass must produce bit-identical logits to original model."
        )


def test_load_checkpoint_missing_key():
    """Verify that missing parameter keys raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer, config, model = _setup_sample_model(tmpdir)
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            name="test_missing",
        )

        # Corrupt npz by removing one parameter
        npz_path = os.path.join(tmpdir, "test_missing.npz")
        with np.load(npz_path) as data:
            loaded_npz = dict(data)
        del loaded_npz["lm_head.W"]
        np.savez(npz_path, **loaded_npz)

        with pytest.raises(ValueError, match="Missing keys in checkpoint"):
            load_checkpoint(checkpoint_dir=tmpdir, name="test_missing")


def test_load_checkpoint_extra_key():
    """Verify that unexpected extra parameter keys raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer, config, model = _setup_sample_model(tmpdir)
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            name="test_extra",
        )

        npz_path = os.path.join(tmpdir, "test_extra.npz")
        with np.load(npz_path) as data:
            loaded_npz = dict(data)
        loaded_npz["unexpected_bonus_weight"] = np.zeros((10, 10))
        np.savez(npz_path, **loaded_npz)

        with pytest.raises(ValueError, match="Unexpected extra keys in checkpoint"):
            load_checkpoint(checkpoint_dir=tmpdir, name="test_extra")


def test_load_checkpoint_shape_mismatch():
    """Verify that parameter shape mismatches raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer, config, model = _setup_sample_model(tmpdir)
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            name="test_shape",
        )

        npz_path = os.path.join(tmpdir, "test_shape.npz")
        with np.load(npz_path) as data:
            loaded_npz = dict(data)
        # Replace lm_head.W with wrong shape
        loaded_npz["lm_head.W"] = np.zeros((config.n_embd, config.vocab_size + 5))
        np.savez(npz_path, **loaded_npz)

        with pytest.raises(ValueError, match=r"Shape mismatch for parameter 'lm_head\.W'"):
            load_checkpoint(checkpoint_dir=tmpdir, name="test_shape")


def test_load_checkpoint_hash_mismatch():
    """Verify that tampered config raises ValueError due to hash mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer, config, model = _setup_sample_model(tmpdir)
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            name="test_hash",
        )

        meta_path = os.path.join(tmpdir, "test_hash_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Tamper with n_layer in config without recomputing hash
        meta["config"]["n_layer"] = 99
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="Checkpoint configuration hash mismatch"):
            load_checkpoint(checkpoint_dir=tmpdir, name="test_hash")


def test_load_checkpoint_vocab_mismatch():
    """Verify that tokenizer vocab size mismatch with config raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer, config, model = _setup_sample_model(tmpdir)
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            name="test_vocab",
        )

        vocab_path = os.path.join(tmpdir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vdata = json.load(f)
        # Add a dummy token to vocab data to create size mismatch
        vdata["char_to_id"]["<DUMMY>"] = 9999
        vdata["vocab_size"] = len(vdata["char_to_id"])
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vdata, f)

        with pytest.raises(ValueError, match="Vocabulary size mismatch"):
            load_checkpoint(checkpoint_dir=tmpdir, name="test_vocab")
