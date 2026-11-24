"""
Unit tests for the full GPT Language Model.
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.model import GPTLanguageModel


def test_model_forward_shapes():
    config = GPTConfig.get_tiny_config(vocab_size=30)
    model = GPTLanguageModel(config)

    B, T = 2, 8
    idx = np.random.randint(0, config.vocab_size, size=(B, T))

    logits = model.forward(idx)
    # Target shape: [B, T, vocab_size]
    assert logits.shape == (B, T, config.vocab_size)
    assert not np.isnan(logits).any()
    assert not np.isinf(logits).any()


def test_model_backward_and_gradients():
    config = GPTConfig.get_tiny_config(vocab_size=20)
    model = GPTLanguageModel(config)

    B, T = 2, 4
    idx = np.random.randint(0, config.vocab_size, size=(B, T))
    logits = model.forward(idx)

    d_logits = np.random.randn(*logits.shape).astype(np.float32)
    model.backward(d_logits)

    params = model.get_params_and_grads()
    assert len(params) > 0

    for name, (param, grad) in params.items():
        assert param.shape == grad.shape, f"Shape mismatch for {name}: {param.shape} vs {grad.shape}"
        assert not np.isnan(grad).any(), f"NaN found in gradient for {name}"

    # Verify zero_grad
    model.zero_grad()
    for name, (param, grad) in params.items():
        assert np.allclose(grad, 0.0), f"Grad not zeroed for {name}"


def test_model_num_parameters():
    config = GPTConfig.get_tiny_config(vocab_size=50)
    model = GPTLanguageModel(config)
    num_params = model.num_parameters()
    assert num_params > 0
    print(f"Tiny GPT parameters: {num_params:,}")
