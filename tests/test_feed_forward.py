"""
Unit tests and gradient checks for Feed-Forward Network.
"""

import numpy as np
import pytest
from src.feed_forward import FeedForward


def test_ffn_shapes():
    B, T, D, ffn_dim = 2, 4, 16, 64
    ffn = FeedForward(n_embd=D, ffn_dim=ffn_dim)
    x = np.random.randn(B, T, D).astype(np.float32)

    out = ffn.forward(x)
    assert out.shape == (B, T, D)


def test_ffn_backward_numerical():
    B, T, D, ffn_dim = 1, 2, 4, 8
    ffn = FeedForward(n_embd=D, ffn_dim=ffn_dim)

    x = np.random.randn(B, T, D).astype(np.float64)
    d_out = np.random.randn(B, T, D).astype(np.float64)

    # Cast parameters to float64
    ffn.W1 = ffn.W1.astype(np.float64)
    ffn.b1 = ffn.b1.astype(np.float64)
    ffn.W2 = ffn.W2.astype(np.float64)
    ffn.b2 = ffn.b2.astype(np.float64)

    _ = ffn.forward(x)
    dx_analytical = ffn.backward(d_out)

    # Numerical gradient check on input X
    eps = 1e-6
    dx_num = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                x_pos = x.copy()
                x_neg = x.copy()
                x_pos[b, t, d] += eps
                x_neg[b, t, d] -= eps

                out_pos = ffn.forward(x_pos)
                out_neg = ffn.forward(x_neg)

                dx_num[b, t, d] = np.sum((out_pos - out_neg) * d_out) / (2 * eps)

    assert np.allclose(dx_analytical, dx_num, atol=1e-5)
