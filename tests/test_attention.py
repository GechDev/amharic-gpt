"""
Unit tests and gradient checks for Causal Self-Attention.
"""

import numpy as np
import pytest
from src.attention import CausalSelfAttention


def test_causal_attention_shapes_and_mask():
    B, T, D = 2, 4, 8
    attn = CausalSelfAttention(n_embd=D)
    x = np.random.randn(B, T, D).astype(np.float32)

    out = attn.forward(x)
    assert out.shape == (B, T, D)

    # Check that attention weights have zeros for future positions (above diagonal)
    weights = attn.cached_weights
    assert weights.shape == (B, T, T)
    for i in range(T):
        for j in range(i + 1, T):
            assert np.allclose(weights[:, i, j], 0.0)


def test_causal_attention_backward_gradient_numerical():
    B, T, D = 1, 3, 4
    attn = CausalSelfAttention(n_embd=D)

    x = np.random.randn(B, T, D).astype(np.float64)
    d_out = np.random.randn(B, T, D).astype(np.float64)

    # Cast parameters to float64 for high-precision finite difference check
    attn.W_q = attn.W_q.astype(np.float64)
    attn.b_q = attn.b_q.astype(np.float64)
    attn.W_k = attn.W_k.astype(np.float64)
    attn.b_k = attn.b_k.astype(np.float64)
    attn.W_v = attn.W_v.astype(np.float64)
    attn.b_v = attn.b_v.astype(np.float64)

    _ = attn.forward(x)
    dx_analytical = attn.backward(d_out)

    # Finite difference gradient check on input X
    eps = 1e-6
    dx_num = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                x_pos = x.copy()
                x_neg = x.copy()
                x_pos[b, t, d] += eps
                x_neg[b, t, d] -= eps

                out_pos = attn.forward(x_pos)
                out_neg = attn.forward(x_neg)

                dx_num[b, t, d] = np.sum((out_pos - out_neg) * d_out) / (2 * eps)

    assert np.allclose(dx_analytical, dx_num, atol=1e-5)
