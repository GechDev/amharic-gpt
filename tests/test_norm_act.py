"""
Unit tests and numerical gradient checks for LayerNorm and GELU.
"""

import numpy as np
import pytest
from src.normalization import LayerNorm
from src.activations import GELU


def test_layernorm_forward():
    B, T, D = 2, 4, 16
    ln = LayerNorm(n_embd=D)

    x = np.random.randn(B, T, D).astype(np.float64) * 5.0 + 3.0
    out = ln.forward(x)

    # Check that output along feature dimension has mean ~0 and var ~1
    mean = np.mean(out, axis=-1)
    var = np.var(out, axis=-1)

    assert np.allclose(mean, 0.0, atol=1e-5)
    assert np.allclose(var, 1.0, atol=1e-3)


def test_layernorm_backward_numerical():
    B, T, D = 2, 3, 4
    ln = LayerNorm(n_embd=D)
    # Set random weights for gamma and beta
    ln.gamma = np.random.uniform(0.5, 1.5, size=D).astype(np.float64)
    ln.beta = np.random.uniform(-0.5, 0.5, size=D).astype(np.float64)

    x = np.random.randn(B, T, D).astype(np.float64)
    d_out = np.random.randn(B, T, D).astype(np.float64)

    # Analytical gradient
    _ = ln.forward(x)
    dx_analytical = ln.backward(d_out)
    dgamma_analytical = ln.grad_gamma.copy()
    dbeta_analytical = ln.grad_beta.copy()

    # Numerical gradient check for x
    eps = 1e-6
    dx_num = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                x_pos = x.copy()
                x_neg = x.copy()
                x_pos[b, t, d] += eps
                x_neg[b, t, d] -= eps

                out_pos = ln.gamma * ((x_pos - np.mean(x_pos, axis=-1, keepdims=True)) / np.sqrt(np.var(x_pos, axis=-1, keepdims=True) + ln.eps)) + ln.beta
                out_neg = ln.gamma * ((x_neg - np.mean(x_neg, axis=-1, keepdims=True)) / np.sqrt(np.var(x_neg, axis=-1, keepdims=True) + ln.eps)) + ln.beta

                dx_num[b, t, d] = np.sum((out_pos - out_neg) * d_out) / (2 * eps)

    assert np.allclose(dx_analytical, dx_num, atol=1e-5)


def test_gelu_forward_and_backward():
    gelu = GELU()
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
    out = gelu.forward(x)

    # GELU(0) should be 0.0
    assert np.isclose(out[2], 0.0)
    # For large positive x, GELU(x) ~ x
    assert np.isclose(out[4], 3.0, atol=0.01)
    # For large negative x, GELU(x) ~ 0
    assert np.isclose(out[0], 0.0, atol=0.01)

    # Numerical gradient check
    d_out = np.random.randn(*x.shape).astype(np.float64)
    dx_analytical = gelu.backward(d_out)

    eps = 1e-6
    dx_num = np.zeros_like(x)
    for i in range(len(x)):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[i] += eps
        x_neg[i] -= eps
        out_pos = gelu.forward(x_pos)
        out_neg = gelu.forward(x_neg)
        dx_num[i] = np.sum((out_pos - out_neg) * d_out) / (2 * eps)

    assert np.allclose(dx_analytical, dx_num, atol=1e-5)
