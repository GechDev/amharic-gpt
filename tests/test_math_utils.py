"""
Unit tests for NumPy math utilities.
"""

import numpy as np
import pytest
from src.math_utils import softmax, softmax_backward, create_causal_mask, init_weights


def test_softmax_probabilities_sum_to_one():
    # Test 1D, 2D, and 3D inputs
    x_1d = np.array([2.0, 1.0, 0.1])
    p_1d = softmax(x_1d)
    assert np.allclose(np.sum(p_1d), 1.0)
    assert np.all(p_1d >= 0.0)

    # 3D tensor: (Batch, Time, Dim)
    x_3d = np.random.randn(2, 4, 10)
    p_3d = softmax(x_3d, axis=-1)
    sums = np.sum(p_3d, axis=-1)
    assert np.allclose(sums, np.ones((2, 4)))


def test_softmax_numerical_stability():
    # Large numbers that would overflow vanilla exp(1000) -> inf
    x_huge = np.array([1000.0, 1001.0, 999.0])
    p_huge = softmax(x_huge)
    assert not np.isnan(p_huge).any()
    assert not np.isinf(p_huge).any()
    assert np.allclose(np.sum(p_huge), 1.0)
    # The largest element (1001.0) must have the highest probability
    assert np.argmax(p_huge) == 1


def test_softmax_backward():
    # Numerical gradient check for softmax
    z = np.array([1.5, -0.5, 2.0], dtype=np.float64)
    probs = softmax(z)
    d_out = np.array([0.2, -0.4, 0.1], dtype=np.float64)

    # Analytical gradient
    d_z_analytical = softmax_backward(d_out, probs)

    # Finite difference numerical gradient
    eps = 1e-6
    d_z_numerical = np.zeros_like(z)
    for i in range(len(z)):
        z_pos = z.copy()
        z_neg = z.copy()
        z_pos[i] += eps
        z_neg[i] -= eps
        p_pos = softmax(z_pos)
        p_neg = softmax(z_neg)
        loss_pos = np.sum(p_pos * d_out)
        loss_neg = np.sum(p_neg * d_out)
        d_z_numerical[i] = (loss_pos - loss_neg) / (2 * eps)

    assert np.allclose(d_z_analytical, d_z_numerical, atol=1e-5)


def test_causal_mask():
    seq_len = 4
    mask = create_causal_mask(seq_len)
    assert mask.shape == (4, 4)

    # Diagonal and lower triangle must be 0.0
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert mask[i, j] == 0.0
            else:
                assert mask[i, j] == -1e9

    # When added to scores and passed through softmax, future tokens have probability 0.0
    scores = np.ones((4, 4)) + mask
    attn_weights = softmax(scores, axis=-1)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_weights[i, j] == 0.0
