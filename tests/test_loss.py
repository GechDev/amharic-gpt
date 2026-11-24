"""
Unit tests and gradient checks for Cross-Entropy Loss.
"""

import numpy as np
import pytest
from src.loss import cross_entropy_loss


def test_cross_entropy_finite_and_expected_value():
    B, T, V = 2, 3, 5
    # Uniform logits: loss should equal log(V)
    logits = np.zeros((B, T, V))
    targets = np.random.randint(0, V, size=(B, T))

    loss, d_logits = cross_entropy_loss(logits, targets)
    expected_loss = np.log(V)
    assert np.isclose(loss, expected_loss, atol=1e-5)
    assert np.isfinite(loss)
    assert loss > 0.0
    assert d_logits.shape == logits.shape


def test_cross_entropy_perfect_prediction():
    # If correct target logit is very large, loss approaches 0
    B, T, V = 1, 2, 4
    logits = np.zeros((B, T, V))
    targets = np.array([[1, 3]])
    logits[0, 0, 1] = 50.0
    logits[0, 1, 3] = 50.0

    loss, _ = cross_entropy_loss(logits, targets)
    assert np.isclose(loss, 0.0, atol=1e-4)


def test_cross_entropy_gradient_numerical():
    B, T, V = 1, 2, 3
    logits = np.random.randn(B, T, V).astype(np.float64)
    targets = np.array([[0, 2]], dtype=np.int64)

    loss, d_logits_analytical = cross_entropy_loss(logits, targets)

    # Numerical finite-difference gradient check
    eps = 1e-6
    d_logits_num = np.zeros_like(logits)
    for b in range(B):
        for t in range(T):
            for v in range(V):
                l_pos = logits.copy()
                l_neg = logits.copy()
                l_pos[b, t, v] += eps
                l_neg[b, t, v] -= eps

                loss_pos, _ = cross_entropy_loss(l_pos, targets)
                loss_neg, _ = cross_entropy_loss(l_neg, targets)

                d_logits_num[b, t, v] = (loss_pos - loss_neg) / (2 * eps)

    assert np.allclose(d_logits_analytical, d_logits_num, atol=1e-5)
