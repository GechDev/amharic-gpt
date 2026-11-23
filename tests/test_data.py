"""
Unit tests for the Next-Token Dataset and Batching Pipeline.
"""

import numpy as np
import pytest
from src.data import AmharicDataset


def test_next_token_shift():
    # Sequence: 10, 20, 30, 40, 50
    tokens = [10, 20, 30, 40, 50]
    block_size = 3
    dataset = AmharicDataset(tokens, block_size=block_size)

    # There should be len(tokens) - block_size = 5 - 3 = 2 items
    assert len(dataset) == 2

    # Item 0: input [10, 20, 30], target [20, 30, 40]
    x0, y0 = dataset[0]
    assert np.array_equal(x0, np.array([10, 20, 30]))
    assert np.array_equal(y0, np.array([20, 30, 40]))

    # Target is shifted by exactly 1 position
    assert np.array_equal(x0[1:], y0[:-1])

    # Item 1: input [20, 30, 40], target [30, 40, 50]
    x1, y1 = dataset[1]
    assert np.array_equal(x1, np.array([20, 30, 40]))
    assert np.array_equal(y1, np.array([30, 40, 50]))


def test_batch_shapes():
    tokens = list(range(100))
    block_size = 16
    batch_size = 4
    dataset = AmharicDataset(tokens, block_size=block_size)

    X, Y = dataset.get_batch(batch_size=batch_size)
    assert X.shape == (batch_size, block_size)
    assert Y.shape == (batch_size, block_size)
    assert X.dtype == np.int64
    assert Y.dtype == np.int64

    # Each row in Y must be the right-shift of corresponding row in X
    for b in range(batch_size):
        assert np.array_equal(X[b, 1:], Y[b, :-1])
