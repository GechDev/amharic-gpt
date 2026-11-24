"""
Unit tests for SGD, AdamW, and gradient clipping.
"""

import numpy as np
import pytest
from src.optimizer import SGD, AdamW, clip_grad_norm


def test_sgd_step():
    param = np.array([10.0, 20.0])
    grad = np.array([1.0, 2.0])
    params = {"w": (param, grad)}

    sgd = SGD(params, lr=0.1)
    sgd.step()

    # 10 - 0.1 * 1.0 = 9.9
    # 20 - 0.1 * 2.0 = 19.8
    assert np.allclose(param, np.array([9.9, 19.8]))


def test_adamw_step_and_state():
    param = np.array([[2.0, -1.0], [0.5, 1.5]])  # 2D weight matrix
    grad = np.array([[0.1, -0.2], [0.3, -0.4]])
    params = {"w": (param, grad)}

    adamw = AdamW(params, lr=1e-3, weight_decay=0.01)
    initial_param = param.copy()
    adamw.step()

    assert adamw.t == 1
    assert not np.allclose(param, initial_param)

    # State saving and loading
    state = adamw.state_dict()
    adamw2 = AdamW(params, lr=1e-3)
    adamw2.load_state_dict(state)
    assert adamw2.t == 1
    assert np.allclose(adamw2.m["w"], adamw.m["w"])
    assert np.allclose(adamw2.v["w"], adamw.v["w"])


def test_clip_grad_norm():
    p1 = np.zeros(2)
    g1 = np.array([3.0, 4.0])  # Norm = 5.0
    params = {"w": (p1, g1)}

    norm = clip_grad_norm(params, max_norm=2.5)
    assert np.isclose(norm, 5.0)

    # Gradient should be scaled down by 2.5 / 5.0 = 0.5
    new_norm = np.linalg.norm(g1)
    assert np.isclose(new_norm, 2.5)
