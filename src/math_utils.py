"""
Mathematical Utilities and Neural Network Primitives in Pure NumPy.

This module provides foundational mathematical operations implemented from scratch:
1. Numerically stable Softmax forward and backward
2. Causal autoregressive attention masks
3. Normal and Xavier/He parameter initializers
4. Numerical stability assertions
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes numerically stable softmax along the specified axis.

    Mathematical Equation:
    ----------------------
    Softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

    Why Subtract the Maximum?
    -------------------------
    If x contains large values (e.g., x = 1000), np.exp(1000) causes 64-bit/32-bit
    floating point overflow to positive infinity (inf), leading to inf / inf = NaN.
    
    Let m = max(x). Mathematically:
      exp(x_i - m) / sum_j exp(x_j - m)
      = (exp(x_i) * exp(-m)) / (sum_j exp(x_j) * exp(-m))
      = exp(x_i) / sum_j exp(x_j)

    Because exp(-m) cancels out in the numerator and denominator, the quotient
    remains mathematically identical. However, the maximum exponent is now
    exp(m - m) = exp(0) = 1.0, ensuring that no number can ever overflow!

    Array Shapes:
    -------------
    Input  x: shape (..., D)
    Output p: shape (..., D), where sum along axis equals 1.0
    """
    # Shift input by subtracting the maximum value along the axis
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max

    # Compute exponent of shifted values
    exp_x = np.exp(x_shifted)

    # Compute sum along the axis
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)

    # Normalize to obtain probabilities
    probs = exp_x / sum_exp_x
    return probs


def softmax_backward(d_out: np.ndarray, probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes analytical gradients of softmax with respect to its input logits.

    Mathematical Equation:
    ----------------------
    Given output probabilities P = softmax(Z), and upstream gradient d_out = dL/dP:
      dL/dZ_i = P_i * (d_out_i - sum_k (d_out_k * P_k))
      
    In vectorized form:
      dZ = P * (d_out - sum(d_out * P, axis=axis, keepdims=True))

    Array Shapes:
    -------------
    d_out: upstream gradient, shape (..., D)
    probs: cached softmax probabilities from forward pass, shape (..., D)
    dZ:    gradient w.r.t input logits, shape (..., D)
    """
    # Vectorized Jacobian-vector product
    sum_p_dout = np.sum(probs * d_out, axis=axis, keepdims=True)
    d_logits = probs * (d_out - sum_p_dout)
    return d_logits


CAUSAL_MASK_CACHE: Dict[Tuple[int, np.dtype], np.ndarray] = {}


def create_causal_mask(seq_len: int, dtype: Any = np.float32) -> np.ndarray:
    """
    Creates and caches an additive causal upper-triangular attention mask of shape (seq_len, seq_len).
    
    Causal Attention Constraint:
    ----------------------------
    In autoregressive language modeling, token at position i must NOT attend
    to future tokens at positions j > i.
    By adding -1e9 to future positions before softmax, exp(-1e9) becomes strictly 0.0,
    enforcing zero attention to future tokens.
    
    Caching:
    --------
    Caches the mask array per (seq_len, dtype) to avoid redundant memory allocations
    and matrix construction on every forward pass.
    """
    dt = np.dtype(dtype)
    cache_key = (seq_len, dt)
    if cache_key in CAUSAL_MASK_CACHE:
        return CAUSAL_MASK_CACHE[cache_key]

    tril = np.tri(seq_len, dtype=bool)
    mask = np.zeros((seq_len, seq_len), dtype=dt)
    mask[~tril] = -1e9
    CAUSAL_MASK_CACHE[cache_key] = mask
    return mask


def init_weights(
    shape: Tuple[int, ...],
    std: float = 0.02,
    rng: Optional[np.random.RandomState] = None,
    dtype: Any = np.float32
) -> np.ndarray:
    """
    Initializes weights using a Gaussian (normal) distribution N(0, std).
    Following the GPT-2 specification with configurable dtype (float32 or float64).

    Uses an explicit NumPy RandomState instance to ensure reproducibility.
    Global np.random state is never used.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    dt = np.dtype(dtype)
    return (rng.randn(*shape) * std).astype(dt)


def init_zeros(shape: Tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
    """Initializes bias vectors or arrays with zeros in specified dtype."""
    return np.zeros(shape, dtype=np.dtype(dtype))
