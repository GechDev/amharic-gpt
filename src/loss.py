"""
Numerically Stable Cross-Entropy Loss and Analytical Logit Gradients in Pure NumPy.

Mathematical Formulation:
--------------------------
In autoregressive language modeling, at each token position t in each batch b,
the model produces unnormalized logit scores:
  z in R^{vocab_size}
and the true next token is:
  y in {0, 1, ..., vocab_size - 1}

The probability assigned to class j is given by the Softmax function:
  p_j = exp(z_j) / sum_k exp(z_k)

The Cross-Entropy Loss for a single token position is:
  L = -log(p_y)
    = -log( exp(z_y) / sum_k exp(z_k) )
    = -z_y + log( sum_k exp(z_k) )

Numerical Stability (Log-Sum-Exp Trick):
----------------------------------------
Directly computing sum_k exp(z_k) can overflow floating-point representation
if z contains large positive numbers (e.g., z = 1000 -> exp(1000) = inf).

To prevent overflow, let m = max_k(z_k):
  log( sum_k exp(z_k) )
  = log( sum_k exp(z_k - m + m) )
  = log( exp(m) * sum_k exp(z_k - m) )
  = m + log( sum_k exp(z_k - m) )

Substituting back into the loss:
  L = m + log( sum_k exp(z_k - m) ) - z_y

Analytical Derivative w.r.t Logits:
-----------------------------------
dL/dz_j = p_j - 1  (if j == y, the correct class)
dL/dz_j = p_j      (if j != y)

When averaged across N = B * T valid tokens:
  dL/dz_j = (p_j - delta_{j, y}) / N
"""

import numpy as np
from typing import Tuple, Optional
from src.math_utils import softmax


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100
) -> Tuple[float, np.ndarray]:
    """
    Computes cross-entropy loss and its analytical gradient w.r.t. logits.

    Args:
        logits (np.ndarray): Unnormalized scores of shape (B, T, vocab_size).
        targets (np.ndarray): Integer ground-truth token IDs of shape (B, T).
        ignore_index (int): Optional index to mask out (e.g. padding).

    Returns:
        loss (float): Scalar mean cross-entropy loss over valid tokens.
        d_logits (np.ndarray): Analytical gradient dL/d(logits) of shape (B, T, vocab_size).
    """
    B, T, V = logits.shape
    assert targets.shape == (B, T), f"Targets shape {targets.shape} must match logits (B, T) = ({B}, {T})."

    # Step 1: Numerically stable Log-Sum-Exp
    # Shift logits by max along vocabulary axis
    max_logits = np.max(logits, axis=-1, keepdims=True)  # (B, T, 1)
    shifted_logits = logits - max_logits

    exp_shifted = np.exp(shifted_logits)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)  # (B, T, 1)
    log_sum_exp = np.log(sum_exp) + max_logits             # (B, T, 1)

    # Step 2: Extract target logit z_y
    # Use advanced integer indexing to extract logits at target indices
    # Shape of target_logits: (B, T)
    batch_idx = np.arange(B)[:, None]
    time_idx = np.arange(T)[None, :]
    target_logits = logits[batch_idx, time_idx, targets]

    # Per-token loss: L = log_sum_exp - z_y
    # Shape: (B, T)
    token_losses = (log_sum_exp.squeeze(-1) - target_logits)

    # Step 3: Handle masking (if ignore_index is used)
    mask = (targets != ignore_index)
    num_valid = np.sum(mask)

    if num_valid == 0:
        return 0.0, np.zeros_like(logits)

    loss = float(np.sum(token_losses * mask) / num_valid)

    # Step 4: Analytical gradient w.r.t logits
    # p = softmax(logits)
    probs = exp_shifted / sum_exp  # (B, T, V)

    # Gradient: (p_j - delta_{j, y}) / num_valid
    d_logits = probs.copy()
    d_logits[batch_idx, time_idx, targets] -= 1.0

    # Zero out gradients for ignored positions
    if not np.all(mask):
        d_logits[~mask] = 0.0

    # Average over total valid tokens
    d_logits /= num_valid

    return loss, d_logits
