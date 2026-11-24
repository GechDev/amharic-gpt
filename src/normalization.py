"""
Layer Normalization implemented from scratch in pure NumPy.

What Does Layer Normalization Do?
---------------------------------
In deep neural networks, activations across layers can fluctuate in scale,
causing gradients to vanish or explode during backpropagation.

Unlike Batch Normalization (which normalizes across batch items and depends on
batch size), Layer Normalization normalizes independently across the feature
dimension for each token position:
  1. Computes mean and variance across the embedding dimension D for each token.
  2. Normalizes activations to zero-mean and unit-variance.
  3. Scales and shifts using learnable parameters gamma and beta.

Why Do Transformers Use LayerNorm?
----------------------------------
Transformers use LayerNorm because:
1. It is invariant to batch size: works identically for training and single-token inference.
2. It stabilizes the forward activations and backward gradient flow across deep stacks.
3. Pre-LayerNorm placement ensures the identity residual path remains unhindered.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class LayerNorm:
    """
    Layer Normalization module.

    Mathematical Equations:
    -----------------------
    Given input x of shape (..., D):
      mean     mu   = (1 / D) * sum_{i=1}^D x_i
      variance var  = (1 / D) * sum_{i=1}^D (x_i - mu)^2
      normalized x_hat = (x - mu) / sqrt(var + eps)
      output   y    = gamma * x_hat + beta

    Trainable Parameters:
      gamma: scale vector of shape (D,), initialized to 1.0
      beta:  shift vector of shape (D,), initialized to 0.0
    """

    def __init__(self, n_embd: int, eps: float = 1e-5, dtype: Any = np.float32):
        self.n_embd = n_embd
        self.dtype = np.dtype(dtype)
        self.eps = self.dtype.type(eps)

        # Trainable parameters
        self.gamma = np.ones(n_embd, dtype=self.dtype)
        self.beta = np.zeros(n_embd, dtype=self.dtype)

        # Gradients
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        # Cache for backward pass
        self.cached_x: Optional[np.ndarray] = None
        self.cached_x_hat: Optional[np.ndarray] = None
        self.cached_var: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for LayerNorm.

        Input  x: shape (B, T, D)
        Output y: shape (B, T, D)
        """
        self.cached_x = x

        # Mean across feature dimension (axis=-1)
        # Shape: (B, T, 1)
        mean = np.mean(x, axis=-1, keepdims=True)

        # Variance across feature dimension
        # Shape: (B, T, 1)
        var = np.var(x, axis=-1, keepdims=True)
        self.cached_var = var

        # Normalize to zero mean, unit variance
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        self.cached_x_hat = x_hat

        # Scale and shift with gamma and beta
        # Broadcasting: (B, T, D) * (D,) + (D,)
        out = self.gamma * x_hat + self.beta
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for LayerNorm.

        Mathematical Derivation:
        ------------------------
        Let D = n_embd, and inv_std = 1 / sqrt(var + eps).
        dL/dgamma = sum(d_out * x_hat)
        dL/dbeta  = sum(d_out)
        
        dL/dx_hat = d_out * gamma
        dL/dx = inv_std * [ dL/dx_hat - (1/D) * sum(dL/dx_hat) - (1/D) * x_hat * sum(dL/dx_hat * x_hat) ]

        Input  d_out: upstream gradient, shape (B, T, D)
        Output dx:    gradient w.r.t input x, shape (B, T, D)
        """
        assert self.cached_x_hat is not None and self.cached_var is not None, "Call forward first."

        D = self.n_embd
        x_hat = self.cached_x_hat
        inv_std = 1.0 / np.sqrt(self.cached_var + self.eps)

        # Accumulate parameter gradients across all batch and sequence dimensions
        self.grad_gamma += np.sum(d_out * x_hat, axis=(0, 1))
        self.grad_beta += np.sum(d_out, axis=(0, 1))

        # Gradient w.r.t normalized activations x_hat
        dx_hat = d_out * self.gamma

        # Efficient vectorized gradient w.r.t input x
        sum_dx_hat = np.sum(dx_hat, axis=-1, keepdims=True)
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)

        dx = inv_std * (dx_hat - (1.0 / D) * sum_dx_hat - (1.0 / D) * x_hat * sum_dx_hat_x_hat)
        return dx

    def get_params_and_grads(self, prefix: str = "") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        p = f"{prefix}." if prefix else ""
        return {
            f"{p}gamma": (self.gamma, self.grad_gamma),
            f"{p}beta": (self.beta, self.grad_beta),
        }

    def zero_grad(self) -> None:
        self.grad_gamma.fill(0.0)
        self.grad_beta.fill(0.0)
