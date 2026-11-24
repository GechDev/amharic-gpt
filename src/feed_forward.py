"""
Transformer Feed-Forward Network (FFN) implemented from scratch in pure NumPy.

Intuition: Attention vs Feed-Forward Network:
---------------------------------------------
1. Attention is Token Communication:
   Attention allows tokens across different positions to interact and pool information
   from one another (e.g., matching verbs with subjects).
   
2. Feed-Forward Network is Position-Wise Transformation:
   The FFN applies two linear projections separated by a nonlinear GELU activation
   independently to each individual token vector:
     x -> Linear(D -> 4D) -> GELU -> Linear(4D -> D)
     
   While attention decides "where to look", the FFN decides "what to do with the gathered facts",
   transforming the token representations into richer semantic features.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from src.activations import GELU
from src.math_utils import init_weights, init_zeros


class FeedForward:
    """
    Position-Wise Feed-Forward Network.

    Mathematical Equations:
    -----------------------
    Given input X of shape (B, T, n_embd):
      hidden = X @ W1 + b1        # (B, T, ffn_dim)
      act    = GELU(hidden)       # (B, T, ffn_dim)
      output = act @ W2 + b2      # (B, T, n_embd)

    Trainable Parameters:
      W1: shape (n_embd, ffn_dim), initialized with N(0, 0.02)
      b1: shape (ffn_dim,), initialized with 0
      W2: shape (ffn_dim, n_embd), initialized with N(0, 0.02)
      b2: shape (n_embd,), initialized with 0
    """

    def __init__(self, n_embd: int, ffn_dim: Optional[int] = None, rng: Optional[np.random.RandomState] = None):
        if ffn_dim is None or ffn_dim <= 0:
            ffn_dim = 4 * n_embd

        self.n_embd = n_embd
        self.ffn_dim = ffn_dim

        # Weights and biases
        self.W1 = init_weights((n_embd, ffn_dim), std=0.02, rng=rng)
        self.b1 = init_zeros((ffn_dim,))
        self.W2 = init_weights((ffn_dim, n_embd), std=0.02, rng=rng)
        self.b2 = init_zeros((n_embd,))

        # GELU activation instance
        self.gelu = GELU()

        # Gradients
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

        # Caches for backward pass
        self.cached_X: Optional[np.ndarray] = None
        self.cached_hidden: Optional[np.ndarray] = None
        self.cached_act: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass for FFN.
        
        Input  X: shape (B, T, n_embd)
        Output:   shape (B, T, n_embd)
        """
        self.cached_X = X

        # First linear projection: n_embd -> ffn_dim
        # (B, T, n_embd) @ (n_embd, ffn_dim) + (ffn_dim,) -> (B, T, ffn_dim)
        hidden = X @ self.W1 + self.b1
        self.cached_hidden = hidden

        # Nonlinear GELU activation
        act = self.gelu.forward(hidden)
        self.cached_act = act

        # Second linear projection: ffn_dim -> n_embd
        # (B, T, ffn_dim) @ (ffn_dim, n_embd) + (n_embd,) -> (B, T, n_embd)
        out = act @ self.W2 + self.b2
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for FFN.
        
        Input d_out: upstream gradient of shape (B, T, n_embd)
        Returns dX:  gradient w.r.t input X of shape (B, T, n_embd)
        """
        assert self.cached_X is not None and self.cached_act is not None, "Call forward first."

        X = self.cached_X
        act = self.cached_act
        B, T, _ = X.shape

        # Flatten batch and time dimensions for 2D matrix multiplication
        X_flat = X.reshape(-1, self.n_embd)          # (B*T, n_embd)
        act_flat = act.reshape(-1, self.ffn_dim)      # (B*T, ffn_dim)
        d_out_flat = d_out.reshape(-1, self.n_embd)  # (B*T, n_embd)

        # Step 2 Backward: output = act @ W2 + b2
        self.grad_W2 += act_flat.T @ d_out_flat
        self.grad_b2 += np.sum(d_out_flat, axis=0)

        # Gradient w.r.t act
        d_act_flat = d_out_flat @ self.W2.T
        d_act = d_act_flat.reshape(B, T, self.ffn_dim)

        # Step 1 Backward: act = GELU(hidden)
        d_hidden = self.gelu.backward(d_act)
        d_hidden_flat = d_hidden.reshape(-1, self.ffn_dim)

        # First projection: hidden = X @ W1 + b1
        self.grad_W1 += X_flat.T @ d_hidden_flat
        self.grad_b1 += np.sum(d_hidden_flat, axis=0)

        # Gradient w.r.t input X
        dX_flat = d_hidden_flat @ self.W1.T
        dX = dX_flat.reshape(B, T, self.n_embd)
        return dX

    def get_params_and_grads(self, prefix: str = "") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        p = f"{prefix}." if prefix else ""
        return {
            f"{p}W1": (self.W1, self.grad_W1),
            f"{p}b1": (self.b1, self.grad_b1),
            f"{p}W2": (self.W2, self.grad_W2),
            f"{p}b2": (self.b2, self.grad_b2),
        }

    def zero_grad(self) -> None:
        self.grad_W1.fill(0.0)
        self.grad_b1.fill(0.0)
        self.grad_W2.fill(0.0)
        self.grad_b2.fill(0.0)
