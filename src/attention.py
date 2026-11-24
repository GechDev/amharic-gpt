"""
Causal Self-Attention mechanism implemented from scratch in pure NumPy.

Plain-English Explanation of Causal Self-Attention:
---------------------------------------------------
Self-attention allows every token in an Amharic sentence to dynamically look at
other tokens to gather context.

Three vectors are created for each token:
1. Query (Q): What the current token is looking for.
   "I am the verb 'አደረገ' (did/made); who is my subject?"
2. Key (K): What information the token holds.
   "I am the noun 'ተማሪው' (the student); I can be a subject."
3. Value (V): The actual contextual representation to be retrieved and combined.

The dot product Q @ K.T measures the compatibility/relevance between queries and keys.
We divide by sqrt(head_dim) so variance of the dot products does not explode with dimension.
Then we apply a causal mask: future tokens receive -1e9 so that after softmax,
the model places 0% attention on tokens that haven't occurred yet!
Finally, attention weights are multiplied by V to produce the context vector.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from src.math_utils import softmax, softmax_backward, create_causal_mask, init_weights, init_zeros


class CausalSelfAttention:
    """
    Causal Self-Attention layer.

    Mathematical Equations:
    -----------------------
    Given input X of shape (B, T, D):
      Q = X @ W_q + b_q
      K = X @ W_k + b_k
      V = X @ W_v + b_v

      scores = (Q @ K.T) / sqrt(d_k)
      masked_scores = scores + causal_mask
      weights = softmax(masked_scores, axis=-1)
      output = weights @ V

    Array Shapes:
    -------------
    X:             (B, T, D)
    W_q, W_k, W_v: (D, D)
    b_q, b_k, b_v: (D,)
    Q, K, V:       (B, T, D)
    scores:        (B, T, T)
    weights:       (B, T, T)
    output:        (B, T, D)
    """

    def __init__(self, n_embd: int, rng: Optional[np.random.RandomState] = None):
        self.n_embd = n_embd
        self.scale = 1.0 / np.sqrt(n_embd)

        # Projections for Q, K, V
        self.W_q = init_weights((n_embd, n_embd), std=0.02, rng=rng)
        self.b_q = init_zeros((n_embd,))
        self.W_k = init_weights((n_embd, n_embd), std=0.02, rng=rng)
        self.b_k = init_zeros((n_embd,))
        self.W_v = init_weights((n_embd, n_embd), std=0.02, rng=rng)
        self.b_v = init_zeros((n_embd,))

        # Parameter gradients
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_b_v = np.zeros_like(self.b_v)

        # Caches for backward pass
        self.cached_X: Optional[np.ndarray] = None
        self.cached_Q: Optional[np.ndarray] = None
        self.cached_K: Optional[np.ndarray] = None
        self.cached_V: Optional[np.ndarray] = None
        self.cached_weights: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass for single-head causal self-attention.
        """
        B, T, D = X.shape
        self.cached_X = X

        # Linear projections
        # (B, T, D) @ (D, D) -> (B, T, D)
        Q = X @ self.W_q + self.b_q
        K = X @ self.W_k + self.b_k
        V = X @ self.W_v + self.b_v

        self.cached_Q = Q
        self.cached_K = K
        self.cached_V = V

        # Scaled dot-product scores: Q @ K.T
        # (B, T, D) @ (B, D, T) -> (B, T, T)
        scores = (Q @ K.swapaxes(-1, -2)) * self.scale

        # Apply causal mask: prevents attending to future positions
        mask = create_causal_mask(T)
        masked_scores = scores + mask  # broadcast (T, T) to (B, T, T)

        # Attention weights via numerically stable softmax
        weights = softmax(masked_scores, axis=-1)
        self.cached_weights = weights

        # Context output: weighted sum of values
        # (B, T, T) @ (B, T, D) -> (B, T, D)
        out = weights @ V
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for causal self-attention.

        Derivation:
        -----------
        Given d_out = dL/d(output) of shape (B, T, D):
          1. output = weights @ V
             d_weights = d_out @ V.T
             dV = weights.T @ d_out
          2. weights = softmax(masked_scores)
             d_scores = softmax_backward(d_weights, weights)
             Future positions in mask receive 0 gradient.
          3. scores = (Q @ K.T) * scale
             dQ = (d_scores @ K) * scale
             dK = (d_scores.T @ Q) * scale
          4. Linear projections:
             dW_q = X.T @ dQ,  db_q = sum(dQ)
             dW_k = X.T @ dK,  db_k = sum(dK)
             dW_v = X.T @ dV,  db_v = sum(dV)
             dX = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T
        """
        assert self.cached_X is not None and self.cached_weights is not None, "Call forward first."

        X = self.cached_X
        Q = self.cached_Q
        K = self.cached_K
        V = self.cached_V
        weights = self.cached_weights
        B, T, D = X.shape

        # Step 1: Gradients through output = weights @ V
        # d_weights: (B, T, D) @ (B, D, T) -> (B, T, T)
        d_weights = d_out @ V.swapaxes(-1, -2)
        # dV: (B, T, T).T @ (B, T, D) -> (B, T, D)
        dV = weights.swapaxes(-1, -2) @ d_out

        # Step 2: Gradients through Softmax
        # d_scores: (B, T, T)
        d_scores = softmax_backward(d_weights, weights, axis=-1)

        # Step 3: Gradients through scores = (Q @ K.T) * scale
        # dQ: (B, T, T) @ (B, T, D) -> (B, T, D)
        dQ = (d_scores @ K) * self.scale
        # dK: (B, T, T).T @ (B, T, D) -> (B, T, D)
        dK = (d_scores.swapaxes(-1, -2) @ Q) * self.scale

        # Step 4: Gradients through linear projections
        # Flatten batch and time dimensions for matrix multiplication with weights
        X_flat = X.reshape(-1, D)    # (B*T, D)
        dQ_flat = dQ.reshape(-1, D)  # (B*T, D)
        dK_flat = dK.reshape(-1, D)  # (B*T, D)
        dV_flat = dV.reshape(-1, D)  # (B*T, D)

        # Parameter gradients
        self.grad_W_q += X_flat.T @ dQ_flat
        self.grad_b_q += np.sum(dQ_flat, axis=0)

        self.grad_W_k += X_flat.T @ dK_flat
        self.grad_b_k += np.sum(dK_flat, axis=0)

        self.grad_W_v += X_flat.T @ dV_flat
        self.grad_b_v += np.sum(dV_flat, axis=0)

        # Input gradient dX: sum of gradients coming from Q, K, and V paths
        dX_flat = dQ_flat @ self.W_q.T + dK_flat @ self.W_k.T + dV_flat @ self.W_v.T
        dX = dX_flat.reshape(B, T, D)
        return dX

    def get_params_and_grads(self, prefix: str = "") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        p = f"{prefix}." if prefix else ""
        return {
            f"{p}W_q": (self.W_q, self.grad_W_q),
            f"{p}b_q": (self.b_q, self.grad_b_q),
            f"{p}W_k": (self.W_k, self.grad_W_k),
            f"{p}b_k": (self.b_k, self.grad_b_k),
            f"{p}W_v": (self.W_v, self.grad_W_v),
            f"{p}b_v": (self.b_v, self.grad_b_v),
        }

    def zero_grad(self) -> None:
        self.grad_W_q.fill(0.0)
        self.grad_b_q.fill(0.0)
        self.grad_W_k.fill(0.0)
        self.grad_b_k.fill(0.0)
        self.grad_W_v.fill(0.0)
        self.grad_b_v.fill(0.0)


class CausalMultiHeadAttention:
    """
    Causal Multi-Head Self-Attention.

    Why Multi-Head Attention?
    -------------------------
    Instead of performing a single attention pooling across the full embedding
    dimension, Multi-Head Attention projects Q, K, and V into H independent
    subspaces (heads) each of dimension head_dim = n_embd // n_head.
    
    This allows different heads to simultaneously focus on different types of
    relationships in Amharic text:
      - Head 1: Syntactic agreement (e.g. subject-verb agreement 'እሷ' -> 'ሄደች')
      - Head 2: Prepositional dependencies ('በ...' or 'ከ...')
      - Head 3: Long-range semantic topic relations
      - Head 4: Immediate neighboring character/word transitions

    Architecture:
    -------------
    Input X (B, T, D)
      ↓
    Linear projections for Q, K, V -> (B, T, D)
      ↓
    Split into H heads: (B, H, T, head_dim)
      ↓
    Scaled dot-product attention per head:
      scores = (Q @ K.T) / sqrt(head_dim)  -> (B, H, T, T)
      scores = scores + causal_mask
      weights = softmax(scores, axis=-1)   -> (B, H, T, T)
      head_out = weights @ V               -> (B, H, T, head_dim)
      ↓
    Concatenate all H heads: (B, T, D)
      ↓
    Output projection: concat @ W_o + b_o  -> (B, T, D)
    """

    def __init__(self, n_embd: int, n_head: int, rng: Optional[np.random.RandomState] = None):
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})."

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Q, K, V projections
        self.W_q = init_weights((n_embd, n_embd), std=0.02, rng=rng)
        self.b_q = init_zeros((n_embd,))
        self.W_k = init_weights((n_embd, n_embd), std=0.02, rng=rng)
        self.b_k = init_zeros((n_embd,))
        self.W_v = init_weights((n_embd, n_embd), std=0.02, rng=rng)
        self.b_v = init_zeros((n_embd,))

        # Output projection W_o, b_o
        self.W_o = init_weights((n_embd, n_embd), std=0.02, rng=rng)
        self.b_o = init_zeros((n_embd,))

        # Gradients
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_b_v = np.zeros_like(self.b_v)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.grad_b_o = np.zeros_like(self.b_o)

        # Caches for backward pass
        self.cached_X: Optional[np.ndarray] = None
        self.cached_Q: Optional[np.ndarray] = None
        self.cached_K: Optional[np.ndarray] = None
        self.cached_V: Optional[np.ndarray] = None
        self.cached_weights: Optional[np.ndarray] = None
        self.cached_concat: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass for Multi-Head Causal Attention.
        
        Input  X: shape (B, T, D)
        Output:   shape (B, T, D)
        """
        B, T, D = X.shape
        H = self.n_head
        d_k = self.head_dim
        self.cached_X = X

        # Step 1: Project input to Q, K, V
        # Shape: (B, T, D)
        Q = X @ self.W_q + self.b_q
        K = X @ self.W_k + self.b_k
        V = X @ self.W_v + self.b_v

        # Step 2: Reshape and transpose into heads
        # (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        Q = Q.reshape(B, T, H, d_k).swapaxes(1, 2)
        K = K.reshape(B, T, H, d_k).swapaxes(1, 2)
        V = V.reshape(B, T, H, d_k).swapaxes(1, 2)

        self.cached_Q = Q
        self.cached_K = K
        self.cached_V = V

        # Step 3: Scaled dot-product attention per head
        # (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        scores = (Q @ K.swapaxes(-1, -2)) * self.scale

        # Step 4: Apply causal mask
        # Mask shape (T, T) broadcasts across B and H to (B, H, T, T)
        mask = create_causal_mask(T)
        masked_scores = scores + mask

        # Step 5: Softmax to get attention weights
        # Shape: (B, H, T, T)
        weights = softmax(masked_scores, axis=-1)
        self.cached_weights = weights

        # Step 6: Attention output per head
        # (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        head_out = weights @ V

        # Step 7: Concatenate heads
        # (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, D)
        concat = head_out.swapaxes(1, 2).reshape(B, T, D)
        self.cached_concat = concat

        # Step 8: Output linear projection
        # (B, T, D) @ (D, D) + (D,) -> (B, T, D)
        out = concat @ self.W_o + self.b_o
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for Multi-Head Causal Attention.
        
        Input d_out: upstream gradient of shape (B, T, D)
        Returns dX:  gradient w.r.t input X of shape (B, T, D)
        """
        assert self.cached_X is not None and self.cached_concat is not None, "Call forward first."

        X = self.cached_X
        Q = self.cached_Q          # (B, H, T, d_k)
        K = self.cached_K          # (B, H, T, d_k)
        V = self.cached_V          # (B, H, T, d_k)
        weights = self.cached_weights  # (B, H, T, T)
        concat = self.cached_concat    # (B, T, D)

        B, T, D = X.shape
        H = self.n_head
        d_k = self.head_dim

        # Step 8 Backward: Gradients through output projection: out = concat @ W_o + b_o
        concat_flat = concat.reshape(-1, D)
        d_out_flat = d_out.reshape(-1, D)

        self.grad_W_o += concat_flat.T @ d_out_flat
        self.grad_b_o += np.sum(d_out_flat, axis=0)

        # Gradient w.r.t concat
        d_concat = (d_out_flat @ self.W_o.T).reshape(B, T, D)

        # Step 7 Backward: Un-concatenate heads
        # (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        d_head_out = d_concat.reshape(B, T, H, d_k).swapaxes(1, 2)

        # Step 6 Backward: head_out = weights @ V
        # d_weights: (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        d_weights = d_head_out @ V.swapaxes(-1, -2)
        # dV: (B, H, T, T).T @ (B, H, T, d_k) -> (B, H, T, d_k)
        dV = weights.swapaxes(-1, -2) @ d_head_out

        # Step 5 Backward: Softmax
        # d_scores: (B, H, T, T)
        d_scores = softmax_backward(d_weights, weights, axis=-1)

        # Step 3 Backward: scores = (Q @ K.T) * scale
        # dQ: (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        dQ = (d_scores @ K) * self.scale
        # dK: (B, H, T, T).T @ (B, H, T, d_k) -> (B, H, T, d_k)
        dK = (d_scores.swapaxes(-1, -2) @ Q) * self.scale

        # Step 2 Backward: Combine heads back to (B, T, D)
        # (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, D)
        dQ = dQ.swapaxes(1, 2).reshape(B, T, D)
        dK = dK.swapaxes(1, 2).reshape(B, T, D)
        dV = dV.swapaxes(1, 2).reshape(B, T, D)

        # Step 1 Backward: Linear projections Q, K, V
        X_flat = X.reshape(-1, D)
        dQ_flat = dQ.reshape(-1, D)
        dK_flat = dK.reshape(-1, D)
        dV_flat = dV.reshape(-1, D)

        self.grad_W_q += X_flat.T @ dQ_flat
        self.grad_b_q += np.sum(dQ_flat, axis=0)

        self.grad_W_k += X_flat.T @ dK_flat
        self.grad_b_k += np.sum(dK_flat, axis=0)

        self.grad_W_v += X_flat.T @ dV_flat
        self.grad_b_v += np.sum(dV_flat, axis=0)

        # Input gradient dX
        dX_flat = dQ_flat @ self.W_q.T + dK_flat @ self.W_k.T + dV_flat @ self.W_v.T
        dX = dX_flat.reshape(B, T, D)
        return dX

    def get_params_and_grads(self, prefix: str = "") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        p = f"{prefix}." if prefix else ""
        return {
            f"{p}W_q": (self.W_q, self.grad_W_q),
            f"{p}b_q": (self.b_q, self.grad_b_q),
            f"{p}W_k": (self.W_k, self.grad_W_k),
            f"{p}b_k": (self.b_k, self.grad_b_k),
            f"{p}W_v": (self.W_v, self.grad_W_v),
            f"{p}b_v": (self.b_v, self.grad_b_v),
            f"{p}W_o": (self.W_o, self.grad_W_o),
            f"{p}b_o": (self.b_o, self.grad_b_o),
        }

    def zero_grad(self) -> None:
        self.grad_W_q.fill(0.0)
        self.grad_b_q.fill(0.0)
        self.grad_W_k.fill(0.0)
        self.grad_b_k.fill(0.0)
        self.grad_W_v.fill(0.0)
        self.grad_b_v.fill(0.0)
        self.grad_W_o.fill(0.0)
        self.grad_b_o.fill(0.0)
