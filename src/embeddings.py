"""
Token and Positional Embeddings in Pure NumPy.

Why Does a Transformer Need Positional Information?
--------------------------------------------------
Self-attention is fundamentally a permutation-invariant operation:
it computes attention scores purely from pairwise dot products Q @ K.T.
Without positional embeddings, a sentence like "ውሻው ድመቷን አባረረ" (The dog chased the cat)
and "ድመቷ ውሻውን አባረረች" (The cat chased the dog) would produce identical representations
at corresponding tokens!

By adding learned positional vectors to token embeddings:
  x = Token_Embedding(token_id) + Positional_Embedding(position_id)
every token representation carries both:
  1. Semantic information (what character/token it is)
  2. Syntactic order information (where it appears in the sentence)
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from src.math_utils import init_weights
from src.activations import Dropout


class TokenEmbedding:
    """
    Learned Token Embedding Table.

    Mathematical Representation:
    -----------------------------
    Weight matrix W_token of shape (vocab_size, n_embd).
    For each token ID i in the input, row W_token[i] is selected:
      E_t = W_token[token_id_t]

    Array Shapes:
    -------------
    Input token_ids: shape (B, T), integer IDs in range [0, vocab_size - 1]
    Weight W:        shape (vocab_size, n_embd)
    Output:          shape (B, T, n_embd)
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        rng: Optional[np.random.RandomState] = None,
        dtype: Any = np.float32,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.dtype = np.dtype(dtype)
        # Initialize token embedding weights with N(0, 0.02)
        self.W = init_weights((vocab_size, n_embd), std=0.02, rng=rng, dtype=self.dtype)
        
        # Gradient cache
        self.grad_W = np.zeros_like(self.W)
        self.cached_ids: Optional[np.ndarray] = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward lookup: retrieves embedding vectors for given token IDs.
        """
        self.cached_ids = token_ids
        # NumPy advanced indexing selects rows corresponding to token IDs
        # Shape: (B, T, n_embd)
        return self.W[token_ids]

    def backward(self, d_out: np.ndarray) -> None:
        """
        Backward pass for token embedding table.
        Accumulates gradients into self.grad_W using np.add.at for duplicate token IDs.
        
        d_out shape: (B, T, n_embd)
        """
        assert self.cached_ids is not None, "Must call forward before backward."
        # Accumulate gradients into corresponding rows of dW
        # np.add.at correctly sums gradients when the same token appears multiple times
        np.add.at(self.grad_W, self.cached_ids, d_out)


class PositionalEmbedding:
    """
    Learned Positional Embedding Table.

    Mathematical Representation:
    -----------------------------
    Weight matrix W_pos of shape (block_size, n_embd).
    For position index t in [0, T-1], vector W_pos[t] is selected.

    Array Shapes:
    -------------
    Input seq_len T: integer <= block_size
    Weight W:        shape (block_size, n_embd)
    Output:          shape (1, T, n_embd) or (T, n_embd)
    """

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        rng: Optional[np.random.RandomState] = None,
        dtype: Any = np.float32,
    ):
        self.block_size = block_size
        self.n_embd = n_embd
        self.dtype = np.dtype(dtype)
        # Initialize position embedding weights with N(0, 0.02)
        self.W = init_weights((block_size, n_embd), std=0.02, rng=rng, dtype=self.dtype)
        
        # Gradient cache
        self.grad_W = np.zeros_like(self.W)
        self.cached_seq_len: int = 0

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Forward lookup: retrieves position vectors for positions [0, seq_len - 1].
        """
        assert seq_len <= self.block_size, (
            f"Sequence length ({seq_len}) exceeds maximum context length ({self.block_size})."
        )
        self.cached_seq_len = seq_len
        # Slice the first seq_len positions and expand dimensions for batch broadcasting
        # Shape: (1, seq_len, n_embd)
        return self.W[:seq_len][np.newaxis, :, :]

    def backward(self, d_out: np.ndarray) -> None:
        """
        Backward pass for positional embedding table.
        
        d_out shape: (B, T, n_embd)
        Sum over batch dimension B because the same position embeddings are shared across all batches.
        """
        # Sum upstream gradient across batch dimension B
        # Shape: (T, n_embd)
        d_pos = np.sum(d_out, axis=0)
        self.grad_W[:self.cached_seq_len] += d_pos


class GPTEmbeddings:
    """
    Combined Token + Positional Embeddings module.
    
    Forward:
      x = Dropout(TokenEmbedding(ids) + PositionalEmbedding(positions))
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        rng: Optional[np.random.RandomState] = None,
        dropout: float = 0.0,
        dtype: Any = np.float32,
    ):
        self.token_emb = TokenEmbedding(vocab_size, n_embd, rng=rng, dtype=dtype)
        self.pos_emb = PositionalEmbedding(block_size, n_embd, rng=rng, dtype=dtype)
        self.dropout = Dropout(p=dropout, rng=rng)

    def forward(self, token_ids: np.ndarray, training: bool = True) -> np.ndarray:
        """
        token_ids: shape (B, T)
        returns:   shape (B, T, n_embd)
        """
        B, T = token_ids.shape
        tok_out = self.token_emb.forward(token_ids)
        pos_out = self.pos_emb.forward(T)
        # Elementwise addition of token semantics and position index
        out = tok_out + pos_out
        return self.dropout.forward(out, training=training)

    def backward(self, d_out: np.ndarray) -> None:
        """
        d_out: shape (B, T, n_embd)
        Backpropagates gradient through dropout, then to both token and positional embedding tables.
        """
        d_out = self.dropout.backward(d_out)
        self.token_emb.backward(d_out)
        self.pos_emb.backward(d_out)

    def get_params_and_grads(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Returns parameter arrays and corresponding gradient arrays for optimization."""
        return {
            "wte": (self.token_emb.W, self.token_emb.grad_W),
            "wpe": (self.pos_emb.W, self.pos_emb.grad_W),
        }

    def zero_grad(self) -> None:
        """Zeros out gradients before a new backward pass."""
        self.token_emb.grad_W.fill(0.0)
        self.pos_emb.grad_W.fill(0.0)
