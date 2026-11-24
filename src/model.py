"""
Full GPT Language Model Architecture in Pure NumPy.

Architecture Overview:
----------------------
The GPT model predicts the probability distribution of the next token
conditioned on the sequence of previous tokens.

Data Flow:
----------
Input Token IDs (B, T)
      ↓
Token Embeddings + Positional Embeddings  [B, T, n_embd]
      ↓
Transformer Block 0                       [B, T, n_embd]
Transformer Block 1                       [B, T, n_embd]
...
Transformer Block (L-1)                   [B, T, n_embd]
      ↓
Final Layer Normalization (ln_f)          [B, T, n_embd]
      ↓
Vocabulary Projection (lm_head)           [B, T, vocab_size]
      ↓
Logits for Next-Token Prediction
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from src.config import GPTConfig
from src.embeddings import GPTEmbeddings
from src.transformer_block import TransformerBlock
from src.normalization import LayerNorm
from src.math_utils import init_weights, init_zeros


class GPTLanguageModel:
    """
    GPT-Style Autoregressive Language Model in pure NumPy.
    
    Attributes:
        config (GPTConfig): Model hyperparameter configuration.
        embeddings (GPTEmbeddings): Combined token and position embedding module.
        blocks (List[TransformerBlock]): Stack of L Transformer blocks.
        ln_f (LayerNorm): Final layer normalization.
        W_vocab (np.ndarray): Vocabulary projection weights of shape (n_embd, vocab_size).
        b_vocab (np.ndarray): Vocabulary projection biases of shape (vocab_size,).
    """

    def __init__(self, config: GPTConfig, rng: Optional[np.random.RandomState] = None):
        self.config = config
        self.training: bool = True
        dt = config.np_dtype

        # GPT-2 style initialization scale for residual output projections:
        # Each Transformer block has 2 residual sublayers (attention and FFN).
        # Across L layers, 2L additions occur. To stabilize variance growth,
        # residual projections (W_o and W2) are scaled by 1 / sqrt(2 * n_layer).
        residual_scale = 1.0 / np.sqrt(2.0 * config.n_layer)

        # 1. Token and Positional Embeddings
        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_embd=config.n_embd,
            rng=rng,
            dropout=config.dropout,
            dtype=dt,
        )

        # 2. Stack of L Transformer Blocks
        self.blocks = [
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                ffn_dim=config.ffn_dim,
                rng=rng,
                residual_scale=residual_scale,
                dropout=config.dropout,
                ln_eps=config.ln_eps,
                dtype=dt,
            )
            for _ in range(config.n_layer)
        ]

        # 3. Final LayerNorm before vocabulary projection
        self.ln_f = LayerNorm(n_embd=config.n_embd, eps=config.ln_eps, dtype=dt)

        # 4. Final Vocabulary Projection (LM Head)
        # Weight tying (Lecture Slide 14 / Press & Wolf 2017):
        # When tie_embeddings is True, the output projection matrix is wte.T (reusing
        # token embeddings), reducing parameters by vocab_size * n_embd and improving
        # representation alignment between input tokens and output logits.
        if not config.tie_embeddings:
            self.W_vocab = init_weights((config.n_embd, config.vocab_size), std=0.02, rng=rng, dtype=dt)
            self.grad_W_vocab = np.zeros_like(self.W_vocab)
        else:
            self.W_vocab = None
            self.grad_W_vocab = None

        self.b_vocab = init_zeros((config.vocab_size,), dtype=dt)
        self.grad_b_vocab = np.zeros_like(self.b_vocab)

        # Cache for backward pass
        self.cached_final_hidden: Optional[np.ndarray] = None

    def train(self, mode: bool = True) -> "GPTLanguageModel":
        """Sets the model in training mode (enabling dropout)."""
        self.training = mode
        return self

    def eval(self) -> "GPTLanguageModel":
        """Sets the model in evaluation mode (disabling dropout)."""
        self.training = False
        return self

    def init_kv_caches(self) -> List[Any]:
        """
        Initializes per-layer Key-Value caches for incremental autoregressive decoding (Lecture Slide 15).
        """
        from src.attention import KVCache
        return [KVCache(max_seq_len=self.config.block_size) for _ in range(self.config.n_layer)]

    def forward(
        self,
        idx: np.ndarray,
        training: Optional[bool] = None,
        kv_caches: Optional[List[Any]] = None,
        start_pos: int = 0,
    ) -> np.ndarray:
        """
        Forward pass of the language model with optional per-layer Key-Value caching.
        
        Args:
            idx (np.ndarray): Integer token IDs of shape (B, T), where T <= block_size.
            training (Optional[bool]): Explicit override for training mode. If None,
                                      uses self.training.
            kv_caches (Optional[List[KVCache]]): Optional per-layer KV caches for incremental decoding.
            start_pos (int): Starting position offset for positional embeddings.
            
        Returns:
            logits (np.ndarray): Next-token scores of shape (B, T, vocab_size).
        """
        if training is None:
            training = self.training

        B, T = idx.shape
        if kv_caches is None:
            assert T <= self.config.block_size, (
                f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}."
            )

        # 1. Embeddings: (B, T) -> (B, T, n_embd)
        x = self.embeddings.forward(idx, start_pos=start_pos, training=training)

        # 2. Sequential pass through L Transformer blocks
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x = block.forward(
                x,
                training=training,
                kv_cache=layer_cache,
                start_pos=start_pos,
            )

        # 3. Final LayerNorm
        x = self.ln_f.forward(x)
        self.cached_final_hidden = x

        # 4. Vocabulary Projection (LM Head)
        # If tied, project using wte.T: (B, T, n_embd) @ (n_embd, vocab_size)
        if self.config.tie_embeddings:
            wte = self.embeddings.token_emb.W
            logits = x @ wte.T + self.b_vocab
        else:
            logits = x @ self.W_vocab + self.b_vocab

        return logits

    def backward(self, d_logits: np.ndarray) -> None:
        """
        Complete analytical backward pass through the full GPT architecture.
        
        Args:
            d_logits (np.ndarray): Upstream gradient dL/d(logits) of shape (B, T, vocab_size).
        """
        assert self.cached_final_hidden is not None, "Call forward before backward."

        B, T, V = d_logits.shape
        D = self.config.n_embd
        hidden = self.cached_final_hidden  # (B, T, D)

        # 1. Gradients through LM Head: logits = hidden @ W_head + b_vocab
        hidden_flat = hidden.reshape(-1, D)
        d_logits_flat = d_logits.reshape(-1, V)

        self.grad_b_vocab += np.sum(d_logits_flat, axis=0)

        if self.config.tie_embeddings:
            # When tied, LM head weights are wte.T.
            # Forward: logits = hidden @ wte.T + b_vocab
            # Gradient w.r.t wte (shape: (vocab_size, n_embd)):
            # d(wte) = d_logits_flat.T @ hidden_flat
            # We ACCUMULATE this gradient into the shared embedding gradient array:
            self.embeddings.token_emb.grad_W += d_logits_flat.T @ hidden_flat
            # Gradient w.r.t final normalized hidden states:
            d_hidden = (d_logits_flat @ self.embeddings.token_emb.W).reshape(B, T, D)
        else:
            self.grad_W_vocab += hidden_flat.T @ d_logits_flat
            d_hidden = (d_logits_flat @ self.W_vocab.T).reshape(B, T, D)

        # 2. Backward through final LayerNorm
        d_x = self.ln_f.backward(d_hidden)

        # 3. Backward through Transformer blocks in reverse order
        for block in reversed(self.blocks):
            d_x = block.backward(d_x)

        # 4. Backward through Token and Positional Embeddings
        # TokenEmbedding will accumulate its lookup gradients into self.embeddings.token_emb.grad_W
        self.embeddings.backward(d_x)

    def num_parameters(self) -> int:
        """Returns total number of trainable parameters in the model."""
        params = self.get_params_and_grads()
        return sum(p.size for p, _ in params.values())

    def get_params_and_grads(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a dictionary of all parameter arrays and their corresponding gradient arrays.
        Keys are formatted with standard dot-notation (e.g. 'blocks.0.attn.W_q').
        When tie_embeddings=True, lm_head.W is excluded as it is tied to wte.
        """
        params: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Embeddings
        params.update(self.embeddings.get_params_and_grads())

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            params.update(block.get_params_and_grads(f"blocks.{i}"))

        # Final LayerNorm
        params.update(self.ln_f.get_params_and_grads("ln_f"))

        # LM Head
        if not self.config.tie_embeddings:
            params["lm_head.W"] = (self.W_vocab, self.grad_W_vocab)
        params["lm_head.b"] = (self.b_vocab, self.grad_b_vocab)

        return params

    def zero_grad(self) -> None:
        """Resets all parameter gradients to zero before computing a new backward pass."""
        self.embeddings.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.ln_f.zero_grad()
        if not self.config.tie_embeddings and self.grad_W_vocab is not None:
            self.grad_W_vocab.fill(0.0)
        self.grad_b_vocab.fill(0.0)
