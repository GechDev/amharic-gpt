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
from typing import Optional, Tuple, Dict, List
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

        # 1. Token and Positional Embeddings
        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_embd=config.n_embd,
            rng=rng,
        )

        # 2. Stack of L Transformer Blocks
        self.blocks = [
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                ffn_dim=config.ffn_dim,
                rng=rng,
            )
            for _ in range(config.n_layer)
        ]

        # 3. Final LayerNorm before vocabulary projection
        self.ln_f = LayerNorm(n_embd=config.n_embd, eps=config.eps)

        # 4. Final Vocabulary Projection (LM Head)
        self.W_vocab = init_weights((config.n_embd, config.vocab_size), std=0.02, rng=rng)
        self.b_vocab = init_zeros((config.vocab_size,))

        # Parameter gradients
        self.grad_W_vocab = np.zeros_like(self.W_vocab)
        self.grad_b_vocab = np.zeros_like(self.b_vocab)

        # Cache for backward pass
        self.cached_final_hidden: Optional[np.ndarray] = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        Forward pass of the language model.
        
        Args:
            idx (np.ndarray): Integer token IDs of shape (B, T), where T <= block_size.
            
        Returns:
            logits (np.ndarray): Next-token scores of shape (B, T, vocab_size).
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}."
        )

        # 1. Embeddings: (B, T) -> (B, T, n_embd)
        x = self.embeddings.forward(idx)

        # 2. Sequential pass through L Transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # 3. Final LayerNorm
        x = self.ln_f.forward(x)
        self.cached_final_hidden = x

        # 4. Vocabulary Projection: (B, T, n_embd) @ (n_embd, vocab_size) + (vocab_size,)
        # Shape: (B, T, vocab_size)
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

        # 1. Gradients through LM Head: logits = hidden @ W_vocab + b_vocab
        hidden_flat = hidden.reshape(-1, D)
        d_logits_flat = d_logits.reshape(-1, V)

        self.grad_W_vocab += hidden_flat.T @ d_logits_flat
        self.grad_b_vocab += np.sum(d_logits_flat, axis=0)

        # Gradient w.r.t final normalized hidden states
        d_hidden = (d_logits_flat @ self.W_vocab.T).reshape(B, T, D)

        # 2. Backward through final LayerNorm
        d_x = self.ln_f.backward(d_hidden)

        # 3. Backward through Transformer blocks in reverse order
        for block in reversed(self.blocks):
            d_x = block.backward(d_x)

        # 4. Backward through Token and Positional Embeddings
        self.embeddings.backward(d_x)

    def num_parameters(self) -> int:
        """Returns total number of trainable parameters in the model."""
        params = self.get_params_and_grads()
        return sum(p.size for p, _ in params.values())

    def get_params_and_grads(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a dictionary of all parameter arrays and their corresponding gradient arrays.
        Keys are formatted with standard dot-notation (e.g. 'blocks.0.attn.W_q').
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
        params["lm_head.W"] = (self.W_vocab, self.grad_W_vocab)
        params["lm_head.b"] = (self.b_vocab, self.grad_b_vocab)

        return params

    def zero_grad(self) -> None:
        """Resets all parameter gradients to zero before computing a new backward pass."""
        self.embeddings.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.ln_f.zero_grad()
        self.grad_W_vocab.fill(0.0)
        self.grad_b_vocab.fill(0.0)
