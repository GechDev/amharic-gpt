"""
Configuration dataclasses and settings for the Amharic GPT model.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class GPTConfig:
    """
    Hyperparameter configuration for the GPT Language Model.
    
    Attributes:
        vocab_size (int): Size of the tokenizer vocabulary.
        block_size (int): Maximum sequence length (context window).
        n_embd (int): Embedding dimension (hidden state size).
        n_head (int): Number of attention heads.
        n_layer (int): Number of stacked Transformer blocks.
        ffn_dim (int): Inner dimension of Feed-Forward Network (typically 4 * n_embd).
        dropout (float): Dropout probability (0.0 for deterministic NumPy inference).
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): L2 regularization weight decay factor.
        beta1 (float): First moment exponential decay for Adam.
        beta2 (float): Second moment exponential decay for Adam.
        eps (float): Numerical stability constant (epsilon).
    """
    vocab_size: int = 256
    block_size: int = 128
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        # Validate that embedding dimension is divisible by number of heads
        assert self.n_embd % self.n_head == 0, (
            f"Embedding dimension ({self.n_embd}) must be divisible by "
            f"number of heads ({self.n_head})."
        )
        if self.ffn_dim is None or self.ffn_dim <= 0:
            self.ffn_dim = 4 * self.n_embd

    @property
    def head_dim(self) -> int:
        """Dimensionality of each individual attention head."""
        return self.n_embd // self.n_head

    @classmethod
    def get_tiny_config(cls, vocab_size: int = 128) -> "GPTConfig":
        """
        Tiny configuration for fast debugging, sanity tests, and verification.
        Computes quickly on CPU in pure NumPy.
        """
        return cls(
            vocab_size=vocab_size,
            block_size=32,
            n_embd=64,
            n_head=2,
            n_layer=2,
            ffn_dim=256,
            dropout=0.0,
            learning_rate=1e-3,
        )

    @classmethod
    def get_small_config(cls, vocab_size: int = 256) -> "GPTConfig":
        """
        Standard small configuration suitable for Google Colab or local CPU training.
        """
        return cls(
            vocab_size=vocab_size,
            block_size=128,
            n_embd=256,
            n_head=4,
            n_layer=4,
            ffn_dim=1024,
            dropout=0.0,
            learning_rate=3e-4,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GPTConfig":
        return cls(**data)

    def save(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> "GPTConfig":
        """Load configuration from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
