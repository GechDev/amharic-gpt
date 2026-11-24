"""
Unit tests and Overfitting Sanity Test for Training.

Requirement 29: Overfitting Sanity Test
---------------------------------------
Trains the model repeatedly on a tiny dataset.
The model should easily memorize it and the cross-entropy loss must decrease substantially.
This rigorously verifies that:
  Forward Pass + Loss + Analytical Backpropagation + Optimizer
are working correctly together without bugs.
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.data import AmharicDataset
from src.model import GPTLanguageModel
from src.train import train, estimate_loss


def test_overfitting_sanity_check():
    """
    Overfitting sanity test:
    Verify that a tiny model can overfit and memorize a small Amharic sentence.
    """
    text = "ሰላም ለኢትዮጵያ እና ለሕዝቧ ይሁን። ሰላም ለኢትዮጵያ እና ለሕዝቧ ይሁን።"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)

    token_ids = tokenizer.encode(text)
    block_size = 16
    dataset = AmharicDataset(token_ids, block_size=block_size)

    # Tiny configuration
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=32,
        n_head=2,
        n_layer=2,
        ffn_dim=64,
        learning_rate=3e-3,
    )
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))

    # Evaluate initial loss before training
    initial_loss = estimate_loss(model, dataset, eval_iters=5, batch_size=2)
    assert initial_loss > 2.0, f"Expected initial loss > 2.0, got {initial_loss}"

    # Train for 80 steps
    history = train(
        model=model,
        train_dataset=dataset,
        val_dataset=None,
        max_iters=80,
        batch_size=2,
        learning_rate=3e-3,
        eval_interval=20,
        verbose=False,
    )

    final_loss = history["train_loss"][-1]
    print(f"\n[Sanity Check] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")

    # Verify that loss decreased substantially (at least 60% reduction)
    assert final_loss < initial_loss * 0.4, (
        f"Loss did not decrease sufficiently: initial={initial_loss}, final={final_loss}"
    )
    assert np.isfinite(final_loss)
