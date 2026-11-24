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


def test_overfit_single_batch_below_threshold():
    """
    Overfit test (Task 7 Requirement 3):
    A tiny model must reach loss below 0.1 on one repeated batch within N steps.
    """
    from src.optimizer import AdamW
    from src.loss import cross_entropy_loss

    rng = np.random.RandomState(42)
    vocab_size = 12
    block_size = 6
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=24,
        n_head=2,
        n_layer=2,
        ffn_dim=48,
        learning_rate=0.01,
        dropout=0.0,
    )
    model = GPTLanguageModel(config, rng=rng)
    optimizer = AdamW(model.get_params_and_grads(), lr=0.01)

    # Single fixed batch of shape (1, 6)
    X = np.array([[1, 4, 7, 2, 8, 3]], dtype=np.int64)
    Y = np.array([[4, 7, 2, 8, 3, 5]], dtype=np.int64)

    loss_history = []
    # Train on this single repeated batch for at most 100 steps
    for step in range(100):
        logits = model.forward(X, training=True)
        loss, d_logits = cross_entropy_loss(logits, Y)
        model.zero_grad()
        model.backward(d_logits)
        optimizer.step()
        loss_history.append(loss)
        if loss < 0.1:
            break

    final_loss = loss_history[-1]
    assert final_loss < 0.1, (
        f"Model failed to overfit single batch below 0.1 within 100 steps. Final loss: {final_loss:.4f}"
    )

