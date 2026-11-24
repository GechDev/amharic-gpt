"""
Causal Leakage Tests:
Mathematically prove that future tokens CANNOT influence past token predictions.
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.model import GPTLanguageModel


def test_causal_leakage_full_model_exact_equality():
    """
    Rigorously tests the causal attention invariant (No Causal Leakage):
    Modifying tokens at future positions (t >= k) MUST NOT alter the model's logits
    at any earlier positions (t < k).
    
    Proof by Contradiction:
    If logits at position t change when modifying token at t + d (d > 0),
    then information has leaked backward in time, violating autoregressive causality.
    """
    rng = np.random.RandomState(42)
    vocab_size = 50
    block_size = 16
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=3,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPTLanguageModel(config, rng=rng)
    model.eval()

    # Base sequence of length 8
    seq_base = np.array([[5, 12, 23, 34, 18, 41, 7, 29]], dtype=np.int64)
    logits_base = model.forward(seq_base)

    # Cutoff position k = 4. Positions 0, 1, 2, 3 must remain untouched.
    # We alter positions 4, 5, 6, 7 to completely different tokens.
    seq_perturbed = seq_base.copy()
    seq_perturbed[0, 4:] = [1, 2, 3, 4]  # Change future tokens

    logits_perturbed = model.forward(seq_perturbed)

    # 1. Past positions (0 to 3) MUST BE 100% BIT-FOR-BIT IDENTICAL
    np.testing.assert_array_equal(
        logits_base[0, :4, :],
        logits_perturbed[0, :4, :],
        err_msg="Causal leakage detected! Changing future tokens altered logits at earlier positions."
    )

    # 2. Future positions (4 to 7) SHOULD differ because their input tokens changed
    diff_future = np.max(np.abs(logits_base[0, 4:, :] - logits_perturbed[0, 4:, :]))
    assert diff_future > 1e-3, "Perturbed future tokens did not affect future logits as expected."


def test_causal_leakage_single_token_mutation():
    """
    Tests that mutating only the very last token in a sequence leaves all preceding
    logits perfectly unchanged.
    """
    config = GPTConfig.get_tiny_config(vocab_size=30)
    model = GPTLanguageModel(config, rng=np.random.RandomState(123))
    model.eval()

    T = 10
    seq1 = np.random.randint(0, 30, size=(1, T), dtype=np.int64)
    seq2 = seq1.copy()
    # Mutate only the last token
    seq2[0, -1] = (seq1[0, -1] + 5) % 30

    l1 = model.forward(seq1)
    l2 = model.forward(seq2)

    # Logits at positions 0 .. T-2 must be exactly identical
    np.testing.assert_array_equal(
        l1[0, :-1, :],
        l2[0, :-1, :],
        err_msg="Mutating the final token changed logits at preceding positions!"
    )
