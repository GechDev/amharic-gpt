"""
Unit tests and numerical verification for Task 4 model refinements:
1. Causal mask caching, strict zero future attention, and row-sum normalization.
2. Inverted dropout forward, backward, eval determinism, and zero-fraction statistics.
3. Weight tying forward, backward gradient accumulation, and parameter consistency.
4. Float64 finite-difference check of EVERY parameter group across the full model
   (tied and untied) with max relative error < 1e-5.
5. Parameter counts reporting for tiny and small configurations (tied vs untied).
"""

import tempfile
import numpy as np
import pytest

from src.config import GPTConfig
from src.math_utils import create_causal_mask, softmax, CAUSAL_MASK_CACHE
from src.activations import Dropout
from src.model import GPTLanguageModel
from src.loss import cross_entropy_loss
from src.utils import save_checkpoint, load_checkpoint


# -----------------------------------------------------------------------------
# 1. Causal Mask Validation
# -----------------------------------------------------------------------------

def test_causal_mask_caching_and_properties():
    """Verify mask caching, zero future weights, and row sum to 1.0."""
    T = 16
    # Check caching
    mask1 = create_causal_mask(T, dtype=np.float32)
    mask2 = create_causal_mask(T, dtype=np.float32)
    assert mask1 is mask2, "Mask should be retrieved from cache"
    assert (T, np.dtype(np.float32)) in CAUSAL_MASK_CACHE

    # Check mask values
    for i in range(T):
        for j in range(T):
            if j <= i:
                assert mask1[i, j] == 0.0
            else:
                assert mask1[i, j] == -1e9

    # Check softmax behavior: future positions strictly 0.0, rows sum to 1.0
    dummy_scores = np.random.randn(2, 4, T, T).astype(np.float32)
    masked_scores = dummy_scores + mask1
    weights = softmax(masked_scores, axis=-1)

    # Check row sum equals 1.0 everywhere
    row_sums = np.sum(weights, axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    # Check future attention weights are strictly 0.0
    for i in range(T):
        for j in range(i + 1, T):
            np.testing.assert_allclose(weights[:, :, i, j], 0.0, atol=1e-7)


# -----------------------------------------------------------------------------
# 2. Inverted Dropout Validation
# -----------------------------------------------------------------------------

def test_dropout_eval_mode_deterministic():
    """Verify eval mode acts as an exact deterministic identity mapping."""
    drop = Dropout(p=0.4, rng=np.random.RandomState(42))
    x = np.random.randn(10, 20).astype(np.float32)

    # In eval mode (training=False), output should be bit-identical to input
    out1 = drop.forward(x, training=False)
    out2 = drop.forward(x, training=False)
    assert np.array_equal(out1, x)
    assert np.array_equal(out2, x)
    assert drop.cached_mask is None

    # Backward in eval mode passes gradient unchanged
    d_out = np.random.randn(*x.shape).astype(np.float32)
    dx = drop.backward(d_out)
    assert np.array_equal(dx, d_out)


def test_dropout_p_zero_equals_eval():
    """Verify train mode with p=0.0 is an exact identity mapping."""
    drop = Dropout(p=0.0, rng=np.random.RandomState(42))
    x = np.random.randn(8, 16).astype(np.float32)

    out_train = drop.forward(x, training=True)
    out_eval = drop.forward(x, training=False)
    assert np.array_equal(out_train, x)
    assert np.array_equal(out_eval, x)


def test_dropout_zero_fraction_statistics():
    """Verify expected zero fraction matches p under training."""
    p = 0.3
    drop = Dropout(p=p, rng=np.random.RandomState(1234))
    x = np.ones((100, 1000), dtype=np.float32)  # 100,000 elements

    out = drop.forward(x, training=True)
    zero_fraction = float(np.mean(out == 0.0))
    # Expect approximately 30% zeros (tolerance 0.02)
    assert abs(zero_fraction - p) < 0.02, f"Expected ~{p}, got {zero_fraction}"

    # Verify scaling of surviving elements: 1 / (1 - p)
    survivors = out[out > 0.0]
    expected_scale = 1.0 / (1.0 - p)
    np.testing.assert_allclose(survivors, expected_scale, rtol=1e-5)


def test_dropout_numerical_gradient():
    """Verify analytical gradient matches finite difference for fixed dropout mask."""
    p = 0.25
    rng = np.random.RandomState(42)
    drop = Dropout(p=p, rng=rng)

    x = np.random.randn(4, 5).astype(np.float64)
    # Forward to fix the mask
    out = drop.forward(x, training=True)
    cached_mask = drop.cached_mask.copy()

    d_out = np.random.randn(*x.shape).astype(np.float64)
    dx_analytical = drop.backward(d_out)

    # Numerical gradient w.r.t x using fixed mask
    eps = 1e-6
    dx_numerical = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, j] += eps
            x_minus[i, j] -= eps

            # Compute scalar objective: sum(out * d_out)
            loss_plus = np.sum((x_plus * cached_mask) * d_out)
            loss_minus = np.sum((x_minus * cached_mask) * d_out)
            dx_numerical[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

    np.testing.assert_allclose(dx_analytical, dx_numerical, rtol=1e-5, atol=1e-7)


# -----------------------------------------------------------------------------
# 3. Weight Tying Validation
# -----------------------------------------------------------------------------

def test_weight_tying_forward_and_params():
    """Verify weight tying reduces parameters by vocab_size * n_embd and matches tied projection."""
    vocab_size = 50
    config_tied = GPTConfig.get_tiny_config(vocab_size=vocab_size)
    config_tied.tie_embeddings = True

    config_untied = GPTConfig.get_tiny_config(vocab_size=vocab_size)
    config_untied.tie_embeddings = False

    rng = np.random.RandomState(42)
    model_tied = GPTLanguageModel(config_tied, rng=rng)
    params_tied = model_tied.get_params_and_grads()

    rng = np.random.RandomState(42)
    model_untied = GPTLanguageModel(config_untied, rng=rng)
    params_untied = model_untied.get_params_and_grads()

    # Tied model should NOT have 'lm_head.W'
    assert "lm_head.W" not in params_tied
    assert "lm_head.b" in params_tied

    # Untied model MUST have 'lm_head.W'
    assert "lm_head.W" in params_untied
    assert "lm_head.b" in params_untied

    # Parameter count difference must equal exactly vocab_size * n_embd
    expected_diff = vocab_size * config_tied.n_embd
    actual_diff = model_untied.num_parameters() - model_tied.num_parameters()
    assert actual_diff == expected_diff, f"Expected {expected_diff} diff, got {actual_diff}"


def test_weight_tying_gradient_accumulation():
    """
    Verify that in tied mode, token_emb.grad_W accumulates gradients from
    both embedding lookup and the LM head projection.
    """
    config = GPTConfig.get_tiny_config(vocab_size=10)
    config.tie_embeddings = True
    config.dropout = 0.0
    config.dtype = "float64"

    model = GPTLanguageModel(config, rng=np.random.RandomState(42))
    idx = np.array([[1, 2, 3, 2]], dtype=np.int64)

    logits = model.forward(idx)
    d_logits = np.random.randn(*logits.shape).astype(np.float64)

    # Perform full backward
    model.zero_grad()
    model.backward(d_logits)

    grad_accumulated = model.embeddings.token_emb.grad_W.copy()

    # Verify manual accumulation:
    # 1. LM head contribution: d_logits_flat.T @ hidden_flat
    hidden_flat = model.cached_final_hidden.reshape(-1, config.n_embd)
    d_logits_flat = d_logits.reshape(-1, config.vocab_size)
    head_grad = d_logits_flat.T @ hidden_flat

    # 2. Lookup contribution: lookup gradient from backpropagating through model
    # The accumulated gradient must equal head_grad + lookup_grad
    # Therefore, grad_accumulated - head_grad must be non-zero for indices [1, 2, 3]
    # and exactly zero for unused token IDs [0, 4, 5, 6, 7, 8, 9]
    lookup_part = grad_accumulated - head_grad

    # Used tokens in sequence are 1, 2, 3
    used_tokens = {1, 2, 3}
    for tok_id in range(config.vocab_size):
        if tok_id not in used_tokens:
            np.testing.assert_allclose(
                lookup_part[tok_id],
                0.0,
                atol=1e-12,
                err_msg=f"Token {tok_id} was not in input but has lookup grad!",
            )
        else:
            assert np.linalg.norm(lookup_part[tok_id]) > 1e-6, f"Token {tok_id} should have non-zero lookup grad"


def test_tied_checkpoint_save_and_load():
    """Verify save_checkpoint and load_checkpoint work seamlessly for tied models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = GPTConfig.get_tiny_config(vocab_size=25)
        config.tie_embeddings = True
        model = GPTLanguageModel(config, rng=np.random.RandomState(42))

        idx = np.array([[0, 1, 2, 3]], dtype=np.int64)
        logits_orig = model.forward(idx)

        save_checkpoint(tmpdir, model, config, name="tied_ckpt")
        restored_model, restored_cfg, _, _ = load_checkpoint(tmpdir, name="tied_ckpt")

        assert restored_cfg.tie_embeddings is True
        assert "lm_head.W" not in restored_model.get_params_and_grads()

        logits_restored = restored_model.forward(idx)
        np.testing.assert_allclose(logits_orig, logits_restored, atol=1e-6)


# -----------------------------------------------------------------------------
# 4. Comprehensive Float64 Finite-Difference Check of EVERY Parameter Group
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_full_model_finite_difference_float64(tie_embeddings: bool):
    """
    Finite-difference check of EVERY parameter group of the full model
    (embeddings, attention Q/K/V/O, ln_1, ln_2, ln_f, ffn W1/b1/W2/b2, head tied/untied)
    in float64 precision.
    Verifies relative error is below 1e-5.
    """
    config = GPTConfig(
        vocab_size=8,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=2,
        ffn_dim=32,
        dropout=0.0,
        ln_eps=1e-5,
        tie_embeddings=tie_embeddings,
        dtype="float64",
    )

    rng = np.random.RandomState(1337)
    model = GPTLanguageModel(config, rng=rng)
    model.eval()

    B, T = 2, 4
    idx = np.array([[1, 2, 0, 3], [2, 1, 3, 0]], dtype=np.int64)
    target = np.array([[2, 0, 3, 1], [1, 3, 0, 2]], dtype=np.int64)

    # Forward pass and cross-entropy loss
    logits = model.forward(idx)
    loss, d_logits = cross_entropy_loss(logits, target)

    # Analytical backward pass
    model.zero_grad()
    model.backward(d_logits)

    params = model.get_params_and_grads()
    h = 1e-5

    # Verify every parameter group
    checked_groups = set()
    for name, (param, grad) in params.items():
        checked_groups.add(name)
        # Select sample coordinates to test (to keep test runtime under control)
        # We test up to 5 elements per parameter (corners, center, edges)
        flat_size = param.size
        indices = np.linspace(0, flat_size - 1, min(5, flat_size), dtype=int)

        flat_param = param.ravel()
        flat_grad = grad.ravel()

        for idx_flat in indices:
            orig_val = flat_param[idx_flat]

            # f(x + h)
            flat_param[idx_flat] = orig_val + h
            logits_pos = model.forward(idx)
            loss_pos, _ = cross_entropy_loss(logits_pos, target)

            # f(x - h)
            flat_param[idx_flat] = orig_val - h
            logits_neg = model.forward(idx)
            loss_neg, _ = cross_entropy_loss(logits_neg, target)

            # Restore original parameter
            flat_param[idx_flat] = orig_val

            num_grad = (loss_pos - loss_neg) / (2.0 * h)
            ana_grad = flat_grad[idx_flat]

            # Relative error (with standard 1e-5 scale threshold for small gradients)
            denom = max(1e-5, abs(ana_grad) + abs(num_grad))
            rel_err = abs(ana_grad - num_grad) / denom

            assert rel_err < 1e-5, (
                f"Gradient check failed for {name} at index {idx_flat}:\n"
                f"  analytical = {ana_grad:.8e}\n"
                f"  numerical  = {num_grad:.8e}\n"
                f"  rel_error  = {rel_err:.8e} (threshold 1e-5)"
            )

    # Ensure all required parameter groups were checked
    expected_prefixes = [
        "wte", "wpe",
        "blocks.0.ln_1.gamma", "blocks.0.ln_1.beta",
        "blocks.0.attn.W_q", "blocks.0.attn.b_q",
        "blocks.0.attn.W_k", "blocks.0.attn.b_k",
        "blocks.0.attn.W_v", "blocks.0.attn.b_v",
        "blocks.0.attn.W_o", "blocks.0.attn.b_o",
        "blocks.0.ln_2.gamma", "blocks.0.ln_2.beta",
        "blocks.0.ffn.W1", "blocks.0.ffn.b1",
        "blocks.0.ffn.W2", "blocks.0.ffn.b2",
        "ln_f.gamma", "ln_f.beta",
        "lm_head.b",
    ]
    if not tie_embeddings:
        expected_prefixes.append("lm_head.W")

    for expected in expected_prefixes:
        assert expected in checked_groups, f"Expected parameter {expected} was not checked!"


# -----------------------------------------------------------------------------
# 5. Parameter Counts Reporting (Tied vs Untied)
# -----------------------------------------------------------------------------

def test_parameter_counts_reporting():
    """Calculates and prints parameter counts for tiny and small configs (tied vs untied)."""
    configs = [
        ("Tiny", GPTConfig.get_tiny_config(vocab_size=128)),
        ("Small", GPTConfig.get_small_config(vocab_size=256)),
    ]

    print("\n" + "=" * 60)
    print(f"{'Config':<10} | {'Tied Params':<15} | {'Untied Params':<15} | {'Difference':<12}")
    print("-" * 60)

    for name, cfg in configs:
        cfg.tie_embeddings = True
        model_tied = GPTLanguageModel(cfg)
        tied_params = model_tied.num_parameters()

        cfg.tie_embeddings = False
        model_untied = GPTLanguageModel(cfg)
        untied_params = model_untied.num_parameters()

        diff = untied_params - tied_params
        assert diff == cfg.vocab_size * cfg.n_embd
        print(f"{name:<10} | {tied_params:<15,} | {untied_params:<15,} | {diff:<12,}")

    print("=" * 60)
