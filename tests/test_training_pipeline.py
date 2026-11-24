"""
Unit tests for Task 5: Production-quality training loop.
1. Learning rate schedule (linear warmup, cosine annealing, terminal clamp).
2. Gradient accumulation mathematical equivalence (loss & gradient scaling).
3. Full deterministic validation metrics (loss, perplexity, bits-per-character).
4. Resume-equivalence: 20 steps -> save -> resume 20 steps equals 40 steps straight (same seed).
5. Early stopping with patience.
6. Metrics logging to CSV and plotting script execution.
"""

import os
import tempfile
import numpy as np
import pytest

from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.data import AmharicDataset, create_datasets_from_text
from src.model import GPTLanguageModel
from src.optimizer import AdamW, get_cosine_schedule_with_warmup_lr
from src.loss import cross_entropy_loss
from src.train import train, evaluate_validation
from src.utils import load_checkpoint
from scripts.plot_curves import plot_training_curves


def test_lr_schedule_warmup_and_cosine():
    """Verify linear warmup and cosine decay properties."""
    max_iters = 100
    warmup_iters = 20
    lr_max = 1e-3
    lr_min = 1e-4

    # Step 0: start at 0.0
    assert get_cosine_schedule_with_warmup_lr(0, max_iters, warmup_iters, lr_max, lr_min) == 0.0

    # Step 10: exactly halfway through warmup
    lr_10 = get_cosine_schedule_with_warmup_lr(10, max_iters, warmup_iters, lr_max, lr_min)
    np.testing.assert_allclose(lr_10, 0.5 * lr_max, rtol=1e-5)

    # Step 20: exact peak
    lr_20 = get_cosine_schedule_with_warmup_lr(20, max_iters, warmup_iters, lr_max, lr_min)
    np.testing.assert_allclose(lr_20, lr_max, rtol=1e-5)

    # Step 60: halfway through cosine decay (20 + (100 - 20) / 2 = 60)
    # Cosine factor at 0.5 is 0.5 * (1 + cos(pi/2)) = 0.5
    lr_60 = get_cosine_schedule_with_warmup_lr(60, max_iters, warmup_iters, lr_max, lr_min)
    expected_mid = lr_min + 0.5 * (lr_max - lr_min)
    np.testing.assert_allclose(lr_60, expected_mid, rtol=1e-5)

    # Step 100: exact minimum
    lr_100 = get_cosine_schedule_with_warmup_lr(100, max_iters, warmup_iters, lr_max, lr_min)
    np.testing.assert_allclose(lr_100, lr_min, rtol=1e-5)

    # Beyond max_iters: clamped to min_lr
    lr_150 = get_cosine_schedule_with_warmup_lr(150, max_iters, warmup_iters, lr_max, lr_min)
    assert lr_150 == lr_min


def test_gradient_accumulation_mathematical_equivalence():
    """
    Verify that gradient accumulation over K micro-batches with (d_logits / K) scaling
    yields identical gradients to evaluating all micro-batches simultaneously.
    """
    config = GPTConfig.get_tiny_config(vocab_size=20)
    config.dropout = 0.0

    # Initial model
    model_accum = GPTLanguageModel(config, rng=np.random.RandomState(42))
    model_direct = GPTLanguageModel(config, rng=np.random.RandomState(42))

    # Two micro-batches of size 2
    X1 = np.array([[1, 2, 3, 4], [2, 3, 4, 1]], dtype=np.int64)
    Y1 = np.array([[2, 3, 4, 1], [3, 4, 1, 2]], dtype=np.int64)

    X2 = np.array([[3, 4, 1, 2], [4, 1, 2, 3]], dtype=np.int64)
    Y2 = np.array([[4, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)

    # Full combined batch of size 4
    X_full = np.concatenate([X1, X2], axis=0)
    Y_full = np.concatenate([Y1, Y2], axis=0)

    # 1. Direct pass on full batch of size 4
    model_direct.zero_grad()
    logits_full = model_direct.forward(X_full, training=True)
    loss_full, d_logits_full = cross_entropy_loss(logits_full, Y_full)
    model_direct.backward(d_logits_full)

    # 2. Accumulated pass on 2 micro-batches of size 2 (grad_accum_steps=2)
    model_accum.zero_grad()
    # Micro-batch 1
    logits1 = model_accum.forward(X1, training=True)
    loss1, d_logits1 = cross_entropy_loss(logits1, Y1)
    model_accum.backward(d_logits1 / 2.0)

    # Micro-batch 2
    logits2 = model_accum.forward(X2, training=True)
    loss2, d_logits2 = cross_entropy_loss(logits2, Y2)
    model_accum.backward(d_logits2 / 2.0)

    # Accumulated loss must match full loss
    mean_accum_loss = 0.5 * (loss1 + loss2)
    np.testing.assert_allclose(mean_accum_loss, loss_full, rtol=1e-5)

    # All parameter gradients must match
    params_direct = model_direct.get_params_and_grads()
    params_accum = model_accum.get_params_and_grads()

    for name in params_direct:
        _, grad_direct = params_direct[name]
        _, grad_accum = params_accum[name]
        np.testing.assert_allclose(
            grad_accum,
            grad_direct,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Gradient accumulation mismatch for parameter '{name}'"
        )


def test_deterministic_full_validation_metrics():
    """Verify deterministic validation evaluation produces identical metrics and correct BPC."""
    config = GPTConfig.get_tiny_config(vocab_size=30)
    config.block_size = 8
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))

    # Token sequence of length 50
    tokens = list(range(1, 25)) * 2 + [1, 2]
    raw_text = "አማርኛ የኢትዮጵያ ቋንቋ ነው። " * 3
    dataset = AmharicDataset(token_ids=tokens, block_size=config.block_size)

    # Run 1
    m1 = evaluate_validation(model, dataset, batch_size=2, val_raw_text=raw_text)
    # Run 2
    m2 = evaluate_validation(model, dataset, batch_size=2, val_raw_text=raw_text)

    # Determinism: bit-identical metrics
    for k in m1:
        assert m1[k] == m2[k], f"Metric {k} was not deterministic: {m1[k]} vs {m2[k]}"

    # Verify BPC and PPL mathematical relations
    val_loss = m1["val_loss"]
    assert val_loss > 0.0
    expected_ppl = np.exp(val_loss)
    np.testing.assert_allclose(m1["val_ppl"], expected_ppl, rtol=1e-5)

    expected_bpt = val_loss / np.log(2.0)
    np.testing.assert_allclose(m1["val_bpt"], expected_bpt, rtol=1e-5)

    expected_bpc = expected_bpt * (len(tokens) / len(raw_text))
    np.testing.assert_allclose(m1["val_bpc"], expected_bpc, rtol=1e-5)


def test_resume_training_equivalence():
    """
    Acceptance Criterion:
    Train 20 steps -> save -> resume 20 steps equals training 40 steps straight
    (using the same seed) to floating-point tolerance.
    """
    raw_text = "ይህ ለአማርኛ ጂፒቲ ሞዴል የተዘጋጀ የሙከራ ጽሑፍ ነው። ሰላም ለዓለም ይሁን። " * 15
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(raw_text)

    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    config.block_size = 16
    config.dropout = 0.0  # Deterministic test

    # -------------------------------------------------------------
    # Run A: Train 40 steps straight
    # -------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp_a:
        train_ds_a, val_ds_a = create_datasets_from_text(raw_text, raw_text, tokenizer, block_size=config.block_size)
        rng_a = np.random.RandomState(1337)
        model_a = GPTLanguageModel(config, rng=np.random.RandomState(1337))

        train(
            model=model_a,
            train_dataset=train_ds_a,
            val_dataset=val_ds_a,
            max_iters=40,
            batch_size=2,
            grad_accum_steps=1,
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_iters=5,
            eval_interval=10,
            output_dir=os.path.join(tmp_a, "outputs"),
            checkpoint_dir=os.path.join(tmp_a, "checkpoints"),
            rng=rng_a,
            verbose=False,
        )

        params_a = {k: p.copy() for k, (p, _) in model_a.get_params_and_grads().items()}

    # -------------------------------------------------------------
    # Run B: Train 20 steps -> Save -> Resume 20 steps (to step 40)
    # -------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp_b:
        ckpt_dir_b = os.path.join(tmp_b, "checkpoints")
        out_dir_b = os.path.join(tmp_b, "outputs")

        train_ds_b, val_ds_b = create_datasets_from_text(raw_text, raw_text, tokenizer, block_size=config.block_size)
        rng_b = np.random.RandomState(1337)
        model_b = GPTLanguageModel(config, rng=np.random.RandomState(1337))

        # First segment: steps 1..20
        train(
            model=model_b,
            train_dataset=train_ds_b,
            val_dataset=val_ds_b,
            max_iters=20,
            total_iters=40,
            batch_size=2,
            grad_accum_steps=1,
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_iters=5,
            eval_interval=10,
            output_dir=out_dir_b,
            checkpoint_dir=ckpt_dir_b,
            rng=rng_b,
            verbose=False,
        )

        # Second segment: resume from step 20 up to step 40
        # Fresh model instance restored from latest checkpoint
        model_b_resumed, _, _, _ = load_checkpoint(ckpt_dir_b, name="checkpoint_latest")

        train(
            model=model_b_resumed,
            train_dataset=train_ds_b,
            val_dataset=val_ds_b,
            max_iters=40,
            total_iters=40,
            batch_size=2,
            grad_accum_steps=1,
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_iters=5,
            eval_interval=10,
            output_dir=out_dir_b,
            checkpoint_dir=ckpt_dir_b,
            resume_from=os.path.join(ckpt_dir_b, "checkpoint_latest_meta.json"),
            rng=rng_b,
            verbose=False,
        )

        params_b = {k: p.copy() for k, (p, _) in model_b_resumed.get_params_and_grads().items()}

    # -------------------------------------------------------------
    # Compare weights between Run A (40 straight) and Run B (20+20 resumed)
    # -------------------------------------------------------------
    assert set(params_a.keys()) == set(params_b.keys())
    for name in sorted(params_a.keys()):
        p_a = params_a[name]
        p_b = params_b[name]
        max_diff = np.max(np.abs(p_a - p_b))
        assert max_diff < 1e-6, (
            f"Resume equivalence violated for parameter '{name}': max abs diff = {max_diff:.8e}"
        )


def test_early_stopping_halts_training():
    """Verify early stopping terminates when validation loss stops improving."""
    train_text = "ተማሪው መጽሐፉን በትጋት አነበበ። " * 15
    # Distinct validation domain so model quickly overfits train_text and val_loss stops improving
    val_text = "የጠፈር ምርምር ጣቢያ ወደ ህዋ ተመጠቀ። " * 15
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(train_text + val_text)

    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    config.learning_rate = 1e-2  # Aggressive learning rate to trigger fast overfitting
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))

    train_ds, val_ds = create_datasets_from_text(train_text, val_text, tokenizer, block_size=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        history = train(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            max_iters=200,
            batch_size=2,
            learning_rate=1e-2,
            eval_interval=5,
            early_stopping_patience=2,  # Stop after 2 consecutive non-improvements
            output_dir=tmpdir,
            verbose=False,
        )

        # Should halt well before step 200
        steps = history["steps"]
        assert len(steps) > 0
        assert steps[-1] < 200, f"Expected early stop before 200, but completed up to {steps[-1]}"


def test_plot_curves_script_execution():
    """Verify scripts/plot_curves.py reads metrics.csv and outputs a valid PNG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "metrics.csv")
        png_path = os.path.join(tmpdir, "test_curves.png")

        # Create dummy metrics CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,train_loss,val_loss,train_val_gap,val_ppl,val_bpc,lr,grad_norm,step_time_sec\n")
            f.write("1,3.50,3.60,+0.10,36.59,5.19,1.0e-4,1.20,0.02\n")
            f.write("2,3.20,,,,1.5e-4,1.15,0.02\n")
            f.write("3,3.00,3.10,+0.10,22.19,4.47,2.0e-4,1.10,0.02\n")
            f.write("4,2.80,,,,1.8e-4,1.05,0.02\n")
            f.write("5,2.60,2.75,+0.15,15.64,3.96,1.5e-4,0.98,0.02\n")

        plot_training_curves(csv_path, png_path, title="Unit Test Curves")
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 1000, "PNG file is empty or too small"
