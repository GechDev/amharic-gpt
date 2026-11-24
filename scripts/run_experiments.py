#!/usr/bin/env python3
"""
Ablation Experiments Runner (Task 8 Requirement 1).
Runs controlled comparative experiments with identical budgets and fixed seeds.

Experiments Grid:
-----------------
(a) Tokenization: Char-level vs BPE-500 vs BPE-1000
(b) Positional Embeddings: With vs Without (No-WPE)
(c) Architecture Grid: 2 layers vs 4 layers; 2 heads vs 4 heads
(d) Parameterization: Tied vs Untied LM Head; Dropout 0.0 vs 0.1
Seeds: Key baseline run on 2 distinct seeds (42 and 1337) to quantify stochastic noise.

Outputs:
--------
Saved to outputs/experiments.csv and reported in README.
"""

import os
import sys
import time
import csv
import json
from typing import Dict, Any, List
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.data import AmharicDataset
from src.model import GPTLanguageModel
from src.optimizer import AdamW, clip_grad_norm, get_cosine_schedule_with_warmup_lr
from src.loss import cross_entropy_loss
from src.train import evaluate_validation


def run_single_experiment(
    exp_id: str,
    name: str,
    tokenizer_type: str,
    target_vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    use_pos_emb: bool,
    tie_embeddings: bool,
    dropout: float,
    seed: int,
    max_steps: int = 40,
    batch_size: int = 4,
    block_size: int = 64,
    train_text: str = "",
    val_text: str = "",
) -> Dict[str, Any]:
    print(f"\n[{exp_id}] Running: {name} (seed={seed})...")
    rng = np.random.RandomState(seed)

    # 1. Initialize tokenizer
    if tokenizer_type == "char":
        tokenizer = AmharicTokenizer()
        tokenizer.build_vocab(train_text)
    else:
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, target_vocab_size=target_vocab_size, verbose=False)

    vocab_size = tokenizer.vocab_size

    # 2. Tokenize splits
    train_ids = tokenizer.encode(train_text, add_special_tokens=False)
    val_ids = tokenizer.encode(val_text, add_special_tokens=False)

    train_ds = AmharicDataset(train_ids, block_size=block_size)
    val_ds = AmharicDataset(val_ids, block_size=block_size)

    # 3. Model setup
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        ffn_dim=4 * n_embd,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
        use_pos_emb=use_pos_emb,
        learning_rate=5e-4,
    )
    model = GPTLanguageModel(config, rng=rng)
    param_count = model.num_parameters()
    optimizer = AdamW(model.get_params_and_grads(), lr=5e-4, weight_decay=0.01)

    # 4. Training loop
    model.train()
    step_times = []
    train_losses = []

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        lr = get_cosine_schedule_with_warmup_lr(step, max_steps, warmup_iters=5, lr_max=5e-4, lr_min=5e-5)
        optimizer.set_lr(lr)

        X, Y = train_ds.get_batch(batch_size, rng=rng)
        logits = model.forward(X, training=True)
        loss, d_logits = cross_entropy_loss(logits, Y)
        model.zero_grad()
        model.backward(d_logits)
        clip_grad_norm(model.get_params_and_grads(), max_norm=1.0)
        optimizer.step()

        step_times.append(time.perf_counter() - t0)
        train_losses.append(loss)

    # 5. Full validation evaluation
    val_metrics = evaluate_validation(
        model=model,
        val_dataset=val_ds,
        batch_size=batch_size,
        tokenizer_type=tokenizer_type,
        val_raw_text=val_text[:2000],
    )

    avg_step_time = float(np.mean(step_times))
    final_train_loss = float(train_losses[-1])
    val_loss = float(val_metrics["val_loss"])
    val_ppl = float(val_metrics["val_ppl"])
    val_bpc = float(val_metrics.get("val_bpc", val_loss / np.log(2.0)))

    print(f"    Params: {param_count:,} | Train Loss: {final_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | BPC: {val_bpc:.3f} | {avg_step_time:.3f}s/step")

    return {
        "exp_id": exp_id,
        "name": name,
        "tokenizer": f"{tokenizer_type}-{vocab_size}",
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "use_pos_emb": use_pos_emb,
        "tied": tie_embeddings,
        "dropout": dropout,
        "seed": seed,
        "params": param_count,
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(val_loss, 4),
        "val_ppl": round(val_ppl, 2),
        "val_bpc": round(val_bpc, 3),
        "step_time_sec": round(avg_step_time, 4),
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "experiments.csv")

    # Load real corpus splits
    train_path = os.path.join("data", "processed", "train.txt")
    val_path = os.path.join("data", "processed", "val.txt")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Processed splits not found, running prepare_corpus.py...")
        from scripts.prepare_corpus import main as prep_main
        prep_main()

    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read(150000)  # Use representative partition for consistent ablation benchmarking
    with open(val_path, "r", encoding="utf-8") as f:
        val_text = f.read(25000)

    experiments_plan = [
        # Baseline & Stochastic Seed Variance
        ("EXP-01", "Baseline (Seed 42)", "char", 0, 2, 2, 64, True, True, 0.0, 42),
        ("EXP-02", "Baseline (Seed 1337 - Noise check)", "char", 0, 2, 2, 64, True, True, 0.0, 1337),

        # (a) Tokenizer Comparison
        ("EXP-03", "BPE-300 Subword", "bpe", 300, 2, 2, 64, True, True, 0.0, 42),
        ("EXP-04", "BPE-600 Subword", "bpe", 600, 2, 2, 64, True, True, 0.0, 42),

        # (b) Positional Embeddings Ablation
        ("EXP-05", "Without Positional Embeddings (No-WPE)", "char", 0, 2, 2, 64, False, True, 0.0, 42),

        # (c) Architecture Grid: Depth & Heads
        ("EXP-06", "Deeper Model (4 Layers)", "char", 0, 4, 2, 64, True, True, 0.0, 42),
        ("EXP-07", "Wider Multi-Head (4 Heads)", "char", 0, 2, 4, 64, True, True, 0.0, 42),

        # (d) Weight Tying & Regularization
        ("EXP-08", "Untied Embeddings (Head != WTE)", "char", 0, 2, 2, 64, True, False, 0.0, 42),
        ("EXP-09", "With Dropout (p=0.1)", "char", 0, 2, 2, 64, True, True, 0.1, 42),
    ]

    results = []
    for plan in experiments_plan:
        res = run_single_experiment(
            exp_id=plan[0],
            name=plan[1],
            tokenizer_type=plan[2],
            target_vocab_size=plan[3],
            n_layer=plan[4],
            n_head=plan[5],
            n_embd=plan[6],
            use_pos_emb=plan[7],
            tie_embeddings=plan[8],
            dropout=plan[9],
            seed=plan[10],
            max_steps=40,
            batch_size=4,
            block_size=64,
            train_text=train_text,
            val_text=val_text,
        )
        results.append(res)

    # Save to outputs/experiments.csv
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Ablations Complete] Successfully exported all results to: {out_csv}")


if __name__ == "__main__":
    main()
