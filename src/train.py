"""
Production-Quality Training and Validation Loop for Amharic GPT in Pure NumPy.

Architecture & Features:
------------------------
1. Learning Rate Schedule: Linear warmup followed by cosine annealing decay to min_lr.
2. Gradient Accumulation: Micro-batch gradient accumulation with mathematically correct
   1 / K loss and gradient scaling, paired with global-norm gradient clipping.
3. Deterministic Validation Evaluation: Evaluates over the FULL validation set in non-overlapping
   sequential chunks; computes validation loss, perplexity, bits-per-token, and bits-per-character (BPC).
   For BPE runs, normalizes metrics by character length so char and BPE runs are directly comparable.
4. Checkpointing & Resume: Saves best-validation checkpoint and latest checkpoint, capturing
   model weights, optimizer state (m, v, t, lr), RNG state, step, and configuration. Full resume support.
5. Early Stopping & Overfitting Indicator: Monitors validation loss with patience counter and
   reports train/validation loss generalization gap (Val - Train).
6. Metrics Logging: Logs metrics incrementally to CSV (outputs/<run>/metrics.csv) and config JSON.
7. CLI and JSON Configuration: Fully configurable with zero hard-coded paths.
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Optional, Dict, Any, Tuple, Union, List

from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.data import AmharicDataset, create_datasets_from_text, preprocess_file
from src.model import GPTLanguageModel
from src.loss import cross_entropy_loss
from src.optimizer import AdamW, clip_grad_norm, get_cosine_schedule_with_warmup_lr
from src.utils import save_checkpoint, load_checkpoint


def evaluate_validation(
    model: GPTLanguageModel,
    val_dataset: AmharicDataset,
    batch_size: int = 4,
    tokenizer_type: str = "char",
    val_raw_text: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluates the model deterministically across the FULL validation dataset.

    Sequential Chunk Evaluation:
    -----------------------------
    Instead of sampling random batches, this function divides the complete validation
    token stream into sequential, non-overlapping chunks of length block_size.
    Every token is evaluated exactly once in canonical sequence order.

    Metrics Computed:
    -----------------
    - val_loss: Cross-entropy loss in nats per token: L = - (1/N) sum log P(y_t | past)
    - val_ppl: Perplexity per token: exp(L)
    - val_bpt: Bits per token: L / ln(2)
    - val_bpc: Bits per character:
        For char tokenizer, 1 token = 1 char, so BPC = BPT.
        For BPE tokenizer: BPC = (N_tokens / N_chars) * (L / ln(2)).
    - val_ppl_char: Perplexity per character: 2^(BPC) = exp(L * N_tokens / N_chars).

    Args:
        model (GPTLanguageModel): Model instance to evaluate.
        val_dataset (AmharicDataset): Validation dataset containing full token sequence.
        batch_size (int): Batch size for parallel sequence evaluation.
        tokenizer_type (str): Tokenizer type ('char' or 'bpe').
        val_raw_text (Optional[str]): Raw validation text for character-level normalization.

    Returns:
        Dict[str, float]: Computed metrics dictionary.
    """
    was_training = getattr(model, "training", True)
    model.eval()

    block_size = model.config.block_size
    data = val_dataset.data
    total_tokens = len(data)

    # Determine sequential chunks: [i : i + block_size] -> target [i+1 : i + block_size + 1]
    # Non-overlapping strides: step by block_size
    chunk_starts = list(range(0, total_tokens - block_size, block_size))
    if not chunk_starts:
        # Fallback if sequence is shorter than 2*block_size
        chunk_starts = [0] if total_tokens > block_size else []

    total_loss_sum = 0.0
    total_eval_tokens = 0

    try:
        # Batch chunks together for vectorization
        for b_start in range(0, len(chunk_starts), batch_size):
            b_indices = chunk_starts[b_start : b_start + batch_size]
            current_batch_size = len(b_indices)

            batch_X = np.empty((current_batch_size, block_size), dtype=np.int64)
            batch_Y = np.empty((current_batch_size, block_size), dtype=np.int64)

            for i, start_idx in enumerate(b_indices):
                batch_X[i] = data[start_idx : start_idx + block_size]
                batch_Y[i] = data[start_idx + 1 : start_idx + block_size + 1]

            logits = model.forward(batch_X, training=False)
            loss, _ = cross_entropy_loss(logits, batch_Y)

            num_tokens = current_batch_size * block_size
            total_loss_sum += loss * num_tokens
            total_eval_tokens += num_tokens

        mean_loss = (total_loss_sum / max(1, total_eval_tokens)) if total_eval_tokens > 0 else 0.0

        # Per-token metrics
        ppl = float(np.exp(mean_loss))
        bpt = float(mean_loss / np.log(2.0))

        # Per-character normalization
        if val_raw_text is not None and len(val_raw_text) > 0:
            char_count = len(val_raw_text)
            compression_ratio = float(total_tokens) / float(char_count)
            bpc = float(bpt * compression_ratio)
            ppl_char = float(np.exp(mean_loss * compression_ratio))
        else:
            bpc = bpt
            ppl_char = ppl

        return {
            "val_loss": float(mean_loss),
            "val_ppl": float(ppl),
            "val_bpt": float(bpt),
            "val_bpc": float(bpc),
            "val_ppl_char": float(ppl_char),
            "eval_tokens": total_eval_tokens,
        }
    finally:
        if was_training:
            model.train()


# Legacy compatibility wrapper
def estimate_loss(
    model: GPTLanguageModel,
    dataset: AmharicDataset,
    eval_iters: int = 10,
    batch_size: int = 4,
    rng: Optional[np.random.RandomState] = None
) -> float:
    """Wrapper for backward compatibility with existing tests."""
    metrics = evaluate_validation(model, dataset, batch_size=batch_size)
    return metrics["val_loss"]


def train(
    model: GPTLanguageModel,
    train_dataset: AmharicDataset,
    val_dataset: Optional[AmharicDataset] = None,
    val_raw_text: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    max_iters: int = 500,
    total_iters: Optional[int] = None,
    stop_step: Optional[int] = None,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    learning_rate: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_iters: int = 50,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    eval_interval: int = 50,
    early_stopping_patience: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_name: str = "run",
    resume_from: Optional[str] = None,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = True
) -> Dict[str, list]:
    """
    Executes the production-quality GPT training loop.

    Key Implementation Details:
    ---------------------------
    - Cosine learning rate schedule with warmup.
    - Gradient accumulation over `grad_accum_steps` micro-batches.
    - Loss scaling: micro-loss scaled by 1 / K, gradients scaled by 1 / K.
    - Global gradient norm tracking & pre-clip logging.
    - Full deterministic validation evaluation (loss, perplexity, BPC).
    - Best and latest checkpointing with optimizer and RNG state.
    - Early stopping with patience.
    - Continuous metrics logging to outputs/<run_name>/metrics.csv.

    Returns:
        Dict[str, list]: Training history containing recorded metrics.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    assert grad_accum_steps >= 1, f"grad_accum_steps must be >= 1, got {grad_accum_steps}."

    # Setup directories
    if output_dir is None:
        output_dir = os.path.join("outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    params_and_grads = model.get_params_and_grads()
    optimizer = AdamW(
        params_and_grads=params_and_grads,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_step = 1
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume from checkpoint if specified
    if resume_from:
        print(f"[Resume] Resuming training state from: {resume_from}")
        res_dir = resume_from if os.path.isdir(resume_from) else os.path.dirname(resume_from)
        res_name = os.path.basename(resume_from).replace("_meta.json", "").replace(".npz", "")
        if not res_name or os.path.isdir(resume_from):
            res_name = "checkpoint_latest" if os.path.exists(os.path.join(res_dir, "checkpoint_latest_meta.json")) else "checkpoint"

        _, _, _, meta = load_checkpoint(
            checkpoint_dir=res_dir,
            name=res_name,
            model=model,
            rng=rng,
            optimizer=optimizer,
        )
        start_step = int(meta.get("step", 0)) + 1
        best_val_loss = float(meta.get("best_val_loss", meta.get("val_loss", float("inf"))))
        if total_iters is None and meta.get("total_iters"):
            total_iters = int(meta["total_iters"])
        print(f"[Resume] Restored model, optimizer, and RNG state. Resuming from step {start_step} (best_val_loss={best_val_loss:.4f}).")

    schedule_iters = total_iters or max_iters

    # Save run configuration
    run_config = {
        "run_name": run_name,
        "max_iters": max_iters,
        "total_iters": schedule_iters,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate": learning_rate,
        "min_lr": min_lr,
        "warmup_iters": warmup_iters,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "eval_interval": eval_interval,
        "early_stopping_patience": early_stopping_patience,
        "model_config": model.config.to_dict(),
        "num_parameters": model.num_parameters(),
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # Initialize metrics CSV file
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file_exists = os.path.exists(metrics_csv_path) and start_step > 1
    metrics_file = open(metrics_csv_path, "a" if csv_file_exists else "w", newline="", encoding="utf-8")
    metrics_writer = csv.writer(metrics_file)

    if not csv_file_exists:
        metrics_writer.writerow([
            "step",
            "train_loss",
            "val_loss",
            "train_val_gap",
            "val_ppl",
            "val_bpc",
            "lr",
            "grad_norm",
            "step_time_sec",
        ])
        metrics_file.flush()

    history: Dict[str, list] = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "train_val_gap": [],
        "val_ppl": [],
        "val_bpc": [],
        "lr": [],
        "grad_norm": [],
    }

    start_time = time.time()
    tokenizer_type = getattr(tokenizer, "tokenizer_type", "char") if tokenizer else "char"

    if verbose:
        print(f"\n[Training] Model parameters: {model.num_parameters():,}")
        print(f"[Training] Steps: {start_step}..{max_iters} | Micro-Batch: {batch_size} | Accum Steps: {grad_accum_steps} | Effective Batch: {batch_size * grad_accum_steps}")
        print(f"[Training] Peak LR: {learning_rate:.2e} | Min LR: {min_lr:.2e} | Warmup Steps: {warmup_iters}")
        print("=" * 105)
        print(f"{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Gap':>8} | {'Val PPL':>8} | {'Val BPC':>8} | {'LR':>9} | {'GradNorm':>8} | {'Time':>7}")
        print("-" * 105)

    try:
        for step in range(start_step, max_iters + 1):
            if stop_step is not None and step > stop_step:
                if verbose:
                    print(f"[Training] Reached stop_step {stop_step}. Halting early.")
                break
            step_start = time.time()

            # 1. Update learning rate per cosine schedule with warmup
            current_lr = get_cosine_schedule_with_warmup_lr(
                step=step,
                max_iters=schedule_iters,
                warmup_iters=warmup_iters,
                lr_max=learning_rate,
                lr_min=min_lr,
            )
            optimizer.lr = current_lr

            # 2. Gradient Accumulation over micro-batches
            model.zero_grad()
            accum_train_loss = 0.0

            for _ in range(grad_accum_steps):
                X, Y = train_dataset.get_batch(batch_size=batch_size, rng=rng)
                logits = model.forward(X, training=True)
                micro_loss, d_logits = cross_entropy_loss(logits, Y)

                # Mathematical Gradient Scaling:
                # Upstream gradient divided by grad_accum_steps accumulates exact mean gradient
                scaled_d_logits = d_logits / float(grad_accum_steps)
                model.backward(scaled_d_logits)

                accum_train_loss += float(micro_loss) / float(grad_accum_steps)

            # 3. Compute pre-clip global gradient norm and apply clipping
            pre_clip_norm = clip_grad_norm(params_and_grads, max_norm=max_grad_norm)

            # 4. Optimizer update step
            optimizer.step()

            step_duration = time.time() - step_start

            # 5. Periodic evaluation and checkpointing
            is_eval_step = (step % eval_interval == 0 or step == 1 or step == max_iters)
            if is_eval_step:
                val_loss_str = "N/A"
                gap_str = "N/A"
                val_ppl_str = "N/A"
                val_bpc_str = "N/A"
                val_loss = None
                val_ppl = None
                val_bpc = None
                train_val_gap = None
                val_metrics = None

                if val_dataset is not None:
                    val_metrics = evaluate_validation(
                        model=model,
                        val_dataset=val_dataset,
                        batch_size=batch_size,
                        tokenizer_type=tokenizer_type,
                        val_raw_text=val_raw_text,
                    )
                    val_loss = val_metrics["val_loss"]
                    val_ppl = val_metrics["val_ppl"]
                    val_bpc = val_metrics["val_bpc"]

                    # Overfitting indicator: Generalization Gap (Val Loss - Train Loss)
                    train_val_gap = val_loss - accum_train_loss

                    val_loss_str = f"{val_loss:.4f}"
                    gap_str = f"{train_val_gap:+.4f}"
                    val_ppl_str = f"{val_ppl:.2f}"
                    val_bpc_str = f"{val_bpc:.3f}"

                    # Early stopping and best checkpoint tracking
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best-val checkpoint
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            model=model,
                            config=model.config,
                            tokenizer=tokenizer,
                            step=step,
                            val_loss=val_loss,
                            best_val_loss=best_val_loss,
                            optimizer=optimizer,
                            rng=rng,
                            val_metrics=val_metrics,
                            total_iters=schedule_iters,
                            name="checkpoint_best",
                        )
                    else:
                        patience_counter += 1

                # Save latest checkpoint
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    config=model.config,
                    tokenizer=tokenizer,
                    step=step,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    optimizer=optimizer,
                    rng=rng,
                    val_metrics=val_metrics,
                    total_iters=schedule_iters,
                    name="checkpoint_latest",
                )

                # Record in history dict
                history["steps"].append(step)
                history["train_loss"].append(accum_train_loss)
                history["val_loss"].append(val_loss)
                history["train_val_gap"].append(train_val_gap)
                history["val_ppl"].append(val_ppl)
                history["val_bpc"].append(val_bpc)
                history["lr"].append(current_lr)
                history["grad_norm"].append(pre_clip_norm)

                # Append to metrics.csv
                metrics_writer.writerow([
                    step,
                    f"{accum_train_loss:.6f}",
                    f"{val_loss:.6f}" if val_loss is not None else "",
                    f"{train_val_gap:.6f}" if train_val_gap is not None else "",
                    f"{val_ppl:.4f}" if val_ppl is not None else "",
                    f"{val_bpc:.4f}" if val_bpc is not None else "",
                    f"{current_lr:.6e}",
                    f"{pre_clip_norm:.4f}",
                    f"{step_duration:.4f}",
                ])
                metrics_file.flush()

                if verbose:
                    print(
                        f"{step:>6} | {accum_train_loss:>10.4f} | {val_loss_str:>10} | {gap_str:>8} | "
                        f"{val_ppl_str:>8} | {val_bpc_str:>8} | {current_lr:>9.2e} | {pre_clip_norm:>8.3f} | {step_duration:>6.3f}s"
                    )

                # Early stopping condition
                if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n[Early Stopping] Validation loss did not improve for {early_stopping_patience} consecutive evaluations. Stopping early at step {step}.")
                    break

            else:
                # Log step without full eval
                metrics_writer.writerow([
                    step,
                    f"{accum_train_loss:.6f}",
                    "",
                    "",
                    "",
                    "",
                    f"{current_lr:.6e}",
                    f"{pre_clip_norm:.4f}",
                    f"{step_duration:.4f}",
                ])
                metrics_file.flush()

    finally:
        metrics_file.close()

    total_time = time.time() - start_time
    completed_steps = len(history["steps"])
    if verbose:
        print("=" * 105)
        print(f"[Training] Completed in {total_time:.2f}s ({max_iters / max(1e-3, total_time):.2f} steps/s). Best Val Loss: {best_val_loss:.4f}")
        print(f"[Training] Metrics logged to -> {metrics_csv_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Production Training Pipeline for Amharic GPT")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file (CLI flags override JSON)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing train.txt & val.txt")
    parser.add_argument("--raw_corpus", type=str, default="data/raw/sample_amharic.txt", help="Path to raw corpus file")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to save tokenizer vocabulary")
    parser.add_argument("--config_type", type=str, default="small", choices=["tiny", "small"], help="Model configuration profile")
    parser.add_argument("--max_iters", type=int, default=1000, help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Number of micro-batches to accumulate")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum terminal learning rate")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient L2 norm clipping threshold")
    parser.add_argument("--eval_interval", type=int, default=50, help="Interval for full validation evaluation")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (eval intervals without improvement)")
    parser.add_argument("--run_name", type=str, default="small_run", help="Unique identifier for run outputs")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save run metrics and logs")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory or meta file to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"], help="Tokenizer type: char or bpe")
    parser.add_argument("--bpe_vocab_size", type=int, default=500, help="Target vocabulary size for BPE tokenizer")

    args = parser.parse_args()

    # Load JSON config if provided, letting CLI override
    if args.config and os.path.exists(args.config):
        print(f"[Config] Loading base configuration from JSON: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            json_cfg = json.load(f)
        for k, v in json_cfg.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)

    # Resolve paths
    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "val.txt")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"[Data] Processed data not found in {args.data_dir}. Preprocessing from {args.raw_corpus}...")
        preprocess_file(args.raw_corpus, output_dir=args.data_dir)

    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(val_path, "r", encoding="utf-8") as f:
        val_text = f.read()

    # Build or restore tokenizer
    vocab_file = args.vocab_path or os.path.join(args.data_dir, f"vocab_{args.tokenizer}.json")
    if args.tokenizer == "bpe":
        print(f"[Tokenizer] Initializing BPE tokenizer (target vocab_size={args.bpe_vocab_size})...")
        tokenizer = BPETokenizer()
        if os.path.exists(vocab_file):
            print(f"[Tokenizer] Loading cached vocabulary from {vocab_file}...")
            tokenizer.load_vocab(vocab_file)
        else:
            print("[Tokenizer] Training BPE tokenizer on TRAIN split only...")
            tokenizer.train(train_text, target_vocab_size=args.bpe_vocab_size)
            tokenizer.save_vocab(vocab_file)
    else:
        print("[Tokenizer] Initializing Character tokenizer...")
        tokenizer = AmharicTokenizer()
        if os.path.exists(vocab_file):
            tokenizer.load_vocab(vocab_file)
        else:
            tokenizer.build_vocab(train_text + val_text)
            tokenizer.save_vocab(vocab_file)

    # Setup model configuration
    if args.config_type == "tiny":
        config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    else:
        config = GPTConfig.get_small_config(vocab_size=tokenizer.vocab_size)

    # Create dataset objects
    train_ds, val_ds = create_datasets_from_text(
        train_text=train_text,
        val_text=val_text,
        tokenizer=tokenizer,
        block_size=config.block_size,
    )

    rng = np.random.RandomState(args.seed)
    model = GPTLanguageModel(config, rng=rng)

    train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        val_raw_text=val_text,
        tokenizer=tokenizer,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        early_stopping_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume_from=args.resume,
        rng=rng,
        verbose=True,
    )


if __name__ == "__main__":
    main()
