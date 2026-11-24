"""
Training and Validation Loop for Amharic GPT in Pure NumPy.

Training Loop Lifecycle:
------------------------
For each iteration step:
  1. Sample mini-batch of token indices (X, Y) from training set.
  2. Forward pass: logits = model.forward(X)  [B, T, vocab_size]
  3. Loss calculation: loss, d_logits = cross_entropy_loss(logits, Y)
  4. Reset gradients: model.zero_grad()
  5. Analytical backward pass: model.backward(d_logits)
  6. Gradient clipping: clip_grad_norm(params, max_norm=1.0)
  7. Parameter update: optimizer.step()
  8. Periodic evaluation of validation loss.
"""

import os
import time
import argparse
import numpy as np
from typing import Optional, Dict

from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.data import AmharicDataset, create_datasets_from_text, preprocess_file
from src.model import GPTLanguageModel
from src.loss import cross_entropy_loss
from src.optimizer import AdamW, clip_grad_norm
from src.utils import save_checkpoint


def estimate_loss(
    model: GPTLanguageModel,
    dataset: AmharicDataset,
    eval_iters: int = 10,
    batch_size: int = 4,
    rng: Optional[np.random.RandomState] = None
) -> float:
    """
    Evaluates the mean loss over a given dataset without updating weights.
    
    Why Validation Loss is Essential:
    ---------------------------------
    Training loss only measures how well the model fits the training examples.
    Validation loss measures generalizability to unseen Amharic text.
    If training loss drops while validation loss increases, the model is overfitting.
    """
    losses = []
    was_training = getattr(model, "training", True)
    model.eval()
    try:
        for _ in range(eval_iters):
            X, Y = dataset.get_batch(batch_size=batch_size, rng=rng)
            logits = model.forward(X)
            loss, _ = cross_entropy_loss(logits, Y)
            losses.append(loss)
    finally:
        if was_training:
            model.train()
    return float(np.mean(losses))


def train(
    model: GPTLanguageModel,
    train_dataset: AmharicDataset,
    val_dataset: Optional[AmharicDataset] = None,
    max_iters: int = 500,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    eval_interval: int = 50,
    eval_iters: int = 10,
    max_grad_norm: float = 1.0,
    checkpoint_dir: Optional[str] = None,
    tokenizer: Optional[AmharicTokenizer] = None,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = True
) -> Dict[str, list]:
    """
    Executes the training loop.
    
    Returns:
        history (dict): Dictionary with recorded 'train_loss', 'val_loss', and 'steps'.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    params_and_grads = model.get_params_and_grads()
    optimizer = AdamW(
        params_and_grads=params_and_grads,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {"steps": [], "train_loss": [], "val_loss": []}
    start_time = time.time()

    if verbose:
        print(f"[Training] Model parameters: {model.num_parameters():,}")
        print(f"[Training] Max iterations: {max_iters} | Batch size: {batch_size} | Learning rate: {learning_rate}")
        print("-" * 65)
        print(f"{'Step':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Step Time':>10}")
        print("-" * 65)

    for step in range(1, max_iters + 1):
        step_start = time.time()

        # 1. Load batch
        X, Y = train_dataset.get_batch(batch_size=batch_size, rng=rng)

        # 2. Forward pass
        logits = model.forward(X)

        # 3. Compute loss and analytical gradient w.r.t logits
        loss, d_logits = cross_entropy_loss(logits, Y)

        # 4. Zero gradients
        model.zero_grad()

        # 5. Backward pass
        model.backward(d_logits)

        # 6. Clip gradient norm
        clip_grad_norm(params_and_grads, max_norm=max_grad_norm)

        # 7. Optimizer step
        optimizer.step()

        step_duration = time.time() - step_start

        # Periodic logging and validation
        if step % eval_interval == 0 or step == 1 or step == max_iters:
            val_loss_str = "N/A"
            val_loss = None
            if val_dataset is not None:
                val_loss = estimate_loss(model, val_dataset, eval_iters=eval_iters, batch_size=batch_size, rng=rng)
                val_loss_str = f"{val_loss:.4f}"

            history["steps"].append(step)
            history["train_loss"].append(loss)
            if val_loss is not None:
                history["val_loss"].append(val_loss)

            if verbose:
                print(f"{step:>6} | {loss:>12.4f} | {val_loss_str:>12} | {step_duration:>9.3f}s")

    total_time = time.time() - start_time
    if verbose:
        print("-" * 65)
        print(f"[Training] Completed {max_iters} iterations in {total_time:.2f}s ({max_iters/total_time:.2f} it/s).")

    # Save checkpoint if directory is specified
    if checkpoint_dir:
        last_val = history["val_loss"][-1] if history["val_loss"] else None
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            config=model.config,
            tokenizer=tokenizer,
            step=max_iters,
            val_loss=last_val,
            name="checkpoint",
        )

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Amharic GPT Model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing train.txt & val.txt")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json", help="Path to save tokenizer vocabulary")
    parser.add_argument("--config_type", type=str, default="tiny", choices=["tiny", "small"], help="Model size configuration")
    parser.add_argument("--max_iters", type=int, default=200, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"], help="Tokenizer type: char or bpe")
    parser.add_argument("--bpe_vocab_size", type=int, default=500, help="Target vocabulary size for BPE tokenizer")
    args = parser.parse_args()

    # Preprocess if data files do not exist
    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "val.txt")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[Data] Processed data not found. Running preprocessing...")
        preprocess_file("data/raw/sample_amharic.txt", output_dir=args.data_dir)

    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(val_path, "r", encoding="utf-8") as f:
        val_text = f.read()

    # Build tokenizer
    if args.tokenizer == "bpe":
        print(f"[Tokenizer] Training BPE tokenizer on TRAIN split only (target vocab_size={args.bpe_vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, target_vocab_size=args.bpe_vocab_size)
    else:
        print("[Tokenizer] Building Character-level tokenizer...")
        tokenizer = AmharicTokenizer()
        tokenizer.build_vocab(train_text + val_text)

    os.makedirs(os.path.dirname(os.path.abspath(args.vocab_path)), exist_ok=True)
    tokenizer.save_vocab(args.vocab_path)

    # Configure model
    if args.config_type == "tiny":
        config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    else:
        config = GPTConfig.get_small_config(vocab_size=tokenizer.vocab_size)

    # Create datasets
    train_ds, val_ds = create_datasets_from_text(
        train_text=train_text,
        val_text=val_text,
        tokenizer=tokenizer,
        block_size=config.block_size
    )

    rng = np.random.RandomState(args.seed)
    model = GPTLanguageModel(config, rng=rng)
    train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer=tokenizer,
        rng=rng,
    )


if __name__ == "__main__":
    main()
