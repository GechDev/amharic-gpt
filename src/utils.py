"""
Model Checkpointing and Utility Functions in Pure NumPy.

This module provides simple, standard serialization:
1. save_checkpoint: saves model weights as .npz, plus config and metadata JSON.
2. load_checkpoint: reconstructs model and restores NumPy parameter arrays.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.tokenizer import AmharicTokenizer


def save_checkpoint(
    checkpoint_dir: str,
    model: GPTLanguageModel,
    config: GPTConfig,
    tokenizer: Optional[AmharicTokenizer] = None,
    step: int = 0,
    val_loss: Optional[float] = None,
    optimizer_state: Optional[dict] = None,
    name: str = "checkpoint"
) -> str:
    """
    Saves model parameters and metadata using NumPy-compatible storage (.npz + .json).
    
    Args:
        checkpoint_dir (str): Directory where checkpoint files will be written.
        model (GPTLanguageModel): Model whose parameters to save.
        config (GPTConfig): Model configuration.
        tokenizer (Optional[AmharicTokenizer]): Tokenizer instance.
        step (int): Current training step.
        val_loss (Optional[float]): Latest validation loss.
        optimizer_state (Optional[dict]): Optimizer state if available.
        name (str): Base name for checkpoint files.
        
    Returns:
        npz_path (str): Path to saved .npz weight archive.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Extract all parameter arrays
    params = model.get_params_and_grads()
    weights_dict = {name: param for name, (param, _) in params.items()}

    # 2. Save weights to compressed NumPy archive
    npz_path = os.path.join(checkpoint_dir, f"{name}.npz")
    np.savez_compressed(npz_path, **weights_dict)

    # 3. Save config and metadata
    metadata = {
        "step": step,
        "val_loss": val_loss,
        "config": config.to_dict(),
        "weights_file": f"{name}.npz",
    }
    meta_path = os.path.join(checkpoint_dir, f"{name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 4. Save tokenizer vocabulary if provided
    if tokenizer is not None:
        vocab_path = os.path.join(checkpoint_dir, "vocab.json")
        tokenizer.save_vocab(vocab_path)

    print(f"[Checkpoint] Saved model checkpoint to {checkpoint_dir} (step {step}, val_loss={val_loss})")
    return npz_path


def load_checkpoint(
    checkpoint_dir: str,
    name: str = "checkpoint"
) -> Tuple[GPTLanguageModel, GPTConfig, Optional[AmharicTokenizer], Dict[str, Any]]:
    """
    Loads model, configuration, and vocabulary from checkpoint directory.
    
    Args:
        checkpoint_dir (str): Path to checkpoint directory.
        name (str): Base name for checkpoint files.
        
    Returns:
        model (GPTLanguageModel): Model with restored weights.
        config (GPTConfig): Restored configuration.
        tokenizer (Optional[AmharicTokenizer]): Restored tokenizer.
        metadata (dict): Checkpoint metadata (step, loss, etc.).
    """
    meta_path = os.path.join(checkpoint_dir, f"{name}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Reconstruct configuration and model
    config = GPTConfig.from_dict(metadata["config"])
    model = GPTLanguageModel(config)

    # Load NumPy weights
    weights_file = metadata.get("weights_file", f"{name}.npz")
    npz_path = os.path.join(checkpoint_dir, weights_file)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Weights file not found: {npz_path}")

    archive = np.load(npz_path)
    model_params = model.get_params_and_grads()

    for param_name, (param_arr, _) in model_params.items():
        if param_name in archive:
            param_arr[...] = archive[param_name]
        else:
            print(f"[Warning] Parameter {param_name} not found in checkpoint archive.")

    # Load tokenizer if vocab.json exists
    tokenizer = None
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    if os.path.exists(vocab_path):
        tokenizer = AmharicTokenizer(vocab_file=vocab_path)

    print(f"[Checkpoint] Successfully loaded checkpoint from {checkpoint_dir} (step {metadata.get('step')}).")
    return model, config, tokenizer, metadata
