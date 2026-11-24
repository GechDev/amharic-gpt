"""
Model Checkpointing and Utility Functions in Pure NumPy.

This module provides hardened, standard serialization:
1. save_checkpoint: saves model weights as compressed .npz, plus config,
   format_version, and a cryptographic sha256 hash in metadata JSON.
2. load_checkpoint: reconstructs model and restores NumPy parameter arrays
   with strict validation of every parameter's key and shape against the model.
   Raises explicit ValueErrors on missing, extra, or mismatched keys.
"""

import os
import json
import hashlib
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.tokenizer import AmharicTokenizer

# Current checkpoint metadata schema version
FORMAT_VERSION = 1


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 hash of the configuration dictionary.
    Keys are sorted to ensure identical hashes regardless of dict insertion order.
    """
    canonical_json = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


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
    weights_dict = {param_name: param_arr for param_name, (param_arr, _) in params.items()}

    # 2. Save weights to compressed NumPy archive
    npz_path = os.path.join(checkpoint_dir, f"{name}.npz")
    np.savez_compressed(npz_path, **weights_dict)

    # 3. Compute config hash and format version
    config_dict = config.to_dict()
    config_hash = compute_config_hash(config_dict)

    metadata = {
        "format_version": FORMAT_VERSION,
        "config_hash": config_hash,
        "step": step,
        "val_loss": val_loss,
        "config": config_dict,
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
    name: str = "checkpoint",
    rng: Optional[np.random.RandomState] = None
) -> Tuple[GPTLanguageModel, GPTConfig, Optional[AmharicTokenizer], Dict[str, Any]]:
    """
    Loads model, configuration, and vocabulary from checkpoint directory.
    Validates parameter keys, shapes, and config hash. Raises ValueError on any mismatch.
    
    Args:
        checkpoint_dir (str): Path to checkpoint directory.
        name (str): Base name for checkpoint files.
        rng (Optional[np.random.RandomState]): Optional random state for model initialization.
        
    Returns:
        model (GPTLanguageModel): Model with restored weights.
        config (GPTConfig): Restored configuration.
        tokenizer (Optional[AmharicTokenizer]): Restored tokenizer.
        metadata (dict): Checkpoint metadata (step, loss, format_version, etc.).
    """
    meta_path = os.path.join(checkpoint_dir, f"{name}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Checkpoint metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 1. Validate format_version
    version = metadata.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version: {version}. Expected version {FORMAT_VERSION}."
        )

    # 2. Validate config integrity
    if "config" not in metadata:
        raise ValueError(f"Checkpoint metadata in {meta_path} is missing 'config' dictionary.")

    config_dict = metadata["config"]
    expected_hash = compute_config_hash(config_dict)
    stored_hash = metadata.get("config_hash")
    if stored_hash is not None and stored_hash != expected_hash:
        raise ValueError(
            f"Checkpoint configuration hash mismatch in {meta_path}. "
            f"Expected {expected_hash}, found {stored_hash}."
        )

    # 3. Reconstruct configuration and model
    config = GPTConfig.from_dict(config_dict)
    model = GPTLanguageModel(config, rng=rng)

    # 4. Load NumPy weights archive
    weights_file = metadata.get("weights_file", f"{name}.npz")
    npz_path = os.path.join(checkpoint_dir, weights_file)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Checkpoint weights file not found: {npz_path}")

    archive = np.load(npz_path)
    try:
        archive_keys = set(archive.files)
        model_params = model.get_params_and_grads()
        model_keys = set(model_params.keys())

        # 5. Strict key validation: no missing or extra keys allowed
        missing_keys = model_keys - archive_keys
        if missing_keys:
            raise ValueError(
                f"Missing keys in checkpoint '{npz_path}': {sorted(missing_keys)}"
            )

        extra_keys = archive_keys - model_keys
        if extra_keys:
            raise ValueError(
                f"Unexpected extra keys in checkpoint '{npz_path}': {sorted(extra_keys)}"
            )

        # 6. Strict shape validation: every array must match model expectation
        for param_name, (param_arr, _) in model_params.items():
            loaded_arr = archive[param_name]
            if loaded_arr.shape != param_arr.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{param_name}': "
                    f"model expects {param_arr.shape}, but checkpoint contains {loaded_arr.shape}."
                )
            param_arr[...] = loaded_arr
    finally:
        archive.close()

    # 7. Load tokenizer if vocab.json exists and validate vocab size
    tokenizer = None
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vdata = json.load(f)
        if vdata.get("tokenizer_type") == "bpe":
            from src.bpe_tokenizer import BPETokenizer
            tokenizer = BPETokenizer(vocab_file=vocab_path)
        else:
            tokenizer = AmharicTokenizer(vocab_file=vocab_path)

        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: tokenizer has {tokenizer.vocab_size} tokens, "
                f"but model configuration expects vocab_size={config.vocab_size}."
            )

    print(f"[Checkpoint] Successfully loaded checkpoint from {checkpoint_dir} (step {metadata.get('step')}).")
    return model, config, tokenizer, metadata
