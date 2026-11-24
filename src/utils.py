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
    tokenizer: Optional[Any] = None,
    step: int = 0,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    optimizer: Optional[Any] = None,
    optimizer_state: Optional[dict] = None,
    rng: Optional[np.random.RandomState] = None,
    val_metrics: Optional[Dict[str, float]] = None,
    total_iters: Optional[int] = None,
    name: str = "checkpoint"
) -> str:
    """
    Saves model parameters, optimizer state, RNG state, and metadata using NumPy-compatible storage.
    
    Files Created:
    --------------
    1. {name}.npz: Compressed model parameter weights.
    2. {name}_meta.json: Configuration, hashes, step, losses, RNG state, optimizer scalar metadata.
    3. {name}_optimizer.npz: Compressed first (m) and second (v) optimizer moment arrays.
    4. vocab.json: Tokenizer vocabulary (if tokenizer provided).
    
    Args:
        checkpoint_dir (str): Directory where checkpoint files will be written.
        model (GPTLanguageModel): Model whose parameters to save.
        config (GPTConfig): Model configuration.
        tokenizer (Optional[Any]): Tokenizer instance (AmharicTokenizer or BPETokenizer).
        step (int): Current training step.
        val_loss (Optional[float]): Latest validation loss.
        best_val_loss (Optional[float]): Best validation loss achieved so far.
        optimizer (Optional[Any]): Optimizer instance (e.g. AdamW).
        optimizer_state (Optional[dict]): Pre-serialized optimizer state dictionary.
        rng (Optional[np.random.RandomState]): RNG instance whose internal state is preserved.
        val_metrics (Optional[dict]): Comprehensive validation metrics (PPL, BPC, etc.).
        name (str): Base name for checkpoint files (e.g., 'checkpoint', 'checkpoint_best', 'checkpoint_latest').
        
    Returns:
        npz_path (str): Path to saved model weights archive.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Extract and save all model parameter arrays
    params = model.get_params_and_grads()
    weights_dict = {param_name: param_arr for param_name, (param_arr, _) in params.items()}
    npz_path = os.path.join(checkpoint_dir, f"{name}.npz")
    np.savez_compressed(npz_path, **weights_dict)

    # 2. Extract and save optimizer state if provided
    opt_meta_dict = None
    has_opt_file = False
    if optimizer is not None or optimizer_state is not None:
        state = optimizer.state_dict() if optimizer is not None else optimizer_state
        if state is not None:
            opt_arrays = {}
            for k, arr in state.get("m", {}).items():
                opt_arrays[f"m.{k}"] = arr
            for k, arr in state.get("v", {}).items():
                opt_arrays[f"v.{k}"] = arr
            
            if opt_arrays:
                opt_npz_path = os.path.join(checkpoint_dir, f"{name}_optimizer.npz")
                np.savez_compressed(opt_npz_path, **opt_arrays)
                has_opt_file = True

            opt_meta_dict = {
                "t": int(state.get("t", step)),
                "lr": float(state.get("lr", 0.0)),
                "beta1": float(state.get("beta1", 0.9)),
                "beta2": float(state.get("beta2", 0.999)),
                "eps": float(state.get("eps", 1e-8)),
                "weight_decay": float(state.get("weight_decay", 0.0)),
            }

    # 3. Serialize RNG state if provided
    serializable_rng_state = None
    if rng is not None:
        r_state = rng.get_state()
        serializable_rng_state = [
            r_state[0],
            r_state[1].tolist(),
            int(r_state[2]),
            int(r_state[3]),
            float(r_state[4]),
        ]

    # 4. Compute config hash and format version
    config_dict = config.to_dict()
    config_hash = compute_config_hash(config_dict)

    metadata: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "config_hash": config_hash,
        "step": step,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "config": config_dict,
        "weights_file": f"{name}.npz",
    }
    if has_opt_file:
        metadata["optimizer_file"] = f"{name}_optimizer.npz"
    if opt_meta_dict is not None:
        metadata["optimizer"] = opt_meta_dict
    if serializable_rng_state is not None:
        metadata["rng_state"] = serializable_rng_state
    if val_metrics is not None:
        metadata["val_metrics"] = val_metrics
    if total_iters is not None:
        metadata["total_iters"] = int(total_iters)

    meta_path = os.path.join(checkpoint_dir, f"{name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 5. Save tokenizer vocabulary if provided
    if tokenizer is not None:
        vocab_path = os.path.join(checkpoint_dir, "vocab.json")
        tokenizer.save_vocab(vocab_path)

    print(f"[Checkpoint] Saved checkpoint '{name}' to {checkpoint_dir} (step {step}, val_loss={val_loss})")
    return npz_path


def load_checkpoint(
    checkpoint_dir: str,
    name: str = "checkpoint",
    rng: Optional[np.random.RandomState] = None,
    optimizer: Optional[Any] = None,
    model: Optional[GPTLanguageModel] = None,
) -> Tuple[GPTLanguageModel, GPTConfig, Optional[Any], Dict[str, Any]]:
    """
    Loads model, configuration, and vocabulary from checkpoint directory.
    Validates parameter keys, shapes, and config hash. Raises ValueError on any mismatch.
    Optionally restores RNG state and optimizer state in-place.
    
    Args:
        checkpoint_dir (str): Path to checkpoint directory.
        name (str): Base name for checkpoint files (e.g. 'checkpoint', 'checkpoint_best', 'checkpoint_latest').
        rng (Optional[np.random.RandomState]): Optional random state instance to restore in-place.
        optimizer (Optional[Any]): Optional AdamW optimizer instance to restore in-place.
        model (Optional[GPTLanguageModel]): Optional existing model instance to load weights into in-place.
        
    Returns:
        model (GPTLanguageModel): Model with restored weights.
        config (GPTConfig): Restored configuration.
        tokenizer (Optional[Any]): Restored tokenizer.
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

    # 3. Reconstruct configuration and model if model not provided
    if model is None:
        config = GPTConfig.from_dict(config_dict)
        # Use an isolated dummy RNG state so passed rng is never advanced by init_weights
        model = GPTLanguageModel(config, rng=np.random.RandomState(0))
    else:
        config = model.config

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

        # Strict key validation: no missing or extra keys allowed
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

        # Strict shape validation: every array must match model expectation
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

    # 5. Restore optimizer state if requested and available
    if optimizer is not None:
        opt_file = metadata.get("optimizer_file", f"{name}_optimizer.npz")
        opt_path = os.path.join(checkpoint_dir, opt_file)
        if os.path.exists(opt_path):
            with np.load(opt_path) as opt_data:
                opt_meta = metadata.get("optimizer", {})
                optimizer.t = opt_meta.get("t", optimizer.t)
                optimizer.lr = opt_meta.get("lr", optimizer.lr)
                optimizer.beta1 = opt_meta.get("beta1", optimizer.beta1)
                optimizer.beta2 = opt_meta.get("beta2", optimizer.beta2)
                optimizer.eps = opt_meta.get("eps", optimizer.eps)
                optimizer.weight_decay = opt_meta.get("weight_decay", optimizer.weight_decay)

                for file_key in opt_data.files:
                    if file_key.startswith("m."):
                        param_name = file_key[2:]
                        if param_name in optimizer.m:
                            optimizer.m[param_name][...] = opt_data[file_key]
                    elif file_key.startswith("v."):
                        param_name = file_key[2:]
                        if param_name in optimizer.v:
                            optimizer.v[param_name][...] = opt_data[file_key]

    # 6. Load tokenizer if vocab.json exists and validate vocab size
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

    # 7. Restore RNG state if requested and available (after all model/tokenizer setup)
    if rng is not None and "rng_state" in metadata:
        s = metadata["rng_state"]
        restored_tuple = (
            s[0],
            np.array(s[1], dtype=np.uint32),
            int(s[2]),
            int(s[3]),
            float(s[4]),
        )
        rng.set_state(restored_tuple)

    print(f"[Checkpoint] Successfully loaded checkpoint from {checkpoint_dir} (step {metadata.get('step')}).")
    return model, config, tokenizer, metadata
