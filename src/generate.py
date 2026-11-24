"""
Autoregressive Amharic Text Generation in Pure NumPy.

Autoregressive Generation Concept:
----------------------------------
Autoregressive means the model generates text token by token:
  Prompt: "ሰላም"
    ↓
  [Encode prompt to IDs] -> [x_0, x_1, ...]
    ↓
  Loop for max_new_tokens:
    1. Crop context to at most block_size: idx_cond = idx[:, -block_size:]
    2. Forward pass: logits = model.forward(idx_cond)
    3. Extract logits at the final time position: next_logits = logits[:, -1, :]
    4. Apply temperature scaling: next_logits = next_logits / temperature
    5. Optionally filter to top_k tokens
    6. Convert to probabilities via softmax: probs = softmax(next_logits)
    7. Select next token (argmax for greedy, or rng.choice for sampling)
    8. Append next token: idx = np.concatenate([idx, [[next_token]]], axis=1)
    9. If next_token is <EOS>, stop generation early.
    ↓
  [Decode token IDs back to Amharic characters]

Sampling Strategies Explained:
------------------------------
1. Greedy Generation:
   Always selects argmax(probs). Deterministic and repetitive.
2. Temperature Scaling (T):
   Scaling logits by 1/T before softmax:
     - Low Temperature (T < 1.0, e.g. 0.5): Sharpens the distribution, prioritizing
       high-confidence tokens.
     - High Temperature (T > 1.0, e.g. 1.2): Flattens the distribution, introducing
       variety and creative variations.
3. Top-k Filtering:
   Keeps only the top k highest probability tokens and sets the rest to -inf.
   Eliminates bizarre out-of-context tokens from the tail of the distribution.
"""

import os
import argparse
import numpy as np
from typing import Optional, List

from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.model import GPTLanguageModel
from src.math_utils import softmax
from src.utils import load_checkpoint


def generate(
    model: GPTLanguageModel,
    tokenizer: AmharicTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 20,
    greedy: bool = False,
    stop_at_eos: bool = True,
    rng: Optional[np.random.RandomState] = None
) -> str:
    """
    Generates text autoregressively starting from an Amharic prompt.
    
    Args:
        model (GPTLanguageModel): Trained model instance.
        tokenizer (AmharicTokenizer): Tokenizer for encoding prompt and decoding output.
        prompt (str): Starting Amharic text prompt.
        max_new_tokens (int): Maximum number of characters to generate.
        temperature (float): Softmax temperature (> 0.0).
        top_k (Optional[int]): Number of highest probability tokens to keep.
        greedy (bool): If True, uses greedy argmax decoding.
        stop_at_eos (bool): If True, halts generation when <EOS> token is generated.
        rng (Optional[np.random.RandomState]): Optional random state for reproducible sampling.
        
    Returns:
        generated_text (str): Complete generated text including prompt.
    """
    if hasattr(model, "eval"):
        model.eval()

    # 1. Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.bos_id]

    idx = np.array([prompt_ids], dtype=np.int64)  # Shape: (1, T)

    block_size = model.config.block_size
    eos_id = tokenizer.eos_id

    for _ in range(max_new_tokens):
        # Crop context window if sequence length exceeds block_size
        idx_cond = idx[:, -block_size:]

        # Forward pass through model to get logits
        # Shape: (1, T_cond, vocab_size)
        logits = model.forward(idx_cond)

        # Focus exclusively on logits at the last position
        # Shape: (vocab_size,)
        next_logits = logits[0, -1, :].copy()

        if greedy:
            # Greedy decoding: choose highest scoring token
            next_token_id = int(np.argmax(next_logits))
        else:
            # Temperature scaling
            temperature = max(temperature, 1e-5)
            next_logits = next_logits / temperature

            # Optional Top-k truncation
            if top_k is not None and top_k > 0 and top_k < len(next_logits):
                # Find the threshold value for top_k
                k_val = np.partition(next_logits, -top_k)[-top_k]
                next_logits[next_logits < k_val] = -1e9

            # Convert logits to probabilities
            probs = softmax(next_logits, axis=-1)

            # Sample categorical distribution using explicit RNG
            if rng is None:
                rng = np.random.RandomState(42)
            next_token_id = int(rng.choice(len(probs), p=probs))

        # Check early stopping at EOS
        if stop_at_eos and next_token_id == eos_id:
            break

        # Append new token to sequence
        # Shape: (1, T + 1)
        idx = np.concatenate([idx, np.array([[next_token_id]], dtype=np.int64)], axis=1)

    # Decode complete token ID list to Amharic string
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate Amharic Text Autoregressively")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where model is saved")
    parser.add_argument("--prompt", type=str, default="ሰላም", help="Amharic starting prompt")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k filtering (0 to disable)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy argmax decoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    args = parser.parse_args()

    # Load checkpoint
    if not os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint_meta.json")):
        print(f"Error: No checkpoint found in '{args.checkpoint_dir}'. Please run training first:")
        print("  python -m src.train --max_iters 200")
        return

    rng = np.random.RandomState(args.seed)
    model, config, tokenizer, meta = load_checkpoint(args.checkpoint_dir, rng=rng)
    assert tokenizer is not None, "Tokenizer vocabulary missing from checkpoint."

    top_k_val = None if args.top_k <= 0 else args.top_k

    print(f"\n[Generation] Prompt: \"{args.prompt}\"")
    print(f"[Generation] Strategy: {'Greedy' if args.greedy else f'Sampling (T={args.temperature}, top_k={top_k_val})'} | Seed: {args.seed}")
    print("-" * 50)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k_val,
        greedy=args.greedy,
        rng=rng,
    )

    print("Generated Text:")
    print(output)
    print("-" * 50)


if __name__ == "__main__":
    main()
