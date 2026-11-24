"""
Autoregressive Amharic Text Generation in Pure NumPy.

Features:
---------
1. Key-Value (KV) Caching (Lecture Slide 15):
   - Incremental decoding storing per-layer keys and values.
   - Reduces decoding complexity from O(T^2) per token to O(T).
   - Sliding window policy handles context overflow beyond block_size.
2. Advanced Sampling Algorithms:
   - Greedy (argmax)
   - Temperature scaling (T > 0, T -> 0 falls back to greedy)
   - Top-k filtering (keeps exactly k highest probability tokens)
   - Top-p / Nucleus filtering (keeps minimal set with cumulative mass >= p)
   - Repetition penalty (penalizes recently generated tokens)
   - Special token masking: <PAD>, <UNK>, and <BOS> are masked to -inf
3. Termination Policies:
   - Halts on <EOS> or <DOC> tokens, or upon reaching max_new_tokens.
4. Quantitative Sanity Metrics:
   - Valid Ethiopic character ratio
   - Repeated trigram rate
   - Greedy sample perplexity under the model
5. Full CLI:
   - Single prompt (--prompt) or batch prompt file (--prompt_file)
   - Multiple completions (--num_samples)
   - Seed control (--seed)
   - Export results to outputs/samples_<run>.txt with exact run settings
   - KV Cache benchmark (--benchmark) for 200 tokens
"""

import os
import sys
import time
import argparse
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.model import GPTLanguageModel
from src.math_utils import softmax
from src.loss import cross_entropy_loss
from src.utils import load_checkpoint
from src.amharic_normalizer import is_ethiopic_character


# ---------------------------------------------------------------------------
# Sampling Utilities
# ---------------------------------------------------------------------------

def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
    context_token_ids: Optional[List[int]] = None,
    masked_token_ids: Optional[List[int]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """
    Samples the next token ID from an unnormalized logit vector from scratch.
    
    Mathematical Steps:
    -------------------
    1. Repetition Penalty:
       For token ID i in context_token_ids:
         logits[i] = logits[i] / repetition_penalty  if logits[i] > 0
         logits[i] = logits[i] * repetition_penalty  if logits[i] <= 0
         
    2. Special Token Masking:
       Sets logits of <PAD>, <UNK>, <BOS> to -inf so they are never sampled.
       
    3. Greedy Decoding:
       If greedy is True or temperature < 1e-5: return argmax(logits).
       
    4. Temperature Scaling:
       logits = logits / temperature
       
    5. Top-k Truncation:
       Keeps exactly the top k tokens; sets all other logits to -inf.
       
    6. Top-p (Nucleus) Truncation:
       Sorts tokens descending by probability; keeps the minimal set whose
       cumulative probability mass >= top_p; sets the remaining tail to -inf.
       
    7. Softmax & Categorical Sampling:
       probs = softmax(finite_logits)
       next_token ~ Categorical(probs) via explicit rng.choice.
    """
    # Work on a local copy of logits
    logits = np.array(logits, dtype=np.float64, copy=True)
    vocab_size = len(logits)

    # 1. Repetition penalty
    if repetition_penalty != 1.0 and context_token_ids:
        seen_tokens = set(context_token_ids)
        for token_id in seen_tokens:
            if 0 <= token_id < vocab_size:
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    # 2. Mask special tokens (<PAD>, <UNK>, <BOS>) to -inf
    if masked_token_ids:
        for mid in masked_token_ids:
            if mid is not None and 0 <= mid < vocab_size:
                logits[mid] = -np.inf

    # 3. Greedy decoding shortcut
    if greedy or temperature < 1e-5:
        return int(np.argmax(logits))

    # 4. Temperature scaling
    temperature = max(temperature, 1e-5)
    logits = logits / temperature

    # 5. Top-k filtering
    if top_k is not None and 0 < top_k < vocab_size:
        # Find indices of top-k elements using partition
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.ones(vocab_size, dtype=bool)
        mask[top_k_indices] = False
        logits[mask] = -np.inf

    # 6. Top-p (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]

        # Compute softmax over sorted logits safely
        finite_mask = np.isfinite(sorted_logits)
        if np.any(finite_mask):
            max_l = np.max(sorted_logits[finite_mask])
            exp_l = np.zeros_like(sorted_logits)
            exp_l[finite_mask] = np.exp(sorted_logits[finite_mask] - max_l)
            sum_exp = np.sum(exp_l)
            sorted_probs = exp_l / sum_exp if sum_exp > 0 else np.zeros_like(exp_l)
            cumulative_probs = np.cumsum(sorted_probs)

            # Mask tokens whose cumulative probability exceeds top_p
            # Shift by 1 so the first token exceeding threshold is preserved
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -np.inf

    # 7. Convert remaining finite logits to categorical probabilities
    finite = np.isfinite(logits)
    if not np.any(finite):
        # Fallback: uniform across entire vocabulary
        probs = np.ones(vocab_size, dtype=np.float64) / vocab_size
    else:
        max_l = np.max(logits[finite])
        exp_l = np.zeros_like(logits)
        exp_l[finite] = np.exp(logits[finite] - max_l)
        probs = exp_l / np.sum(exp_l)

    if rng is None:
        rng = np.random.RandomState(42)

    return int(rng.choice(vocab_size, p=probs))


# ---------------------------------------------------------------------------
# Core Generation Pipeline
# ---------------------------------------------------------------------------

def generate(
    model: GPTLanguageModel,
    tokenizer: AmharicTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 20,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
    stop_at_eos: bool = True,
    stop_tokens: Optional[List[str]] = None,
    use_cache: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> str:
    """
    Generates Amharic text autoregressively with optional KV cache and advanced sampling.
    
    Args:
        model (GPTLanguageModel): Model instance.
        tokenizer (AmharicTokenizer): Tokenizer.
        prompt (str): Input prompt string.
        max_new_tokens (int): Maximum new tokens to generate.
        temperature (float): Sampling temperature.
        top_k (Optional[int]): Number of highest probability tokens to consider.
        top_p (Optional[float]): Nucleus sampling cumulative threshold.
        repetition_penalty (float): Penalty for previously seen tokens (> 1.0 reduces repetitions).
        greedy (bool): If True, greedy argmax decoding is used.
        stop_at_eos (bool): If True, stops on EOS or DOC tokens.
        stop_tokens (Optional[List[str]]): Additional tokens to halt on (e.g. ["<EOS>", "<DOC>"]).
        use_cache (bool): If True, uses incremental per-layer KV cache.
        rng (Optional[np.random.RandomState]): Random state for sampling.
        
    Returns:
        generated_text (str): Complete generated text including prompt.
    """
    if hasattr(model, "eval"):
        model.eval()

    if rng is None:
        rng = np.random.RandomState(42)

    # 1. Resolve stop token IDs
    target_stop_tokens = ["<EOS>", "<DOC>"]
    if stop_tokens:
        target_stop_tokens.extend(stop_tokens)

    stop_ids = set()
    if hasattr(tokenizer, "eos_id") and tokenizer.eos_id is not None:
        stop_ids.add(tokenizer.eos_id)
    if hasattr(tokenizer, "vocab"):
        for st in target_stop_tokens:
            if st in tokenizer.vocab:
                stop_ids.add(tokenizer.vocab[st])

    # 2. Resolve masked special token IDs (<PAD>, <UNK>, <BOS>)
    masked_ids = []
    for attr in ["pad_id", "unk_id", "bos_id"]:
        if hasattr(tokenizer, attr):
            val = getattr(tokenizer, attr)
            if val is not None:
                masked_ids.append(val)

    # 3. Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.bos_id if hasattr(tokenizer, "bos_id") else 0]

    block_size = model.config.block_size
    generated_ids = list(prompt_ids)

    # -----------------------------------------------------------------------
    # Generation Mode A: Incremental Decoding with Key-Value Cache (O(T))
    # -----------------------------------------------------------------------
    if use_cache:
        kv_caches = model.init_kv_caches()

        # Prefill phase: process full prompt to initialize KV caches
        prefill_input = np.array([prompt_ids], dtype=np.int64)
        if prefill_input.shape[1] > block_size:
            # Crop prompt if longer than block_size
            prefill_input = prefill_input[:, -block_size:]

        prefill_logits = model.forward(prefill_input, kv_caches=kv_caches, start_pos=0)
        next_logits = prefill_logits[0, -1, :].copy()

        # Sample first token
        next_token = sample_next_token(
            logits=next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            greedy=greedy,
            context_token_ids=generated_ids,
            masked_token_ids=masked_ids,
            rng=rng,
        )
        generated_ids.append(next_token)

        if stop_at_eos and next_token in stop_ids:
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Decode phase: generate one token per step using cached KV states
        for step in range(1, max_new_tokens):
            curr_pos = len(generated_ids) - 1
            step_input = np.array([[next_token]], dtype=np.int64)

            # Incremental forward pass (only 1 token passed!)
            step_logits = model.forward(
                step_input,
                kv_caches=kv_caches,
                start_pos=curr_pos,
            )
            next_logits = step_logits[0, -1, :].copy()

            next_token = sample_next_token(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                greedy=greedy,
                context_token_ids=generated_ids,
                masked_token_ids=masked_ids,
                rng=rng,
            )
            generated_ids.append(next_token)

            if stop_at_eos and next_token in stop_ids:
                break

    # -----------------------------------------------------------------------
    # Generation Mode B: Full Context Forward Without Cache (O(T^2))
    # -----------------------------------------------------------------------
    else:
        for _ in range(max_new_tokens):
            # Crop context window if sequence length exceeds block_size
            idx_cond = np.array([generated_ids[-block_size:]], dtype=np.int64)
            logits = model.forward(idx_cond)
            next_logits = logits[0, -1, :].copy()

            next_token = sample_next_token(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                greedy=greedy,
                context_token_ids=generated_ids,
                masked_token_ids=masked_ids,
                rng=rng,
            )
            generated_ids.append(next_token)

            if stop_at_eos and next_token in stop_ids:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Quantitative Sanity Checks
# ---------------------------------------------------------------------------

def calculate_fraction_ethiopic(text: str) -> float:
    """
    Computes the fraction of non-whitespace characters that are valid Ethiopic.
    """
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    ethiopic_count = sum(1 for c in non_ws if is_ethiopic_character(c))
    return float(ethiopic_count) / float(len(non_ws))


def calculate_repeated_trigram_rate(text: str) -> float:
    """
    Calculates the repeated-trigram rate across character 3-grams:
      rate = 1.0 - (unique_trigrams / total_trigrams)
    """
    if len(text) < 3:
        return 0.0
    trigrams = [text[i : i + 3] for i in range(len(text) - 2)]
    total_trigrams = len(trigrams)
    unique_trigrams = len(set(trigrams))
    return 1.0 - (float(unique_trigrams) / float(total_trigrams))


def calculate_greedy_perplexity(
    model: GPTLanguageModel,
    tokenizer: AmharicTokenizer,
    text: str,
) -> float:
    """
    Evaluates the teacher-forcing perplexity of generated text under the model:
      PPL = exp(CrossEntropyLoss)
    """
    if hasattr(model, "eval"):
        model.eval()

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 2:
        return 1.0

    block_size = model.config.block_size
    # Truncate to block_size if necessary
    token_ids = token_ids[:block_size]

    inputs = np.array([token_ids[:-1]], dtype=np.int64)
    targets = np.array([token_ids[1:]], dtype=np.int64)

    logits = model.forward(inputs)
    loss, _ = cross_entropy_loss(logits, targets)
    return float(np.exp(loss))


def compute_generation_metrics(
    model: GPTLanguageModel,
    tokenizer: AmharicTokenizer,
    text: str,
) -> Dict[str, float]:
    """
    Runs full quantitative sanity checks on generated Amharic text.
    """
    frac_eth = calculate_fraction_ethiopic(text)
    rep_trigram = calculate_repeated_trigram_rate(text)
    ppl = calculate_greedy_perplexity(model, tokenizer, text)
    return {
        "fraction_ethiopic": frac_eth,
        "repeated_trigram_rate": rep_trigram,
        "greedy_perplexity": ppl,
    }


# ---------------------------------------------------------------------------
# Timing Comparison Benchmark: 200 Tokens (Cache vs No-Cache)
# ---------------------------------------------------------------------------

def compare_cache_speed(
    model: GPTLanguageModel,
    tokenizer: AmharicTokenizer,
    prompt: str = "ሰላም",
    num_tokens: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Benchmarks incremental KV Cache decoding vs No-Cache full forward for 200 tokens.
    """
    # 1. Run with KV Cache
    rng_cache = np.random.RandomState(seed)
    t0 = time.perf_counter()
    out_cache = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=num_tokens,
        greedy=True,
        stop_at_eos=False,
        use_cache=True,
        rng=rng_cache,
    )
    time_cache = time.perf_counter() - t0

    # 2. Run without Cache
    rng_nocache = np.random.RandomState(seed)
    t0 = time.perf_counter()
    out_nocache = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=num_tokens,
        greedy=True,
        stop_at_eos=False,
        use_cache=False,
        rng=rng_nocache,
    )
    time_nocache = time.perf_counter() - t0

    speedup = time_nocache / max(time_cache, 1e-6)

    return {
        "num_tokens": num_tokens,
        "time_cache_sec": time_cache,
        "time_nocache_sec": time_nocache,
        "speedup_factor": speedup,
        "tokens_per_sec_cache": num_tokens / max(time_cache, 1e-6),
        "tokens_per_sec_nocache": num_tokens / max(time_nocache, 1e-6),
        "output_match": out_cache == out_nocache,
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autoregressive Amharic Text Generation (Pure NumPy)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where model is saved")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_best", help="Checkpoint base name")
    parser.add_argument("--prompt", type=str, default="ሰላም", help="Amharic starting prompt")
    parser.add_argument("--prompt_file", type=str, default=None, help="Optional text file with prompts (one per line)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of completions to generate per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Maximum number of new characters/tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k filtering (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus filtering (0 to disable)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty factor")
    parser.add_argument("--greedy", action="store_true", help="Use greedy argmax decoding")
    parser.add_argument("--no_cache", action="store_true", help="Disable KV caching (use full forward pass)")
    parser.add_argument("--benchmark", action="store_true", help="Run 200 token cache speed benchmark")
    parser.add_argument("--output_file", type=str, default=None, help="Custom output path for saving samples")
    parser.add_argument("--run_name", type=str, default="amharic_eval", help="Run identifier used in default output filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    args = parser.parse_args()

    # Determine checkpoint path
    ckpt_name = args.checkpoint_name
    meta_path = os.path.join(args.checkpoint_dir, f"{ckpt_name}_meta.json")
    if not os.path.exists(meta_path):
        if os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint_meta.json")):
            ckpt_name = "checkpoint"
            meta_path = os.path.join(args.checkpoint_dir, "checkpoint_meta.json")
        else:
            print(f"Error: No checkpoint found in '{args.checkpoint_dir}' (looked for {ckpt_name}_meta.json).")
            print("Please run training first:")
            print("  python -m src.train --config configs/train_small.json")
            return

    rng = np.random.RandomState(args.seed)
    model, config, tokenizer, meta = load_checkpoint(args.checkpoint_dir, name=ckpt_name, rng=rng)
    assert tokenizer is not None, "Tokenizer vocabulary missing from checkpoint."

    top_k_val = None if args.top_k <= 0 else args.top_k
    top_p_val = None if args.top_p <= 0.0 or args.top_p >= 1.0 else args.top_p
    use_cache = not args.no_cache

    # 1. Benchmark option
    if args.benchmark:
        print("\n" + "=" * 60)
        print("Running KV Cache Timing Benchmark (200 tokens)...")
        print("=" * 60)
        bench = compare_cache_speed(model, tokenizer, prompt=args.prompt, num_tokens=200, seed=args.seed)
        print(f"Time with KV Cache:    {bench['time_cache_sec']:.3f}s ({bench['tokens_per_sec_cache']:.1f} tok/s)")
        print(f"Time without KV Cache: {bench['time_nocache_sec']:.3f}s ({bench['tokens_per_sec_nocache']:.1f} tok/s)")
        print(f"Speedup Factor:        {bench['speedup_factor']:.2f}x faster")
        print(f"Outputs Match:         {bench['output_match']}")
        print("=" * 60 + "\n")

    # 2. Gather prompts
    prompts = []
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        prompts = [args.prompt]

    # 3. Destination file
    os.makedirs("outputs", exist_ok=True)
    out_file = args.output_file or os.path.join("outputs", f"samples_{args.run_name}.txt")

    results_log = []
    header = (
        f"=== Amharic GPT Generation Log ===\n"
        f"Checkpoint: {args.checkpoint_dir}/{ckpt_name}\n"
        f"Parameters: {model.num_parameters():,}\n"
        f"Settings: greedy={args.greedy}, T={args.temperature}, top_k={top_k_val}, "
        f"top_p={top_p_val}, rep_penalty={args.repetition_penalty}, KV_cache={use_cache}, seed={args.seed}\n"
        f"===================================\n\n"
    )
    results_log.append(header)
    print(header)

    sample_count = 0
    for p_idx, prompt_text in enumerate(prompts):
        print(f"Prompt [{p_idx + 1}/{len(prompts)}]: \"{prompt_text}\"")
        results_log.append(f"--- Prompt: \"{prompt_text}\" ---\n")

        for s_idx in range(args.num_samples):
            sample_count += 1
            sample_seed = args.seed + sample_count - 1
            s_rng = np.random.RandomState(sample_seed)

            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k_val,
                top_p=top_p_val,
                repetition_penalty=args.repetition_penalty,
                greedy=args.greedy,
                use_cache=use_cache,
                rng=s_rng,
            )

            # Quantitative sanity metrics
            metrics = compute_generation_metrics(model, tokenizer, output)

            sample_str = (
                f"Sample {s_idx + 1} (seed={sample_seed}):\n"
                f"{output}\n"
                f"[Sanity Metrics] Ethiopic Ratio: {metrics['fraction_ethiopic']:.3f} | "
                f"Repeated Trigrams: {metrics['repeated_trigram_rate']:.3f} | "
                f"Perplexity: {metrics['greedy_perplexity']:.2f}\n\n"
            )
            print(sample_str)
            results_log.append(sample_str)

    # 4. Save results to disk
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(results_log)
    print(f"Results successfully saved to: {out_file}\n")


if __name__ == "__main__":
    main()
