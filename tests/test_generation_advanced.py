"""
Comprehensive tests for KV Cache, incremental decoding, advanced sampling,
sliding window overflow, and quantitative sanity metrics (Task 6).
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.model import GPTLanguageModel
from src.attention import KVCache
from src.generate import (
    sample_next_token,
    generate,
    calculate_fraction_ethiopic,
    calculate_repeated_trigram_rate,
    calculate_greedy_perplexity,
    compute_generation_metrics,
    compare_cache_speed,
)


def test_kv_cache_logits_equal_no_cache_full_forward():
    """
    REQUIRED TEST:
    Logits for every generated step equal the no-cache full forward within 1e-5.
    (Lecture Slide 15)
    """
    rng = np.random.RandomState(42)
    vocab_size = 35
    block_size = 64
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPTLanguageModel(config, rng=rng)
    model.eval()

    # Initial prompt sequence
    prompt_ids = [5, 12, 19, 24, 7]
    idx_history = list(prompt_ids)

    # 1. Prefill step comparison
    kv_caches = model.init_kv_caches()
    prefill_input = np.array([prompt_ids], dtype=np.int64)
    logits_cache = model.forward(prefill_input, kv_caches=kv_caches, start_pos=0)
    logits_nocache = model.forward(prefill_input)

    max_diff_step0 = np.max(np.abs(logits_cache[0, -1, :] - logits_nocache[0, -1, :]))
    assert max_diff_step0 < 1e-5, f"Step 0 prefill mismatch: max diff {max_diff_step0:.2e}"

    # 2. Sequential generation comparison for 25 steps
    for step in range(25):
        # Determine next token deterministically (greedy from current cached logits)
        next_token = int(np.argmax(logits_cache[0, -1, :]))
        idx_history.append(next_token)

        # Full forward pass without cache
        full_input = np.array([idx_history], dtype=np.int64)
        full_logits = model.forward(full_input)
        expected_next_logits = full_logits[0, -1, :]

        # Incremental forward pass with KV cache (passing only the single new token!)
        step_input = np.array([[next_token]], dtype=np.int64)
        curr_pos = len(idx_history) - 1
        step_logits = model.forward(step_input, kv_caches=kv_caches, start_pos=curr_pos)
        actual_next_logits = step_logits[0, -1, :]

        max_diff = np.max(np.abs(actual_next_logits - expected_next_logits))
        assert max_diff < 1e-5, (
            f"Step {step + 1} logits mismatch: max diff = {max_diff:.2e} (exceeds 1e-5 tolerance)"
        )
        logits_cache = step_logits


def test_sampling_distribution_top_k():
    """
    Verifies that top-k filtering keeps EXACTLY k highest probability tokens
    and assigns zero probability (-inf logit) to all others.
    """
    vocab_size = 20
    k = 5
    rng = np.random.RandomState(42)
    # Logits: index i has logit i (so indices 15, 16, 17, 18, 19 are top-5)
    logits = np.arange(vocab_size, dtype=np.float64)

    samples = [
        sample_next_token(logits, top_k=k, temperature=1.0, rng=np.random.RandomState(seed))
        for seed in range(500)
    ]
    sampled_set = set(samples)

    # Expected top-k tokens are strictly {15, 16, 17, 18, 19}
    expected_top_k = set(range(vocab_size - k, vocab_size))
    assert sampled_set.issubset(expected_top_k), f"Sampled token outside top-{k}: {sampled_set - expected_top_k}"
    assert len(sampled_set) == k, f"Expected all {k} tokens to be sampled across 500 draws, got {len(sampled_set)}"


def test_sampling_distribution_top_p_nucleus():
    """
    Verifies that top-p (nucleus) filtering keeps the minimal set of tokens
    whose cumulative probability mass >= top_p.
    """
    # Create probabilities: [0.50, 0.25, 0.15, 0.07, 0.03]
    # For top_p = 0.60:
    #   Token 0 has cumsum = 0.50 (< 0.60)
    #   Token 1 has cumsum = 0.75 (>= 0.60)
    # Minimal nucleus set is {0, 1}.
    probs = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
    logits = np.log(probs)

    samples = [
        sample_next_token(logits, top_p=0.60, temperature=1.0, rng=np.random.RandomState(seed))
        for seed in range(500)
    ]
    sampled_set = set(samples)

    assert sampled_set == {0, 1}, f"Expected nucleus tokens {{0, 1}}, got {sampled_set}"


def test_temperature_approaching_zero_equals_greedy():
    """
    Verifies that as temperature -> 0, sampling matches greedy decoding bit-for-bit.
    """
    text = "ሰላም ለኢትዮጵያ"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)
    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))

    prompt = "ሰ"
    gen_greedy = generate(model, tokenizer, prompt, max_new_tokens=15, greedy=True)
    gen_temp_zero = generate(model, tokenizer, prompt, max_new_tokens=15, temperature=1e-7, greedy=False)

    assert gen_temp_zero == gen_greedy, (
        f"Temperature ~ 0 output differs from greedy:\nGreedy: {gen_greedy}\nTemp:   {gen_temp_zero}"
    )


def test_special_tokens_masked_from_sampling():
    """
    Verifies that <PAD>, <UNK>, and <BOS> are always masked to -inf and cannot be sampled.
    """
    vocab_size = 10
    logits = np.ones(vocab_size) * 0.0
    # Make special tokens artificially have massive logits
    pad_id, unk_id, bos_id = 0, 1, 2
    logits[pad_id] = 1000.0
    logits[unk_id] = 1000.0
    logits[bos_id] = 1000.0

    samples = [
        sample_next_token(
            logits,
            masked_token_ids=[pad_id, unk_id, bos_id],
            temperature=1.0,
            rng=np.random.RandomState(seed),
        )
        for seed in range(200)
    ]

    for s in samples:
        assert s not in {pad_id, unk_id, bos_id}, f"Masked special token {s} was sampled!"


def test_repetition_penalty_reduces_repeat_likelihood():
    """
    Verifies that repetition penalty discounts logits of seen tokens.
    """
    logits = np.array([2.0, 2.0, 0.5, 0.1])
    # Token 0 is already in context
    context = [0]

    # Without repetition penalty: argmax is token 0 (tie-broken to 0)
    token_no_pen = sample_next_token(logits, repetition_penalty=1.0, greedy=True, context_token_ids=context)
    assert token_no_pen == 0

    # With repetition penalty = 2.0: token 0 logit drops from 2.0 to 1.0, token 1 stays 2.0
    token_with_pen = sample_next_token(logits, repetition_penalty=2.0, greedy=True, context_token_ids=context)
    assert token_with_pen == 1


def test_stop_on_eos_and_doc():
    """
    Verifies that generation terminates when <EOS> or <DOC> is encountered.
    """
    text = "ሰላም"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)
    doc_id = tokenizer.vocab_size
    tokenizer.char_to_id["<DOC>"] = doc_id
    tokenizer.id_to_char[doc_id] = "<DOC>"

    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config)

    # Force model to output <DOC> on first step
    doc_id = tokenizer.vocab["<DOC>"]
    model.b_vocab.fill(-100.0)
    model.b_vocab[doc_id] = 100.0

    gen_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="ሰ",
        max_new_tokens=20,
        greedy=True,
        stop_at_eos=True,
        stop_tokens=["<DOC>"],
    )
    # Stops immediately after first token (<DOC>)
    assert gen_text in ["ሰ", "ሰ<DOC>"]


def test_context_overflow_sliding_window():
    """
    Verifies that the sliding window policy handles sequence lengths exceeding block_size.
    """
    vocab_size = 20
    small_block_size = 8
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=small_block_size,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))
    model.eval()

    # Generate 15 tokens when block_size is only 8
    kv_caches = model.init_kv_caches()
    tokens = [1, 2, 3]
    # Prefill
    model.forward(np.array([tokens]), kv_caches=kv_caches, start_pos=0)

    # Decode past block_size
    for step in range(12):
        curr_token = tokens[-1]
        step_logits = model.forward(
            np.array([[curr_token]]),
            kv_caches=kv_caches,
            start_pos=len(tokens) - 1,
        )
        assert step_logits.shape == (1, 1, vocab_size)
        tokens.append(int(np.argmax(step_logits[0, -1, :])))

    # Cache should have capped at small_block_size
    assert kv_caches[0].seq_len == small_block_size


def test_timing_comparison_200_tokens():
    """
    Verifies the timing comparison for 200 new tokens (cache vs no-cache).
    """
    vocab_size = 30
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))
    tokenizer = AmharicTokenizer()
    # Dummy alphabet
    tokenizer.build_vocab("ሀሁሂሃሄህሆለሉሊላሌልሎ")

    bench = compare_cache_speed(model, tokenizer, prompt="ሀ", num_tokens=200, seed=42)
    assert bench["num_tokens"] == 200
    assert bench["output_match"] is True, "KV cache and no-cache outputs did not match!"
    assert bench["speedup_factor"] > 1.0, f"Expected cache speedup > 1.0x, got {bench['speedup_factor']:.2f}x"
    assert bench["time_cache_sec"] < bench["time_nocache_sec"]


def test_quantitative_sanity_metrics():
    """
    Verifies calculation of Ethiopic character ratio, repeated trigram rate,
    and teacher-forcing perplexity.
    """
    # 1. Pure Ethiopic
    amharic_text = "ሰላም ለኢትዮጵያ"
    eth_ratio = calculate_fraction_ethiopic(amharic_text)
    assert eth_ratio == 1.0

    # 2. Mixed English / Amharic
    mixed_text = "ሰላም hello"
    mixed_ratio = calculate_fraction_ethiopic(mixed_text)
    assert 0.0 < mixed_ratio < 1.0

    # 3. Repeated trigrams
    repetitive_text = "ሰላምሰላምሰላምሰላም"
    rep_rate = calculate_repeated_trigram_rate(repetitive_text)
    assert rep_rate > 0.5, f"Expected high trigram repetition rate, got {rep_rate}"

    # 4. Perplexity under model
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(amharic_text)
    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))

    ppl = calculate_greedy_perplexity(model, tokenizer, amharic_text)
    assert np.isfinite(ppl)
    assert ppl >= 1.0
