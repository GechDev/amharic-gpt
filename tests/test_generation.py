"""
Unit tests for Autoregressive Text Generation.
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.model import GPTLanguageModel
from src.generate import generate


def test_autoregressive_generation_greedy_and_temperature():
    text = "ሰላም ለኢትዮጵያ እና ለመላው ዓለም ይሁን።"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)

    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))

    prompt = "ሰላም"
    # Greedy generation test
    gen_greedy = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=15,
        greedy=True,
    )
    assert gen_greedy.startswith(prompt)
    assert len(gen_greedy) > len(prompt)

    # Temperature sampling test
    gen_sample = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=15,
        temperature=0.8,
        top_k=5,
        greedy=False,
        rng=np.random.RandomState(123),
    )
    assert gen_sample.startswith(prompt)
    assert len(gen_sample) > len(prompt)


def test_generation_stopping_at_eos():
    text = "ሰላም"
    tokenizer = AmharicTokenizer()
    tokenizer.build_vocab(text)

    config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config)

    # Overwrite lm_head biases so EOS is always selected
    model.b_vocab.fill(-100.0)
    model.b_vocab[tokenizer.eos_id] = 100.0

    prompt = "ሰ"
    gen_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        greedy=True,
        stop_at_eos=True,
    )
    # Since EOS is selected at the very first step, output should just be the prompt
    assert gen_text == prompt
