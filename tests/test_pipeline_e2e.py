"""
Comprehensive End-to-End Pipeline Integration Test.

Tests the complete lifecycle:
  Amharic Raw Text
       ↓
  Preprocessing
       ↓
  Tokenizer Build & Encode
       ↓
  Dataset Creation
       ↓
  Model Initialization
       ↓
  Forward Pass & Loss
       ↓
  Backpropagation & Optimization (Loss Decreases)
       ↓
  Checkpoint Saving & Loading
       ↓
  Autoregressive Generation
"""

import os
import tempfile
import numpy as np
import pytest

from src.config import GPTConfig
from src.tokenizer import AmharicTokenizer
from src.data import clean_amharic_text, AmharicDataset
from src.model import GPTLanguageModel
from src.train import train
from src.utils import save_checkpoint, load_checkpoint
from src.generate import generate


def test_complete_end_to_end_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Raw text with extra spaces, punctuation, fidels
        raw_amharic = (
            "ሰላም   ለኢትዮጵያ!   ይህ   አነስተኛ   የቋንቋ   ሞዴል   ነው።\n\n"
            "ትምህርት   ለሰው   ልጅ   እድገት   ወሳኝ   ነው።\n"
            "ሰላም   ለኢትዮጵያ!   ይህ   አነስተኛ   የቋንቋ   ሞዴል   ነው።\n"
        )

        # 2. Preprocessing
        clean_text = clean_amharic_text(raw_amharic)
        assert "   " not in clean_text
        assert len(clean_text) > 0

        # 3. Tokenizer
        tokenizer = AmharicTokenizer()
        tokenizer.build_vocab(clean_text)
        assert tokenizer.vocab_size > 10

        # 4. Token IDs and Next-Token Dataset
        token_ids = tokenizer.encode(clean_text)
        block_size = 16
        dataset = AmharicDataset(token_ids, block_size=block_size)
        assert len(dataset) > 0

        # 5. Model Configuration
        config = GPTConfig.get_tiny_config(vocab_size=tokenizer.vocab_size)
        config.block_size = block_size
        config.n_embd = 32
        config.n_head = 2
        config.n_layer = 2
        config.ffn_dim = 64

        model = GPTLanguageModel(config, rng=np.random.RandomState(42))

        # 6. Training & Verification of Loss Reduction
        history = train(
            model=model,
            train_dataset=dataset,
            val_dataset=None,
            max_iters=60,
            batch_size=2,
            learning_rate=3e-3,
            eval_interval=20,
            verbose=False,
        )

        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        assert final_loss < initial_loss, f"Training loss did not decrease: {initial_loss} -> {final_loss}"

        # 7. Checkpointing
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            step=60,
            val_loss=final_loss,
            name="pipeline_model",
        )

        # 8. Load Checkpoint
        loaded_model, loaded_config, loaded_tok, meta = load_checkpoint(ckpt_dir, name="pipeline_model")
        assert loaded_config.n_embd == config.n_embd
        assert loaded_tok.vocab_size == tokenizer.vocab_size

        # 9. Autoregressive Generation
        prompt = "ሰላም"
        generated = generate(
            model=loaded_model,
            tokenizer=loaded_tok,
            prompt=prompt,
            max_new_tokens=20,
            greedy=True,
        )

        assert generated.startswith(prompt)
        assert len(generated) > len(prompt)
        print(f"\n[E2E Pipeline Test] Prompt: '{prompt}' -> Generated: '{generated}'")
