"""
Shape and Dtype Contract Tests for Every Module in the Repository.

Verifies that for every module:
1. Output tensor shapes strictly match mathematical contracts.
2. Output dtypes strictly preserve configuration dtype (float32 vs float64).
3. Parameter and gradient shapes match exactly.
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.normalization import LayerNorm
from src.activations import GELU, Dropout
from src.feed_forward import FeedForward
from src.attention import CausalSelfAttention, CausalMultiHeadAttention
from src.embeddings import TokenEmbedding, PositionalEmbedding, GPTEmbeddings
from src.transformer_block import TransformerBlock


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_layernorm_shape_and_dtype_contract(dtype_str: str):
    dt = getattr(np, dtype_str)
    B, T, D = 3, 5, 16
    ln = LayerNorm(n_embd=D, dtype=dt)
    x = np.random.randn(B, T, D).astype(dt)
    out = ln.forward(x)
    assert out.shape == (B, T, D)
    assert out.dtype == dt

    d_out = np.random.randn(B, T, D).astype(dt)
    dx = ln.backward(d_out)
    assert dx.shape == (B, T, D)
    assert dx.dtype == dt
    assert ln.grad_gamma.shape == ln.gamma.shape
    assert ln.grad_beta.shape == ln.beta.shape


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_activations_shape_and_dtype_contract(dtype_str: str):
    dt = getattr(np, dtype_str)
    x = np.random.randn(2, 4, 8).astype(dt)
    gelu = GELU()
    out = gelu.forward(x)
    assert out.shape == x.shape
    assert out.dtype == dt

    d_out = np.random.randn(2, 4, 8).astype(dt)
    dx = gelu.backward(d_out)
    assert dx.shape == x.shape
    assert dx.dtype == dt

    dropout = Dropout(p=0.1)
    out_drop = dropout.forward(x, training=True)
    assert out_drop.shape == x.shape
    assert out_drop.dtype == dt
    dx_drop = dropout.backward(d_out)
    assert dx_drop.shape == x.shape
    assert dx_drop.dtype == dt


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_attention_shape_and_dtype_contract(dtype_str: str):
    dt = getattr(np, dtype_str)
    B, T, D, H = 2, 6, 16, 4
    attn = CausalMultiHeadAttention(n_embd=D, n_head=H, dtype=dt)
    x = np.random.randn(B, T, D).astype(dt)

    out = attn.forward(x, training=False)
    assert out.shape == (B, T, D)
    assert out.dtype == dt

    d_out = np.random.randn(B, T, D).astype(dt)
    dx = attn.backward(d_out)
    assert dx.shape == (B, T, D)
    assert dx.dtype == dt


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_feedforward_shape_and_dtype_contract(dtype_str: str):
    dt = getattr(np, dtype_str)
    B, T, D, H = 2, 4, 12, 48
    ffn = FeedForward(n_embd=D, ffn_dim=H, dtype=dt)
    x = np.random.randn(B, T, D).astype(dt)

    out = ffn.forward(x, training=False)
    assert out.shape == (B, T, D)
    assert out.dtype == dt

    d_out = np.random.randn(B, T, D).astype(dt)
    dx = ffn.backward(d_out)
    assert dx.shape == (B, T, D)
    assert dx.dtype == dt


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_embeddings_shape_and_dtype_contract(dtype_str: str):
    dt = getattr(np, dtype_str)
    vocab_size = 25
    block_size = 10
    n_embd = 16
    B, T = 2, 7

    emb = GPTEmbeddings(vocab_size, block_size, n_embd, dtype=dt)
    token_ids = np.random.randint(0, vocab_size, size=(B, T), dtype=np.int64)

    out = emb.forward(token_ids, training=False)
    assert out.shape == (B, T, n_embd)
    assert out.dtype == dt

    d_out = np.random.randn(B, T, n_embd).astype(dt)
    emb.backward(d_out)
    assert emb.token_emb.grad_W.shape == (vocab_size, n_embd)
    assert emb.pos_emb.grad_W.shape == (block_size, n_embd)


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_transformer_block_and_model_contract(dtype_str: str):
    dt = getattr(np, dtype_str)
    vocab_size = 20
    block_size = 8
    n_embd = 16
    B, T = 2, 5

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        n_embd=n_embd,
        dtype=dtype_str,
    )
    model = GPTLanguageModel(config, rng=np.random.RandomState(42))
    token_ids = np.random.randint(0, vocab_size, size=(B, T), dtype=np.int64)

    logits = model.forward(token_ids)
    assert logits.shape == (B, T, vocab_size)
    assert logits.dtype == dt
