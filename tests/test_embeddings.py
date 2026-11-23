"""
Unit tests for Token and Positional Embeddings.
"""

import numpy as np
import pytest
from src.embeddings import TokenEmbedding, PositionalEmbedding, GPTEmbeddings


def test_token_embedding_forward_and_backward():
    vocab_size = 50
    n_embd = 16
    emb = TokenEmbedding(vocab_size=vocab_size, n_embd=n_embd)

    token_ids = np.array([[5, 12, 5], [0, 5, 49]])  # Batch 2, Seq 3, note duplicate token 5
    out = emb.forward(token_ids)
    assert out.shape == (2, 3, n_embd)

    # Output rows should equal embedding rows
    assert np.allclose(out[0, 0], emb.W[5])
    assert np.allclose(out[0, 2], emb.W[5])

    # Backward gradient accumulation test
    d_out = np.ones_like(out)
    emb.backward(d_out)
    
    # Token 5 appears 3 times in total across batch, so its gradient sum should be 3 * 1.0 = 3.0
    assert np.allclose(emb.grad_W[5], 3.0)
    # Token 12 appears once, gradient should be 1.0
    assert np.allclose(emb.grad_W[12], 1.0)
    # Token 1 never appears, gradient should be 0.0
    assert np.allclose(emb.grad_W[1], 0.0)


def test_positional_embedding_forward_and_backward():
    block_size = 32
    n_embd = 8
    pos_emb = PositionalEmbedding(block_size=block_size, n_embd=n_embd)

    seq_len = 4
    out = pos_emb.forward(seq_len)
    assert out.shape == (1, seq_len, n_embd)

    # Backward with batch size 2
    d_out = np.ones((2, seq_len, n_embd))
    pos_emb.backward(d_out)

    # Each position across 2 batches should sum to 2.0
    assert np.allclose(pos_emb.grad_W[:seq_len], 2.0)
    # Positions beyond seq_len should remain 0.0
    assert np.allclose(pos_emb.grad_W[seq_len:], 0.0)


def test_gpt_embeddings_combined():
    vocab_size = 100
    block_size = 32
    n_embd = 16
    gpt_emb = GPTEmbeddings(vocab_size, block_size, n_embd)

    token_ids = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
    out = gpt_emb.forward(token_ids)
    assert out.shape == (2, 4, n_embd)

    d_out = np.ones_like(out)
    gpt_emb.backward(d_out)

    params = gpt_emb.get_params_and_grads()
    assert "wte" in params and "wpe" in params
    wte, grad_wte = params["wte"]
    wpe, grad_wpe = params["wpe"]
    assert not np.allclose(grad_wte, 0.0)
    assert not np.allclose(grad_wpe, 0.0)

    gpt_emb.zero_grad()
    assert np.allclose(grad_wte, 0.0)
    assert np.allclose(grad_wpe, 0.0)
