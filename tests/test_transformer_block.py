"""
Unit tests and gradient checks for TransformerBlock.
"""

import numpy as np
import pytest
from src.transformer_block import TransformerBlock


def test_transformer_block_shapes():
    B, T, D, H = 2, 4, 16, 4
    block = TransformerBlock(n_embd=D, n_head=H, ffn_dim=64)
    x = np.random.randn(B, T, D).astype(np.float32)

    out = block.forward(x)
    assert out.shape == (B, T, D)


def test_transformer_block_backward_numerical():
    B, T, D, H = 1, 3, 4, 2
    block = TransformerBlock(n_embd=D, n_head=H, ffn_dim=8)

    # Cast to float64 for finite difference check
    block.ln_1.gamma = block.ln_1.gamma.astype(np.float64)
    block.ln_1.beta = block.ln_1.beta.astype(np.float64)
    block.attn.W_q = block.attn.W_q.astype(np.float64)
    block.attn.b_q = block.attn.b_q.astype(np.float64)
    block.attn.W_k = block.attn.W_k.astype(np.float64)
    block.attn.b_k = block.attn.b_k.astype(np.float64)
    block.attn.W_v = block.attn.W_v.astype(np.float64)
    block.attn.b_v = block.attn.b_v.astype(np.float64)
    block.attn.W_o = block.attn.W_o.astype(np.float64)
    block.attn.b_o = block.attn.b_o.astype(np.float64)
    block.ln_2.gamma = block.ln_2.gamma.astype(np.float64)
    block.ln_2.beta = block.ln_2.beta.astype(np.float64)
    block.ffn.W1 = block.ffn.W1.astype(np.float64)
    block.ffn.b1 = block.ffn.b1.astype(np.float64)
    block.ffn.W2 = block.ffn.W2.astype(np.float64)
    block.ffn.b2 = block.ffn.b2.astype(np.float64)

    x = np.random.randn(B, T, D).astype(np.float64)
    d_out = np.random.randn(B, T, D).astype(np.float64)

    _ = block.forward(x)
    dx_analytical = block.backward(d_out)

    # Numerical gradient check on input X
    eps = 1e-6
    dx_num = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                x_pos = x.copy()
                x_neg = x.copy()
                x_pos[b, t, d] += eps
                x_neg[b, t, d] -= eps

                out_pos = block.forward(x_pos)
                out_neg = block.forward(x_neg)

                dx_num[b, t, d] = np.sum((out_pos - out_neg) * d_out) / (2 * eps)

    assert np.allclose(dx_analytical, dx_num, atol=1e-5)
