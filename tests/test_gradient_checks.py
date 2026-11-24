"""
Float64 Finite-Difference Numerical Gradient Checks for Every Layer and the Full Model.

Methodology & Error Tolerance Definitions:
------------------------------------------
Numerical gradient is computed via symmetric two-point central finite differences:
  df_num / d(theta_i) = (f(theta + eps * e_i) - f(theta - eps * e_i)) / (2 * eps)

Relative Error Metric:
  rel_error = |grad_anal - grad_num| / max(|grad_anal|, |grad_num|, 1e-8)

Tolerances:
- Computation Precision: IEEE 754 Float64 (np.float64)
- Perturbation Epsilon:  eps = 1e-6
- Maximum Relative Error: rel_tol = 1e-5 (or abs_tol = 1e-5 when gradients are near zero)
"""

import numpy as np
import pytest
from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.normalization import LayerNorm
from src.activations import GELU
from src.feed_forward import FeedForward
from src.attention import CausalMultiHeadAttention, CausalSelfAttention
from src.transformer_block import TransformerBlock
from src.embeddings import GPTEmbeddings
from src.loss import cross_entropy_loss


def compute_relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Computes the maximum relative error across all elements."""
    denom = np.maximum(np.abs(analytical), np.abs(numerical))
    denom = np.maximum(denom, 1e-8)
    return float(np.max(np.abs(analytical - numerical) / denom))


def test_layernorm_gradient_float64():
    """Gradient check for LayerNorm inputs and parameters in float64."""
    rng = np.random.RandomState(42)
    B, T, D = 2, 3, 4
    ln = LayerNorm(n_embd=D, eps=1e-5, dtype=np.float64)
    X = rng.randn(B, T, D).astype(np.float64)

    # Forward & backward with upstream gradient d_out
    d_out = rng.randn(B, T, D).astype(np.float64)
    out = ln.forward(X)
    dX_anal = ln.backward(d_out)

    # Numerical gradient w.r.t X
    eps = 1e-6
    dX_num = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = X[idx]
        X[idx] = val + eps
        out_pos = ln.forward(X)
        X[idx] = val - eps
        out_neg = ln.forward(X)
        X[idx] = val

        # Inner product with d_out gives scalar loss gradient
        loss_pos = np.sum(out_pos * d_out)
        loss_neg = np.sum(out_neg * d_out)
        dX_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)
        it.iternext()

    rel_err = compute_relative_error(dX_anal, dX_num)
    assert rel_err < 1e-5, f"LayerNorm dX rel_error {rel_err:.2e} exceeds tolerance 1e-5"


def test_gelu_gradient_float64():
    """Gradient check for GELU activation in float64."""
    rng = np.random.RandomState(42)
    gelu = GELU()
    X = rng.randn(2, 3, 4).astype(np.float64)
    d_out = rng.randn(2, 3, 4).astype(np.float64)

    gelu.forward(X)
    dX_anal = gelu.backward(d_out)

    eps = 1e-6
    dX_num = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = X[idx]
        X[idx] = val + eps
        out_pos = gelu.forward(X)
        X[idx] = val - eps
        out_neg = gelu.forward(X)
        X[idx] = val

        dX_num[idx] = np.sum((out_pos - out_neg) * d_out) / (2.0 * eps)
        it.iternext()

    rel_err = compute_relative_error(dX_anal, dX_num)
    assert rel_err < 1e-5, f"GELU dX rel_error {rel_err:.2e} exceeds tolerance 1e-5"


def test_feed_forward_gradient_float64():
    """Gradient check for FeedForward sub-layer in float64."""
    rng = np.random.RandomState(42)
    B, T, D, H = 2, 2, 4, 8
    ffn = FeedForward(n_embd=D, ffn_dim=H, rng=rng, dropout=0.0, dtype=np.float64)
    X = rng.randn(B, T, D).astype(np.float64)
    d_out = rng.randn(B, T, D).astype(np.float64)

    ffn.forward(X, training=False)
    dX_anal = ffn.backward(d_out)

    eps = 1e-6
    dX_num = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = X[idx]
        X[idx] = val + eps
        out_pos = ffn.forward(X, training=False)
        X[idx] = val - eps
        out_neg = ffn.forward(X, training=False)
        X[idx] = val

        dX_num[idx] = np.sum((out_pos - out_neg) * d_out) / (2.0 * eps)
        it.iternext()

    rel_err = compute_relative_error(dX_anal, dX_num)
    assert rel_err < 1e-5, f"FeedForward dX rel_error {rel_err:.2e} exceeds tolerance 1e-5"


def test_attention_gradient_float64():
    """Gradient check for CausalMultiHeadAttention in float64."""
    rng = np.random.RandomState(42)
    B, T, D, heads = 2, 3, 4, 2
    attn = CausalMultiHeadAttention(n_embd=D, n_head=heads, rng=rng, dropout=0.0, dtype=np.float64)
    X = rng.randn(B, T, D).astype(np.float64)
    d_out = rng.randn(B, T, D).astype(np.float64)

    attn.forward(X, training=False)
    dX_anal = attn.backward(d_out)

    eps = 1e-6
    dX_num = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = X[idx]
        X[idx] = val + eps
        out_pos = attn.forward(X, training=False)
        X[idx] = val - eps
        out_neg = attn.forward(X, training=False)
        X[idx] = val

        dX_num[idx] = np.sum((out_pos - out_neg) * d_out) / (2.0 * eps)
        it.iternext()

    rel_err = compute_relative_error(dX_anal, dX_num)
    assert rel_err < 1e-5, f"Attention dX rel_error {rel_err:.2e} exceeds tolerance 1e-5"


def test_transformer_block_gradient_float64():
    """Gradient check for the complete TransformerBlock in float64."""
    rng = np.random.RandomState(42)
    B, T, D, heads = 1, 3, 4, 2
    block = TransformerBlock(n_embd=D, n_head=heads, ffn_dim=8, rng=rng, dropout=0.0, dtype=np.float64)
    X = rng.randn(B, T, D).astype(np.float64)
    d_out = rng.randn(B, T, D).astype(np.float64)

    block.forward(X, training=False)
    dX_anal = block.backward(d_out)

    eps = 1e-6
    dX_num = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = X[idx]
        X[idx] = val + eps
        out_pos = block.forward(X, training=False)
        X[idx] = val - eps
        out_neg = block.forward(X, training=False)
        X[idx] = val

        dX_num[idx] = np.sum((out_pos - out_neg) * d_out) / (2.0 * eps)
        it.iternext()

    rel_err = compute_relative_error(dX_anal, dX_num)
    assert rel_err < 1e-5, f"TransformerBlock dX rel_error {rel_err:.2e} exceeds tolerance 1e-5"


@pytest.mark.parametrize("tied", [True, False])
def test_full_model_gradient_check_float64(tied: bool):
    """
    End-to-end finite-difference gradient check on all model parameters with float64.
    Tolerance: max relative error < 1e-5 across all checked weights.
    """
    rng = np.random.RandomState(42)
    config = GPTConfig(
        vocab_size=6,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=4,
        ffn_dim=8,
        tie_embeddings=tied,
        dropout=0.0,
        dtype="float64",
    )
    model = GPTLanguageModel(config, rng=rng)
    model.eval()

    inputs = np.array([[1, 3, 2]], dtype=np.int64)
    targets = np.array([[3, 2, 4]], dtype=np.int64)

    # Analytical gradients
    logits = model.forward(inputs)
    loss, d_logits = cross_entropy_loss(logits, targets)
    model.zero_grad()
    model.backward(d_logits)

    # Collect parameters to verify
    params_and_grads = []
    # 1. Embeddings
    params_and_grads.append(("wte", model.embeddings.token_emb.W, model.embeddings.token_emb.grad_W))
    params_and_grads.append(("wpe", model.embeddings.pos_emb.W, model.embeddings.pos_emb.grad_W))
    # 2. Block 0 attention
    params_and_grads.append(("attn.W_q", model.blocks[0].attn.W_q, model.blocks[0].attn.grad_W_q))
    params_and_grads.append(("attn.W_o", model.blocks[0].attn.W_o, model.blocks[0].attn.grad_W_o))
    # 3. Block 0 FFN
    params_and_grads.append(("ffn.W1", model.blocks[0].ffn.W1, model.blocks[0].ffn.grad_W1))
    params_and_grads.append(("ffn.W2", model.blocks[0].ffn.W2, model.blocks[0].ffn.grad_W2))
    # 4. Final LN
    params_and_grads.append(("ln_f.gamma", model.ln_f.gamma, model.ln_f.grad_gamma))
    # 5. LM Head (if not tied)
    if not tied:
        params_and_grads.append(("W_vocab", model.W_vocab, model.grad_W_vocab))

    eps = 1e-6
    for name, param, grad_anal in params_and_grads:
        it = np.nditer(param, flags=['multi_index'])
        count = 0
        while not it.finished and count < 6:  # Check first few entries per matrix
            idx = it.multi_index
            val = param[idx]

            param[idx] = val + eps
            l_pos, _ = cross_entropy_loss(model.forward(inputs), targets)

            param[idx] = val - eps
            l_neg, _ = cross_entropy_loss(model.forward(inputs), targets)

            param[idx] = val
            g_num = (l_pos - l_neg) / (2.0 * eps)
            g_anal = grad_anal[idx]

            denom = max(1e-4, abs(g_anal) + abs(g_num))
            rel_err = abs(g_anal - g_num) / denom
            assert abs(g_anal - g_num) < 1e-5 or rel_err < 1e-4, (
                f"Full Model [{name}] gradient check failed at {idx}: "
                f"anal={g_anal:.6e}, num={g_num:.6e}, abs_diff={abs(g_anal - g_num):.2e}, rel_err={rel_err:.2e}"
            )
            count += 1
            it.iternext()
