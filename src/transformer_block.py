"""
Transformer Block with Pre-LayerNorm and Residual Connections in Pure NumPy.

Why Pre-LayerNorm?
------------------
The original Transformer (Vaswani et al., 2017) placed LayerNorm *after* the
residual addition (Post-LN):
  x = LayerNorm(x + SubLayer(x))
In Post-LN, gradients passing through the residual stream get multiplied by
normalization factors, which often destabilizes deep networks and requires
delicate learning rate warmup.

Modern GPT models (GPT-2, GPT-3, LLaMA) use **Pre-LayerNorm**:
  x = x + SubLayer(LayerNorm(x))
In Pre-LN, the identity shortcut x -> x is uninterrupted. Information and gradients
can propagate directly through the residual "highway" without attenuation, making
deep networks exceptionally stable and easy to train.

Why Residual Connections?
-------------------------
In a deep stack of L layers:
  x_L = x_0 + sum_{l=1}^L SubLayer_l(x)
During backpropagation, the derivative of the identity mapping is 1.0:
  dL/dx_0 = dL/dx_L * (1 + sum_l d(SubLayer_l)/dx_0)
This guarantees that the gradient has a direct path of 1.0 back to earlier layers,
completely eliminating the vanishing gradient problem!
"""

import numpy as np
from typing import Optional, Tuple, Dict
from src.normalization import LayerNorm
from src.attention import CausalMultiHeadAttention
from src.feed_forward import FeedForward


class TransformerBlock:
    """
    A single GPT-style Pre-LN Transformer Block.

    Structure:
    ----------
    Input x
      ↓
    x_norm1 = ln_1(x)
    attn_out = attention(x_norm1)
    x1 = x + attn_out           # First residual connection
      ↓
    x_norm2 = ln_2(x1)
    ffn_out = feed_forward(x_norm2)
    output = x1 + ffn_out       # Second residual connection

    Array Shapes:
    -------------
    Input x: shape (B, T, n_embd)
    Output:  shape (B, T, n_embd)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        ffn_dim: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None
    ):
        self.n_embd = n_embd
        self.n_head = n_head
        self.ffn_dim = ffn_dim or (4 * n_embd)

        # Pre-attention LayerNorm and Multi-Head Attention
        self.ln_1 = LayerNorm(n_embd=n_embd)
        self.attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, rng=rng)

        # Pre-FFN LayerNorm and Feed-Forward Network
        self.ln_2 = LayerNorm(n_embd=n_embd)
        self.ffn = FeedForward(n_embd=n_embd, ffn_dim=self.ffn_dim, rng=rng)

        # Cache for backward pass
        self.cached_x0: Optional[np.ndarray] = None
        self.cached_x1: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Transformer block.
        
        Input x: shape (B, T, n_embd)
        Returns: shape (B, T, n_embd)
        """
        self.cached_x0 = x

        # 1. Pre-LN Attention sub-layer with residual connection
        # x1 = x + Attention(LN_1(x))
        norm_x0 = self.ln_1.forward(x)
        attn_out = self.attn.forward(norm_x0)
        x1 = x + attn_out
        self.cached_x1 = x1

        # 2. Pre-LN Feed-Forward sub-layer with residual connection
        # output = x1 + FFN(LN_2(x1))
        norm_x1 = self.ln_2.forward(x1)
        ffn_out = self.ffn.forward(norm_x1)
        out = x1 + ffn_out
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for the Transformer block.
        
        Derivation:
        -----------
        Sub-layer 2: out = x1 + ffn(ln_2(x1))
          dL/d_ffn_out = d_out
          dL/dx1 = d_out + ln_2.backward(ffn.backward(d_out))
          
        Sub-layer 1: x1 = x0 + attn(ln_1(x0))
          dL/d_attn_out = dL/dx1
          dL/dx0 = dL/dx1 + ln_1.backward(attn.backward(dL/dx1))
          
        Returns dL/dx0 of shape (B, T, n_embd).
        """
        assert self.cached_x0 is not None and self.cached_x1 is not None, "Call forward first."

        # Backward through Sub-layer 2 (FFN + LN_2 + Residual)
        d_ffn_out = d_out.copy()
        d_norm_x1 = self.ffn.backward(d_ffn_out)
        d_ln2_in = self.ln_2.backward(d_norm_x1)
        
        # Residual gradient splits into identity path and FFN path
        dx1 = d_out + d_ln2_in

        # Backward through Sub-layer 1 (Attention + LN_1 + Residual)
        d_attn_out = dx1.copy()
        d_norm_x0 = self.attn.backward(d_attn_out)
        d_ln1_in = self.ln_1.backward(d_norm_x0)

        # Residual gradient splits into identity path and Attention path
        dx0 = dx1 + d_ln1_in
        return dx0

    def get_params_and_grads(self, prefix: str = "") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        p = f"{prefix}." if prefix else ""
        params = {}
        params.update(self.ln_1.get_params_and_grads(f"{p}ln_1"))
        params.update(self.attn.get_params_and_grads(f"{p}attn"))
        params.update(self.ln_2.get_params_and_grads(f"{p}ln_2"))
        params.update(self.ffn.get_params_and_grads(f"{p}ffn"))
        return params

    def zero_grad(self) -> None:
        self.ln_1.zero_grad()
        self.attn.zero_grad()
        self.ln_2.zero_grad()
        self.ffn.zero_grad()
