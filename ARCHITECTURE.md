# Amharic GPT Architecture Specification & Tensor Dimensions

This document defines the mathematical architecture, modular components, and exact tensor shapes for **Amharic GPT**, a decoder-only autoregressive language model implemented entirely in **pure Python and NumPy**.

---

## 1. High-Level System Data-Flow Diagram

```
Input Amharic Tokens: [B, T]
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ GPTEmbeddings Module                                        │
│   Token Lookup:      W_te[idx]               -> [B, T, D]   │
│   Positional Lookup: W_pe[0..T-1]            -> [1, T, D]   │
│   Sum + Dropout:     Dropout(W_te + W_pe)    -> [B, T, D]   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼  x_0: [B, T, D]
┌─────────────────────────────────────────────────────────────┐
│ TransformerBlock (Repeated for l = 0 ... L - 1)             │
│                                                             │
│   1. Pre-LN Attention Sub-Layer:                            │
│      norm1 = LN_1(x)                         -> [B, T, D]   │
│      Q = norm1 @ W_q + b_q                   -> [B, H, T, d]│
│      K = norm1 @ W_k + b_k                   -> [B, H, T, d]│
│      V = norm1 @ W_v + b_v                   -> [B, H, T, d]│
│      Scores = (Q @ K^T) / sqrt(d) + Mask     -> [B, H, T, T]│
│      Weights = Softmax(Scores, axis=-1)      -> [B, H, T, T]│
│      HeadOut = (Dropout(Weights) @ V)        -> [B, H, T, d]│
│      Concat = ConcatHeads(HeadOut)           -> [B, T, D]   │
│      AttnOut = Dropout(Concat @ W_o + b_o)   -> [B, T, D]   │
│      x_1 = x_0 + AttnOut  (Residual 1)       -> [B, T, D]   │
│                                                             │
│   2. Pre-LN Feed-Forward Sub-Layer:                         │
│      norm2 = LN_2(x_1)                       -> [B, T, D]   │
│      h1 = GELU(norm2 @ W_1 + b_1)            -> [B, T, D_ff]│
│      FFNOut = Dropout(h1 @ W_2 + b_2)        -> [B, T, D]   │
│      x_out = x_1 + FFNOut (Residual 2)       -> [B, T, D]   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼  x_L: [B, T, D]
┌─────────────────────────────────────────────────────────────┐
│ Final LayerNorm & Vocabulary Projection (LM Head)           │
│   h_final = LN_f(x_L)                        -> [B, T, D]   │
│   Logits  = h_final @ W_head + b_vocab       -> [B, T, V]   │
│             (W_head = W_te^T when tied)                     │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
Next-Token Categorical Distribution: Logits [B, T, V]
```

---

## 2. Dimensional Definitions

| Symbol | Meaning | Example Small Config | Example Tiny Config |
|:---:|:---|:---:|:---:|
| $B$ | Batch Size (Parallel independent sequences) | $4$ | $2$ |
| $T$ | Sequence Context Window (`block_size`) | $128$ | $32$ |
| $V$ | Vocabulary Size (`vocab_size`) | $176$ (char) / $500$ (BPE) | $32$ |
| $D$ | Model Hidden Dimension (`n_embd`) | $256$ | $64$ |
| $H$ | Number of Attention Heads (`n_head`) | $4$ | $2$ |
| $d_k$ | Attention Head Dimension ($D / H$) | $64$ | $32$ |
| $D_{\text{ff}}$ | Feed-Forward Inner Dimension (`ffn_dim` = $4D$) | $1024$ | $256$ |
| $L$ | Number of Transformer Layers (`n_layer`) | $4$ | $2$ |

---

## 3. Explicit Tensor Shape Contract Table

### 3.1 Input & Embedding Layer
| Operation | Input Tensor(s) | Output Tensor | Mathematical Expression |
|:---|:---|:---|:---|
| **Input IDs** | - | $(B, T)$ [int64] | Sequence of integer token IDs |
| **Token Embedding** | $(B, T)$ | $(B, T, D)$ [float32] | $E_{\text{tok}} = W_{\text{te}}[\text{idx}]$ |
| **Positional Embedding** | Scalar $T$ | $(1, T, D)$ [float32] | $E_{\text{pos}} = W_{\text{pe}}[0..T-1]$ |
| **Sum & Dropout** | $(B, T, D), (1, T, D)$ | $(B, T, D)$ [float32] | $x_0 = \text{Dropout}(E_{\text{tok}} + E_{\text{pos}})$ |

### 3.2 Transformer Block (Layer $l$)
| Operation | Input Tensor(s) | Output Tensor | Notes / Shape Derivation |
|:---|:---|:---|:---|
| **Pre-LN 1** | $(B, T, D)$ | $(B, T, D)$ | $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ |
| **Q, K, V Projections** | $(B, T, D)$ | $(B, T, D)$ each | Linear: $X W_q + b_q, X W_k + b_k, X W_v + b_v$ |
| **Head Reshape** | $(B, T, D)$ | $(B, H, T, d_k)$ each | Split $D$ into $H \times d_k$, swap axes $(1, 2)$ |
| **Attention Scores** | $Q, K \in (B, H, T, d_k)$ | $(B, H, T, T_{\text{past}} + T)$ | $S = (Q @ K^T) / \sqrt{d_k}$ |
| **Causal Masking** | $(B, H, T, T_{\text{total}})$ | $(B, H, T, T_{\text{total}})$ | $S_{i, j} = -\infty$ for $j > i$ (future) |
| **Attention Softmax** | $(B, H, T, T_{\text{total}})$ | $(B, H, T, T_{\text{total}})$ | Numerically stable softmax along axis $-1$ |
| **Attention Weighted Sum**| Weights, $V$ | $(B, H, T, d_k)$ | $\text{Weights} @ V$ |
| **Head Concatenation** | $(B, H, T, d_k)$ | $(B, T, D)$ | Swap axes $(1, 2)$ and reshape to full $D$ |
| **Output Projection** | $(B, T, D)$ | $(B, T, D)$ | Linear: $\text{Concat} @ W_o + b_o$ |
| **Residual Addition 1** | $(B, T, D), (B, T, D)$ | $(B, T, D)$ | $x_1 = x_0 + \text{AttnOut}$ |
| **Pre-LN 2** | $(B, T, D)$ | $(B, T, D)$ | $\text{LN}_2(x_1)$ before Feed-Forward |
| **FFN Projection 1** | $(B, T, D)$ | $(B, T, D_{\text{ff}})$ | Linear: $X W_1 + b_1$ |
| **GELU Activation** | $(B, T, D_{\text{ff}})$ | $(B, T, D_{\text{ff}})$ | $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$ |
| **FFN Projection 2** | $(B, T, D_{\text{ff}})$ | $(B, T, D)$ | Linear: $h_1 W_2 + b_2$ |
| **Residual Addition 2** | $(B, T, D), (B, T, D)$ | $(B, T, D)$ | $x_{\text{out}} = x_1 + \text{FFNOut}$ |

### 3.3 Output Head & Loss
| Operation | Input Tensor(s) | Output Tensor | Notes |
|:---|:---|:---|:---|
| **Final LayerNorm** | $(B, T, D)$ | $(B, T, D)$ | Normalizes hidden states across layers |
| **LM Head Projection** | $(B, T, D)$ | $(B, T, V)$ | $x @ W_{\text{head}} + b_{\text{vocab}}$ ($W_{\text{head}} = W_{\text{te}}^T$ if tied) |
| **Cross-Entropy Loss** | $(B, T, V), (B, T)$ | Scalar $\mathbb{R}$ | $-\frac{1}{N} \sum_{i} \log P(y_i \mid x_{<i})$ with Log-Sum-Exp |

---

## 4. Key-Value (KV) Cache Tensor Dynamics (Slide 15)

During incremental autoregressive generation:
- **Prefill Step ($t = 0 \dots T_{\text{prompt}}-1$):**
  - Input: $(1, T_{\text{prompt}})$
  - $K, V$ cached in memory: shape $(1, H, T_{\text{prompt}}, d_k)$ per layer.
  - Generates first new token.
- **Incremental Decode Step ($t \ge T_{\text{prompt}}$):**
  - Input: $(1, 1)$ (only the newly sampled token).
  - $Q_{\text{new}} \in (1, H, 1, d_k)$
  - $K_{\text{new}}, V_{\text{new}} \in (1, H, 1, d_k)$
  - Cache append: $K_{\text{total}} = [K_{\text{cache}}, K_{\text{new}}] \in (1, H, T_{\text{total}}, d_k)$
  - Attention scores: $(Q_{\text{new}} @ K_{\text{total}}^T) \in (1, H, 1, T_{\text{total}})$
  - **No future tokens exist**: causal mask is not required; query attends to all $0 \dots T_{\text{total}}-1$.
  - Output logits: $(1, 1, V)$.
  - **Sliding Window Overflow Policy**: When $T_{\text{total}} > \text{block\_size}$, the oldest tokens are sliced: $K = K[:, :, -\text{block\_size}:, :]$, $V = V[:, :, -\text{block\_size}:, :]$.

---

## 5. Analytical Backpropagation Gradient Flow

Every layer implements an exact analytical backward pass matching matrix calculus:

$$\frac{\partial \mathcal{L}}{\partial X} = \text{LN.backward}\left(\text{Attn.backward}\left(\frac{\partial \mathcal{L}}{\partial x_1}\right)\right) + \frac{\partial \mathcal{L}}{\partial x_1}$$

$$\frac{\partial \mathcal{L}}{\partial x_1} = \text{LN.backward}\left(\text{FFN.backward}\left(\frac{\partial \mathcal{L}}{\partial x_{\text{out}}}\right)\right) + \frac{\partial \mathcal{L}}{\partial x_{\text{out}}}$$

All parameter gradient arrays ($\nabla W, \nabla b$) maintain exact shape correspondence with their parameter counterparts.
