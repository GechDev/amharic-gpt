# Small GPT-Style Language Model for Amharic

An educational, transparent, from-scratch implementation of an autoregressive GPT-style Transformer language model for the Amharic language (አማርኛ), built using **Python and pure NumPy**.

Developed as a university assignment to demonstrate the inner mathematical and algorithmic workings of generative Transformer models without hiding the implementation behind deep-learning frameworks (no PyTorch `nn.MultiheadAttention`, no Hugging Face Transformers, no `tiktoken`, and no pre-trained weights).

---

## Table of Contents

- [Overview](#overview)
- [Objective](#objective)
- [Dataset](#dataset)
- [Tokenization](#tokenization)
- [Architecture](#architecture)
- [Causal Self-Attention](#causal-self-attention)
- [Multi-Head Attention](#multi-head-attention)
- [Feed-Forward Network](#feed-forward-network)
- [Transformer Block & Residual Connections](#transformer-block--residual-connections)
- [Training & Backpropagation](#training--backpropagation)
- [Autoregressive Generation](#autoregressive-generation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Output](#example-output)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a complete generative language model for Amharic from first mathematical principles. Every core component—from Unicode text preprocessing and character tokenization, to self-attention matrix projections, causal triangular masking, layer normalization, GELU activations, analytical backpropagation, AdamW parameter updates, and autoregressive nucleus/temperature sampling—is implemented in pure NumPy.

---

## Objective

1. **Pedagogical Clarity**: Demystify how generative pre-trained Transformers (GPT) work internally by exposing all mathematical tensor operations, derivatives, and algorithmic steps.
2. **From-Scratch Mathematical Implementation**: Use NumPy for vectors, matrices, dot products, transposes, exponentials, softmax, causal masking, layer normalization, loss, and analytical gradients.
3. **Amharic Language Modeling**: Build an end-to-end pipeline tailored to the Ethiopic syllabary (ግዕዝ ፊደላት), demonstrating that small autoregressive models can learn natural character transitions in low-resource African languages.

---

## Dataset

### Source & Description
- **Recommended Corpus**: Publicly available Amharic news and encyclopedic corpora, such as the [IsraelAbebe Amharic News Dataset](https://github.com/IsraelAbebe/Amharic-News-Text-Classification-Dataset) and [Amharic Wikipedia](https://dumps.wikimedia.org/amwiki/).
- **Bundled Corpus**: An authentic, diverse sample corpus is included at `data/raw/sample_amharic.txt` (exact size: 3,940 characters, 741 words, 71 lines, 170 unique characters) covering culture, education, science, history, computing, and everyday Ethiopian life.
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0) / Open Access.

### Preprocessing Pipeline (`src/data.py`)
1. **Unicode NFKC Normalization**: Harmonizes composite Unicode glyphs into canonical Ethiopic code points (`unicodedata.normalize('NFKC', text)`).
2. **Noise Filtering**: Removes HTML tags, URLs, email addresses, zero-width spaces (`\u200b`, `\ufeff`), and non-printable control characters.
3. **Fidel & Punctuation Preservation**: Preserves the complete Ethiopic syllabary (U+1200 to U+137F), Ethiopic punctuation (`፡`, `።`, `፣`, `፤`, `፥`, `፦`), Arabic numerals, and standard punctuation (`?`, `!`, `,`, `.`).
4. **Whitespace Regularization**: Consolidates consecutive spaces into single spaces and collapses redundant blank lines into clean paragraph boundaries.
5. **Train/Validation Splitting**: Splits the preprocessed text sequentially into 90% training (`data/processed/train.txt`) and 10% validation (`data/processed/val.txt`).

---

## Tokenization

The tokenizer (`src/tokenizer.py`) is a custom, from-scratch **character-level Unicode-aware tokenizer**.

### Why Character-Level for Amharic?
Amharic is written in an *abugida* (syllabary), where each character represents a consonant-vowel syllable:
- Example: **ለ** (lä), **ሉ** (lu), **ሊ** (li), **ላ** (la), **ሌ** (le), **ል** (l), **ሎ** (lo).

Because each character already encapsulates a syllable:
1. **Compact Vocabulary**: The entire Amharic alphabet, numerals, and punctuation fit in ~170 to 300 unique tokens.
2. **Zero Out-of-Vocabulary (OOV)**: Any Amharic word—no matter how rare—can be represented without unknown tokens.
3. **No External Dependencies**: Eliminates the need for BPE merge tables, SentencePiece, or Hugging Face tokenizers.

### Limitations
- **Sequence Length**: Character-level tokenization results in longer sequences (5–8 tokens per word) compared to subword tokenization, reducing the effective word count that fits within a fixed context window ($T=128$).

### Special Tokens
- `<PAD>` (ID 0): Padding token for batch alignment.
- `<UNK>` (ID 1): Unknown token fallback.
- `<BOS>` (ID 2): Beginning of sequence.
- `<EOS>` (ID 3): End of sequence.

---

## Architecture

The model follows the modern **GPT (decoder-only Transformer)** architecture with **Pre-Layer Normalization**:

```
Input Token IDs: [x_0, x_1, ..., x_{T-1}]   (Shape: [B, T])
                  ↓
          +-----------------------+
          | Token Embeddings (W)  |  (Shape: [B, T, n_embd])
          +-----------------------+
                      +
          +-----------------------+
          | Position Embeddings   |  (Shape: [1, T, n_embd])
          +-----------------------+
                  ↓
      x_0 = Token_Emb + Pos_Emb      (Shape: [B, T, n_embd])
                  ↓
   ┌─────────────────────────────────────┐
   │ Transformer Block 0                 │
   │   x = x + MultiHeadAttn(LayerNorm(x))│
   │   x = x + FeedForward(LayerNorm(x))  │
   └─────────────────────────────────────┘
                  ↓
                 ... (Repeated for L layers)
                  ↓
   ┌─────────────────────────────────────┐
   │ Transformer Block (L - 1)           │
   └─────────────────────────────────────┘
                  ↓
          +-----------------------+
          | Final LayerNorm (ln_f)|  (Shape: [B, T, n_embd])
          +-----------------------+
                  ↓
          +-----------------------+
          | LM Head (W_vocab)     |  (Shape: [B, T, vocab_size])
          +-----------------------+
                  ↓
       Logits for Next-Token Prediction
```

### Model Dimensions
| Configuration | `block_size` | `n_embd` | `n_head` | `head_dim` | `n_layer` | `ffn_dim` | Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tiny** (Debug/Local) | 32 | 64 | 2 | 32 | 2 | 256 | ~124,000 |
| **Small** (Colab/Standard) | 128 | 256 | 4 | 64 | 4 | 1024 | ~3,200,000 |

---

## Causal Self-Attention

Implemented in `src/attention.py`. Self-attention computes compatibility between tokens:

$$Q = X W_q + b_q, \quad K = X W_k + b_k, \quad V = X W_v + b_v$$

$$\text{scores} = \frac{Q K^T}{\sqrt{d_k}}$$

### Causal Masking
To prevent tokens from looking into the future during autoregressive generation, future positions receive $-\infty$ (represented as $-10^9$ in float):

$$\text{Mask}_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -10^9 & \text{if } j > i \end{cases}$$

$$\text{Attention Weights} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right)$$

$$\text{Output} = \text{Attention Weights} \times V$$

---

## Multi-Head Attention

Multi-head attention splits the embedding dimension $D$ across $H$ independent attention heads:

$$d_k = \frac{D}{H}$$

1. Project input $X \in \mathbb{R}^{B \times T \times D}$ into $Q, K, V$.
2. Reshape into $H$ heads: $\mathbb{R}^{B \times H \times T \times d_k}$.
3. Run scaled dot-product causal attention in parallel across all heads.
4. Concatenate head outputs back into $\mathbb{R}^{B \times T \times D}$.
5. Project through output linear matrix $W_o \in \mathbb{R}^{D \times D}$.

---

## Feed-Forward Network

Implemented in `src/feed_forward.py`. While attention communicates between tokens across time, the FFN transforms features position-wise:

$$\text{hidden} = X W_1 + b_1 \quad (D \to 4D)$$

$$\text{activated} = \text{GELU}(\text{hidden})$$

$$\text{output} = \text{activated} W_2 + b_2 \quad (4D \to D)$$

### GELU Activation (`src/activations.py`)
Uses the smooth tanh approximation:

$$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

---

## Transformer Block & Residual Connections

Implemented in `src/transformer_block.py`. Uses **Pre-LayerNorm** and **Residual Connections**:

$$x_1 = x + \text{MultiHeadAttention}(\text{LayerNorm}_1(x))$$

$$x_2 = x_1 + \text{FeedForward}(\text{LayerNorm}_2(x_1))$$

Residual connections provide an identity gradient highway ($dL/dx = dL/dy \cdot 1$), preventing vanishing gradients in deep networks.

---

## Training & Backpropagation

### Next-Token Supervision
Given token sequence $[x_0, x_1, x_2, x_3, x_4]$:
- **Input $X$**: $[x_0, x_1, x_2, x_3]$
- **Target $Y$**: $[x_1, x_2, x_3, x_4]$ (shifted by 1 token)

### Cross-Entropy Loss (`src/loss.py`)
Computed with the numerically stable Log-Sum-Exp formulation:

$$\text{Loss} = -z_y + \max(z) + \log\left(\sum_k e^{z_k - \max(z)}\right)$$

### Analytical Gradients
The gradient with respect to logits is:

$$\frac{\partial L}{\partial z_j} = \frac{p_j - \mathbf{1}(j = y)}{N}$$

Analytical derivatives are backpropagated through all layers (LM head, LayerNorm, FFN, GELU, Attention Q/K/V, Residuals, and Embeddings) using exact tensor calculus.

### Optimizer (`src/optimizer.py`)
Implements **AdamW** from scratch:
- First moment estimate: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
- Second moment estimate: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
- Bias correction: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
- Decoupled weight decay: $\theta \leftarrow \theta - \eta \lambda \theta$
- Parameter update: $\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

---

## Autoregressive Generation

Implemented in `src/generate.py`.

Given a prompt (e.g., `"ሰላም"`):
1. Encode prompt characters to token IDs.
2. Forward pass through model; pluck logits at the final position.
3. Apply decoding strategy:
   - **Greedy**: $\text{argmax}(\text{logits})$
   - **Temperature**: scale logits by $1 / T$ before softmax.
   - **Top-k**: filter out all but the $k$ highest logit scores.
4. Sample next token and append to sequence.
5. Repeat iteratively until `max_new_tokens` is reached or `<EOS>` is emitted.
6. Decode token IDs back to Amharic text.

---

## Project Structure

```
amharic-gpt/
├── README.md               # Comprehensive project documentation
├── requirements.txt        # Minimal dependencies (numpy, pytest)
├── .gitignore              # Git ignore rules
│
├── data/
│   ├── DATASET.md          # Corpus documentation & download instructions
│   ├── raw/
│   │   └── sample_amharic.txt  # Curated authentic Amharic text corpus
│   └── processed/          # Preprocessed train.txt & val.txt
│
├── checkpoints/            # Saved model weights (.npz) and vocab (.json)
├── outputs/                # Generated generation logs
│
├── notebooks/
│   └── demo.ipynb          # Interactive Jupyter demonstration notebook
│
├── src/
│   ├── __init__.py         # Package init
│   ├── config.py           # Model hyperparameter configuration
│   ├── tokenizer.py        # Custom character-level Amharic tokenizer
│   ├── data.py             # Preprocessing & Next-Token batching
│   ├── math_utils.py       # Softmax, causal masks, weight initializers
│   ├── embeddings.py       # Token & learned positional embeddings
│   ├── activations.py      # GELU activation forward & backward
│   ├── normalization.py    # LayerNorm forward & analytical backward
│   ├── attention.py        # Causal single-head & multi-head attention
│   ├── feed_forward.py     # Position-wise Feed-Forward Network
│   ├── transformer_block.py# Pre-LN Transformer block & residuals
│   ├── model.py            # Complete GPT Language Model assembly
│   ├── loss.py             # Cross-entropy loss & logit gradients
│   ├── optimizer.py        # AdamW and SGD optimizers from scratch
│   ├── train.py            # Training and validation loop
│   ├── generate.py         # Autoregressive text generation
│   └── utils.py            # NumPy checkpoint save/load utilities
│
└── tests/
    ├── test_tokenizer.py   # Tokenizer encoding, decoding & special tokens
    ├── test_math_utils.py  # Softmax stability & causal masks
    ├── test_embeddings.py  # Embedding lookups & gradient accumulation
    ├── test_norm_act.py    # LayerNorm & GELU numerical gradient checks
    ├── test_attention.py   # Single & multi-head causal attention checks
    ├── test_feed_forward.py# FFN forward & gradient verification
    ├── test_transformer_block.py # Transformer block gradient verification
    ├── test_model.py       # Full GPT model forward & parameter checks
    ├── test_loss.py        # Cross-entropy loss & numerical checks
    ├── test_optimizer.py   # SGD, AdamW & gradient clipping
    ├── test_training.py    # Overfitting sanity test (loss reduction)
    ├── test_checkpoint.py  # Checkpoint save & restore verification
    ├── test_generation.py  # Autoregressive generation & EOS stopping
    └── test_pipeline_e2e.py# Full end-to-end integration test
```

---

## Installation

### Prerequisites
- Python 3.9+
- NumPy >= 1.24.0
- pytest >= 7.0.0

```bash
# Clone the repository
git clone https://github.com/GechDev/amharic-gpt.git
cd amharic-gpt

# Install requirements
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preprocessing
Preprocesses raw Amharic text, normalizes Unicode, and generates `train.txt` and `val.txt`:
```bash
python -m src.data --input data/raw/sample_amharic.txt --output_dir data/processed
```

### 2. Training the Model
Trains the GPT model from scratch using pure NumPy:
```bash
# Train tiny configuration (fast local CPU training, ~15-30s):
python -m src.train --config_type tiny --max_iters 200 --lr 0.001 --batch_size 4

# Or train small configuration:
python -m src.train --config_type small --max_iters 500 --lr 0.0003 --batch_size 4
```

### 3. Autoregressive Text Generation
Generate Amharic text from an initial prompt:
```bash
# Greedy generation:
python -m src.generate --prompt "ሰላም" --max_new_tokens 60 --greedy

# Sampling with temperature and top-k:
python -m src.generate --prompt "ትምህርት" --max_new_tokens 80 --temperature 0.8 --top_k 20
```

### 4. Running the Tests
Run the 43 unit tests verifying all mathematical operations, gradient backpropagation, checkpoint integrity, and generation:
```bash
pytest tests/ -v
```

---

## Example Output

Real generation output produced by the trained baseline model (`tiny` config, 124,590 parameters) after 200 training steps (final training loss: 2.7511, validation loss: 3.3350):

### 1. Temperature Sampling with Top-k Filtering
```bash
python -m src.generate --seed 42 --checkpoint_dir checkpoints --prompt "ሰላም" --max_new_tokens 60 --temperature 0.8 --top_k 20
```
**Output:**
```
[Tokenizer] Loaded vocabulary from checkpoints\vocab.json (174 tokens).
[Checkpoint] Successfully loaded checkpoint from checkpoints (step 200).

[Generation] Prompt: "ሰላም"
[Generation] Strategy: Sampling (T=0.8, top_k=20) | Seed: 42
--------------------------------------------------
Generated Text:
ሰላም ነው የተር ትተት አየቋ የያት።
የተድው። መን የአንቋ የር ለት ንተበስና የማንንነመን ያመልም 
--------------------------------------------------
```

### 2. Deterministic Greedy Generation
```bash
python -m src.generate --seed 42 --checkpoint_dir checkpoints --prompt "ሰላም" --max_new_tokens 60 --greedy
```
**Output:**
```
[Tokenizer] Loaded vocabulary from checkpoints\vocab.json (174 tokens).
[Checkpoint] Successfully loaded checkpoint from checkpoints (step 200).

[Generation] Prompt: "ሰላም"
[Generation] Strategy: Greedy | Seed: 42
--------------------------------------------------
Generated Text:
ሰላም የተር የተርት የተመረት የተር የተለም የተር የተርት የተርት የተርት የተርት የተርት የተርት የ
--------------------------------------------------
```

---

## Limitations

1. **Model Capacity**: This is an educational university assignment trained on modest hardware with pure NumPy. It is not intended to compete with multi-billion parameter industrial models (such as GPT-4 or Claude).
2. **Speed**: Pure NumPy runs single-threaded or CPU-BLAS vector operations without specialized CUDA tensor core kernel fusions. Training on millions of tokens is best done in PyTorch/JAX for speed.
3. **Context Length**: Small context window ($T=32$ to $128$) restricts long-document coherence.

---

## Future Improvements

1. **Subword Tokenizer**: Implement Byte-Pair Encoding (BPE) specifically adapted for Ethiopic morphemes.
2. **Rotary Position Embeddings (RoPE)**: Replace learned absolute positional embeddings with relative Rotary embeddings.
3. **KV Cache**: Implement Key-Value caching to accelerate autoregressive token generation from $O(N^2)$ to $O(N)$.
4. **Larger Corpora**: Train on the full 20MB+ Amharic news and Wikipedia dumps.
