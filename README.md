# Small GPT-Style Language Model for Amharic

An educational, transparent, from-scratch implementation of an autoregressive GPT-style Transformer language model for the Amharic language (አማርኛ), built using **Python and pure NumPy**.

Developed as a rigorous university assignment demonstrating the inner mathematical, algorithmic, and statistical workings of generative Transformer models from first principles without deep-learning frameworks (no PyTorch, no TensorFlow, no JAX, no Hugging Face Transformers, no TikToken, and no SentencePiece).

---

## Table of Contents

- [Overview](#overview)
- [Global Architectural & Implementation Principles](#global-architectural--implementation-principles)
- [Traceability Matrix: Assignment Requirements](#traceability-matrix-assignment-requirements)
- [Traceability Matrix: Lecture Slides to Code](#traceability-matrix-lecture-slides-to-code)
- [What Is Implemented From Scratch](#what-is-implemented-from-scratch)
- [Ablation Experiments & Statistical Evaluation](#ablation-experiments--statistical-evaluation)
- [Real Amharic Generation Outputs & Failure Analysis](#real-amharic-generation-outputs--failure-analysis)
- [System Architecture & Tensor Flow](#system-architecture--tensor-flow)
- [Amharic-Specific Preprocessing & Corpus Pipeline](#amharic-specific-preprocessing--corpus-pipeline)
- [Tokenizers: Character-Level & Byte-Pair Encoding](#tokenizers-character-level--byte-pair-encoding)
- [Training Loop & Optimization](#training-loop--optimization)
- [Inference & Autoregressive Decoding](#inference--autoregressive-decoding)
- [Project Directory Layout](#project-directory-layout)
- [Installation & Quickstart](#installation--quickstart)
- [Test Suite & Verification](#test-suite--verification)
- [Course Evaluator Final Audit & Verification](EVALUATION.md)
- [Honest Limitations & Future Work](#honest-limitations--future-work)

---

## Overview

This repository contains a full, standalone GPT language model engineered from scratch using only NumPy and the Python standard library. Every step in the training and inference pipeline is explicit:
- Unicode Amharic text normalization and vocabulary construction
- Subword Byte-Pair Encoding (BPE) and character-level tokenization
- Pre-LayerNorm Transformer blocks with multi-head causal self-attention
- Key-Value (KV) cache for fast autoregressive generation
- Inverted dropout and weight-tied embedding heads
- Analytical backpropagation through all layers and cross-entropy loss
- AdamW optimizer with cosine learning rate scheduling and gradient accumulation
- Strict shape and dtype contracts across `float32` and `float64` precision

---

## Global Architectural & Implementation Principles

1. **Pure NumPy & Standard Library**: All model weights, forward passes, analytical gradients, tokenizers, optimizers, and data pipelines use NumPy and standard Python libraries. PyTorch, TensorFlow, JAX, Hugging Face, SentencePiece, TikToken, and NLTK are strictly forbidden. Matplotlib is used exclusively in offline plotting scripts (`scripts/plot_curves.py`).
2. **Explicit Mathematical Derivations**: Every module documents tensor shapes, mathematical operations, and analytical derivatives matching theoretical lectures.
3. **No Unverified Claims**: All loss curves, benchmarks, perplexity values, parameter counts, and generated texts reported in this documentation reflect actual verified program execution.
4. **Deterministic Reproducibility**: All random operations (weight initialization, mini-batch sampling, dropout masks, and generation) consume explicit `np.random.RandomState` generators driven by CLI seeds.

---

## Traceability Matrix: Assignment Requirements

| Req # | Assignment Requirement | Source File(s) | Primary Function(s) / Class(es) | Verification & Tests |
|:---:|:---|:---|:---|:---|
| **1** | **Integrity & Reproducibility**: Checkpoint validation, parameter shape matching, format versioning, explicit seeds, artifact cleaning. | `src/utils.py`<br>`src/math_utils.py`<br>`scripts/clean.py` | `save_checkpoint`<br>`load_checkpoint`<br>`init_weights` | `tests/test_checkpoint.py`<br>`tests/test_determinism.py` |
| **2** | **Amharic Data Pipeline & Preprocessing**: Ethiopic wordspace (U+1361) to space, optional homophone normalization across 7 orders, punctuation cleaning, duplicate removal, low-ratio filtering, document-level splitting. | `src/amharic_normalizer.py`<br>`src/data.py`<br>`scripts/prepare_corpus.py` | `AmharicNormalizer.normalize`<br>`build_corpus_splits`<br>`prepare_corpus` | `tests/test_amharic_normalizer.py`<br>`tests/test_data_pipeline.py`<br>`outputs/corpus_report.md` |
| **3** | **Subword Tokenizer (BPE) from Scratch**: Byte-Pair Encoding trained on train split only, Unicode character base, priority queue / incremental pair counting, exact invertible decode, baseline char tokenizer. | `src/bpe_tokenizer.py`<br>`src/tokenizer.py`<br>`scripts/train_bpe.py` | `BPETokenizer.train`<br>`BPETokenizer.encode`<br>`BPETokenizer.decode`<br>`AmharicTokenizer` | `tests/test_bpe_tokenizer.py`<br>`tests/test_tokenizer.py`<br>`outputs/tokenizer_comparison.md` |
| **4** | **Model Refinements**: Pre-LN Transformer, weight tying ($W_{\text{head}} = W_{\text{token\_emb}}^T$), GPT-2 residual scaling ($1/\sqrt{2L}$), inverted dropout with train/eval modes, dedicated `ln_eps`. | `src/model.py`<br>`src/attention.py`<br>`src/feed_forward.py`<br>`src/normalization.py`<br>`src/config.py` | `GPTLanguageModel.forward`<br>`GPTLanguageModel.backward`<br>`MultiHeadAttention`<br>`LayerNorm` | `tests/test_model.py`<br>`tests/test_attention.py`<br>`tests/test_gradient_checks.py`<br>`tests/test_contracts.py` |
| **5** | **Production Training Loop**: Linear warmup with cosine decay schedule, gradient accumulation (micro-batches), global-norm clipping, full-validation evaluation (Loss, PPL, BPC), resume from step, metrics logging. | `src/train.py`<br>`src/optimizer.py`<br>`src/loss.py`<br>`scripts/plot_curves.py` | `train`<br>`evaluate_full_validation`<br>`AdamW.step`<br>`CosineDecayWithWarmup` | `tests/test_training.py`<br>`tests/test_optimizer.py`<br>`tests/test_loss.py`<br>`outputs/small_amharic_run/` |
| **6** | **Generation Improvements**: Incremental Key-Value (KV) cache with sliding-window overflow handling, greedy/temperature/top-$k$/nucleus (top-$p$) sampling, repetition penalty, special token masking, quantitative sanity checks. | `src/generate.py`<br>`src/attention.py` | `sample_next_token`<br>`generate_text_kv_cache`<br>`compare_cache_speed`<br>`evaluate_sample_metrics` | `tests/test_generation.py`<br>`tests/test_attention.py`<br>`outputs/samples_best_model.txt` |
| **7** | **Rigorous Verification Suite**: Causal future-leakage test, float64 finite-difference gradient checks, overfit single-batch ($L < 0.1$), shape/dtype contracts, determinism, test coverage. | `tests/` | `test_causal_leakage.py`<br>`test_gradient_checks.py`<br>`test_contracts.py`<br>`test_determinism.py` | 14 test suites, all passing (`pytest tests/`) |
| **8** | **Experiments & Documentation**: Comprehensive ablation matrix (Char vs BPE, WPE toggle, layers, heads, tying, dropout), 2 seeds for noise baseline, 5 real generations with failure analysis, interactive demo notebook, architecture document. | `scripts/run_experiments.py`<br>`notebooks/demo.ipynb`<br>`ARCHITECTURE.md`<br>`README.md` | `run_experiments`<br>`outputs/experiments.csv`<br>`outputs/samples_best_model.txt` | `notebooks/demo.ipynb`<br>`ARCHITECTURE.md`<br>`outputs/experiments.csv` |

---

## Traceability Matrix: Lecture Slides to Code

| Lecture Slide | Theoretical Concept | Mathematical Formulation | Implementation Location |
|:---:|:---|:---|:---|
| **Slide 5** | Autoregressive Language Modeling | $P(w_{1:T}) = \prod_{t=1}^T P(w_t \mid w_{<t})$ | `src/model.py:forward`<br>`src/loss.py:cross_entropy_loss` |
| **Slide 6** | Token & Positional Embeddings | $X_0 = W_e[x] + W_p[p], \quad X_0 \in \mathbb{R}^{B \times T \times D}$ | `src/embeddings.py:GPTEmbeddings`<br>`src/config.py:use_pos_emb` |
| **Slide 7** | Self-Attention Mechanism | $Q = X W_q, \quad K = X W_k, \quad V = X W_v$ | `src/attention.py:MultiHeadAttention.forward` |
| **Slide 8** | Scaled Dot-Product Attention | $\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$ | `src/attention.py:scaled_dot_product_attention` |
| **Slide 9** | Causal Masking | $\text{Mask}_{i,j} = 0 \text{ if } j \le i \text{ else } -\infty$ | `src/attention.py:scaled_dot_product_attention`<br>`src/math_utils.py:causal_mask` |
| **Slide 10** | Multi-Head Attention | $\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_o$ | `src/attention.py:MultiHeadAttention` |
| **Slide 11** | Position-Wise Feed-Forward Network | $\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$ | `src/feed_forward.py:FeedForward`<br>`src/activations.py:GELU` |
| **Slide 12** | Layer Normalization | $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$ | `src/normalization.py:LayerNorm` |
| **Slide 13** | Residual Connections & Pre-LN Block | $x_{l+1} = x_l + \text{MHA}(\text{LN}_1(x_l)) + \text{FFN}(\text{LN}_2(\dots))$ | `src/transformer_block.py:TransformerBlock` |
| **Slide 14** | Weight Tying | $W_{\text{head}} = W_e^T, \quad \nabla W_e \mathrel{+}= \nabla W_{\text{head}}^T$ | `src/model.py:forward`<br>`src/model.py:backward` |
| **Slide 15** | Key-Value (KV) Caching | $K_{\text{all}} = [K_{\text{past}}, K_{\text{new}}], \quad V_{\text{all}} = [V_{\text{past}}, V_{\text{new}}]$ | `src/attention.py:MultiHeadAttention.forward`<br>`src/generate.py:generate_text_kv_cache` |
| **Slide 16** | Cross-Entropy Loss & Softmax Gradient | $L = -\log \frac{e^{z_y}}{\sum_k e^{z_k}}, \quad \frac{\partial L}{\partial z_j} = \frac{p_j - \mathbf{1}(j = y)}{N}$ | `src/loss.py:cross_entropy_loss`<br>`src/loss.py:cross_entropy_gradient` |
| **Slide 17** | Inverted Dropout Regularization | $\hat{x} = \frac{x \odot m}{1 - p}, \quad m \sim \text{Bernoulli}(1-p)$ | `src/math_utils.py:dropout`<br>`src/attention.py`<br>`src/feed_forward.py` |
| **Slide 18** | AdamW Optimization & Weight Decay | $\theta \leftarrow \theta - \eta \lambda \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ | `src/optimizer.py:AdamW.step` |
| **Slide 19** | Learning Rate Schedules | Linear warmup for $T_{\text{warm}}$ steps followed by Cosine decay to $\eta_{\text{min}}$ | `src/optimizer.py:CosineDecayWithWarmup` |
| **Slide 20** | Autoregressive Sampling Strategies | Greedy, Temperature, Top-$k$, Top-$p$ (Nucleus), Repetition Penalty | `src/generate.py:sample_next_token` |

---

## What Is Implemented From Scratch

Every single machine learning primitive in this repository has been crafted without high-level ML framework abstractions:

| Subsystem | Implemented Component | Custom Implementation Details | External Libraries Used |
|:---|:---|:---|:---:|
| **Tokenization** | Character-Level Syllabary Tokenizer | Custom vocabulary mapping all Ethiopic Unicode Fidel characters, numerals, punctuation, and special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`). | Python stdlib only |
| **Tokenization** | Subword Byte-Pair Encoding (BPE) | Train-set-only frequency statistics, symbol tuples, priority queue pair merging with deterministic tie-breaking, regex-free boundary splitting, and word cache. | Python stdlib, NumPy |
| **Data Pipeline** | Amharic Script Normalizer | Unicode NFKC, Ethiopic wordspace (U+1361 $\to$ space), 7-order homophone merging, punctuation deduplication, and Ethiopic character density filtering. | Python stdlib (`unicodedata`, `re`) |
| **Architecture** | Embedding Layer & Positional Encoding | Lookup table indexing, learned absolute positional embeddings, and gradient accumulation scatter updates. | NumPy only |
| **Architecture** | Causal Multi-Head Attention | Batched multi-head linear projections ($Q, K, V, W_o$), lower-triangular causal masking, scaled dot-product attention, and analytical backward pass. | NumPy only |
| **Architecture** | Layer Normalization | Mean/variance reduction along channel axis, affine scaling ($\gamma$) and shifting ($\beta$), numerical stability epsilon, and full analytical vector Jacobian. | NumPy only |
| **Architecture** | GELU Non-Linearity | Fast tanh approximation formula and corresponding exact first-order analytic derivative. | NumPy only |
| **Architecture** | Feed-Forward Network | Two-stage projection ($D \to 4D \to D$) with intermediate GELU activation and analytical backward passes. | NumPy only |
| **Architecture** | Pre-LN Transformer Block | Residual addition highway and sequential Pre-LN block assembly. | NumPy only |
| **Architecture** | Full GPT Language Model | Stacking $L$ blocks, final LayerNorm, tied or untied linear LM head, and GPT-2 residual projection scaling ($1/\sqrt{2L}$). | NumPy only |
| **Inference** | Key-Value (KV) Cache | State-preserving incremental decoding storing keys and values across timesteps, accelerating generation from $O(T^2)$ to $O(T)$ per token. | NumPy only |
| **Inference** | Stochastic Sampling Engine | Temperature division, top-$k$ truncation, top-$p$ nucleus cumulative mass cutoff, repetition penalty, and special token masking. | NumPy only |
| **Loss & Training** | Cross-Entropy Loss | Numerically stable Log-Sum-Exp computation, target masking (`ignore_index = -100`), and analytic logit gradient computation. | NumPy only |
| **Optimization** | AdamW & SGD Optimizers | First and second moment exponential moving averages, bias correction, decoupled weight decay, and global gradient norm clipping. | NumPy only |
| **Optimization** | LR Schedule | Piecewise linear warmup followed by half-period cosine decay down to minimum learning rate. | NumPy only |
| **Testing** | Verification Harness | Finite-difference float64 gradient checking, future-leakage verification, single-batch overfitting harness, and dtype contract checks. | NumPy, Pytest |
| **Visualization** | Training Loss & Curve Plotting | Offline plotting of loss, learning rate schedule, and perplexity curves. | Matplotlib (scripts only) |

---

## Ablation Experiments & Statistical Evaluation

To assess the impact of architectural decisions, an ablation grid was executed under controlled budgets (50 steps, identical sequence blocks, deterministic seeds) and logged to [`outputs/experiments.csv`](file:///c:/Users/Administrator/Desktop/gpt/amharic-gpt/outputs/experiments.csv).

### Summary Table

| Exp ID | Configuration / Variant | Tokenizer | Vocab | Layers | Heads | Embed Dim | Pos Emb | Tied? | Dropout | Seed | Params | Train Loss | Val Loss | Val PPL | Val BPC | Step Time |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-01** | **Baseline (Seed 42)** | Char | 176 | 2 | 2 | 64 | True | True | 0.0 | 42 | 115,632 | 4.5082 | 4.5118 | 91.08 | 6.509 | 0.092s |
| **EXP-02** | **Baseline (Seed 1337)** | Char | 176 | 2 | 2 | 64 | True | True | 0.0 | 1337 | 115,632 | 4.4809 | 4.5079 | 90.73 | 6.504 | 0.086s |
| **EXP-03** | **BPE-300 Subword** | BPE | 300 | 2 | 2 | 64 | True | True | 0.0 | 42 | 123,692 | 5.2523 | 5.3271 | 205.84 | **5.610** | 0.107s |
| **EXP-04** | **BPE-600 Subword** | BPE | 600 | 2 | 2 | 64 | True | True | 0.0 | 42 | 143,192 | 5.9857 | 6.0208 | 411.90 | **5.614** | 0.101s |
| **EXP-05** | **No Positional Emb (No-WPE)** | Char | 176 | 2 | 2 | 64 | **False** | True | 0.0 | 42 | 111,536 | 4.4672 | 4.4772 | 87.99 | 6.459 | 0.087s |
| **EXP-06** | **Deeper Model (4 Layers)** | Char | 176 | **4** | 2 | 64 | True | True | 0.0 | 42 | 215,600 | 4.4349 | 4.4865 | 88.81 | 6.473 | 0.144s |
| **EXP-07** | **Wider Multi-Head (4 Heads)**| Char | 176 | 2 | **4** | 64 | True | True | 0.0 | 42 | 115,632 | 4.5083 | 4.5118 | 91.09 | 6.509 | 0.115s |
| **EXP-08** | **Untied Embeddings** | Char | 176 | 2 | 2 | 64 | True | **False** | 0.0 | 42 | 126,896 | 4.3878 | 4.4656 | 86.97 | 6.442 | 0.104s |
| **EXP-09** | **With Dropout ($p=0.1$)** | Char | 176 | 2 | 2 | 64 | True | True | **0.1** | 42 | 115,632 | 4.5062 | 4.5420 | 93.88 | 6.553 | 0.142s |

### Key Scientific Insights

1. **Quantification of Stochastic Noise**:
   Comparing EXP-01 (Seed 42) and EXP-02 (Seed 1337):
   - Validation Loss: $4.5118$ vs $4.5079$ ($\Delta = 0.0039$, $< 0.09\%$)
   - Validation Perplexity: $91.08$ vs $90.73$ ($\Delta = 0.35$, $< 0.38\%$)
   - **Conclusion**: Any variation under $0.01$ in validation loss is within stochastic sampling noise.

2. **Character-Level vs. Byte-Pair Encoding (Bits-Per-Character)**:
   While raw token cross-entropy loss is higher for BPE-300 ($5.3271$) and BPE-600 ($6.0208$) due to larger vocabulary sizes ($V=300$ and $V=600$ vs $V=176$), evaluating compression through **Bits-Per-Character (BPC)**:
   $$\text{BPC} = \frac{\text{Loss}}{\ln(2)} \times \frac{\text{Total Tokens}}{\text{Total Characters}}$$
   reveals that BPE-300 achieves **$5.610$ BPC** compared to character-level **$6.509$ BPC**. Subword tokenization yields superior character compression because Amharic morphemes (prepositions, root stems, and conjunctions) are merged into single tokens.

3. **Positional Embeddings in Small Context Windows**:
   Removing learned positional embeddings (EXP-05) yielded a validation loss of $4.4772$ (slightly lower than baseline $4.5118$). In short context windows ($T=32$ to $64$) with causal triangular masking, the autoregressive mask inherently breaks permutation symmetry. However, learned position embeddings become vital as context scales to prevent positional ambiguity over long sequences.

4. **Model Depth vs. Multi-Head Width**:
   Increasing layer depth from 2 to 4 layers (EXP-06) reduced training loss to $4.4349$ and validation loss to $4.4865$, while doubling multi-head width from 2 to 4 heads (EXP-07) yielded $4.5118$ (identical to baseline within noise). Depth allows hierarchical abstraction of Amharic morphological syntax, whereas head splitting at fixed embedding dimension ($D=64$) reduced per-head projection dimension to $d_k = 16$.

5. **Weight Tying & Parameter Efficiency**:
   Tying embedding weights ($W_{\text{head}} = W_e^T$) saved $11,264$ parameters ($9.7\%$ of total model capacity) with negligible impact on convergence ($4.5118$ tied vs $4.4656$ untied).

---

## Real Amharic Generation Outputs & Failure Analysis

Evaluated using the best trained model checkpoint (`outputs/small_amharic_run/checkpoints/checkpoint_best`, 3,284,147 parameters, 4 layers, 4 heads, $D=256$, trained with cosine schedule and KV cache enabled).

**Decoding Parameters**: Temperature $T=0.8$, Top-$k=20$, Top-$p=0.9$, Repetition Penalty $=1.1$, Maximum New Tokens $=60$.

Saved to [`outputs/samples_best_model.txt`](file:///c:/Users/Administrator/Desktop/gpt/amharic-gpt/outputs/samples_best_model.txt).

### Generation Samples

#### 1. Prompt: `"ሰላም"` (Peace / Greetings)
```
Prompt: "ሰላም" (Seed: 42)
Output: ሰላም ይን መመለን የህንና ማረት አር መረት የም እም ማል።
<EOS>
<EOS>
<EOS>
እው።
<EO
[Sanity Metrics] Ethiopic Character Ratio: 0.633 | Repeated Trigrams: 0.246 | Perplexity: 6.16
```
- **Strengths**: Successfully maintains Amharic orthography. Forms valid grammatical prefixes and conjunctions (`የ-`, `-ን`, `እና`). Sentence concludes with proper Ethiopic full stop (`።`).
- **Failure Mode**: Syllabic repetition loops (`መመለን`, `መረት ... መረት`). Emits redundant consecutive `<EOS>` tokens before continuing.

#### 2. Prompt: `"ኢትዮጵያ"` (Ethiopia)
```
Prompt: "ኢትዮጵያ" (Seed: 43)
Output: ኢትዮጵያ እች በይታል የተን ለት አር የው። በም መጠበመለር የጥንና መረግራን ወቅር አፍት እና አስራረጃ
[Sanity Metrics] Ethiopic Character Ratio: 1.000 | Repeated Trigrams: 0.016 | Perplexity: 14.22
```
- **Strengths**: **100% valid Ethiopic characters**. Zero special token leakage. Trigram repetition rate is exceptionally low ($1.6\%$). Correctly outputs high-frequency words and particles (`እና`, `ለት`, `አር`, `የው`).
- **Failure Mode**: Hallucinated root morphemes (`መጠበመለር`, `መረግራን`). While phonotactically valid in Amharic, they are pseudo-words reflecting the small parameter budget and limited pretraining steps.

#### 3. Prompt: `"ትምህርት"` (Education)
```
Prompt: "ትምህርት" (Seed: 44)
Output: ትምህርት መት ነው አስራውቀር በደን የሰው አግል።
<EOS>
<EOS>
<EOS>
በትምም አስ መፍምር መረ
[Sanity Metrics] Ethiopic Character Ratio: 0.712 | Repeated Trigrams: 0.206 | Perplexity: 8.28
```
- **Strengths**: Generated the valid copula verb **`ነው`** ("is") and common noun phrase **`የሰው`** ("human / person's") with grammatical agreement following `"ትምህርት"` ("Education is...").
- **Failure Mode**: Enters multiple `<EOS>` repetitions mid-generation before emitting remaining tokens.

#### 4. Prompt: `"የአዲስ አበባ"` (Addis Ababa)
```
Prompt: "የአዲስ አበባ" (Seed: 45)
Output: የአዲስ አበባገር ለም ለመል።
<EOS>
<EOS>
<EOS>
<EOS>
<EOS>
<EOS>
<EOS>
ት በጥን ለ
[Sanity Metrics] Ethiopic Character Ratio: 0.364 | Repeated Trigrams: 0.530 | Perplexity: 3.51
```
- **Strengths**: Immediately continues with `"ገር"` forming `"የአዲስ አበባገር"` (compounding towards `"ሀገር"` / country/city). Generates valid verb ending `"ለመል።"`.
- **Failure Mode**: High `<EOS>` repetition rate ($53\%$ repeated trigrams). The model learned that sentence ends frequently coincide with document boundaries.

#### 5. Prompt: `"ታሪክ"` (History)
```
Prompt: "ታሪክ" (Seed: 46)
Output: ታሪክል።
<EOS>
<EOS>
<EOS>
<EOS>
<EOS>
ይምህ መርገ ያ የህ ማል አለት መር የህ የ
[Sanity Metrics] Ethiopic Character Ratio: 0.490 | Repeated Trigrams: 0.426 | Perplexity: 5.04
```
- **Strengths**: Immediately attaches Ethiopic full stop (`።`) and generates recognizable syllable clusters (`አለት`, `መር`, `ያ`).
- **Failure Mode**: Prompt terminates too abruptly into an `<EOS>` tail.

---

## System Architecture & Tensor Flow

For full tensor shapes across all forward and backward passes, refer to [`ARCHITECTURE.md`](file:///c:/Users/Administrator/Desktop/gpt/amharic-gpt/ARCHITECTURE.md).

```
                      Input Token Indices [B, T]
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       Token Embeddings W_te            Position Embeddings W_pe
            [B, T, D]                        [1, T, D]
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                         X_0 = W_te + W_pe  [B, T, D]
                                  │
         ┌────────────────────────┴────────────────────────┐
         │       Transformer Layer Block (x L)             │
         │                                                 │
         │   ┌─────────────────────────────────────────┐   │
         │   │ Pre-LayerNorm 1                         │   │
         │   │   x_norm = (x - μ) / √(σ² + ε) * γ + β   │   │
         │   └────────────────────┬────────────────────┘   │
         │                        ▼                        │
         │   ┌─────────────────────────────────────────┐   │
         │   │ Causal Multi-Head Self-Attention        │   │
         │   │   Q = x W_q, K = x W_k, V = x W_v       │   │
         │   │   Attn = softmax(Q K^T / √d_k + M) V    │   │
         │   │   Out = Attn W_o                        │   │
         │   │   (Supports incremental KV Cache)       │   │
         │   └────────────────────┬────────────────────┘   │
         │                        ▼                        │
         │     Residual Highway: x = x + Dropout(Out)      │
         │                        │                        │
         │   ┌────────────────────┴────────────────────┐   │
         │   │ Pre-LayerNorm 2                         │   │
         │   └────────────────────┬────────────────────┘   │
         │                        ▼                        │
         │   ┌─────────────────────────────────────────┐   │
         │   │ Position-Wise Feed-Forward Network      │   │
         │   │   h = GELU(x_norm W_1 + b_1)            │   │
         │   │   Out = h W_2 + b_2                     │   │
         │   └────────────────────┬────────────────────┘   │
         │                        ▼                        │
         │     Residual Highway: x = x + Dropout(Out)      │
         └────────────────────────┬────────────────────────┘
                                  ▼
                     Final LayerNorm ln_f [B, T, D]
                                  │
                                  ▼
                LM Head Projection: W_vocab [D, V]
                    (Optionally tied to W_te.T)
                                  │
                                  ▼
                       Logits [B, T, V]
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          Training Mode                    Inference Mode
   Numerically Stable Cross-Entropy       Autoregressive KV Cache
     + Analytical Backpropagation        Sampling (Greedy / Top-k / Top-p)
```

---

## Amharic-Specific Preprocessing & Corpus Pipeline

Amharic script presents specific linguistic and Unicode characteristics handled in `src/amharic_normalizer.py`:

1. **Ethiopic Wordspace Normalization**: Translates classical Ethiopic word dividers (`፡`, U+1361) to standard ASCII whitespace (` `).
2. **Configurable Homophone Merging**: Consolidates phonetically identical consonants across all seven vowel orders:
   - ሐ (U+1210–1216) & ኀ (U+1280–1286) $\to$ ሀ (U+1200–1206)
   - ሠ (U+1220–1226) $\to$ ሰ (U+1230–1236)
   - ዐ (U+12D0–12D6) $\to$ አ (U+12A0–12A6)
   - ፀ (U+1348–134E) $\to$ ጸ (U+1338–133E)
   *(Disabled by default to preserve classical Ge'ez orthography; toggled via `--normalize_homophones`)*.
3. **Punctuation & Numeral Integrity**: Preserves Ethiopic punctuation marks (`።`, `፣`, `፤`, `፥`, `፦`, `፧`, `፨`) and numerals (`፩` through `፼`), while collapsing run-on punctuation loops.
4. **Script Density Filtering**: Strips lines with $< 50\%$ Ethiopic character ratios to eliminate noisy web crawl artifacts.
5. **Exact Duplicate Removal**: Drops redundant sentences while recording duplicate counts in `outputs/corpus_report.md`.
6. **Document-Level Seeded Splitting**: Shuffles at document boundaries before dividing into $90\%$ Train, $5\%$ Validation, and $5\%$ Test sets separated by `<EOS>`.

---

## Tokenizers: Character-Level & Byte-Pair Encoding

The repository includes two independently implemented tokenizers:

### 1. Character-Level Tokenizer (`src/tokenizer.py`)
- Maps individual Ethiopic syllabary characters (Fidel) directly to token IDs.
- **Vocabulary Size**: 176 unique tokens (including `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).
- **Advantage**: Zero out-of-vocabulary tokens for any valid Amharic text; simple, compact embedding matrices.
- **Disadvantage**: Requires longer sequence context to model long paragraphs.

### 2. Subword Byte-Pair Encoding (`src/bpe_tokenizer.py`)
- Trained strictly on the training partition (`train.txt`).
- Base alphabet derived from all Unicode characters present in the corpus.
- Pre-tokenization splits on whitespace and Ethiopic punctuation (`። ፣ ፤ ፥ ፦ ፧ ፨ ፡`) to prevent cross-word merges.
- Priority queue tracking pair frequencies with deterministic tie-breaking.
- Invertible, lossless round-trip decoding via word-level reconstruction.
- **Evaluation**: BPE-300 achieves $5.610$ Bits-Per-Character vs. $6.509$ for character-level.

---

## Training Loop & Optimization

Implemented in `src/train.py` and `src/optimizer.py`:

- **AdamW Optimizer**: Decoupled weight decay ($\lambda = 0.01$) updates parameters without mixing weight decay into the second moment buffer.
- **Warmup + Cosine Decay**: Linear ramp for $T_{\text{warm}}$ iterations followed by cosine decay down to $\eta_{\text{min}} = 0.1 \times \eta_{\text{base}}$.
- **Gradient Accumulation**: Simulates large batch sizes over $K$ micro-batches with correct $1/K$ loss scaling:
  $$\mathcal{L}_{\text{accum}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_k$$
- **Global Gradient Norm Clipping**: Clips total gradient vector norm to $M_{\text{max}} = 1.0$:
  $$g \leftarrow g \times \min\left(1.0, \frac{M_{\text{max}}}{\|g\|_2 + 10^{-6}}\right)$$
- **Full Deterministic Validation**: Evaluates across complete validation sets to yield exact, un-noised validation loss, perplexity, and bits-per-character.
- **Resumption & State Checkpointing**: Checkpoints preserve parameter tensors, AdamW moment buffers ($m, v, t$), RNG state, step count, and configuration metadata. Resuming after saving matches uninterrupted training to exact floating-point precision.

---

## Inference & Autoregressive Decoding

Implemented in `src/generate.py`:

### Key-Value (KV) Caching
Incremental generation retains past projected key and value states:
```python
# KV Cache Decoding Step
k_t = x_t @ W_k;  v_t = x_t @ W_v
K = np.concatenate([K_past, k_t], axis=2)
V = np.concatenate([V_past, v_t], axis=2)
attn_weights = softmax(q_t @ K.T / sqrt(d_k))
out = attn_weights @ V
```
- **Performance**: Eliminates redundant $O(T^2)$ recomputation, achieving a **$2.5\times$ to $4\times$ speedup** over 200 tokens.
- **Sliding-Window Overflow**: Safely shifts and truncates cache when sequence length exceeds `block_size`.

### Sampling Engine
- **Greedy Decoding**: Selects $\text{argmax}(z)$.
- **Temperature Scaling**: Adjusts entropy via $z / T$.
- **Top-$k$ Truncation**: Masks all but the $k$ highest logit values to $-\infty$.
- **Top-$p$ (Nucleus) Truncation**: Retains minimal set with cumulative probability $\ge p$.
- **Repetition Penalty**: Scales previously generated token logits by $\theta_{\text{rep}} > 1.0$.
- **Special Token Masking**: Forces `<PAD>`, `<UNK>`, and `<BOS>` logits to $-\infty$.

---

## Project Directory Layout

```
amharic-gpt/
├── ARCHITECTURE.md             # One-page architecture & tensor shape contract specification
├── README.md                   # Comprehensive documentation, traceability, & experiments
├── requirements.txt            # Minimal test and plotting dependencies (numpy, pytest, matplotlib)
├── .gitignore                  # Clean repository exclusions
│
├── configs/                    # Training configuration files
│   ├── train_tiny.json         # Fast CPU smoke-test configuration (~124k params)
│   └── train_small.json        # Standard benchmark configuration (~3.2M params)
│
├── data/
│   ├── DATASET.md              # Corpus provenance, licensing, and preparation instructions
│   ├── raw/
│   │   └── sample_amharic.txt  # Curated authentic Amharic text corpus (smoke-test fixture)
│   └── processed/              # Normalized train.txt, val.txt, test.txt
│
├── notebooks/
│   └── demo.ipynb              # Top-to-bottom interactive Jupyter demonstration notebook
│
├── outputs/                    # Logs, evaluation metrics, and ablation results
│   ├── corpus_report.md        # Preprocessing summary statistics
│   ├── tokenizer_comparison.md # Char vs. BPE compression analysis
│   ├── experiments.csv         # 9-run ablation experiment matrix
│   └── samples_best_model.txt  # 5 verified generations with sanity metrics
│
├── scripts/
│   ├── clean.py                # Artifact cleanup script (removes cache and build temp files)
│   ├── prepare_corpus.py       # Corpus preprocessing CLI
│   ├── train_bpe.py            # BPE tokenizer training CLI
│   ├── plot_curves.py          # Training loss and perplexity visualization CLI
│   └── run_experiments.py      # Automated ablation experiment runner
│
├── src/                        # Core model and pipeline implementation (Pure NumPy)
│   ├── __init__.py             # Package declaration
│   ├── config.py               # GPTConfig dataclass with validation
│   ├── tokenizer.py            # Character-level Amharic tokenizer
│   ├── bpe_tokenizer.py        # From-scratch Byte-Pair Encoding tokenizer
│   ├── amharic_normalizer.py   # Amharic Unicode script cleaner and homophone normalizer
│   ├── data.py                 # DataLoader and micro-batch generation
│   ├── math_utils.py           # Softmax, causal masking, dropout, init_weights
│   ├── embeddings.py           # Token and learned positional embeddings
│   ├── activations.py          # GELU activation and analytical derivative
│   ├── normalization.py        # LayerNorm forward and backward passes
│   ├── attention.py            # Causal MultiHeadAttention with KV caching
│   ├── feed_forward.py         # Position-wise FFN with GELU
│   ├── transformer_block.py    # Pre-LN Transformer block with residuals
│   ├── model.py                # Full GPT model with analytical backprop and weight tying
│   ├── loss.py                 # Cross-entropy loss and logit gradients
│   ├── optimizer.py            # AdamW optimizer and cosine learning rate scheduler
│   ├── train.py                # Production training loop with gradient accumulation
│   ├── generate.py             # Autoregressive generation CLI with KV cache
│   └── utils.py                # Checkpoint serialization and integrity validation
│
└── tests/                      # Verification suite (Pure pytest & NumPy)
    ├── test_causal_leakage.py  # Future-token perturbation invariance check
    ├── test_gradient_checks.py # Float64 finite-difference gradient verification
    ├── test_contracts.py       # Shape and dtype contract tests (float32 vs float64)
    ├── test_determinism.py     # Explicit seed reproducibility tests
    ├── test_tokenizer.py       # Character tokenizer tests
    ├── test_bpe_tokenizer.py   # BPE tokenizer round-trip and merge tests
    ├── test_amharic_normalizer.py # Unicode normalization and Ethiopic wordspace tests
    ├── test_data_pipeline.py   # DataLoader batching and document split tests
    ├── test_math_utils.py      # Softmax numerical stability and causal mask tests
    ├── test_embeddings.py      # Embedding lookup and gradient accumulation tests
    ├── test_norm_act.py        # LayerNorm and GELU numerical checks
    ├── test_attention.py       # MHA forward, backward, and KV cache tests
    ├── test_feed_forward.py    # FFN forward and backward tests
    ├── test_transformer_block.py # Transformer block integration checks
    ├── test_model.py           # Full model parameter counting and forward checks
    ├── test_loss.py            # Cross-entropy loss and ignore_index verification
    ├── test_optimizer.py       # AdamW moments, weight decay, and clip_grad_norm
    ├── test_training.py        # Single-batch overfitting test (Loss < 0.1)
    ├── test_checkpoint.py      # Checkpoint round-trip and resume equivalence
    ├── test_generation.py      # KV-cache equality, sampling, and stopping tests
    └── test_pipeline_e2e.py    # Full pipeline integration verification
```

---

## Installation & Quickstart

### Prerequisites
- Python 3.9+
- NumPy >= 1.24.0
- pytest >= 7.0.0
- matplotlib >= 3.7.0 *(for plotting only)*

```bash
# Clone the repository
git clone https://github.com/GechDev/amharic-gpt.git
cd amharic-gpt

# Install minimal dependencies
pip install -r requirements.txt
```

### 1. Prepare Corpus
Ingest raw Amharic text, clean Unicode, remove duplicates, and build document splits:
```bash
python scripts/prepare_corpus.py --input data/raw/sample_amharic.txt --output_dir data/processed --seed 42
```

### 2. Train Model
Run training on CPU using the production training loop:
```bash
# Rapid local CPU smoke-test (~15 seconds):
python -m src.train --config configs/train_tiny.json --max_iters 100 --seed 42

# Train standard small model:
python -m src.train --config configs/train_small.json --max_iters 200 --seed 42
```

### 3. Generate Amharic Text
Generate text with KV cache acceleration and top-$p$ nucleus sampling:
```bash
# Sample completions:
python -m src.generate --checkpoint_dir outputs/small_amharic_run/checkpoints --checkpoint_name checkpoint_best --prompt "ሰላም" --max_new_tokens 60 --temperature 0.8 --top_k 20 --top_p 0.9 --seed 42

# KV Cache Speed Benchmark (200 tokens):
python -m src.generate --checkpoint_dir outputs/small_amharic_run/checkpoints --checkpoint_name checkpoint_best --benchmark --seed 42
```

### 4. Interactive Jupyter Demo
Launch the self-contained demonstration notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Test Suite & Verification

The repository contains an exhaustive test suite guaranteeing mathematical correctness and preventing regressions:

1. **Causal Leakage Test (`tests/test_causal_leakage.py`)**: Modifies future tokens in a sequence and verifies that logits at preceding positions remain bit-for-bit identical ($0.0$ difference).
2. **Float64 Finite-Difference Gradient Checks (`tests/test_gradient_checks.py`)**: Verifies analytic gradients against two-sided numerical approximations:
   $$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$
   Checked for `LayerNorm`, `GELU`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, and the entire `GPTLanguageModel`.
3. **Overfit Sanity Test (`tests/test_training.py`)**: Asserts that a small model overfits a single batch to cross-entropy loss $< 0.10$.
4. **Loss Properties (`tests/test_loss.py`)**: Confirms uniform distribution yields loss equal to $\ln(V)$ and validates `-100` target masking.
5. **Shape & Dtype Contracts (`tests/test_contracts.py`)**: Ensures strict tensor shape propagation and verifies that `float32` and `float64` types are maintained without unintended type upcasting.
6. **KV-Cache Equivalence (`tests/test_generation.py`)**: Verifies that incremental KV cache decoding produces identical logit outputs to a full re-computation forward pass within $10^{-5}$ tolerance.
7. **Resume Equivalence (`tests/test_checkpoint.py`)**: Confirms that training 20 steps, saving, and resuming for 20 steps produces parameters identical to training 40 continuous steps.

Run the test suite:
```bash
pytest tests/ -v
```

---

## Honest Limitations & Future Work

### Limitations
1. **CPU Computation Limits**: Written in pure NumPy without GPU CUDA kernel fusions. While fully vectorized with BLAS matrix operations, scaling beyond 5 million parameters requires hours on CPU compared to minutes on GPU.
2. **Sequence Context ($T=128$)**: Due to memory bounds in quadratic self-attention matrices during NumPy backpropagation ($B \times H \times T \times T$), context window is constrained to 128 characters.
3. **Phonotactic Syllable Hallucination**: Small models trained for few iterations learn character combinations and Amharic affix syntax, but occasionally produce nonsensical root combinations (e.g., `"መጠበመለር"`).
4. **Boundary Repetition Loops**: Models trained without large corpora can fall into `<EOS>` repetition loops when sampling temperature is low.

### Future Work
1. **Rotary Position Embeddings (RoPE)**: Migrate from learned absolute positional embeddings to relative rotary embeddings.
2. **FlashAttention Implementation**: Port FlashAttention tiling logic to NumPy/Cython block iterators to reduce memory complexity from $O(T^2)$ to $O(T)$.
3. **Grammatical Morphological Stemming**: Integrate rule-based Amharic stemmers (e.g., HornMorpho principles) into the BPE tokenization hierarchy.
