# French → English Neural Machine Translation System
### High-Performance Encoder-Decoder Transformer Built From Scratch

An end-to-end Neural Machine Translation (NMT) system that translates French sentences into English. The entire model, tokenizer, and decoding pipeline are built **strictly from scratch** in Python using PyTorch solely for tensor math, automatic differentiation, and hardware acceleration. No pre-trained models (such as Hugging Face transformers or MarianMT) or pre-existing weights are used.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Architectural Improvements](#-key-architectural-improvements)
- [Mathematical Foundations](#-mathematical-foundations)
  - [1. Sinusoidal Positional Encoding](#1-sinusoidal-positional-encoding)
  - [2. Scaled Dot-Product Attention & Fused Kernels](#2-scaled-dot-product-attention--fused-kernels)
  - [3. Multi-Head Attention (8 Heads)](#3-multi-head-attention-8-heads)
  - [4. Attention Masks](#4-attention-masks)
  - [5. Pre-LN Stacks with Residual Scaling](#5-pre-ln-stacks-with-residual-scaling)
  - [6. Weight Tying](#6-weight-tying)
- [Byte-Pair Encoding (BPE) Tokenizer](#-byte-pair-encoding-bpe-tokenizer)
- [Dataset Pipeline & Kaggle / Web Integration](#-dataset-pipeline--kaggle--web-integration)
- [Training Strategy](#-training-strategy)
  - [AdamW with Decoupled Weight Decay](#adamw-with-decoupled-weight-decay)
  - [Teacher Forcing & Label Smoothing](#teacher-forcing--label-smoothing)
  - [Noam Learning Rate Scheduler](#noam-learning-rate-scheduler)
  - [Overfitting / Memorization Sanity Check](#overfitting--memorization-sanity-check)
- [Inference: Optimized Beam Search](#-inference-optimized-beam-search)
  - [Length Normalization (Wu et al.)](#length-normalization-wu-et-al)
  - [Repetition Blocking](#repetition-blocking)
- [Evaluation & Quality Metrics](#-evaluation--quality-metrics)
- [Project Structure](#-project-structure)
- [Installation & Quick Start](#-installation--quick-start)
  - [1. Installation](#1-installation)
  - [2. Running the Full Pipeline](#2-running-the-full-pipeline)
  - [3. Interactive Translation Chatbot](#3-interactive-translation-chatbot)
- [CLI Arguments Reference](#-cli-arguments-reference)

---

## 🏛️ Overview & Architecture

The system implements an enhanced **Seq2Seq Transformer** architecture with an auto-regressive decoder:

```text
French Source Sentence ("Je voudrais un café.")
         │
         ▼
[ BPE Tokenizer (French Vocab) ]
         │
         ▼ Token IDs: [2, 45, 128, 12, 89, 3] (<BOS> ... <EOS>)
         │
    ┌────┴────────────────────────────────────────────┐
    │              ENCODER                            │
    │  Token Embeddings * sqrt(d_model)               │
    │  + Sinusoidal Positional Encodings              │
    │  ┌───────────────────────────────────────────┐  │
    │  │  Multi-Head Self-Attention (8 heads)      │  │ x 4 Layers
    │  │  Pre-LN + Residual Scaling                │  │
    │  │  Position-wise Feed-Forward (GELU, d_ff)  │  │
    │  └───────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────┘
                         │ Encoder Memory Representations
                         ▼
    ┌─────────────────────────────────────────────────┐
    │              DECODER                            │
    │  Shifted Target Tokens (<BOS> I would...)       │
    │  + Sinusoidal Positional Encodings              │
    │  ┌───────────────────────────────────────────┐  │
    │  │  Masked Causal Self-Attention (Pre-LN)    │  │
    │  │  Cross-Attention over Encoder Memory      │  │ x 4 Layers
    │  │  Position-wise Feed-Forward (GELU, d_ff)  │  │
    │  │  Residual Scaling                         │  │
    │  └───────────────────────────────────────────┘  │
    │  Tied Output Linear Head (Shares Tgt Embeddings)│
    └────────────────────┬────────────────────────────┘
                         │
                         ▼
             Optimized Beam Search
         (Length Penalty + Repetition Blocking)
                         │
                         ▼
English Translation ("I would like a coffee.")
```

---

## ⚡ Key Architectural Improvements

1. **GELU Activations**: Replaced standard ReLU with Gaussian Error Linear Units (`nn.GELU()`) in feed-forward networks, providing smoother gradients and richer representation capacity.
2. **Weight Tying**: The target token embedding matrix is tied directly with the output linear projection matrix (`generator.weight = tgt_tok_embed.embedding.weight`). This cuts parameter count in half for vocabulary projections and accelerates vocabulary learning.
3. **Fused Attention Kernels**: Utilizes PyTorch's native `F.scaled_dot_product_attention` for 2-4x higher execution throughput and reduced memory consumption on both CPU and GPU.
4. **Pre-LN with Deep Residual Scaling**: Normalization is computed prior to attention and feed-forward sub-layers (Pre-LN), combined with depth-dependent residual scaling factors ($1/\sqrt{2N_{enc}}$ and $1/\sqrt{3N_{dec}}$), guaranteeing stable gradient flow.
5. **Decoupled Weight Decay (AdamW)**: Weight decay ($10^{-2}$) is applied strictly to multi-dimensional weight matrices while disabling decay on LayerNorm parameters and biases.
6. **Optimized Beam Search Exclusively**: Greedy decoding has been replaced across the pipeline by Beam Search featuring length normalization and 3-gram repetition blocking.

---

## 📐 Mathematical Foundations

### 1. Sinusoidal Positional Encoding
Deterministic sinusoidal encodings provide positional order without learned parameters:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

### 2. Scaled Dot-Product Attention & Fused Kernels
Given Query ($Q$), Key ($K$), and Value ($V$) matrices:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Accelerated via hardware-optimized fused attention primitives.

### 3. Multi-Head Attention (8 Heads)
Multi-Head Attention projects $Q$, $K$, and $V$ into $h = 8$ distinct representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_8)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 4. Attention Masks
- **Padding Mask**: Prevents the model from attending to `<PAD>` positions.
- **Causal (Look-Ahead) Mask**: Upper-triangular boolean mask ensuring position $t$ only attends to positions $\le t$.

### 5. Pre-LN Stacks with Residual Scaling
$$x^{(l+1)} = x^{(l)} + \lambda \cdot \text{SubLayer}(\text{LayerNorm}(x^{(l)}))$$

Where $\lambda = \frac{1}{\sqrt{2 N_{layers}}}$ for encoder layers and $\lambda = \frac{1}{\sqrt{3 N_{layers}}}$ for decoder layers.

### 6. Weight Tying
Target embedding weights are shared with the final projection head:

$$W_{generator} = W_{tgt\_embedding}$$

---

## 🔤 Byte-Pair Encoding (BPE) Tokenizer

Built from scratch (`app/tokenizer/`):
- Subword segmentation via iterative pair merges.
- End-of-word markers `</w>` preserve morphology and avoid word boundary ambiguity.
- Special tokens: `<PAD>` (0), `<UNK>` (1), `<BOS>` (2), `<EOS>` (3).

---

## 🌐 Dataset Pipeline & Kaggle / Web Integration

- **Automatic Internet Download**: Downloads the Tatoeba French-English corpus (~240,500 parallel pairs, identical to Kaggle datasets `devansodariya/english-french-translations` and `dhruvildave/french-english-bilingual-pairs`).
- **Local Kaggle Support**: Load any local Kaggle CSV/TSV via `--source local --kaggle-file path/to/dataset.csv`.
- **Cleaning & Splits**: Normalizes unicode (`NFKC`), filters by length (1 to 50 words) and length ratio, with 80/10/10 Train/Validation/Test splits.

---

## 🏋️ Training Strategy

### AdamW with Decoupled Weight Decay
Weights with dimension $\ge 2$ receive $10^{-2}$ weight decay, while 1D biases and LayerNorm parameters receive $0.0$ decay.

### Teacher Forcing & Label Smoothing
Cross-entropy loss with `label_smoothing=0.1` and `ignore_index=PAD_IDX`.

### Noam Learning Rate Scheduler
$$\text{lr} = d_{model}^{-0.5} \cdot \min\left(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$

---

## 🔍 Inference: Optimized Beam Search

The system uses **Beam Search** as its definitive decoding strategy:

### Length Normalization (Wu et al.)
$$\text{score}(Y) = \frac{\log P(Y)}{\text{LP}(Y)}, \quad \text{LP}(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha} \quad (\alpha = 0.7)$$

### Repetition Blocking
An active n-gram filter (`no_repeat_ngram_size=3`) discards any candidate token that would create a repeated trigram, eliminating repetitive loops.

---

## 📁 Project Structure

```text
Translation/
├── app/
│   ├── tokenizer/               # Custom BPE Tokenizer
│   │   ├── vocabulary.py        # Vocabulary mapping & special tokens
│   │   ├── tokenizer.py         # Subword segmentation, encode & decode
│   │   └── train_bpe.py         # BPE merge training algorithm
│   ├── data/                    # Dataset processing & acquisition
│   │   ├── download.py          # Internet / Kaggle dataset downloader
│   │   ├── clean.py             # Unicode normalization & filtering
│   │   ├── split.py             # Train / Validation / Test splitter
│   │   └── dataset.py           # PyTorch Dataset & dynamic batch collator
│   ├── model/                   # High-Performance Transformer Seq2Seq
│   │   ├── embeddings.py        # TokenEmbedding & PositionalEncoding
│   │   ├── attention.py         # Fused MultiHeadAttention (8 heads)
│   │   ├── masks.py             # Padding & Causal look-ahead masks
│   │   ├── encoder.py           # Pre-LN EncoderLayer (GELU) & TransformerEncoder
│   │   ├── decoder.py           # Pre-LN DecoderLayer (GELU) & TransformerDecoder
│   │   └── transformer.py       # Seq2Seq Transformer with Weight Tying
│   ├── training/                # Training pipeline
│   │   ├── scheduler.py         # Noam learning rate scheduler
│   │   ├── validate.py          # Validation evaluation loop
│   │   ├── checkpoint.py        # Checkpoint serialization
│   │   └── train.py             # AdamW training loop with Teacher Forcing
│   ├── inference/               # Autoregressive generation
│   │   └── beam_search.py       # Optimized Beam Search with repetition blocking
│   ├── evaluation/              # Evaluation metrics & error reports
│   │   ├── chrf.py              # chrF character F-score implementation
│   │   └── error_analysis.py    # Repetition, omission & hallucination analysis
│   └── interface/               # User interface & High-level API
│       ├── api.py               # Translator wrapper class (Beam Search)
│       └── app.py               # Interactive CLI chatbot
├── main.py                      # Automated end-to-end pipeline runner
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch and NumPy
pip install torch numpy
```

### 2. Running the Full Pipeline

```bash
# Standard training run (10,000 sentences, 10 epochs, 8 heads)
.venv/bin/python main.py --source internet --max-samples 10000 --epochs 10 --vocab-size 2000

# Scaled training run (50,000 sentences, 20 epochs)
.venv/bin/python main.py --source internet --max-samples 50000 --epochs 20 --vocab-size 4000
```

### 3. Interactive Translation Chatbot

Launch the interactive chat interface directly:

```bash
.venv/bin/python main.py --chat
```

Or launch directly from a saved checkpoint:

```bash
.venv/bin/python app/interface/app.py checkpoints/best_model.pt data/tokenizers/tokenizer_fr.json data/tokenizers/tokenizer_en.json
```

**Chatbot Commands:**
- Type any French sentence to translate it into English.
- `/beam <size>` : Adjust beam search width (e.g. `/beam 5`).
- `/aide` : Display command help.
- `/quitter` : Exit the chatbot.

---

## ⚙️ CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source` | `str` | `"internet"` | Data source (`"internet"` for Tatoeba/Kaggle download, or `"local"`). |
| `--max-samples` | `int` | `10000` | Number of sentence pairs to extract. |
| `--kaggle-file` | `str` | `None` | Path to a local CSV/TSV Kaggle dataset when `--source local`. |
| `--epochs` | `int` | `10` | Number of training epochs. |
| `--vocab-size` | `int` | `2000` | Target BPE vocabulary size. |
| `--batch-size` | `int` | `32` | Training batch size. |
| `--beam-size` | `int` | `5` | Beam search width for inference. |
| `--skip-memorize` | `flag` | `False` | Skip the step 4 memorization sanity check. |
| `--chat` | `flag` | `False` | Automatically launch the interactive chatbot. |
| `--force-train` | `flag` | `False` | Force retraining even if a checkpoint exists when using `--chat`. |
