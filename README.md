# French → English Neural Machine Translation System
### Encoder-Decoder Transformer Architecture Built From Scratch (No Pre-trained Weights)

An end-to-end Neural Machine Translation (NMT) system that translates French sentences into English. The entire model, tokenizer, and decoding pipeline are built **strictly from scratch** in Python using PyTorch solely for tensor math, automatic differentiation, and hardware acceleration. No pre-trained models (such as Hugging Face transformers or MarianMT) or pre-existing weights are used.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Mathematical Foundations](#-mathematical-foundations)
  - [1. Sinusoidal Positional Encoding](#1-sinusoidal-positional-encoding)
  - [2. Scaled Dot-Product Attention](#2-scaled-dot-product-attention)
  - [3. Multi-Head Attention](#3-multi-head-attention)
  - [4. Attention Masks](#4-attention-masks)
  - [5. Pre-LN Encoder & Decoder Stacks](#5-pre-ln-encoder--decoder-stacks)
- [Byte-Pair Encoding (BPE) Tokenizer](#-byte-pair-encoding-bpe-tokenizer)
- [Dataset Pipeline & Kaggle / Web Integration](#-dataset-pipeline--kaggle--web-integration)
- [Training Strategy](#-training-strategy)
  - [Teacher Forcing & Label Smoothing](#teacher-forcing--label-smoothing)
  - [Noam Learning Rate Scheduler](#noam-learning-rate-scheduler)
  - [Overfitting / Memorization Sanity Check](#overfitting--memorization-sanity-check)
- [Inference & Decoding Strategies](#-inference--decoding-strategies)
  - [Greedy Decoding](#greedy-decoding)
  - [Beam Search with Length Normalization](#beam-search-with-length-normalization)
- [Evaluation & Quality Metrics](#-evaluation--quality-metrics)
- [Project Structure](#-project-structure)
- [Installation & Quick Start](#-installation--quick-start)
  - [1. Installation](#1-installation)
  - [2. Running the Full Pipeline](#2-running-the-full-pipeline)
  - [3. Interactive Translation Chatbot](#3-interactive-translation-chatbot)
- [CLI Arguments Reference](#-cli-arguments-reference)

---

## 🏛️ Overview & Architecture

The system implements the classic **Seq2Seq Transformer** architecture with an auto-regressive decoder:

```text
French Source Sentence ("Je voudrais un café.")
         │
         ▼
[ BPE Tokenizer (French Vocab) ]
         │
         ▼ Token IDs: [2, 45, 128, 12, 89, 3] (<BOS> ... <EOS>)
         │
    ┌────┴───────────────────────────────────────┐
    │              ENCODER                       │
    │  Token Embeddings * sqrt(d_model)          │
    │  + Sinusoidal Positional Encodings         │
    │  ┌──────────────────────────────────────┐  │
    │  │  Multi-Head Self-Attention (Pre-LN)  │  │ x N Layers
    │  │  Position-wise Feed-Forward (d_ff)   │  │
    │  └──────────────────────────────────────┘  │
    └────────────────────┬───────────────────────┘
                         │ Encoder Memory Representations
                         ▼
    ┌────────────────────────────────────────────┐
    │              DECODER                       │
    │  Shifted Target Tokens (<BOS> I would...)  │
    │  + Sinusoidal Positional Encodings         │
    │  ┌──────────────────────────────────────┐  │
    │  │  Masked Causal Self-Attention        │  │
    │  │  Cross-Attention (over Enc Memory)   │  │ x N Layers
    │  │  Position-wise Feed-Forward (d_ff)   │  │
    │  └──────────────────────────────────────┘  │
    │  Linear Projection Head -> Target Vocab    │
    └────────────────────┬───────────────────────┘
                         │
                         ▼
             Autoregressive Decoding
             (Greedy or Beam Search)
                         │
                         ▼
English Translation ("I would like a coffee.")
```

---

## 📐 Mathematical Foundations

### 1. Sinusoidal Positional Encoding
Transformers lack inherent recurrence or convolution, making them permutation-invariant. To inject positional order, deterministic sinusoidal encodings are added to the input embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

Where:
- $pos$ is the token index in the sequence ($0 \le pos < \text{max\_len}$).
- $i$ is the dimension index ($0 \le i < d_{model} / 2$).

### 2. Scaled Dot-Product Attention
Given Query ($Q$), Key ($K$), and Value ($V$) matrices:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

- $\sqrt{d_k}$ is the scaling factor preventing vanishing gradients in the softmax function for large dimensions.
- $M$ is an optional additive attention mask ($-\infty$ on masked positions, $0$ elsewhere).

### 3. Multi-Head Attention
Multi-Head Attention projects $Q$, $K$, and $V$ into $h$ distinct representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

With projection parameters:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{model}}$

### 4. Attention Masks
- **Padding Mask**: Prevents the model from attending to `<PAD>` tokens in source and target sequences.
- **Causal (Look-Ahead) Mask**: An upper-triangular boolean mask applied in decoder self-attention to ensure position $t$ can only attend to positions $\le t$.

### 5. Pre-LN Encoder & Decoder Stacks
This implementation utilizes **Pre-Layer Normalization (Pre-LN)**, where normalization is applied before the sub-layers rather than after:

$$x^{(l+1)} = x^{(l)} + \text{SubLayer}(\text{LayerNorm}(x^{(l)}))$$

Pre-LN provides substantially smoother gradient flow during backpropagation and prevents vanishing/exploding gradients when training from scratch.

---

## 🔤 Byte-Pair Encoding (BPE) Tokenizer

The subword tokenizer is implemented completely from scratch (`app/tokenizer/`):

1. **Character Decomposition**: Words are initially split into individual characters followed by an end-of-word marker `</w>` (e.g., `"manger"` $\rightarrow$ `('m', 'a', 'n', 'g', 'e', 'r</w>')`).
2. **Pair Frequency Counting**: The corpus is scanned to count co-occurrences of all adjacent symbol pairs.
3. **Iterative Merging**: The most frequent pair is merged and added to the vocabulary. This process repeats until reaching the target vocabulary size.
4. **Special Tokens**:
   - `<PAD>` (Index 0): Sequence padding for batch collation.
   - `<UNK>` (Index 1): Unknown tokens/characters.
   - `<BOS>` (Index 2): Beginning of sequence.
   - `<EOS>` (Index 3): End of sequence.
5. **Serialization**: Tokenizer states (vocabularies and learned merges) are serialized directly into JSON files.

---

## 🌐 Dataset Pipeline & Kaggle / Web Integration

The data pipeline (`app/data/`) supports both automatic web acquisition and local dataset files:

- **Automatic Internet Download**: Downloads the real-world bilingual sentence dataset curated by the Tatoeba Project (the exact same underlying data featured in popular Kaggle datasets such as `devansodariya/english-french-translations` and `dhruvildave/french-english-bilingual-pairs`).
  - Contains **~240,500 parallel sentence pairs** translated and verified by native speakers.
- **Local Kaggle CSV / TSV Support**: Allows loading any downloaded Kaggle dataset directly via `--source local --kaggle-file path/to/dataset.csv`.
- **Filtering & Cleaning**:
  - Unicode normalization (`NFKC`).
  - Removal of corrupt and empty entries.
  - Length filtering ($1 \le \text{length} \le 50$ words).
  - Source/target length ratio filtering to remove misaligned pairs.
  - Deduplication.
- **Splits**: Automatically splits the cleaned corpus into **Train (80%)**, **Validation (10%)**, and **Test (10%)** sets.

---

## 🏋️ Training Strategy

### Teacher Forcing & Label Smoothing
During training, the decoder receives the ground-truth target sequence shifted by one token:
- **Decoder Input**: `tgt[:, :-1]` (starts with `<BOS>`)
- **Expected Output**: `tgt[:, 1:]` (ends with `<EOS>`)
- **Loss Function**: `CrossEntropyLoss` with `ignore_index=PAD_IDX` and `label_smoothing=0.1` to prevent overconfidence.

### Noam Learning Rate Scheduler
Implements the warmup schedule from *Attention Is All You Need*:

$$\text{lr} = d_{model}^{-0.5} \cdot \min\left(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$

### Overfitting / Memorization Sanity Check
Before full training, the pipeline runs a dedicated verification step on a 50-sentence batch. The model must drive training loss down to $\approx 0.0$ within 200 steps. This strictly proves the mathematical integrity of:
- Causal masking (preventing future information leakage).
- Cross-attention key/value alignment.
- Embedding scaling and gradient backpropagation.

---

## 🔍 Inference & Decoding Strategies

### Greedy Decoding
At each step $t$, the decoder selects the token with the highest predicted probability:

$$\hat{y}_t = \arg\max_{w \in V} P(w \mid y_{<t}, x)$$

Generation terminates when `<EOS>` is predicted or `max_len` is reached.

### Beam Search with Length Normalization
Maintains a beam of $K$ candidate hypotheses. To prevent penalizing longer, grammatically complete sentences, hypotheses are ranked using the length penalty formula of Wu et al. (2016):

$$\text{score}(Y) = \frac{\log P(Y)}{\text{LP}(Y)}$$

$$\text{LP}(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$$

Where $\alpha = 0.6$ by default.

---

## 📊 Evaluation & Quality Metrics

- **chrF (Character n-gram F-score)**: Evaluates character n-grams (1 to 6) between predictions and references. Robust against morphological variations and subword segmentation artifacts.
- **Automated Error Analysis**:
  - **Repetition Rate**: Identifies looping / repetitive generation failures.
  - **Omission Rate**: Detects premature terminations where the generated translation is less than half the reference length.
  - **Hallucination Rate**: Identifies runaway generations that exceed twice the reference length.

---

## 📁 Project Structure

```text
Translation/
├── app/
│   ├── tokenizer/               # Custom BPE Tokenizer
│   │   ├── vocabulary.py        # Vocabulary mapping and special tokens
│   │   ├── tokenizer.py         # Subword segmentation, encode & decode
│   │   └── train_bpe.py         # BPE merge training algorithm
│   │
│   ├── data/                    # Dataset processing & acquisition
│   │   ├── download.py          # Internet / Kaggle dataset downloader
│   │   ├── clean.py             # Unicode normalization & length filtering
│   │   ├── split.py             # Train / Validation / Test splitter
│   │   └── dataset.py           # PyTorch Dataset & dynamic batch collator
│   │
│   ├── model/                   # Transformer Seq2Seq from scratch
│   │   ├── embeddings.py        # TokenEmbedding & PositionalEncoding
│   │   ├── attention.py         # MultiHeadAttention (Self & Cross)
│   │   ├── masks.py             # Padding & Causal look-ahead masks
│   │   ├── encoder.py           # Pre-LN EncoderLayer & TransformerEncoder
│   │   ├── decoder.py           # Pre-LN DecoderLayer & TransformerDecoder
│   │   └── transformer.py       # Full Seq2Seq Transformer model
│   │
│   ├── training/                # Training pipeline
│   │   ├── scheduler.py         # Noam learning rate scheduler
│   │   ├── validate.py          # Validation evaluation loop
│   │   ├── checkpoint.py        # Checkpoint serialization
│   │   └── train.py             # Teacher-forcing training loop
│   │
│   ├── inference/               # Autoregressive generation
│   │   ├── greedy_decode.py     # Greedy decoding algorithm
│   │   └── beam_search.py       # Beam search with length penalty
│   │
│   ├── evaluation/              # Evaluation metrics & error reports
│   │   ├── chrf.py              # chrF character F-score implementation
│   │   └── error_analysis.py    # Repetition, omission & hallucination analysis
│   │
│   └── interface/               # User interface & High-level API
│       ├── api.py               # Translator wrapper class
│       └── app.py               # Interactive CLI chatbot
│
├── main.py                      # Automated end-to-end pipeline runner
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Installation

Ensure Python 3.10+ is available:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch and NumPy
pip install torch numpy
```

### 2. Running the Full Pipeline

Run the automated pipeline to download the dataset, train tokenizers, verify memorization, train the Transformer, and evaluate decoding strategies:

```bash
# Fast training run (10,000 sentences, 10 epochs)
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
- `/mode [greedy|beam]` : Switch decoding strategy on the fly.
- `/beam <size>` : Adjust beam search width (e.g. `/beam 5`).
- `/aide` : Display command help.
- `/quitter` : Exit the chatbot.

---

## ⚙️ CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source` | `str` | `"internet"` | Data source (`"internet"` for Tatoeba/Kaggle download, or `"local"`). |
| `--max-samples` | `int` | `10000` | Number of sentence pairs to extract (use large value or omit for full dataset). |
| `--kaggle-file` | `str` | `None` | Path to a local CSV/TSV Kaggle dataset when `--source local`. |
| `--epochs` | `int` | `10` | Number of training epochs. |
| `--vocab-size` | `int` | `2000` | Target BPE vocabulary size. |
| `--batch-size` | `int` | `32` | Training batch size. |
| `--skip-memorize` | `flag` | `False` | Skip the step 4 memorization sanity check. |
| `--chat` | `flag` | `False` | Automatically launch the interactive chatbot after the pipeline finishes. |
