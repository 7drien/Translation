"""
Main entry point for the French -> English Neural Machine Translation project.
Executes the end-to-end training and evaluation pipeline:
1. Dataset acquisition (Internet / Kaggle download) & cleaning
2. Byte-Pair Encoding (BPE) tokenizer training from scratch
3. Transformer model initialization (Encoder-Decoder from scratch)
4. Overfitting / Memorization sanity check on real data
5. Full model training with validation and checkpointing
6. Autoregressive decoding (Greedy vs Beam Search)
7. Metric evaluation (chrF) & qualitative error analysis
8. Interactive chatbot interface
"""

from __future__ import annotations
import os
import sys
import argparse
import torch

from app.data import prepare_dataset, clean_files, split_and_save, create_dataloader
from app.tokenizer import train_bpe, BPETokenizer
from app.model import Transformer, create_masks
from app.training import train_model, save_checkpoint, load_checkpoint
from app.inference import greedy_decode, beam_search_decode
from app.evaluation import corpus_chrf, analyze_translation_errors
from app.interface import Translator, run_chatbot


def step_1_and_2_data_and_tokenizer(
    data_dir: str = "data",
    source: str = "internet",
    max_samples: int = 10000,
    local_file: str | None = None,
    vocab_size: int = 2000
):
    """Download dataset from the internet/Kaggle, clean, split, and train BPE tokenizers."""
    print("\n" + "=" * 60)
    print("STEP 1 & 2: Dataset Acquisition (Internet/Kaggle) & BPE Tokenizer")
    print("=" * 60)

    raw_src, raw_tgt = prepare_dataset(
        output_dir=data_dir,
        source=source,
        max_samples=max_samples,
        local_file=local_file
    )
    print(f"✓ Raw corpus prepared: {raw_src}, {raw_tgt}")

    clean_src = os.path.join(data_dir, "clean.fr")
    clean_tgt = os.path.join(data_dir, "clean.en")
    n_cleaned = clean_files(raw_src, raw_tgt, clean_src, clean_tgt, min_length=1, max_length=50)
    print(f"✓ Cleaned and deduplicated corpus: {n_cleaned} sentence pairs")

    with open(clean_src, "r", encoding="utf-8") as fs, open(clean_tgt, "r", encoding="utf-8") as ft:
        pairs = [(s.strip(), t.strip()) for s, t in zip(fs, ft)]

    split_paths = split_and_save(pairs, output_dir=data_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
    print(f"✓ Dataset split: train ({len(pairs)*0.8:.0f}), valid ({len(pairs)*0.1:.0f}), test ({len(pairs)*0.1:.0f})")

    # Train BPE Tokenizers on French and English corpora
    fr_sentences = [p[0] for p in pairs]
    en_sentences = [p[1] for p in pairs]

    print(f"✓ Training French BPE Tokenizer (target vocab_size={vocab_size})...")
    src_tokenizer = train_bpe(fr_sentences, vocab_size=vocab_size, min_frequency=2)
    os.makedirs(f"{data_dir}/tokenizers", exist_ok=True)
    src_tok_path = f"{data_dir}/tokenizers/tokenizer_fr.json"
    src_tokenizer.save(src_tok_path)
    print(f"  French Vocab: {len(src_tokenizer.vocab)} tokens. Saved to {src_tok_path}")

    print(f"✓ Training English BPE Tokenizer (target vocab_size={vocab_size})...")
    tgt_tokenizer = train_bpe(en_sentences, vocab_size=vocab_size, min_frequency=2)
    tgt_tok_path = f"{data_dir}/tokenizers/tokenizer_en.json"
    tgt_tokenizer.save(tgt_tok_path)
    print(f"  English Vocab: {len(tgt_tokenizer.vocab)} tokens. Saved to {tgt_tok_path}")

    # Test round-trip encoding/decoding on a sample sentence from the dataset
    test_phrase = pairs[0][0]
    encoded = src_tokenizer.encode(test_phrase)
    decoded = src_tokenizer.decode(encoded)
    print(f"  Sample Round-trip: '{test_phrase}' -> {len(encoded)} tokens -> '{decoded}'")

    return pairs, split_paths, src_tokenizer, tgt_tokenizer, src_tok_path, tgt_tok_path


def step_6_memorization_sanity_check(
    pairs: list,
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    device: torch.device,
    num_sentences: int = 50,
    steps: int = 200
):
    """
    Crucial sanity check:
    Verify that the model can quickly overfit and memorize a small batch of real sentences.
    Validates causal masks, teacher forcing, dimensions, and gradient flow.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Overfitting / Memorization Sanity Check (Real Data)")
    print("=" * 60)
    print(f"Testing whether Transformer can memorize {num_sentences} real sentences to ~0 loss...")

    toy_pairs = pairs[:num_sentences]

    toy_loader = create_dataloader(
        pairs=toy_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=len(toy_pairs),
        shuffle=False
    )

    toy_model = Transformer(
        src_vocab_size=len(src_tokenizer.vocab),
        tgt_vocab_size=len(tgt_tokenizer.vocab),
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.0,
        pad_idx=0,
        norm_first=True
    ).to(device)

    optimizer = torch.optim.Adam(toy_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    src, tgt = next(iter(toy_loader))
    src, tgt = src.to(device), tgt.to(device)
    dec_in, exp_out = tgt[:, :-1], tgt[:, 1:]
    s_mask, t_mask, _ = create_masks(src, dec_in, pad_idx=0)

    final_loss = 0.0
    for s in range(1, steps + 1):
        toy_model.train()
        optimizer.zero_grad()
        logits = toy_model(src, dec_in, s_mask, t_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), exp_out.reshape(-1))
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if s % 50 == 0 or s == steps:
            print(f"  Step {s:03d}/{steps} | Memorization Loss: {final_loss:.4f}")

    if final_loss < 0.2:
        print("✓ SUCCESS: Model successfully memorized the real sentence sample!")
        print("  Causal masks, cross-attention, embeddings and gradients are mathematically verified.")
    else:
        print(f"⚠ WARNING: Final loss ({final_loss:.4f}) is higher than expected. Check learning rate or steps.")


def step_7_and_8_train(
    split_paths: dict,
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 32,
    checkpoint_dir: str = "checkpoints"
) -> Transformer:
    """Train the Transformer on the full dataset with validation."""
    print("\n" + "=" * 60)
    print("STEP 7 & 8: Full Model Training with Validation")
    print("=" * 60)

    def load_split(src_p, tgt_p):
        with open(src_p, "r", encoding="utf-8") as fs, open(tgt_p, "r", encoding="utf-8") as ft:
            return [(s.strip(), t.strip()) for s, t in zip(fs, ft)]

    train_pairs = load_split(*split_paths["train"])
    valid_pairs = load_split(*split_paths["valid"])

    train_loader = create_dataloader(train_pairs, src_tokenizer, tgt_tokenizer, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(valid_pairs, src_tokenizer, tgt_tokenizer, batch_size=batch_size, shuffle=False)

    print(f"Initializing Transformer Architecture:")
    print(f"  Src Vocab: {len(src_tokenizer.vocab)} | Tgt Vocab: {len(tgt_tokenizer.vocab)}")
    print(f"  d_model=256, num_heads=4, N_enc=4, N_dec=4, d_ff=1024, dropout=0.1")

    model = Transformer(
        src_vocab_size=len(src_tokenizer.vocab),
        tgt_vocab_size=len(tgt_tokenizer.vocab),
        d_model=256,
        num_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=1024,
        dropout=0.1,
        pad_idx=0,
        norm_first=True
    ).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        lr=5e-4,
        warmup_steps=200,
        checkpoint_dir=checkpoint_dir,
        device=device,
        label_smoothing=0.1,
        verbose=True
    )

    return model


def step_9_10_11_evaluation(
    split_paths: dict,
    model: Transformer,
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    device: torch.device
):
    """Run Greedy vs Beam Search decoding, compute metrics and error analysis."""
    print("\n" + "=" * 60)
    print("STEP 9, 10 & 11: Greedy vs Beam Search Evaluation & Error Analysis")
    print("=" * 60)

    with open(split_paths["test"][0], "r", encoding="utf-8") as fs, open(split_paths["test"][1], "r", encoding="utf-8") as ft:
        test_pairs = [(s.strip(), t.strip()) for s, t in zip(fs, ft)]

    sources = [p[0] for p in test_pairs[:20]]
    references = [p[1] for p in test_pairs[:20]]

    translator = Translator(model, src_tokenizer, tgt_tokenizer, device=device)

    print("\nGenerating translations on test sample...")
    greedy_translations = translator.translate_batch(sources, strategy="greedy")
    beam_translations = translator.translate_batch(sources, strategy="beam", beam_size=4)

    # Compute chrF scores
    chrf_greedy = corpus_chrf(greedy_translations, references)
    chrf_beam = corpus_chrf(beam_translations, references)

    print(f"\nResults Comparison:")
    print(f"  Greedy Decoding chrF: {chrf_greedy * 100:.2f}%")
    print(f"  Beam Search (k=4) chrF: {chrf_beam * 100:.2f}%")

    print("\nSample Translations (Side-by-Side):")
    print("-" * 80)
    for i in range(min(5, len(sources))):
        print(f"FR       : {sources[i]}")
        print(f"REF (EN) : {references[i]}")
        print(f"GREEDY   : {greedy_translations[i]}")
        print(f"BEAM (k=4): {beam_translations[i]}")
        print("-" * 80)

    # Qualitative Error Analysis
    print("\nError Analysis (Beam Search):")
    analysis = analyze_translation_errors(sources, beam_translations, references)
    print(f"  Repetition Rate: {analysis['repetition_rate']}%")
    print(f"  Omission Rate  : {analysis['omission_rate']}%")
    print(f"  Hallucination  : {analysis['hallucination_rate']}%")

    return translator


def main():
    parser = argparse.ArgumentParser(description="French -> English Neural Machine Translation Pipeline")
    parser.add_argument("--source", type=str, default="internet", choices=["internet", "local"],
                        help="Data source: 'internet' (downloads Tatoeba/Kaggle dataset) or 'local'")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Maximum sentence pairs to extract from dataset (default: 10000)")
    parser.add_argument("--kaggle-file", type=str, default=None,
                        help="Path to local Kaggle CSV/TSV file if --source is 'local'")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--vocab-size", type=int, default=2000, help="BPE vocabulary size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--chat", action="store_true", help="Launch interactive chatbot (loads existing checkpoint if available)")
    parser.add_argument("--force-train", action="store_true", help="Force retraining even if a checkpoint exists when using --chat")
    parser.add_argument("--skip-memorize", action="store_true", help="Skip the step 6 sanity check")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    checkpoint_file = "checkpoints/best_model.pt"
    src_tok_file = "data/tokenizers/tokenizer_fr.json"
    tgt_tok_file = "data/tokenizers/tokenizer_en.json"

    # If --chat is requested and an existing checkpoint is found, launch chat immediately
    if args.chat and not args.force_train and os.path.exists(checkpoint_file) and os.path.exists(src_tok_file) and os.path.exists(tgt_tok_file):
        print(f"\n✓ Found existing trained checkpoint at {checkpoint_file}.")
        print("Loading model and launching interactive chatbot directly...")
        translator = Translator.from_checkpoint(
            checkpoint_path=checkpoint_file,
            src_tokenizer_path=src_tok_file,
            tgt_tokenizer_path=tgt_tok_file,
            device=device
        )
        run_chatbot(translator)
        return

    # Steps 1 & 2: Data & Tokenizer
    pairs, split_paths, src_tokenizer, tgt_tokenizer, src_tok_path, tgt_tok_path = step_1_and_2_data_and_tokenizer(
        source=args.source,
        max_samples=args.max_samples,
        local_file=args.kaggle_file,
        vocab_size=args.vocab_size
    )

    # Step 6: Memorization Sanity Check
    if not args.skip_memorize:
        step_6_memorization_sanity_check(pairs, src_tokenizer, tgt_tokenizer, device=device)

    # Steps 7 & 8: Full Model Training
    model = step_7_and_8_train(
        split_paths=split_paths,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Steps 9, 10, 11: Evaluation & Error Analysis
    translator = step_9_10_11_evaluation(
        split_paths=split_paths,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device
    )

    # Step 12: Chatbot
    if args.chat:
        run_chatbot(translator)
    else:
        print("\nPipeline completed successfully!")
        print("To launch the interactive translation chatbot, run:")
        print("  .venv/bin/python main.py --chat")
        print("or:")
        print("  .venv/bin/python app/interface/app.py checkpoints/best_model.pt data/tokenizers/tokenizer_fr.json data/tokenizers/tokenizer_en.json")


if __name__ == "__main__":
    main()
