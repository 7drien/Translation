"""
Dataset acquisition and download for French-English Translation.
Downloads real datasets from the internet (Tatoeba / Kaggle French-English corpus)
and provides support for local Kaggle CSV/TSV files.
"""

from __future__ import annotations
import os
import io
import zipfile
import urllib.request
from typing import List, Tuple, Optional

# Public URL for the Tatoeba French-English dataset (the exact same dataset published on Kaggle as
# 'devansodariya/english-french-translations' and 'dhruvildave/french-english-bilingual-pairs')
KAGGLE_TATOEBA_URL = "https://www.manythings.org/anki/fra-eng.zip"


def download_internet_dataset(
    output_dir: str = "data",
    max_samples: Optional[int] = 100000,
    url: str = ""
) -> Tuple[str, str]:
    """
    Download a high-quality literary French-English dataset using HuggingFace Datasets.
    Uses 'Helsinki-NLP/opus-100' which contains 1M high-quality sentence pairs.

    Args:
        output_dir: Destination directory.
        max_samples: Maximum number of sentence pairs to extract.
        url: Unused (kept for compatibility).

    Returns:
        Tuple of (src_path, tgt_path) for French and English text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    src_path = os.path.join(output_dir, "corpus.fr")
    tgt_path = os.path.join(output_dir, "corpus.en")

    print(f"Downloading high-quality literary dataset (OPUS-100 en-fr) via HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: The 'datasets' library is required. Please install it using: pip install datasets")
        raise

    # opus-100 has very high quality translated data
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train")
    
    pairs: List[Tuple[str, str]] = []
    
    print(f"✓ Dataset downloaded. Extracting sentence pairs...")
    
    for row in dataset:
        translation = row["translation"]
        en_text = translation.get("en", "").strip()
        fr_text = translation.get("fr", "").strip()
        
        if fr_text and en_text:
            pairs.append((fr_text, en_text))

    # Shuffle the dataset to ensure a diverse distribution
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    # Take the required number of samples
    if max_samples is not None:
        pairs = pairs[:max_samples]

    print(f"✓ {len(pairs)} sentence pairs extracted and shuffled from OPUS-100.")

    with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
        for fr, en in pairs:
            # Replace newlines with spaces to avoid breaking the text file format
            fr = fr.replace('\n', ' ').replace('\r', '')
            en = en.replace('\n', ' ').replace('\r', '')
            fs.write(fr + "\n")
            ft.write(en + "\n")

    return src_path, tgt_path


def load_kaggle_csv(
    csv_path: str,
    output_dir: str = "data",
    src_col: str = "French",
    tgt_col: str = "English",
    max_samples: Optional[int] = None
) -> Tuple[str, str]:
    """
    Load a dataset downloaded from Kaggle (CSV or TSV format).
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    src_path = os.path.join(output_dir, "corpus.fr")
    tgt_path = os.path.join(output_dir, "corpus.en")

    pairs: List[Tuple[str, str]] = []
    delimiter = "\t" if csv_path.endswith(".tsv") else ","

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            fr = row.get(src_col) or row.get("fr") or row.get("french") or row.get("French")
            en = row.get(tgt_col) or row.get("en") or row.get("english") or row.get("English")
            if fr and en:
                pairs.append((fr.strip(), en.strip()))
                if max_samples is not None and len(pairs) >= max_samples:
                    break

    with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
        for fr, en in pairs:
            fs.write(fr + "\n")
            ft.write(en + "\n")

    return src_path, tgt_path


def prepare_dataset(
    output_dir: str = "data",
    source: str = "internet",
    max_samples: Optional[int] = 10000,
    local_file: Optional[str] = None
) -> Tuple[str, str]:
    """
    Prepare the dataset either by downloading it from the web or loading a local file.
    """
    if source == "local" and local_file and os.path.exists(local_file):
        return load_kaggle_csv(local_file, output_dir=output_dir, max_samples=max_samples)
    else:
        return download_internet_dataset(output_dir=output_dir, max_samples=max_samples)
