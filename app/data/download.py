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
    max_samples: Optional[int] = 10000,
    url: str = KAGGLE_TATOEBA_URL
) -> Tuple[str, str]:
    """
    Download the real French-English parallel dataset from the internet.

    Args:
        output_dir: Destination directory.
        max_samples: Maximum number of sentence pairs to extract (None for all ~240,000).
        url: Direct download link.

    Returns:
        Tuple of (src_path, tgt_path) for French and English text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    src_path = os.path.join(output_dir, "corpus.fr")
    tgt_path = os.path.join(output_dir, "corpus.en")

    print(f"Téléchargement du dataset depuis {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=45) as resp:
        content = resp.read()

    print(f"✓ Archive téléchargée ({len(content) / (1024 * 1024):.2f} Mo). Extraction des phrases...")

    pairs: List[Tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open("fra.txt") as f:
            for line in f:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue
                parts = decoded.split("\t")
                if len(parts) >= 2:
                    en_text = parts[0].strip()
                    fr_text = parts[1].strip()
                    if fr_text and en_text:
                        pairs.append((fr_text, en_text))
                        if max_samples is not None and len(pairs) >= max_samples:
                            break

    print(f"✓ {len(pairs)} paires de phrases extraites.")

    with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
        for fr, en in pairs:
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
