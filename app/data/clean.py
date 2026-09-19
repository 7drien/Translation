"""
Data cleaning and preprocessing for parallel translation corpora.
Removes empty lines, trims whitespace, filters based on sequence length and length ratio.
"""

from __future__ import annotations
import unicodedata
import re
from typing import List, Tuple, Optional


def normalize_string(s: str) -> str:
    """Normalize unicode characters and normalize consecutive whitespaces."""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r" +", " ", s)
    return s


def clean_parallel_pairs(
    pairs: List[Tuple[str, str]],
    min_length: int = 1,
    max_length: int = 100,
    max_ratio: float = 2.5
) -> List[Tuple[str, str]]:
    """
    Clean and filter parallel sentence pairs.

    Args:
        pairs: List of (source_sentence, target_sentence).
        min_length: Minimum number of words required.
        max_length: Maximum number of words allowed.
        max_ratio: Maximum ratio between source and target length to filter misalignments.

    Returns:
        Filtered and normalized list of pairs.
    """
    cleaned: List[Tuple[str, str]] = []
    seen = set()

    for src, tgt in pairs:
        src = normalize_string(src)
        tgt = normalize_string(tgt)

        if not src or not tgt:
            continue

        src_words = src.split()
        tgt_words = tgt.split()

        src_len = len(src_words)
        tgt_len = len(tgt_words)

        if src_len < min_length or tgt_len < min_length:
            continue
        if src_len > max_length or tgt_len > max_length:
            continue

        # Check length ratio to discard poorly aligned pairs
        ratio = max(src_len, tgt_len) / max(min(src_len, tgt_len), 1)
        if ratio > max_ratio:
            continue

        # Deduplication
        pair_key = (src.lower(), tgt.lower())
        if pair_key in seen:
            continue
        seen.add(pair_key)

        cleaned.append((src, tgt))

    return cleaned


def clean_files(
    src_path: str,
    tgt_path: str,
    out_src_path: str,
    out_tgt_path: str,
    min_length: int = 1,
    max_length: int = 100,
    max_ratio: float = 2.5
) -> int:
    """Clean parallel corpus files and save the filtered versions."""
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        pairs = [(s.strip(), t.strip()) for s, t in zip(fs, ft)]

    cleaned = clean_parallel_pairs(pairs, min_length=min_length, max_length=max_length, max_ratio=max_ratio)

    with open(out_src_path, "w", encoding="utf-8") as fs, open(out_tgt_path, "w", encoding="utf-8") as ft:
        for s, t in cleaned:
            fs.write(s + "\n")
            ft.write(t + "\n")

    return len(cleaned)
