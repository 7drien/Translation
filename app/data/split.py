"""
Dataset splitting for train, validation, and test sets.
Ensures reproducible splits with random seed.
"""

from __future__ import annotations
import os
import random
from typing import List, Tuple, Dict


def split_data(
    pairs: List[Tuple[str, str]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Split a list of pairs into train, validation, and test subsets.
    """
    assert abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_data = shuffled[:n_train]
    valid_data = shuffled[n_train:n_train + n_valid]
    test_data = shuffled[n_train + n_valid:]

    return {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }


def split_and_save(
    pairs: List[Tuple[str, str]],
    output_dir: str = "data",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Tuple[str, str]]:
    """
    Split pairs and write files:
      train.fr, train.en
      valid.fr, valid.en
      test.fr, test.en
    """
    os.makedirs(output_dir, exist_ok=True)
    splits = split_data(pairs, train_ratio, valid_ratio, test_ratio, seed)
    paths = {}

    for split_name, split_pairs in splits.items():
        src_path = os.path.join(output_dir, f"{split_name}.fr")
        tgt_path = os.path.join(output_dir, f"{split_name}.en")

        with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
            for s, t in split_pairs:
                fs.write(s.strip() + "\n")
                ft.write(t.strip() + "\n")

        paths[split_name] = (src_path, tgt_path)

    return paths
