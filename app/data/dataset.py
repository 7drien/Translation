"""
PyTorch Dataset and dynamic collator for Translation.
Pads sequences dynamically to the batch maximum length.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from ..tokenizer.tokenizer import BPETokenizer
from ..tokenizer.vocabulary import PAD_IDX


class TranslationDataset(Dataset):
    """Dataset of parallel sentences encoded into token ID sequences."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        max_length: int = 128
    ) -> None:
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

        self.encoded_pairs: List[Tuple[List[int], List[int]]] = []
        for src_text, tgt_text in self.pairs:
            src_ids = self.src_tokenizer.encode(src_text, add_bos=True, add_eos=True)
            tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_bos=True, add_eos=True)

            # Truncate if exceeding max_length
            if len(src_ids) > self.max_length:
                src_ids = src_ids[:self.max_length - 1] + [src_ids[-1]]
            if len(tgt_ids) > self.max_length:
                tgt_ids = tgt_ids[:self.max_length - 1] + [tgt_ids[-1]]

            self.encoded_pairs.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.encoded_pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.encoded_pairs[idx]


def collate_translation_batch(
    batch: List[Tuple[List[int], List[int]]],
    pad_idx: int = PAD_IDX
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to dynamically pad source and target sequences to the maximum length in the batch.

    Args:
        batch: List of (src_ids, tgt_ids).
        pad_idx: Padding token index.

    Returns:
        src_padded: Tensor of shape (batch_size, max_src_len)
        tgt_padded: Tensor of shape (batch_size, max_tgt_len)
    """
    src_seqs, tgt_seqs = zip(*batch)

    max_src_len = max(len(s) for s in src_seqs)
    max_tgt_len = max(len(t) for t in tgt_seqs)

    src_padded = []
    for s in src_seqs:
        padded = s + [pad_idx] * (max_src_len - len(s))
        src_padded.append(padded)

    tgt_padded = []
    for t in tgt_seqs:
        padded = t + [pad_idx] * (max_tgt_len - len(t))
        tgt_padded.append(padded)

    return (
        torch.tensor(src_padded, dtype=torch.long),
        torch.tensor(tgt_padded, dtype=torch.long)
    )


def create_dataloader(
    pairs: List[Tuple[str, str]],
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = 128,
    pad_idx: int = PAD_IDX
) -> DataLoader:
    """Convenience helper to create a DataLoader from sentence pairs."""
    dataset = TranslationDataset(
        pairs=pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_translation_batch(b, pad_idx=pad_idx)
    )
