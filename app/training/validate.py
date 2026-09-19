"""
Validation loop for Transformer training.
Evaluates cross-entropy loss on validation/test datasets.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..model.transformer import Transformer
from ..model.masks import create_masks
from ..tokenizer.vocabulary import PAD_IDX


def evaluate_loss(
    model: Transformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = PAD_IDX
) -> float:
    """
    Evaluate average loss on a dataloader with teacher forcing.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing inputs and targets
            decoder_input = tgt[:, :-1]
            expected_output = tgt[:, 1:]

            src_mask, tgt_mask, _ = create_masks(src, decoder_input, pad_idx=pad_idx)

            logits = model(src, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)

            # Reshape for cross entropy
            logits = logits.reshape(-1, logits.size(-1))
            targets = expected_output.reshape(-1)

            loss = criterion(logits, targets)

            # Count non-pad tokens for weighted average
            non_pad_tokens = (targets != pad_idx).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    model.train()
    return total_loss / max(total_tokens, 1)
