"""
Attention mask utilities for Transformer models.
Provides padding masks and look-ahead (causal) masks.
"""

from __future__ import annotations
import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a boolean mask where True indicates padding tokens.

    Args:
        seq: Tensor of shape (batch_size, seq_len)
        pad_idx: The integer index for <PAD>

    Returns:
        mask: Tensor of shape (batch_size, 1, 1, seq_len) with True for PAD positions.
    """
    # (batch_size, 1, 1, seq_len)
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Create an upper-triangular boolean mask for causal / autoregressive decoding.
    Positions where mask is True should NOT be attended to.

    Args:
        size: Sequence length
        device: Torch device

    Returns:
        mask: Tensor of shape (1, 1, size, size)
    """
    # triu with diagonal=1 masks all positions strictly above the main diagonal
    mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(1)


def create_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate all masks needed for training:
    - src_mask: Encoder self-attention mask (batch, 1, 1, src_len)
    - tgt_mask: Decoder self-attention causal + padding mask (batch, 1, tgt_len, tgt_len)
    - memory_mask: Decoder cross-attention mask for encoder memory (batch, 1, 1, src_len)

    Returns:
        (src_mask, tgt_mask, memory_mask)
    """
    src_mask = create_padding_mask(src, pad_idx)

    tgt_len = tgt.size(1)
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)  # (batch, 1, 1, tgt_len)
    tgt_causal_mask = create_causal_mask(tgt_len, tgt.device)  # (1, 1, tgt_len, tgt_len)

    # Broadcasted logical OR: masked if padding OR in the future
    tgt_mask = tgt_pad_mask | tgt_causal_mask

    # Cross attention mask ignores padding in source sequence
    memory_mask = src_mask

    return src_mask, tgt_mask, memory_mask
