"""
Multi-Head Attention mechanism implemented from scratch.
Supports Self-Attention, Masked Self-Attention, and Cross-Attention.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module from 'Attention Is All You Need'.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, Value, and Output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key:   (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask:  (batch_size, 1, seq_len_q, seq_len_k) or broadcastable boolean mask where True = masked

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # 1. Project and reshape to (batch_size, num_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # context: (batch_size, num_heads, seq_len_q, d_k)
        context = torch.matmul(attn_weights_dropped, v)

        # 3. Concatenate heads and project output
        # (batch_size, seq_len_q, num_heads * d_k) -> (batch_size, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)

        return output, attn_weights
