"""
Multi-Head Attention mechanism with fused kernel acceleration.
Supports Self-Attention, Masked Self-Attention, and Cross-Attention.
Optimized for high throughput using PyTorch's native scaled_dot_product_attention.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention module.
    Utilizes PyTorch's fused scaled dot-product attention when available for 2-4x speedup.
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key:   (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask:  (batch_size, 1, seq_len_q, seq_len_k) boolean mask where True indicates masked positions

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attn_weights: Optional attention weights matrix
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # 1. Project and reshape to (batch_size, num_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Optimized Scaled Dot-Product Attention
        if hasattr(F, "scaled_dot_product_attention"):
            # In PyTorch's native kernel, boolean mask True means keep, False means ignore
            attn_mask = ~mask if (mask is not None and mask.dtype == torch.bool) else mask
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
            attn_weights = None
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(self.dropout(attn_weights), v)

        # 3. Concatenate heads and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)

        return output, attn_weights
