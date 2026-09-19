"""
Transformer Encoder and Encoder Layer modules.
Features Pre-LN architecture, GELU activations, and residual scaling for improved stability.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """Feed-Forward Network with GELU activation and dropout."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single layer of the Transformer Encoder with Pre-LN and residual scaling."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        res_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.res_scale = res_scale

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, mask=src_mask)
        x = x + self.dropout(attn_out) * self.res_scale

        # Pre-LN Feed-Forward
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out) * self.res_scale

        return x


class TransformerEncoder(nn.Module):
    """Stack of N Encoder layers with final LayerNorm."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        res_scale = 1.0 / math.sqrt(2.0 * num_layers)
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                res_scale=res_scale
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask=src_mask)
        return self.norm(src)
