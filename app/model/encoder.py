"""
Transformer Encoder and Encoder Layer modules built from scratch.
Supports both Pre-LN and Post-LN architectures.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """Feed-Forward Network applied to each position separately and identically."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single layer of the Transformer Encoder."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            # Pre-LN
            norm_x = self.norm1(x)
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, mask=src_mask)
            x = x + self.dropout(attn_out)

            norm_x = self.norm2(x)
            ff_out = self.feed_forward(norm_x)
            x = x + self.dropout(ff_out)
        else:
            # Post-LN
            attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))

        return x


class TransformerEncoder(nn.Module):
    """Stack of N Encoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                norm_first=norm_first
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask=src_mask)
        return self.norm(src)
