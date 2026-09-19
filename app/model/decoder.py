"""
Transformer Decoder and Decoder Layer modules built from scratch.
Includes Masked Self-Attention and Cross-Attention with Encoder memory.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .encoder import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Single layer of the Transformer Decoder."""

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
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            # Pre-LN
            norm_x = self.norm1(x)
            self_attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, mask=tgt_mask)
            x = x + self.dropout(self_attn_out)

            norm_x = self.norm2(x)
            cross_attn_out, _ = self.cross_attn(norm_x, memory, memory, mask=memory_mask)
            x = x + self.dropout(cross_attn_out)

            norm_x = self.norm3(x)
            ff_out = self.feed_forward(norm_x)
            x = x + self.dropout(ff_out)
        else:
            # Post-LN
            self_attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
            x = self.norm1(x + self.dropout(self_attn_out))

            cross_attn_out, _ = self.cross_attn(x, memory, memory, mask=memory_mask)
            x = self.norm2(x + self.dropout(cross_attn_out))

            ff_out = self.feed_forward(x)
            x = self.norm3(x + self.dropout(ff_out))

        return x


class TransformerDecoder(nn.Module):
    """Stack of N Decoder layers."""

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
            DecoderLayer(
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
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(tgt)
