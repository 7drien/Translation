"""
Transformer Decoder and Decoder Layer modules.
Features Pre-LN architecture, GELU activations, and residual scaling for stable convergence.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .encoder import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Single layer of the Transformer Decoder with Pre-LN and residual scaling."""

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
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.res_scale = res_scale

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN Masked Self-Attention
        norm_x = self.norm1(x)
        self_attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, mask=tgt_mask)
        x = x + self.dropout(self_attn_out) * self.res_scale

        # Pre-LN Cross-Attention over encoder memory
        norm_x = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(norm_x, memory, memory, mask=memory_mask)
        x = x + self.dropout(cross_attn_out) * self.res_scale

        # Pre-LN Positionwise Feed-Forward
        norm_x = self.norm3(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out) * self.res_scale

        return x


class TransformerDecoder(nn.Module):
    """Stack of N Decoder layers with final LayerNorm."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        res_scale = 1.0 / math.sqrt(3.0 * num_layers)
        self.layers = nn.ModuleList([
            DecoderLayer(
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
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(tgt)
