"""
Seq2Seq Transformer architecture built from scratch.
Combines custom Encoder, Decoder, Attention, Embeddings, and Weight Tying.
Optimized for translation performance and training stability.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .embeddings import TokenEmbedding, PositionalEncoding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .masks import create_masks


class Transformer(nn.Module):
    """
    High-Performance Encoder-Decoder Transformer for Neural Machine Translation.
    Features Pre-LN, GELU activations, residual scaling, and target weight tying.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
        tie_weights: bool = True
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.tie_weights = tie_weights

        # Embeddings & Positional Encodings
        self.src_tok_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_embed = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Encoder & Decoder Stacks (Pre-LN with GELU)
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # Output projection to target vocabulary
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Weight tying: share target embedding weights with generator projection head
        # Dramatically reduces parameter count and improves target vocabulary representations
        if tie_weights:
            self.generator.weight = self.tgt_tok_embed.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters using Xavier uniform and normal distributions."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize embeddings with normal distribution std = 0.02
        nn.init.normal_(self.src_tok_embed.embedding.weight, mean=0.0, std=0.02)
        if not self.tie_weights:
            nn.init.normal_(self.tgt_tok_embed.embedding.weight, mean=0.0, std=0.02)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source token sequences into contextual memory representations.

        Args:
            src: (batch_size, src_len)
            src_mask: (batch_size, 1, 1, src_len)

        Returns:
            memory: (batch_size, src_len, d_model)
        """
        src_emb = self.pos_encoder(self.src_tok_embed(src))
        return self.encoder(src_emb, src_mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target token sequences using encoder memory.

        Args:
            tgt: (batch_size, tgt_len)
            memory: (batch_size, src_len, d_model)
            tgt_mask: (batch_size, 1, tgt_len, tgt_len)
            memory_mask: (batch_size, 1, 1, src_len)

        Returns:
            output: (batch_size, tgt_len, d_model)
        """
        tgt_emb = self.pos_encoder(self.tgt_tok_embed(tgt))
        return self.decoder(tgt_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass during training.

        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)

        Returns:
            logits: (batch_size, tgt_len, tgt_vocab_size)
        """
        if src_mask is None or tgt_mask is None:
            s_mask, t_mask, m_mask = create_masks(src, tgt, pad_idx=self.pad_idx)
            src_mask = s_mask if src_mask is None else src_mask
            tgt_mask = t_mask if tgt_mask is None else tgt_mask
            memory_mask = m_mask
        else:
            memory_mask = src_mask

        memory = self.encode(src, src_mask=src_mask)
        dec_out = self.decode(tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.generator(dec_out)
        return logits

    def get_config(self) -> Dict[str, Any]:
        """Return model hyperparameter configuration."""
        return {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout_rate,
            "max_len": self.max_len,
            "pad_idx": self.pad_idx,
            "tie_weights": self.tie_weights
        }
