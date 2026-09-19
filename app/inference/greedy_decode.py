"""
Autoregressive Greedy Decoding for Seq2Seq Transformer.
Generates translation token-by-token using argmax.
"""

from __future__ import annotations
import torch
from typing import List

from ..model.transformer import Transformer
from ..model.masks import create_padding_mask, create_causal_mask
from ..tokenizer.vocabulary import BOS_IDX, EOS_IDX, PAD_IDX


def greedy_decode(
    model: Transformer,
    src_tokens: List[int],
    max_len: int = 60,
    bos_idx: int = BOS_IDX,
    eos_idx: int = EOS_IDX,
    pad_idx: int = PAD_IDX,
    device: torch.device = torch.device("cpu")
) -> List[int]:
    """
    Autoregressive greedy decoding: chooses the most likely token at each step.

    Args:
        model: Trained Transformer instance.
        src_tokens: List of source token IDs (including BOS/EOS).
        max_len: Maximum number of tokens to generate.
        bos_idx: ID of <BOS>.
        eos_idx: ID of <EOS>.
        pad_idx: ID of <PAD>.
        device: PyTorch device.

    Returns:
        List of generated target token IDs.
    """
    model.eval()

    with torch.no_grad():
        src = torch.tensor([src_tokens], dtype=torch.long, device=device)
        src_mask = create_padding_mask(src, pad_idx=pad_idx)

        # 1. Encode source once
        memory = model.encode(src, src_mask=src_mask)

        # 2. Iteratively decode target tokens
        generated = [bos_idx]

        for _ in range(max_len):
            tgt = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_mask = create_causal_mask(tgt.size(1), device=device)

            dec_out = model.decode(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )

            # Predict next token from the last position
            logits = model.generator(dec_out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).item()

            generated.append(next_token)

            if next_token == eos_idx:
                break

    return generated
