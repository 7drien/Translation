"""
High-Performance Beam Search Decoding for Seq2Seq Transformer.
Features length normalization (Wu et al.) and n-gram repetition blocking.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Set

from ..model.transformer import Transformer
from ..model.masks import create_padding_mask, create_causal_mask
from ..tokenizer.vocabulary import BOS_IDX, EOS_IDX, PAD_IDX


class BeamHypothesis:
    """Represents a single candidate hypothesis in the beam."""

    def __init__(self, tokens: List[int], log_prob: float = 0.0) -> None:
        self.tokens = tokens
        self.log_prob = log_prob

    def score(self, alpha: float = 0.7) -> float:
        """Compute length-penalized score using Wu et al. (2016) formula."""
        lp = ((5.0 + len(self.tokens)) ** alpha) / ((5.0 + 1.0) ** alpha)
        return self.log_prob / lp


def has_repeating_ngram(tokens: List[int], n: int = 3) -> bool:
    """Check if the latest token creates a duplicate n-gram in the sequence."""
    if len(tokens) < 2 * n:
        return False
    latest_ngram = tuple(tokens[-n:])
    for i in range(len(tokens) - n):
        if tuple(tokens[i:i + n]) == latest_ngram:
            return True
    return False


def beam_search_decode(
    model: Transformer,
    src_tokens: List[int],
    beam_size: int = 5,
    max_len: int = 60,
    bos_idx: int = BOS_IDX,
    eos_idx: int = EOS_IDX,
    pad_idx: int = PAD_IDX,
    alpha: float = 0.7,
    no_repeat_ngram_size: int = 3,
    device: torch.device = torch.device("cpu")
) -> List[int]:
    """
    Beam Search decoding algorithm with repetition blocking and length normalization.

    Args:
        model: Trained Transformer model.
        src_tokens: Source token IDs.
        beam_size: Number of hypotheses to maintain in the beam (default: 5).
        max_len: Maximum target sequence length.
        bos_idx: BOS token ID.
        eos_idx: EOS token ID.
        pad_idx: PAD token ID.
        alpha: Length penalty exponent (default: 0.7).
        no_repeat_ngram_size: Order of n-grams forbidden from repeating (default: 3).
        device: Torch compute device.

    Returns:
        List of generated target token IDs for the best hypothesis.
    """
    model.eval()

    with torch.no_grad():
        src = torch.tensor([src_tokens], dtype=torch.long, device=device)
        src_mask = create_padding_mask(src, pad_idx=pad_idx)

        # 1. Encode source once
        memory = model.encode(src, src_mask=src_mask)

        # 2. Initialize beam with starting token <BOS>
        active_hypotheses: List[BeamHypothesis] = [BeamHypothesis([bos_idx], 0.0)]
        completed_hypotheses: List[BeamHypothesis] = []

        for _ in range(max_len):
            all_candidates: List[BeamHypothesis] = []

            for hyp in active_hypotheses:
                tgt = torch.tensor([hyp.tokens], dtype=torch.long, device=device)
                tgt_mask = create_causal_mask(tgt.size(1), device=device)

                dec_out = model.decode(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_mask=src_mask
                )

                logits = model.generator(dec_out[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                # Consider top (beam_size * 2) candidates to allow for repetition blocking
                top_probs, top_indices = torch.topk(log_probs, k=min(beam_size * 2, log_probs.size(-1)))

                for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                    candidate_tokens = hyp.tokens + [idx]

                    # Repetition blocking: skip candidate if it generates a loop
                    if no_repeat_ngram_size > 0 and has_repeating_ngram(candidate_tokens, no_repeat_ngram_size):
                        continue

                    candidate = BeamHypothesis(
                        tokens=candidate_tokens,
                        log_prob=hyp.log_prob + prob
                    )
                    all_candidates.append(candidate)

            # Sort all candidates across all hypotheses by length-penalized score
            all_candidates.sort(key=lambda h: h.score(alpha=alpha), reverse=True)

            # Select top-K candidates
            active_hypotheses = []
            for cand in all_candidates:
                if cand.tokens[-1] == eos_idx:
                    completed_hypotheses.append(cand)
                    if len(completed_hypotheses) >= beam_size:
                        break
                else:
                    active_hypotheses.append(cand)
                    if len(active_hypotheses) >= beam_size:
                        break

            if len(completed_hypotheses) >= beam_size or not active_hypotheses:
                break

        # Select the best hypothesis
        candidates = completed_hypotheses if completed_hypotheses else active_hypotheses
        if not candidates:
            return [bos_idx, eos_idx]

        best_hyp = max(candidates, key=lambda h: h.score(alpha=alpha))
        return best_hyp.tokens
