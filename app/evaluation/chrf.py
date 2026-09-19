"""
chrF (character n-gram F-score) evaluation metric.
A robust character-level metric for machine translation quality.
"""

from __future__ import annotations
import collections
from typing import List, Sequence


def get_char_ngrams(text: str, n: int) -> collections.Counter:
    """Extract character n-grams from text, including whitespace."""
    ngrams = collections.Counter()
    for i in range(len(text) - n + 1):
        ngrams[text[i:i + n]] += 1
    return ngrams


def compute_chrf(
    candidate: str,
    reference: str,
    min_n: int = 1,
    max_n: int = 6,
    beta: float = 2.0
) -> float:
    """
    Compute sentence-level chrF score between candidate and reference.

    Args:
        candidate: Predicted translation string.
        reference: Ground truth reference string.
        min_n: Minimum character n-gram order (default 1).
        max_n: Maximum character n-gram order (default 6).
        beta: Beta parameter for F-score (beta=2 gives more weight to recall).

    Returns:
        chrF score between 0.0 and 1.0.
    """
    precisions = []
    recalls = []

    for n in range(min_n, max_n + 1):
        cand_ngrams = get_char_ngrams(candidate, n)
        ref_ngrams = get_char_ngrams(reference, n)

        total_cand = sum(cand_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        if total_cand == 0 or total_ref == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        matches = 0
        for ngram, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))

        precisions.append(matches / total_cand)
        recalls.append(matches / total_ref)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    if avg_precision + avg_recall == 0:
        return 0.0

    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * (avg_precision * avg_recall) / (beta_sq * avg_precision + avg_recall)
    return f_score


def corpus_chrf(
    candidates: List[str],
    references: List[str],
    min_n: int = 1,
    max_n: int = 6,
    beta: float = 2.0
) -> float:
    """Compute average chrF score over a corpus."""
    if not candidates or not references:
        return 0.0
    scores = [
        compute_chrf(cand, ref, min_n=min_n, max_n=max_n, beta=beta)
        for cand, ref in zip(candidates, references)
    ]
    return sum(scores) / len(scores)
