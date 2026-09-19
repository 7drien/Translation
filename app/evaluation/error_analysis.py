"""
Translation error analysis and quality assessment.
Detects common neural machine translation failure modes:
- Repetitions / loops
- Omissions / truncations
- Additions / hallucinations
- Low character overlap
"""

from __future__ import annotations
import collections
from typing import List, Dict, Any
from .chrf import compute_chrf


def detect_repetitions(text: str, max_repeat: int = 3) -> bool:
    """Detect if the translation contains repeated n-grams / looping behavior."""
    words = text.lower().split()
    if len(words) < max_repeat:
        return False

    # Check 1-gram, 2-gram, 3-gram repetitions
    for n in (1, 2, 3):
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        counts = collections.Counter(ngrams)
        if any(c >= max_repeat for c in counts.values()):
            return True
    return False


def analyze_translation_errors(
    sources: List[str],
    candidates: List[str],
    references: List[str]
) -> Dict[str, Any]:
    """
    Perform qualitative and quantitative error analysis across predictions.

    Args:
        sources: Original source French sentences.
        candidates: Predicted English translations.
        references: Ground truth English references.

    Returns:
        Structured analysis report dictionary.
    """
    assert len(sources) == len(candidates) == len(references)
    total = len(sources)
    if total == 0:
        return {"total_samples": 0}

    chrf_scores = []
    repetitions_count = 0
    omissions_count = 0
    hallucinations_count = 0
    sample_analyses = []

    for src, cand, ref in zip(sources, candidates, references):
        score = compute_chrf(cand, ref)
        chrf_scores.append(score)

        has_rep = detect_repetitions(cand)
        if has_rep:
            repetitions_count += 1

        cand_len = len(cand.split())
        ref_len = len(ref.split())

        # Length mismatch checks
        is_omission = (cand_len <= ref_len // 2) and ref_len > 3
        is_hallucination = (cand_len >= ref_len * 2) and cand_len > 5

        if is_omission:
            omissions_count += 1
        if is_hallucination:
            hallucinations_count += 1

        sample_analyses.append({
            "source": src,
            "candidate": cand,
            "reference": ref,
            "chrf": score,
            "has_repetition": has_rep,
            "is_omission": is_omission,
            "is_hallucination": is_hallucination
        })

    # Sort samples by chrF score to identify best and worst cases
    sorted_samples = sorted(sample_analyses, key=lambda x: x["chrf"])
    worst_samples = sorted_samples[:5]
    best_samples = sorted_samples[-5:][::-1]

    avg_chrf = sum(chrf_scores) / total

    return {
        "total_samples": total,
        "average_chrf": round(avg_chrf * 100, 2),
        "repetition_rate": round(repetitions_count / total * 100, 2),
        "omission_rate": round(omissions_count / total * 100, 2),
        "hallucination_rate": round(hallucinations_count / total * 100, 2),
        "worst_samples": worst_samples,
        "best_samples": best_samples
    }
