"""FireRed N-best hypothesis reranking with conservative MBR.

When FireRedASR2-AED generates multiple hypotheses (beam_size > 1, nbest > 1),
this module provides a conservative reranking strategy that only replaces the
baseline (beam=1) result when an alternative clearly dominates under MBR risk.
"""
from __future__ import annotations

import math
from typing import Any


def normalized_edit_distance(hyp: str, ref: str) -> float:
    """Normalized edit distance between two transcripts."""
    from infer import levenshtein
    if not hyp and not ref:
        return 0.0
    return levenshtein(hyp, ref) / max(len(hyp), len(ref), 1)


def conservative_mbr_rerank(
    baseline: dict[str, Any],
    nbest: list[dict[str, Any]],
    min_confidence_gain: float = 0.03,
    min_mbr_risk_reduction: float = 0.05,
) -> dict[str, Any]:
    """Rerank N-best hypotheses using conservative MBR.

    Only replaces baseline when an alternative has:
    1. Higher confidence by at least min_confidence_gain
    2. Lower MBR risk by at least min_mbr_risk_reduction

    Args:
        baseline: {"text": str, "confidence": float}
        nbest: List of {"text": str, "confidence": float}
        min_confidence_gain: Minimum confidence improvement to switch
        min_mbr_risk_reduction: Minimum MBR risk reduction to switch

    Returns:
        Selected hypothesis with reranking metadata
    """
    candidates = [baseline] + nbest
    
    # Compute MBR risk for each candidate
    mbr_risks = []
    for i, cand_i in enumerate(candidates):
        risk = 0.0
        for j, cand_j in enumerate(candidates):
            if i != j:
                # Weight by confidence of the other candidate
                weight = cand_j.get("confidence", 0.0)
                dist = normalized_edit_distance(
                    cand_i.get("text", ""),
                    cand_j.get("text", "")
                )
                risk += weight * dist
        mbr_risks.append(risk)
    
    # Find candidate with minimum MBR risk
    best_idx = min(range(len(mbr_risks)), key=lambda i: mbr_risks[i])
    best_candidate = candidates[best_idx]
    best_risk = mbr_risks[best_idx]
    baseline_risk = mbr_risks[0]
    
    # Decision logic
    if best_idx == 0:
        # Baseline has minimum risk, keep it
        return {
            **baseline,
            "nbest_applied": False,
            "nbest_reason": "baseline_has_minimum_risk",
            "nbest_candidates": candidates,
            "nbest_risks": mbr_risks,
        }
    
    # Check if improvement is significant enough
    baseline_conf = baseline.get("confidence", 0.0)
    best_conf = best_candidate.get("confidence", 0.0)
    conf_gain = best_conf - baseline_conf
    risk_reduction = baseline_risk - best_risk
    
    if conf_gain < min_confidence_gain:
        return {
            **baseline,
            "nbest_applied": False,
            "nbest_reason": f"insufficient_confidence_gain ({conf_gain:.4f} < {min_confidence_gain})",
            "nbest_candidates": candidates,
            "nbest_risks": mbr_risks,
        }
    
    if risk_reduction < min_mbr_risk_reduction:
        return {
            **baseline,
            "nbest_applied": False,
            "nbest_reason": f"insufficient_risk_reduction ({risk_reduction:.4f} < {min_mbr_risk_reduction})",
            "nbest_candidates": candidates,
            "nbest_risks": mbr_risks,
        }
    
    # Switch to best candidate
    return {
        **best_candidate,
        "nbest_applied": True,
        "nbest_reason": f"switched (conf_gain={conf_gain:.4f}, risk_reduction={risk_reduction:.4f})",
        "nbest_candidates": candidates,
        "nbest_risks": mbr_risks,
        "nbest_baseline": baseline,
    }
