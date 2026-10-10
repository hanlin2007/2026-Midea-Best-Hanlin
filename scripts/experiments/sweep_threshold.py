#!/usr/bin/env python3
"""离线重扫融合门控阈值：直接读 detail.json，用每条的 sv_score 重算 PR/RR/CER/Score。

推理很贵、过程数据便宜：detail.json 里存了每条样本的门控分与实际转写，
换个阈值重算指标只是几秒的纯 CPU 计算——这是"事后重扫阈值"的标准做法，
不需要重新跑任何模型。

用法：
    python scripts/experiments/sweep_threshold.py outputs/<NAME>/<NAME>_detail.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def compute_metrics(results: list[dict], threshold: float) -> dict:
    """Compute PR/RR/CER for a given threshold."""
    pos_total = 0
    pos_accepted = 0
    pos_errors = 0
    pos_chars = 0

    neg_total = 0
    neg_rejected = 0

    for item in results:
        is_pos = item.get("sample_type") == "positive"
        label = item.get("label")
        content = item.get("content", "")
        sv_score_str = item.get("sv_score")

        if sv_score_str is None or sv_score_str == "":
            continue

        sv_score = float(sv_score_str)
        accepted = sv_score >= threshold

        if is_pos:
            pos_total += 1
            if label is None or label == "":
                continue

            if accepted and content:
                pos_accepted += 1
                pos_errors += levenshtein(content, str(label))
                pos_chars += len(str(label))
            else:
                # Rejected or empty -> treat as all wrong
                pos_chars += len(str(label))
                pos_errors += len(str(label))
        else:
            neg_total += 1
            if not accepted or not content:
                neg_rejected += 1

    pr = pos_accepted / pos_total if pos_total > 0 else 0.0
    rr = neg_rejected / neg_total if neg_total > 0 else 0.0
    cer = pos_errors / pos_chars if pos_chars > 0 else 1.0

    return {
        "threshold": threshold,
        "PR": pr,
        "RR": rr,
        "CER": cer,
        "score": (1 - cer + rr) / 2,
        "pos_total": pos_total,
        "pos_accepted": pos_accepted,
        "neg_total": neg_total,
        "neg_rejected": neg_rejected,
    }


def main():
    ap = argparse.ArgumentParser(description="Sweep fusion gate thresholds on a detail JSON")
    ap.add_argument("detail", help="管线产出的 detail.json 路径")
    args = ap.parse_args()

    detail_path = Path(args.detail)
    data = json.loads(detail_path.read_text(encoding="utf-8"))
    results = data.get("results", [])

    print(f"Loaded {len(results)} samples")
    print(f"Positive: {sum(1 for r in results if r.get('sample_type') == 'positive')}")
    print(f"Negative: {sum(1 for r in results if r.get('sample_type') == 'negative')}")
    print()

    # Current threshold
    current_thr = float(results[0].get("sv_accept_threshold", "-0.6469"))
    print(f"Current threshold: {current_thr:.6f}")
    print()

    # Sweep thresholds
    thresholds = [round(-1.0 + 0.05 * i, 2) for i in range(41)]  # -1.0 to 1.0
    thresholds.extend([round(-0.8 + 0.01 * i, 2) for i in range(50)])  # -0.8 to -0.3 finer
    thresholds = sorted(set(thresholds))

    print(f"Sweeping {len(thresholds)} thresholds from {min(thresholds):.2f} to {max(thresholds):.2f}")
    print()

    # Compute metrics for all thresholds
    all_metrics = []
    for thr in thresholds:
        metrics = compute_metrics(results, thr)
        all_metrics.append(metrics)

    # Sort by score descending
    all_metrics.sort(key=lambda m: m["score"], reverse=True)

    # Print top 20
    print("Top 20 thresholds:")
    print(f"{'Thr':>8s} | {'PR':>7s} | {'RR':>7s} | {'CER':>7s} | {'Score':>7s} | {'PosAcc':>7s} | {'NegRej':>7s}")
    print("-" * 75)
    for m in all_metrics[:20]:
        marker = " ← CURRENT" if abs(m["threshold"] - current_thr) < 0.001 else ""
        print(f"{m['threshold']:8.3f} | {m['PR']:7.4f} | {m['RR']:7.4f} | {m['CER']:7.4f} | {m['score']:7.4f} | {m['pos_accepted']:7d} | {m['neg_rejected']:7d}{marker}")

    print()
    best = all_metrics[0]
    print(f"Best threshold: {best['threshold']:.3f}")
    print(f"  PR:    {best['PR']:.4f}")
    print(f"  RR:    {best['RR']:.4f}")
    print(f"  CER:   {best['CER']:.4f}")
    print(f"  Score: {best['score']:.4f}")
    print(f"  Positive accepted: {best['pos_accepted']}/{best['pos_total']}")
    print(f"  Negative rejected: {best['neg_rejected']}/{best['neg_total']}")

    # Compare with current
    current_metrics = next((m for m in all_metrics if abs(m["threshold"] - current_thr) < 0.001), None)
    if current_metrics:
        print()
        print(f"Current threshold ({current_thr:.3f}):")
        print(f"  PR:    {current_metrics['PR']:.4f}")
        print(f"  RR:    {current_metrics['RR']:.4f}")
        print(f"  CER:   {current_metrics['CER']:.4f}")
        print(f"  Score: {current_metrics['score']:.4f}")
        print(f"  Positive accepted: {current_metrics['pos_accepted']}/{current_metrics['pos_total']}")
        print(f"  Negative rejected: {current_metrics['neg_rejected']}/{current_metrics['neg_total']}")

        print()
        print(f"Improvement: {best['score'] - current_metrics['score']:+.4f}")


if __name__ == "__main__":
    main()
