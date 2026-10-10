#!/usr/bin/env python3
"""分析 detail.json 里 N-best 重排序的覆盖与机会（离线诊断）。

回答三个问题：
  1. 当前 N-best 在正/负样本上的覆盖率各是多少；
  2. 未触发 N-best 的样本置信度分布如何（调阈值能救到谁、会误伤谁）；
  3. 已触发样本的换人原因（nbest_reason）长什么样。

用法：
    python scripts/experiments/analyze_nbest.py outputs/<NAME>/<NAME>_detail.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze N-best reranking opportunities")
    ap.add_argument("detail", help="管线产出的 detail.json 路径")
    args = ap.parse_args()

    with Path(args.detail).open(encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    tp = [r for r in results if r.get("sample_type") == "positive" and r.get("accepted")]
    fp = [r for r in results if r.get("sample_type") == "negative" and r.get("accepted")]

    print("=== N-best Analysis ===\n")

    # Current state
    nbest_applied_tp = sum(1 for r in tp if r.get("nbest_applied"))
    nbest_applied_fp = sum(1 for r in fp if r.get("nbest_applied"))
    print("Current N-best coverage:")
    print(f"  TP: {nbest_applied_tp}/{len(tp)} ({100*nbest_applied_tp/max(len(tp),1):.1f}%)")
    print(f"  FP: {nbest_applied_fp}/{len(fp)} ({100*nbest_applied_fp/max(len(fp),1):.1f}%)")
    both = nbest_applied_tp + nbest_applied_fp
    print(f"  FP rate with N-best: {100*nbest_applied_fp/both if both > 0 else 0:.1f}%")
    print()

    # Analyze why N-best wasn't applied
    print("Why N-best wasn't applied (for samples without it):")
    tp_no_nbest = [r for r in tp if not r.get("nbest_applied")]
    fp_no_nbest = [r for r in fp if not r.get("nbest_applied")]

    conf_tp = [float(r.get("raw_confidence", 1.0)) for r in tp_no_nbest]
    conf_fp = [float(r.get("raw_confidence", 1.0)) for r in fp_no_nbest]

    if conf_tp and conf_fp:
        print("\nConfidence distribution (samples without N-best):")
        print(f"  TP: min={min(conf_tp):.3f}, median={sorted(conf_tp)[len(conf_tp)//2]:.3f}, max={max(conf_tp):.3f}")
        print(f"  FP: min={min(conf_fp):.3f}, median={sorted(conf_fp)[len(conf_fp)//2]:.3f}, max={max(conf_fp):.3f}")

        # Check what threshold would capture low-confidence samples
        for thr in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            tp_captured = sum(1 for c in conf_tp if c < thr)
            fp_captured = sum(1 for c in conf_fp if c < thr)
            print(f"\n  If N-best threshold = {thr}:")
            print(f"    Would capture {tp_captured}/{len(conf_tp)} TP ({100*tp_captured/len(conf_tp):.1f}%)")
            print(f"    Would capture {fp_captured}/{len(conf_fp)} FP ({100*fp_captured/len(conf_fp):.1f}%)")

    # Check samples that already have N-best
    print("\n\nSamples WITH N-best applied:")
    tp_with_nbest = [r for r in tp if r.get("nbest_applied")]
    for r in tp_with_nbest[:5]:
        print(f"  ID: {r['id']}")
        print(f"    Confidence: {r.get('raw_confidence')}")
        print(f"    N-best reason: {r.get('nbest_reason')}")
        print(f"    Content: {r.get('content', '')[:50]}")
        print()


if __name__ == "__main__":
    main()
