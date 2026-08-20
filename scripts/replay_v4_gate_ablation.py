#!/usr/bin/env python3
"""门控消融重放（零 GPU）：双模型融合 vs 单预训练 eresv2 vs 单微调 eresv2。

输入为 szy_v4 全量运行 detail.json（每条样本含 6 维 gate_features 与下游
content/route）。门控只决定接受/拒绝；接受样本的下游内容不变，拒绝正样本
按空 content 计 CER，因此可在不重跑管线的情况下精确重放任意门控变体。

输出：
  A. 产线复现校验（融合分 @ -0.6469 应完全复现 detail 的 accepted）；
  B. 各变体在外部标定阈值下的 PR/RR/CER/公式分；
  C. 各变体（含 6 个单维）在 DatasetA 上的 oracle 上限（公式分/balanced 双口径）；
  D. 各单维特征的 SV  standalone 质量（EER）。

用法：
  python scripts/replay_v4_gate_ablation.py \
    --detail outputs/sweep_v4/ep6_detail.json \
    --fusion ../dataset_dasynth/out/gate_fusion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from midea_qiji.infer import levenshtein as lev  # noqa: E402
from midea_qiji.infer import normalize_transcript  # noqa: E402

FEATURES = ["pre_raw", "ft_raw", "pre_sep", "ft_sep", "ch_sim_pre", "ch_sim_ft"]
# 历史外部标定阈值（非 DatasetA 调出）
EXT_THRESHOLDS = {"dual": -0.6469, "pre_raw": 0.27, "ft_sep": 0.2662653475999832}


class Trial:
    __slots__ = ("is_pos", "content", "label", "feats", "fused")

    def __init__(self, r: dict):
        self.is_pos = r["sample_type"] == "positive"
        # 与官方口径一致：normalize 后再算编辑距离
        self.content = normalize_transcript(r.get("content", "") or "")
        self.label = normalize_transcript(r.get("label", "") or "")
        self.feats = {k: float(r["gate_features"][k]) for k in FEATURES}
        self.fused = float(r["sv_score"])


class TrialSet:
    """预计算向量化指标所需的每样本常量（lev 与阈值无关，只算一次）。"""

    def __init__(self, trials: list[Trial]):
        self.pos = np.array([t.is_pos for t in trials])
        self.nonempty = np.array([bool(t.content.strip()) for t in trials])
        self.err_accept = np.array(
            [lev(t.content, t.label) if t.is_pos else 0 for t in trials], dtype=np.int64)
        self.len_label = np.array([len(t.label) if t.is_pos else 0 for t in trials], dtype=np.int64)
        self.n_pos = int(self.pos.sum())
        self.n_neg = int((~self.pos).sum())
        self.total_chars = int(self.len_label.sum())

    def metrics(self, scores: np.ndarray, threshold: float) -> dict:
        accept = scores >= threshold
        pr = float((accept & self.pos & self.nonempty).sum() / max(self.n_pos, 1))
        rr = float((~accept & ~self.pos).sum() / max(self.n_neg, 1))
        errs = int(np.where(accept, self.err_accept, self.len_label)[self.pos].sum())
        cer = errs / max(self.total_chars, 1)
        return {
            "threshold": round(float(threshold), 4), "PR": round(pr, 4), "RR": round(rr, 4),
            "CER": round(cer, 4), "formula": round((1 - cer + rr) / 2, 4),
            "balanced": round((pr + rr) / 2, 4),
        }

    def sweep(self, scores: np.ndarray) -> tuple[dict, dict]:
        """全量唯一阈值精确扫描，返回 (公式分最优点, balanced 最优点)。"""
        best_f = best_b = None
        for t in np.unique(scores):
            m = self.metrics(scores, t)
            if best_f is None or m["formula"] > best_f["formula"]:
                best_f = m
            if best_b is None or m["balanced"] > best_b["balanced"]:
                best_b = m
        return best_f, best_b


def eer(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    alls = np.concatenate([pos_scores, neg_scores])
    best_gap, best_eer = 1.0, 1.0
    for t in np.unique(np.quantile(alls, np.linspace(0.001, 0.999, 599))):
        frr = float((pos_scores < t).mean())
        far = float((neg_scores >= t).mean())
        if abs(far - frr) < best_gap:
            best_gap, best_eer = abs(far - frr), (far + frr) / 2
    return round(best_eer, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, required=True)
    ap.add_argument("--fusion", type=Path, required=True)
    args = ap.parse_args()

    results = json.load(args.detail.open(encoding="utf-8"))["results"]
    trials = [Trial(r) for r in results]
    bundle = json.load(args.fusion.open(encoding="utf-8"))
    mean = np.asarray(bundle["mean"], dtype=np.float32)
    std = np.asarray(bundle["std"], dtype=np.float32)
    weight = np.asarray(bundle["weight"], dtype=np.float32)
    bias = float(bundle["bias"])

    X = np.array([[t.feats[k] for k in FEATURES] for t in trials], dtype=np.float32)
    fused = (X - mean) / std @ weight + bias

    ts = TrialSet(trials)

    # A. 产线复现校验
    detail_fused = np.array([t.fused for t in trials], dtype=np.float32)
    max_diff = float(np.abs(fused - detail_fused).max())
    prod = ts.metrics(fused, EXT_THRESHOLDS["dual"])
    n_pos = sum(t.is_pos for t in trials)
    n_neg = len(trials) - n_pos
    print(f"== A. 产线复现校验 ==  样本 {len(trials)}（pos {n_pos} / neg {n_neg}）")
    print(f"  重算融合分 vs detail sv_score 最大差: {max_diff:.2e}")
    print(f"  双模型融合 @ -0.6469: {prod}")

    variants = {"dual": fused}
    for k in FEATURES:
        variants[k] = X[:, FEATURES.index(k)]

    # B. 外部标定阈值
    print("\n== B. 外部标定阈值（非 DatasetA 调出）==")
    print(f"{'variant':12s} {'thr':>9s} {'PR':>7s} {'RR':>7s} {'CER':>7s} {'公式分':>7s}")
    for name, thr in EXT_THRESHOLDS.items():
        m = ts.metrics(variants[name], thr)
        print(f"{name:12s} {thr:9.4f} {m['PR']:7.4f} {m['RR']:7.4f} {m['CER']:7.4f} {m['formula']:7.4f}")

    # C. DatasetA oracle 上限
    print("\n== C. DatasetA oracle 上限（全量唯一阈值精确扫描，标注为 A 上调参）==")
    print(f"{'variant':12s} {'| 公式分最优':>26s} {'| balanced 最优':>26s} {'| EER':>7s}")
    for name in ["dual"] + FEATURES:
        s = variants[name]
        bf, bb = ts.sweep(s)
        e = eer(s[ts.pos], s[~ts.pos])
        print(f"{name:12s} | {bf['formula']:6.4f} @{bf['threshold']:8.4f} | "
              f"PR {bf['PR']:.4f} RR {bf['RR']:.4f} | {bb['balanced']:6.4f} @{bb['threshold']:8.4f} | {e:7.4f}")


if __name__ == "__main__":
    main()
