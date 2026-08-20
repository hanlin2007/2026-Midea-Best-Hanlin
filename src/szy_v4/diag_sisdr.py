#!/usr/bin/env python3
"""诊断：分离器在 dasynth valid 上的 SI-SDR 水位（按腐蚀类型分组）。

对每条样本计算 SI-SDR(dB)：mix 基线（混合 vs 各干净源，取较好值）与
MossFormer2 两路输出 vs 干净源（用 metadata 的 SV 通道归属；归属 none 跳过）。
本脚本读已分离好的 mos0/mos1 wav，不跑模型；用于量化"分离是否起反作用"。

用法（conda env: yuyin）：
  python -m szy_v4.diag_sisdr --data-root ../dataset_dasynth --n 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from szy_v4.metrics import si_sdr

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _REPO_ROOT.parent / "dataset_dasynth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", type=str, default="valid")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    root = args.data_root

    metas = {}
    for line in (root / "metadata" / f"{args.split}.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        metas[d["id"]] = d
    assigns = {}
    for line in (root / "metadata" / f"{args.split}_mossformer2.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        assigns[d["id"]] = d

    rows = []
    ids = sorted(metas)[: args.n]
    for uid in ids:
        m = metas[uid]
        a = assigns.get(uid)
        if a is None:
            continue
        mix, _ = sf.read(root / m["mixed_audio"], dtype="float32")
        refs = {}
        for k in ("s1", "s2"):
            refs[k], _ = sf.read(root / m[f"source_{k[-1]}_audio"], dtype="float32")
        rec = {
            "id": uid,
            "overlap": m["actual_overlap_ratio"],
            "masked": m["corruption"].get("masked", False),
            "reverb": any(m["corruption"].get("reverb", [False, False])),
            "clean": m["corruption"].get("clean", False),
        }
        base = []
        for k in ("s1", "s2"):
            v = si_sdr(mix, refs[k])
            if not np.isnan(v):
                base.append(v)
        rec["mix_sisdr"] = max(base) if base else float("nan")
        for ch in ("mos0", "mos1"):
            ref_key = a.get(f"{ch}_reference", "none")
            if ref_key not in ("s1", "s2"):
                rec[ch] = float("nan")
                continue
            wav, _ = sf.read(root / a[f"{ch}_audio"], dtype="float32")
            rec[ch] = si_sdr(wav, refs[ref_key])
        rows.append(rec)

    def stats(sel, name):
        vals0 = [r["mos0"] for r in sel if not np.isnan(r.get("mos0", float("nan")))]
        vals1 = [r["mos1"] for r in sel if not np.isnan(r.get("mos1", float("nan")))]
        base = [r["mix_sisdr"] for r in sel if not np.isnan(r.get("mix_sisdr", float("nan")))]
        allv = vals0 + vals1
        if not allv:
            return
        print(
            f"{name:28s} n_ch={len(allv):4d}  mix={np.mean(base):6.2f}  "
            f"mos_mean={np.mean(allv):6.2f}  mos_med={np.median(allv):6.2f}  "
            f"p10={np.percentile(allv,10):6.2f}  imp={np.mean(allv)-np.mean(base):+5.2f}"
        )

    print(f"== {len(rows)} samples ({root}, {args.split}) ==")
    stats(rows, "ALL")
    stats([r for r in rows if r["clean"]], "clean")
    stats([r for r in rows if r["masked"]], "masked (SIR -12~-4)")
    stats([r for r in rows if not r["masked"]], "unmasked (SIR -5~5)")
    stats([r for r in rows if r["reverb"]], "reverb")
    stats([r for r in rows if not r["reverb"]], "no-reverb")
    stats([r for r in rows if r["overlap"] >= 0.7], "overlap>=0.7")
    stats([r for r in rows if 0.3 <= r["overlap"] < 0.7], "overlap 0.3~0.7")
    stats([r for r in rows if r["overlap"] < 0.3], "overlap<0.3")


if __name__ == "__main__":
    main()
