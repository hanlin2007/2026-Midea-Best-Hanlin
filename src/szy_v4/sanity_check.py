#!/usr/bin/env python3
"""Sanity 对照：torch 版 MossFormer2（整波推理）vs 产线 ONNX 缓存 wav。

在 dasynth valid 上以 oracle 排列 SI-SDR 对比两者，验证 torch 权重可用且
定位短板子集（reverb / masked）。历史结论：torch 预训练全体 0.42 vs ONNX
-3.94（60 条），两者 reverb 均崩塌（<-12dB）→ 引出 dasynth 微调。

用法（conda env: yuyin）：
  CUDA_VISIBLE_DEVICES=1 python -m szy_v4.sanity_check --n 60 \
    --ckpt ../exp_sep/clearvoice_ckpt/MossFormer2_SS_16K/last_best_checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(_REPO_ROOT / "third_party"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from szy_v4.metrics import oracle_sisdr  # noqa: E402
from szy_v4.separator import load_mossformer2_torch  # noqa: E402

DEFAULT_ROOT = _REPO_ROOT.parent / "dataset_dasynth"
DEFAULT_CKPT = (_REPO_ROOT.parent / "exp_sep" / "clearvoice_ckpt"
                / "MossFormer2_SS_16K" / "last_best_checkpoint.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    root = args.data_root

    model, device = load_mossformer2_torch(args.ckpt, "cuda")
    print(f"loaded {args.ckpt}", flush=True)

    metas = {}
    for line in (root / "metadata" / "valid.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        metas[d["id"]] = d

    rows = []
    for uid in sorted(metas)[: args.n]:
        m = metas[uid]
        mix, _ = sf.read(root / m["mixed_audio"], dtype="float32")
        refs = [sf.read(root / m["source_1_audio"], dtype="float32")[0],
                sf.read(root / m["source_2_audio"], dtype="float32")[0]]
        with torch.no_grad():
            out = model(torch.from_numpy(mix).to(device)[None, :])
        ests = [o.squeeze(0).float().cpu().numpy() for o in out]
        ow = []
        for ch in ("mos0", "mos1"):
            f = root / "valid" / ch / f"{uid}.wav"
            ow.append(sf.read(f, dtype="float32")[0] if f.is_file() else np.zeros(8))
        rows.append({
            "reverb": any(m["corruption"].get("reverb", [False, False])),
            "masked": m["corruption"].get("masked", False),
            "clean": m["corruption"].get("clean", False),
            "torch": oracle_sisdr(ests, refs),
            "onnx": oracle_sisdr(ow, refs),
        })

    def st(sel, name, key):
        v = [r[key] for r in sel if not np.isnan(r[key])]
        if v:
            print(f"{name:12s} {key:5s} n={len(v):3d} mean={np.mean(v):6.2f} "
                  f"med={np.median(v):6.2f} p10={np.percentile(v,10):6.2f}")

    for key in ("torch", "onnx"):
        print(f"== oracle-permutation SI-SDR: {key} ==")
        st(rows, "ALL", key)
        st([r for r in rows if r["clean"]], "clean", key)
        st([r for r in rows if r["reverb"]], "reverb", key)
        st([r for r in rows if not r["reverb"]], "no-reverb", key)
        st([r for r in rows if r["masked"]], "masked", key)


if __name__ == "__main__":
    main()
