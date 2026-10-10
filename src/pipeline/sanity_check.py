#!/usr/bin/env python3
"""Sanity 对照：torch 版 MossFormer2（整波推理）vs 产线 ONNX 缓存 wav。

在 dasynth valid 上以 oracle 排列 SI-SDR 对比两者，验证 torch 权重可用且
定位短板子集（reverb / masked）。历史结论：torch 预训练全体 0.42 vs ONNX
-3.94（60 条），两者 reverb 均崩塌（<-12dB）→ 引出 dasynth 微调。

"sanity check"（健全性检查）的定位：换实现（ONNX → torch）时，第一件
要做的事就是证明"新实现和旧实现一样好、甚至更好"。同一批输入、同一
套指标、两个实现并列打分——分数一致/更好才敢替换，这是工程上换实现
的标准流程。本脚本正是那次迁移的验证工具，如今保留为 torch 权重更新
后的回归检查入口。

用法（conda env: yuyin）：
  CUDA_VISIBLE_DEVICES=1 python -m pipeline.sanity_check --n 60 \
    --ckpt ../exp_sep/clearvoice_ckpt/MossFormer2_SS_16K/last_best_checkpoint.pt
"""
from __future__ import annotations

import argparse                                  # 命令行参数
import json                                      # dasynth 元数据
from pathlib import Path                         # 路径

import numpy as np                               # 统计
import soundfile as sf                           # 读 wav（含 ONNX 缓存）
import torch                                     # torch 推理

_REPO_ROOT = Path(__file__).resolve().parents[2]
# 与 train.py 同理：先注入 third_party/src 再 import szy。
# 本脚本是独立开发工具、本来就必须有 torch，所以不需要 pipeline 那套
# 延迟导入——顶层 import 即可（延迟导入是为"无 torch 场景"服务的）
import sys
sys.path.insert(0, str(_REPO_ROOT / "third_party"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pipeline.metrics import oracle_sisdr  # noqa: E402
from pipeline.separator import load_mossformer2  # noqa: E402

DEFAULT_ROOT = _REPO_ROOT.parent / "dataset_dasynth"
DEFAULT_CKPT = (_REPO_ROOT.parent / "exp_sep" / "clearvoice_ckpt"
                / "MossFormer2_SS_16K" / "last_best_checkpoint.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)  # 待验证的 torch 权重
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    root = args.data_root

    # 加载权重；load_mossformer2 返回 (模型, 实际设备)——请求 cuda、
    # 无 GPU 自动落 cpu，变量 device 以返回值为准
    model, device = load_mossformer2(args.ckpt, "cuda")
    print(f"loaded {args.ckpt}", flush=True)

    # 读 valid 元数据（同 diag.py：id → 样本信息）
    metas = {}
    for line in (root / "metadata" / "valid.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        metas[d["id"]] = d

    rows = []
    # sorted 保证"前 n 条"确定可复现（与历史 60 条口径一致）
    for uid in sorted(metas)[: args.n]:
        m = metas[uid]
        mix, _ = sf.read(root / m["mixed_audio"], dtype="float32")
        refs = [sf.read(root / m["source_1_audio"], dtype="float32")[0],
                sf.read(root / m["source_2_audio"], dtype="float32")[0]]
        # ---- torch 侧：当场整波推理 ----
        # no_grad：纯推理不建图；[None, :] 加 batch 维（同 separator.separate）
        with torch.no_grad():
            out = model(torch.from_numpy(mix).to(device)[None, :])
        # 两路输出转 numpy（squeeze 去 batch 维、float 回 float32、cpu 回主机）
        ests = [o.squeeze(0).float().cpu().numpy() for o in out]
        # ---- ONNX 侧：读历史缓存 wav ----
        # 缓存命名约定：root/valid/{mos0,mos1}/{uid}.wav（历史 ONNX 分离
        # 流程的产物目录结构）
        ow = []
        for ch in ("mos0", "mos1"):
            f = root / "valid" / ch / f"{uid}.wav"
            # 缓存缺失时用 8 点静音占位（避免单条缺失中断整个对照批次；
            # 实际数据里这些缓存应齐全）
            ow.append(sf.read(f, dtype="float32")[0] if f.is_file() else np.zeros(8))
        rows.append({
            # 腐蚀标签（与 diag/train 的 evaluate 完全同一口径）
            "reverb": any(m["corruption"].get("reverb", [False, False])),
            "masked": m["corruption"].get("masked", False),
            "clean": m["corruption"].get("clean", False),
            # 两侧都用 oracle 排列 SI-SDR：公平比较（都不因通道顺序扣分）
            "torch": oracle_sisdr(ests, refs),
            "onnx": oracle_sisdr(ow, refs),
        })

    def st(sel, name, key):
        """打印某子集在某个实现（torch/onnx）下的统计行。"""
        v = [r[key] for r in sel if not np.isnan(r[key])]
        if v:
            print(f"{name:12s} {key:5s} n={len(v):3d} mean={np.mean(v):6.2f} "
                  f"med={np.median(v):6.2f} p10={np.percentile(v,10):6.2f}")

    # 同一组样本、同一套分组，两个实现各报一遍——直接并排对比
    for key in ("torch", "onnx"):
        print(f"== oracle-permutation SI-SDR: {key} ==")
        st(rows, "ALL", key)
        st([r for r in rows if r["clean"]], "clean", key)
        st([r for r in rows if r["reverb"]], "reverb", key)
        st([r for r in rows if not r["reverb"]], "no-reverb", key)
        st([r for r in rows if r["masked"]], "masked", key)


if __name__ == "__main__":
    main()
