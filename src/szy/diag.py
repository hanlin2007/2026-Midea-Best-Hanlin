#!/usr/bin/env python3
"""诊断：分离器在 dasynth valid 上的 SI-SDR 水位（按腐蚀类型分组）。

对每条样本计算 SI-SDR(dB)：mix 基线（混合 vs 各干净源，取较好值）与
MossFormer2 两路输出 vs 干净源（用 metadata 的 SV 通道归属；归属 none 跳过）。
本脚本读已分离好的 mos0/mos1 wav，不跑模型；用于量化"分离是否起反作用"。

"是否起反作用"的量化方法：先算"混合信号本身 vs 干净源"的 SI-SDR 基线，
再算"分离通道 vs 干净源"的 SI-SDR；两者之差（报表里的 imp 列）为正说明
分离确实把目标提纯了，为负说明分离还不如不分离——负值出现在哪类腐蚀
分组里，微调就该重点补哪类数据。

本脚本是纯离线分析：只读 wav 与 metadata，不需要 torch/GPU/模型，
可以在任何装了 numpy+soundfile 的机器上跑——把"分析"与"推理"解耦，
是实验系统里诊断工具的标准形态。

用法（conda env: yuyin）：
  python -m szy.diag_sisdr --data-root ../dataset_dasynth --n 300
（注意：当前模块名为 szy.diag，实际命令应为 python -m szy.diag——
docstring 保留历史命名，若运行报错请以模块名 szy.diag 为准。）
"""
from __future__ import annotations

import argparse                                  # 命令行参数
import json                                      # dasynth 元数据
from pathlib import Path                         # 路径

import numpy as np                               # 统计（均值/中位数/分位数）
import soundfile as sf                           # 读 wav

from szy.metrics import si_sdr                   # 单通道 SI-SDR（numpy 诊断口径）

# 同 train.py：从本文件位置推导仓库根，默认数据目录在仓库上级的
# dataset_dasynth（跨机器路径不同时用 --data-root 覆盖）
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _REPO_ROOT.parent / "dataset_dasynth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", type=str, default="valid")   # 数据切分（valid/train）
    ap.add_argument("--n", type=int, default=300)           # 只诊断前 n 条（sorted 后，确定性顺序）
    args = ap.parse_args()
    root = args.data_root

    # 两类元数据的区别：
    #   {split}.jsonl            样本本体：混合/双源路径、腐蚀标注、重叠率；
    #   {split}_mossformer2.jsonl 分离产物清单：mos0/mos1 wav 路径，
    #                           以及"每路输出对应哪个干净源"的归属
    #                           （SV 声纹判定的，历史流程产出）
    metas = {}
    for line in (root / "metadata" / f"{args.split}.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        metas[d["id"]] = d
    assigns = {}
    for line in (root / "metadata" / f"{args.split}_mossformer2.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        assigns[d["id"]] = d

    rows = []
    # sorted 保证"前 n 条"每次都是同样那 n 条（dict 键序稳定，诊断可复现）
    ids = sorted(metas)[: args.n]
    for uid in ids:
        m = metas[uid]
        a = assigns.get(uid)
        if a is None:
            # 没有分离产物清单的样本跳过（可能分离阶段没跑它）
            continue
        mix, _ = sf.read(root / m["mixed_audio"], dtype="float32")
        # 读两路干净源。k[-1] 取 "s1" 的最后一个字符 "1"，即
        # source_1_audio / source_2_audio——利用命名规律的简洁写法
        refs = {}
        for k in ("s1", "s2"):
            refs[k], _ = sf.read(root / m[f"source_{k[-1]}_audio"], dtype="float32")
        rec = {
            "id": uid,
            "overlap": m["actual_overlap_ratio"],   # 两说话人的实际时间重叠率
            "masked": m["corruption"].get("masked", False),
            "reverb": any(m["corruption"].get("reverb", [False, False])),
            "clean": m["corruption"].get("clean", False),
        }
        # 基线：混合 vs 两个干净源，取较好值（max）。为什么取 max 而不是
        # 指定某一源：混合里"占优的那个说话人"贡献更大，SI-SDR 天然更高；
        # 取 max 表示"混合里最清晰的那路"的质量，作为分离的公平参照
        base = []
        for k in ("s1", "s2"):
            v = si_sdr(mix, refs[k])
            if not np.isnan(v):                 # 静音参考等不可评样本过滤
                base.append(v)
        rec["mix_sisdr"] = max(base) if base else float("nan")
        # 分离两路各自对着"归属的干净源"打分：归属由 metadata 给定
        # （SV 判定的对应关系），此处不做 oracle 排列
        for ch in ("mos0", "mos1"):
            ref_key = a.get(f"{ch}_reference", "none")
            if ref_key not in ("s1", "s2"):
                # 归属未知（"none"）：跳过该通道，记 NaN 表示不可评
                rec[ch] = float("nan")
                continue
            wav, _ = sf.read(root / a[f"{ch}_audio"], dtype="float32")
            rec[ch] = si_sdr(wav, refs[ref_key])
        rows.append(rec)

    def stats(sel, name):
        """对一个样本子集打印统计行：均值/中位数/10 分位数/相对基线提升。

        闭包 + 列表推导：过滤 NaN 后分别对 mos0/mos1 两路合起来统计
        （allv），"n_ch" 是通道数（不是样本数，每样本最多 2 通道）。
        """
        vals0 = [r["mos0"] for r in sel if not np.isnan(r.get("mos0", float("nan")))]
        vals1 = [r["mos1"] for r in sel if not np.isnan(r.get("mos1", float("nan")))]
        base = [r["mix_sisdr"] for r in sel if not np.isnan(r.get("mix_sisdr", float("nan")))]
        allv = vals0 + vals1
        if not allv:
            return                            # 该分组无可评通道：跳过不打印
        # 报表列含义：mix=基线均值，mos_mean/med=分离通道的均值/中位数，
        # p10=第 10 百分位数（最差 10% 的水位，观察尾部崩塌），
        # imp=分离相对基线的平均提升（+5.2f 带符号显示正负）
        print(
            f"{name:28s} n_ch={len(allv):4d}  mix={np.mean(base):6.2f}  "
            f"mos_mean={np.mean(allv):6.2f}  mos_med={np.median(allv):6.2f}  "
            f"p10={np.percentile(allv,10):6.2f}  imp={np.mean(allv)-np.mean(base):+5.2f}"
        )

    print(f"== {len(rows)} samples ({root}, {args.split}) ==")
    # 全量 + 按腐蚀类型/重叠率分组。分组的意义：ALL 可能被容易的子集
    # 平均掉问题，逐组看才知道"模型在哪类输入上崩"（历史结论就是
    # reverb 组崩塌才引出 dasynth 微调）
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
