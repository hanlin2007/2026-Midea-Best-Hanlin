"""szy_v4.metrics：分离质量指标（SI-SDR / SI-SNR 家族）。

历史上 SI-SDR 在 train_mossformer2 / diag_sisdr / sanity_check 三处各有一份
numpy 实现，口径略有出入；现统一收敛到此模块：

  - ``si_sdr``          numpy 诊断口径（dB）。实现与 train/sanity 原版一致
                        （噪声能量 < 1e-18 的完美重建上限取 60.0 dB，沿用
                        diag 的封顶意图）；
  - ``oracle_sisdr``    双通道 oracle 排列 SI-SDR（枚举 2! 排列取均值最大）；
  - ``si_snr``          torch 训练口径（B,T -> B）；
  - ``pit_sisnr_loss``  PIT 排列不变 SI-SNR 训练损失（最小排列负均值）。

依赖：numpy（torch 函数延迟到调用环境具备 torch 时使用）。
"""
from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR(dB)：est/ref 各自去均值，按 ref 投影分解目标/噪声。

    长度不一致时按较短侧截断；ref 能量过小返回 NaN；噪声能量 < 1e-18
    （数值完美重建）封顶 60.0 dB。
    """
    n = min(len(est), len(ref))
    est = est[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    est -= est.mean()
    ref -= ref.mean()
    d = float(ref @ ref)
    if d < 1e-12:
        return float("nan")
    t = float(est @ ref) / d * ref
    no = est - t
    noise_power = float(no @ no)
    if noise_power < 1e-18:
        return 60.0
    return 10.0 * np.log10(float(t @ t) / noise_power)


def oracle_sisdr(ests: list[np.ndarray], refs: list[np.ndarray]) -> float:
    """双通道 oracle 排列 SI-SDR：枚举全部排列，取两条均值最大者（dB）。"""
    best = None
    for p in permutations(range(2)):
        v = float(np.mean([si_sdr(ests[p[i]], refs[i]) for i in range(2)]))
        if best is None or v > best:
            best = v
    return best


def si_snr(est: Any, ref: Any) -> Any:
    """SI-SNR：est/ref 为 torch 张量，形状 (B, T) -> (B,)。"""
    import torch

    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    denom = (ref ** 2).sum(dim=-1).clamp_min(1e-12)
    scale = (est * ref).sum(dim=-1) / denom
    target = scale.unsqueeze(-1) * ref
    noise = est - target
    return 10.0 * torch.log10(
        (target ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1).clamp_min(1e-12) + 1e-12
    )


def pit_sisnr_loss(ests: list, refs: Any) -> Any:
    """PIT SI-SNR 损失：ests 为 2 个 (B,T) 估计，refs 为 (B,2,T) torch 张量。

    返回最小排列（即最高 SI-SNR）的负均值。
    """
    import torch

    perms = list(permutations(range(2)))
    vals = []
    for p in perms:
        v = si_snr(ests[p[0]], refs[:, 0]) + si_snr(ests[p[1]], refs[:, 1])
        vals.append(v)
    stacked = torch.stack(vals, dim=0)  # (P, B)
    best = stacked.max(dim=0).values
    return -(best.mean() / 2.0)
