"""pipeline.metrics：分离质量指标（SI-SDR / SI-SNR 家族）。

历史上 SI-SDR 在训练 / 诊断 / 对照验证三处各有一份 numpy 实现，口径略有
出入；现已统一收敛到此模块：

  - ``si_sdr``          numpy 诊断口径（dB）。噪声能量 < 1e-18 的完美重建
                        上限取 60.0 dB（封顶）；
  - ``oracle_sisdr``    双通道 oracle 排列 SI-SDR（枚举 2! 排列取均值最大）；
  - ``si_snr``          torch 训练口径（B,T -> B）；
  - ``pit_sisnr_loss``  PIT 排列不变 SI-SNR 训练损失（最小排列负均值）。

先讲清"分离质量"为什么难量化：分离器输出两路，我们不知道哪路是说话人
A、哪路是说话人 B（盲分离）。所以要用 SI-SDR 一族指标：把每路估计与
"最像的那路干净源"对齐后再算误差，再以"更好的那种对齐方式"作为该模型
的成绩——这就是 oracle 排列 / PIT 的含义（oracle 用于评估，PIT 用于训练）。

SI-SDR 的直觉：把估计信号 est 投影到参考信号 ref 所在的方向上，投影是
"est 里属于 ref 的成分"（目标），剩余部分是"不属于 ref 的成分"（噪声），
两者能量比取对数即得分，单位 dB，越大越好。0 dB 表示目标与噪声能量
相等，20 dB 已经相当干净。

依赖：numpy（torch 函数在调用时按需 import，保证本模块可被轻量环境使用）。
（延迟 import 的原因：metrics 被 diag/sanity_check 之外还会被无 torch 的
环境（如单测、文档工具）导入；把 import torch 放进函数体，import 本模块
本身零代价。）
"""
from __future__ import annotations

from itertools import permutations                 # 全排列：oracle/PIT 枚举通道对应关系
from typing import Any                              # torch 张量的类型占位（避免顶层依赖 torch）

import numpy as np


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR(dB)：est/ref 各自去均值，按 ref 投影分解目标/噪声。

    长度不一致时按较短侧截断；ref 能量过小返回 NaN；噪声能量 < 1e-18
    （数值完美重建）封顶 60.0 dB。

    推导（向量投影）：去均值后，est 在 ref 方向上的投影系数是
    (est·ref)/(ref·ref)，投影向量 t = 系数 * ref 就是"目标"，残余
    no = est - t 就是"噪声"。SI-SDR = 10*log10(||t||² / ||no||²)。
    """
    # 两段信号长度可能不同（缓存 wav 与源 wav 采样点差几个），按较短侧截断对齐
    n = min(len(est), len(ref))
    # float64 计算：SI-SDR 是能量比，减掉大均值后剩余量级很小，
    # float32 的 7 位有效数字会吃掉精度，float64（约 16 位）才稳
    est = est[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    # 各自去均值（直流分量）：直流不携带说话人信息，却会算进能量污染比值
    est -= est.mean()
    ref -= ref.mean()
    # d = ref·ref（ref 的能量）。@ 是 numpy 矩阵乘运算符，一维数组 @ 一维
    # 数组就是点积（逐元素相乘再求和），一行顶一个 for 循环（向量化）
    d = float(ref @ ref)
    if d < 1e-12:
        # ref 是（近乎）静音：投影方向没有意义，任何估计都"相对静音"
        # 打分失真，返回 NaN 表示"此样本不可评"，由调用方过滤
        return float("nan")
    # 投影系数 * ref = 目标分量 t（numpy 广播：标量乘向量=逐元素乘）
    t = float(est @ ref) / d * ref
    # 噪声 = 估计 - 目标（向量逐元素相减）
    no = est - t
    noise_power = float(no @ no)
    if noise_power < 1e-18:
        # 噪声能量几乎为 0 = 数值上的完美重建。真实信号做不到 0 噪声，
        # 这通常来自补零/占位等数值巧合；若不封顶，log10 会算出 +inf，
        # 一条 +inf 就能把整组均值统计带飞，所以封顶为 60.0 dB（足够好
        # 但有限的分数，与历史口径一致）
        return 60.0
    # dB = 10 * log10(目标能量 / 噪声能量)
    return 10.0 * np.log10(float(t @ t) / noise_power)


def oracle_sisdr(ests: list[np.ndarray], refs: list[np.ndarray]) -> float:
    """双通道 oracle 排列 SI-SDR：枚举全部排列，取两条均值最大者（dB）。

    "oracle" 指知道干净源对齐关系时能拿到的理论上限——分离出的两条通道
    顺序是任意的（谁对应说话人 1 谁对应说话人 2 未知），所以枚举排列。

    为什么"知道对齐关系"反而是评估指标：分离器的任务只是把两个人分开，
    不负责"按固定顺序输出"；按固定顺序打分，模型可能因为顺序反了被
    冤枉扣分。枚举排列取最优，测的是模型"真实能做到的最好情况"。
    """
    best = None
    # permutations(range(2)) 生成 [(0,1), (1,0)]：两种"估计↔参考"对应关系
    for p in permutations(range(2)):
        # 该排列下两条通道的 SI-SDR 均值；si_sdr 返回 NaN 的样本会把均值
        # 也变 NaN，从而自然跳过该排列（NaN 与任何数比较都为 False）
        v = float(np.mean([si_sdr(ests[p[i]], refs[i]) for i in range(2)]))
        if best is None or v > best:
            best = v
    return best


def si_snr(est: Any, ref: Any) -> Any:
    """SI-SNR：est/ref 为 torch 张量，形状 (B, T) -> (B,)。

    numpy 版 si_sdr 的 torch 兄弟：同样"去均值→投影→分解"三步，只是
    全部用张量广播写成批量版（B 条样本一次算完），供训练损失使用。
    二者差值极小，工程上视为同一指标（numpy 版多一个 60 dB 封顶）。
    """
    import torch                                # 延迟导入：见模块 docstring

    # 去均值。dim=-1 指最后一个轴（时间轴 T）；keepdim=True 保留维度，
    # 使 (B,T) 减 (B,1) 可以广播——否则形状对不上会报错
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    # ref 能量；clamp_min(1e-12) 把过小的值抬到 1e-12 再当除数，
    # 防止"静音参考"造成除零/除出 inf（与 numpy 版 d < 1e-12 → NaN 对应，
    # 但训练损失必须处处可导，所以选择钳制而不是返回 NaN）
    denom = (ref ** 2).sum(dim=-1).clamp_min(1e-12)
    # 投影系数 scale 形状 (B,)；sum(dim=-1) 沿时间轴求和 = 点积
    scale = (est * ref).sum(dim=-1) / denom
    # unsqueeze(-1) 把 (B,) 变 (B,1)，再与 (B,T) 的 ref 广播相乘得到目标分量
    target = scale.unsqueeze(-1) * ref
    noise = est - target
    # dB 分数；分子分母各加 1e-12 是双重保险（除数为 0 时 log10 也得有限值）
    return 10.0 * torch.log10(
        (target ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1).clamp_min(1e-12) + 1e-12
    )


def pit_sisnr_loss(ests: list, refs: Any) -> Any:
    """PIT SI-SNR 损失：ests 为 2 个 (B,T) 估计，refs 为 (B,2,T) torch 张量。

    返回最小排列（即最高 SI-SNR）的负均值。训练时通道顺序未知，PIT 每次
    都选"更合理"的对应关系，避免模型被通道顺序惩罚。

    PIT = Permutation Invariant Training（排列不变训练）。设想模型每次都把
    A 输出到通道 0，但标注把 A 放在源 1：按固定对应算损失会巨大，逼模型
    学会"按数据集标注顺序输出"这种无意义技能。PIT 每次从所有排列里挑
    损失最小（=分数最高）的那个来更新梯度，模型只需"把两个人分开"，
    不需要知道标注顺序。
    """
    import torch                                # 延迟导入：见模块 docstring

    # 两种排列 [(0,1), (1,0)]，对应"est0↔ref0 & est1↔ref1"或交叉对应
    perms = list(permutations(range(2)))
    vals = []
    for p in perms:
        # refs[:, 0]：取第 0 个源的所有 B 条样本（高级索引，形状 (B,T)）。
        # 该排列下两条通道的 SI-SNR 相加（和越大越好）
        v = si_snr(ests[p[0]], refs[:, 0]) + si_snr(ests[p[1]], refs[:, 1])
        vals.append(v)
    # 把两个 (B,) 张量沿新维度堆成 (P, B)；dim=0 表示"排列"轴
    stacked = torch.stack(vals, dim=0)  # (P, B)
    # 沿排列轴取每批样本的最大值：best 就是 oracle 分数（可达到的最好值）
    best = stacked.max(dim=0).values
    # 损失 = 负均值 / 2：除以 2 因为两条通道的 SI-SNR 相加了（等价于取
    # 两通道平均分），负号把"最大化分数"变成"最小化损失"（梯度下降惯例）
    return -(best.mean() / 2.0)
