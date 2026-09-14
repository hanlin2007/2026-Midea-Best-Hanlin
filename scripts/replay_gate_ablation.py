#!/usr/bin/env python3
"""门控消融重放（零 GPU）：双模型融合 vs 单预训练 eresv2 vs 单微调 eresv2。

输入为 SZY 全量运行 detail.json（每条样本含 6 维 gate_features 与下游
content/route）。门控只决定接受/拒绝；接受样本的下游内容不变，拒绝正样本
按空 content 计 CER，因此可在不重跑管线的情况下精确重放任意门控变体。

为什么"零 GPU"：detail.json 里已经落盘了每样本的 6 维门控特征与融合分
sv_score——重放门控决策只需要这些数字加一个阈值比较，全是 numpy 向量
运算。这正是重放手法的极致形态：模型都不用加载，任何门控变体（单维
特征、不同阈值）都只是"换一列数、换个阈值再算一遍"，几秒钟出全套
指标。消融（ablation）= 逐个拿掉部件看贡献，这里是拿掉融合、只剩单维
特征，观察每个特征单独能扛多少决策能力。

输出：
  A. 产线复现校验（融合分 @ -0.6469 应完全复现 detail 的 accepted）；
  B. 各变体在外部标定阈值下的 PR/RR/CER/公式分；
  C. 各变体（含 6 个单维）在 DatasetA 上的 oracle 上限（公式分/balanced 双口径）；
  D. 各单维特征的 SV  standalone 质量（EER）。
（A 校验重放器本身对不对；B 看"外部标定"的泛化成绩；C 看同集调参的
理论上限——标了"DatasetA 上调参"以诚实区分二者；D 用 EER 衡量单维特征
本身的区分能力。）

用法：
  python scripts/replay_gate_ablation.py \
    --detail outputs/sweep/ep6_detail.json \
    --fusion ../dataset_dasynth/out/gate_fusion.json
说明：CPU 即可运行，无需 GPU/模型。

依赖：midea_qiji.text（levenshtein/normalize_transcript，靠下方 sys.path
注入导入）、numpy、标准库 argparse/json/sys/pathlib。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---- 手动扩展模块搜索路径（必须在 import 仓库包之前）----
# 脚本在 scripts/ 里，不在包内。直接执行时 Python 的模块搜索路径只有
# scripts/ 与标准库，import midea_qiji 会失败。解决办法：在 import 前
# 把 src 目录插进 sys.path。Path(__file__).resolve().parents[1] =
# scripts/ 上一级 = 仓库根；insert(0, ...) 插到最前给最高优先级。
# 必须放在 import 之前：import 语句执行时才去 sys.path 里找包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# noqa: E402：告诉 linter 这个"不在文件顶部的 import"是有意为之。
from midea_qiji.text import levenshtein as lev  # noqa: E402
from midea_qiji.text import normalize_transcript  # noqa: E402

# 门控特征的 6 个维度（列顺序与 detail 的 gate_features、融合包权重对齐）：
# pre_* = 预训练基础模型的余弦；ft_* = 微调模型的余弦；
# raw = 唤醒 vs 原始指令；sep = 唤醒 vs 分离通道（取大）；ch_sim = 两通道间。
FEATURES = ["pre_raw", "ft_raw", "pre_sep", "ft_sep", "ch_sim_pre", "ch_sim_ft"]
# 历史外部标定阈值（非 DatasetA 调出）
# ——这些数字是历史实验产物（在 DatasetA 之外的数据上标定），
# 本脚本原样引用、不许改动：改了就失去"外部标定"的意义，B 节会失真。
EXT_THRESHOLDS = {"dual": -0.6469, "pre_raw": 0.27, "ft_sep": 0.2662653475999832}


class Trial:
    # __slots__：告诉 Python"这个类的实例只有这几个属性，不要给我建
    # 实例字典（__dict__）"。收益有二：
    # 1) 内存：普通对象每个实例背后挂一个 dict（几十到上百字节），本脚本
    #    处理数千条样本，省下的是可观的常驻内存；__slots__ 让属性存储
    #    变成固定槽位布局，无 dict 开销；
    # 2) 纪律：不允许再往实例上塞未声明的属性——拼错属性名会立刻抛
    #    AttributeError，而不是像普通对象那样静默创建一个新属性
    #    （后者是极难排查的 bug 来源）。
    # 代价：不能用 __dict__、弱引用等少数特性，对本用例无影响。
    __slots__ = ("is_pos", "content", "label", "feats", "fused")

    def __init__(self, r: dict):
        # 从 detail 的一条结果构造一个"试验样本"。
        self.is_pos = r["sample_type"] == "positive"
        # 与官方口径一致：normalize 后再算编辑距离。
        # content/label 都先归一化，编辑距离才反映真实识别错误
        # （数字写法、标点差异不算错）——这是全仓库统一的 CER 口径。
        self.content = normalize_transcript(r.get("content", "") or "")
        self.label = normalize_transcript(r.get("label", "") or "")
        # 6 维门控特征取出来存成 {特征名: float}，float() 把 detail 里
        # 的字符串/JSON 数值统一成 Python float。
        self.feats = {k: float(r["gate_features"][k]) for k in FEATURES}
        # 历史管线已算好的融合分（产线复现校验的对照基准）。
        self.fused = float(r["sv_score"])


class TrialSet:
    """预计算向量化指标所需的每样本常量（lev 与阈值无关，只算一次）。

    （扩充）设计动机：C 节的 oracle 扫描要在成百上千个阈值上重复计算
    PR/RR/CER。朴素写法是"每个阈值 × 每样本"都算一遍编辑距离——而
    编辑距离是 O(文本长度²) 的重活，全部样本乘全部阈值会慢到不可接受。
    观察：编辑距离只依赖样本本身，与阈值无关。所以把"每样本的编辑错误
    数、标签长度、是否非空"等常量一次性算好存成 numpy 数组，之后任何
    阈值都只做向量化的布尔运算（O(样本数)，微秒级）——"把与参数无关的
    计算提出循环"是性能优化的经典套路。
    """

    def __init__(self, trials: list[Trial]):
        # 以下每个属性都是一维 numpy 数组，行号与 trials 一一对应。
        self.pos = np.array([t.is_pos for t in trials])
        # nonempty：接受后能否给出非空内容（PR 的口径是"非空"）。
        self.nonempty = np.array([bool(t.content.strip()) for t in trials])
        # err_accept：该样本若被接受，其编辑错误数（正样本 = 编辑距离，
        # 负样本填 0——负样本不参与 CER）。
        self.err_accept = np.array(
            [lev(t.content, t.label) if t.is_pos else 0 for t in trials], dtype=np.int64)
        # len_label：正样本的标注长度（CER 分母/拒绝时的全错错误数），
        # 负样本填 0。
        self.len_label = np.array([len(t.label) if t.is_pos else 0 for t in trials], dtype=np.int64)
        # int(bool 数组).sum() 数 True 的个数——numpy 里计数的最快写法。
        self.n_pos = int(self.pos.sum())
        self.n_neg = int((~self.pos).sum())
        self.total_chars = int(self.len_label.sum())

    def metrics(self, scores: np.ndarray, threshold: float) -> dict:
        # 给定一列分数和一个阈值，向量化计算全部指标。
        # 布尔数组运算（& ~ >=）都是逐元素、一次完成的 C 层循环。
        accept = scores >= threshold
        # PR：接受 且 是正样本 且 内容非空 / 正样本数。
        pr = float((accept & self.pos & self.nonempty).sum() / max(self.n_pos, 1))
        # RR：拒绝 且 是负样本 / 负样本数。
        rr = float((~accept & ~self.pos).sum() / max(self.n_neg, 1))
        # CER 分子：接受的正样本算编辑错误数，拒绝的正样本整条算错
        # （空输出对全部标注字符全错）。np.where 三元向量化选择。
        errs = int(np.where(accept, self.err_accept, self.len_label)[self.pos].sum())
        cer = errs / max(self.total_chars, 1)
        # formula = (1 - CER + RR) / 2：竞赛官方公式。
        # balanced = (PR + RR) / 2：另一口径（oracle 扫描双口径各报一个）。
        return {
            "threshold": round(float(threshold), 4), "PR": round(pr, 4), "RR": round(rr, 4),
            "CER": round(cer, 4), "formula": round((1 - cer + rr) / 2, 4),
            "balanced": round((pr + rr) / 2, 4),
        }

    def sweep(self, scores: np.ndarray) -> tuple[dict, dict]:
        """全量唯一阈值精确扫描，返回 (公式分最优点, balanced 最优点)。

        （扩充）为什么"唯一值"就是全量：阈值落在相邻两个分数之间时，
        没有任何样本的决策发生变化，指标不变；所以只需在样本实际取到的
        分数值上试一遍，就是精确的最优解（oracle = 事后在测试集上调参
        的理论上限）。两个口径各追一个最优点。
        """
        best_f = best_b = None
        for t in np.unique(scores):
            m = self.metrics(scores, t)
            if best_f is None or m["formula"] > best_f["formula"]:
                best_f = m
            if best_b is None or m["balanced"] > best_b["balanced"]:
                best_b = m
        return best_f, best_b


def eer(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    # EER（等错误率）：找 FAR == FRR 的阈值点，两个错误率相等时的值。
    # 它是说话人验证领域的标准指标，衡量"分数列本身"区分正负的能力
    # （与业务阈值无关）。这里用分位数网格近似：在 0.1% 到 99.9%
    # 分位之间取 599 个候选阈值，找 |FAR - FRR| 最小的点，取两率均值。
    # 用分位数而不是在原始分数上均匀取点：分数分布可能有极端离群值，
    # 分位网格把候选点均匀铺在"数据密度"上，覆盖更均衡。
    alls = np.concatenate([pos_scores, neg_scores])
    best_gap, best_eer = 1.0, 1.0
    for t in np.unique(np.quantile(alls, np.linspace(0.001, 0.999, 599))):
        frr = float((pos_scores < t).mean())   # 假拒绝率：正样本被拒
        far = float((neg_scores >= t).mean())  # 假接受率：负样本被收
        if abs(far - frr) < best_gap:
            best_gap, best_eer = abs(far - frr), (far + frr) / 2
    return round(best_eer, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    # type=Path：argparse 把参数值直接解析成 Path 对象。
    ap.add_argument("--detail", type=Path, required=True)
    ap.add_argument("--fusion", type=Path, required=True)
    args = ap.parse_args()

    # 读入历史运行的全部结果行，构造成 Trial 列表。
    results = json.load(args.detail.open(encoding="utf-8"))["results"]
    trials = [Trial(r) for r in results]
    # 融合包（生产 logistic 的参数）——用它重算融合分，与 detail 里
    # 落盘的 sv_score 对照（A 节），两者一致才说明重放口径正确。
    bundle = json.load(args.fusion.open(encoding="utf-8"))
    mean = np.asarray(bundle["mean"], dtype=np.float32)
    std = np.asarray(bundle["std"], dtype=np.float32)
    weight = np.asarray(bundle["weight"], dtype=np.float32)
    bias = float(bundle["bias"])

    # 所有样本的 6 维特征拼成 (N, 6) 矩阵——向量化运算的基础。
    X = np.array([[t.feats[k] for k in FEATURES] for t in trials], dtype=np.float32)
    # 融合分 = 标准化后加权求和 + 偏置（与生产管线/重放器同一公式）。
    # 优先级：先 (X - mean) / std（逐元素广播），再 @ weight（点积），
    # 最后 + bias。
    fused = (X - mean) / std @ weight + bias

    # TrialSet 预计算所有与阈值无关的量（编辑距离等），后续每个阈值
    # 的指标都是纯向量运算。
    ts = TrialSet(trials)

    # A. 产线复现校验
    # detail 里落盘的融合分 vs 现在重算的融合分：最大绝对差应当接近 0
    # （浮点舍入量级）。这一步是"重放器的可信度证明"：连产线分数都
    # 复现不了，后面的所有消融数字都无从谈起。
    detail_fused = np.array([t.fused for t in trials], dtype=np.float32)
    max_diff = float(np.abs(fused - detail_fused).max())
    prod = ts.metrics(fused, EXT_THRESHOLDS["dual"])
    n_pos = sum(t.is_pos for t in trials)
    n_neg = len(trials) - n_pos
    print(f"== A. 产线复现校验 ==  样本 {len(trials)}（pos {n_pos} / neg {n_neg}）")
    print(f"  重算融合分 vs detail sv_score 最大差: {max_diff:.2e}")
    print(f"  双模型融合 @ -0.6469: {prod}")

    # 变体集：dual（6 维融合）+ 每个单维特征各成一列分数。
    # 消融 = 拿掉 5 维、只剩 1 维，看单维能保留多少决策能力。
    variants = {"dual": fused}
    for k in FEATURES:
        variants[k] = X[:, FEATURES.index(k)]

    # B. 外部标定阈值
    # 用历史标定阈值（非本集调出）评价每个变体——这是"泛化视角"，
    # 数字可与其他外部实验横向比较。
    print("\n== B. 外部标定阈值（非 DatasetA 调出）==")
    # 格式化打印：{:12s} 左对齐 12 列宽等，把表格排整齐。
    print(f"{'variant':12s} {'thr':>9s} {'PR':>7s} {'RR':>7s} {'CER':>7s} {'公式分':>7s}")
    for name, thr in EXT_THRESHOLDS.items():
        m = ts.metrics(variants[name], thr)
        print(f"{name:12s} {thr:9.4f} {m['PR']:7.4f} {m['RR']:7.4f} {m['CER']:7.4f} {m['formula']:7.4f}")

    # C. DatasetA oracle 上限
    # 同集精确扫描的最优阈值 = 理论上限。标题明确标注"DatasetA 上调参"，
    # 提醒读者这不是公平的外部估计（oracle 永远高估真实性能），
    # 只能用作"这个特征最多能到哪"的参考。
    print("\n== C. DatasetA oracle 上限（全量唯一阈值精确扫描，标注为 A 上调参）==")
    print(f"{'variant':12s} {'| 公式分最优':>26s} {'| balanced 最优':>26s} {'| EER':>7s}")
    for name in ["dual"] + FEATURES:
        s = variants[name]
        bf, bb = ts.sweep(s)
        # EER 用正负样本各自的分数列（D 节指标，与阈值口径无关）。
        e = eer(s[ts.pos], s[~ts.pos])
        print(f"{name:12s} | {bf['formula']:6.4f} @{bf['threshold']:8.4f} | "
              f"PR {bf['PR']:.4f} RR {bf['RR']:.4f} | {bb['balanced']:6.4f} @{bb['threshold']:8.4f} | {e:7.4f}")


if __name__ == "__main__":
    main()
