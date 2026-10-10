"""core.firered_nbest：FireRed N-best 假设的保守 MBR 重排序。

背景：beam search（束搜索）在解码时会同时维护多条候选转写
（N-best 列表）。beam=1 只返回概率最高的一条，可能错过"概率接近但
字更对"的候选。MBR（Minimum Bayes Risk，最小贝叶斯风险）重排序的
思路是：不选"模型分最高"的候选，而选"与其他候选平均差异最小"的
候选——它像候选集合的"中位数"，期望错误最少。

"保守"体现在两道闸（缺一不可才允许换掉基线）：
  1. 置信度增益 >= min_confidence_gain（新候选要明显更自信）；
  2. MBR 风险下降 >= min_mbr_risk_reduction（风险要明显更低）。

两道闸的意义：N-best 重排只在"明显更好"时才改动结果，防止重排序
本身引入抖动——宁可不换，不可换错。生产上它只作用在路由选定后的
那一条通道上（见 pipeline.pipeline.PipelineRunner._nbest_redecode），不参与
门控/路由决策。

依赖：仅标准库 + core.text（levenshtein）。
"""
from __future__ import annotations

from typing import Any

from .text import levenshtein


def normalized_edit_distance(hyp: str, ref: str) -> float:
    """归一化编辑距离：0 = 完全一致，1 = 完全不同（MBR 风险的"损失"）。

    MBR 的"风险"是候选两两之间的期望差异，直接用原始编辑距离会让长
    文本天然风险更高，所以先按长度归一化（与 moss_candidate 里的同名
    函数同口径，但这里输入一定非 None，故不需要 None 兜底）。
    """
    if not hyp and not ref:
        return 0.0
    return levenshtein(hyp, ref) / max(len(hyp), len(ref), 1)


def conservative_mbr_rerank(
    baseline: dict[str, Any],
    nbest: list[dict[str, Any]],
    min_confidence_gain: float = 0.03,
    min_mbr_risk_reduction: float = 0.05,
) -> dict[str, Any]:
    """对 N-best 候选做保守 MBR 重排序。

    参数：
      baseline               基线候选 {"text", "confidence"}（通常就是
                             beam 搜索的 top-1）；
      nbest                  其余候选列表（已剔除与基线文本相同者）；
      min_confidence_gain    允许换人的最低置信度增益；
      min_mbr_risk_reduction 允许换人的最低风险下降。

    返回：被选中的候选（加上重排序元数据）：
      nbest_applied      是否真的换掉了基线；
      nbest_reason       人类可读的决策原因（写进 detail.json 供排查）；
      nbest_candidates   参与评比的全部候选；
      nbest_risks        每个候选的 MBR 风险值。
    """
    candidates = [baseline] + nbest

    # ---- 计算每个候选的 MBR 风险 ----
    # 候选 i 的风险 = Σ_j≠i confidence(j) × distance(i, j)：
    # 用其他候选的置信度当权重，度量"如果真相在别处，选 i 会错多少"。
    # 置信度高的对手发言权重更大——它更可能是真相。
    mbr_risks = []
    for i, cand_i in enumerate(candidates):
        risk = 0.0
        for j, cand_j in enumerate(candidates):
            if i != j:
                weight = cand_j.get("confidence", 0.0)
                dist = normalized_edit_distance(cand_i.get("text", ""), cand_j.get("text", ""))
                risk += weight * dist
        mbr_risks.append(risk)

    # 风险最小的候选 = MBR 最优
    best_idx = min(range(len(mbr_risks)), key=lambda i: mbr_risks[i])
    best_candidate = candidates[best_idx]
    best_risk = mbr_risks[best_idx]
    baseline_risk = mbr_risks[0]

    if best_idx == 0:
        # 基线自己就是风险最小者，不换
        return {
            **baseline,
            "nbest_applied": False,
            "nbest_reason": "baseline_has_minimum_risk",
            "nbest_candidates": candidates,
            "nbest_risks": mbr_risks,
        }

    # 第一道闸：置信度增益不足不换
    baseline_conf = baseline.get("confidence", 0.0)
    best_conf = best_candidate.get("confidence", 0.0)
    conf_gain = best_conf - baseline_conf
    risk_reduction = baseline_risk - best_risk

    if conf_gain < min_confidence_gain:
        return {
            **baseline,
            "nbest_applied": False,
            "nbest_reason": f"insufficient_confidence_gain ({conf_gain:.4f} < {min_confidence_gain})",
            "nbest_candidates": candidates,
            "nbest_risks": mbr_risks,
        }

    # 第二道闸：风险下降不足不换
    if risk_reduction < min_mbr_risk_reduction:
        return {
            **baseline,
            "nbest_applied": False,
            "nbest_reason": f"insufficient_risk_reduction ({risk_reduction:.4f} < {min_mbr_risk_reduction})",
            "nbest_candidates": candidates,
            "nbest_risks": mbr_risks,
        }

    # 两闸都过：换成 MBR 最优候选，并把基线一并记录供事后审计
    return {
        **best_candidate,
        "nbest_applied": True,
        "nbest_reason": f"switched (conf_gain={conf_gain:.4f}, risk_reduction={risk_reduction:.4f})",
        "nbest_candidates": candidates,
        "nbest_risks": mbr_risks,
        "nbest_baseline": baseline,
    }
