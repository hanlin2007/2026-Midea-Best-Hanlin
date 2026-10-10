"""Shared decision rules for the cached MossFormer candidate experiment.

This module intentionally contains only the two decisions that produced the
historical offline result:

* wake source selection is conditioned on the supplied wake text;
* command replacement is enabled only when the original FireRed hypothesis is
  unreliable, then uses same-separation differential speaker evidence.

MossFormer itself is a blind candidate generator.  It is not a replacement
for the primary dual-MLP gate and this module must not be enabled in online
inference until a separate end-to-end evaluation passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infer import levenshtein, normalize_transcript
from phoneme import phoneme_similarity


DECISION_SCHEMA_VERSION = "moss-candidate-reproduction-v1"


@dataclass(frozen=True)
class WakeAnchorDecision:
    """Wake-source decision before differential command scoring."""

    selected_source: int
    contrast_source: int
    raw_wake_fallback: bool
    text_tie: bool
    source0_sim: float | None
    source1_sim: float | None
    reason: str


@dataclass(frozen=True)
class CommandCandidateDecision:
    """Which transcript is used for one primary-gate accepted positive."""

    selected_candidate: str
    attempted_switch: bool
    command_source_empty: bool
    reason: str


def select_wake_anchor(
    wake_text: str | None,
    source0_text: str | None,
    source1_text: str | None,
) -> WakeAnchorDecision:
    """Select the source that uttered ``wake_text`` using phoneme similarity.

    A missing transcript marks the corresponding separated source as unusable
    for enrollment.  The full wake waveform is then the anchor, while the
    silent source remains a same-separation contrast term.  When both sources
    contain speech, phoneme similarity (pinyin initial/final/tone features)
    selects the best match.
    """

    reference = normalize_transcript(wake_text or "")
    source0 = str(source0_text or "").strip()
    source1 = str(source1_text or "").strip()
    has0, has1 = bool(source0), bool(source1)

    if has0 and not has1:
        return WakeAnchorDecision(0, 1, True, False, None, None, "source1_empty_raw_wake")
    if has1 and not has0:
        return WakeAnchorDecision(1, 0, True, False, None, None, "source0_empty_raw_wake")

    sim0 = phoneme_similarity(source0, reference)
    sim1 = phoneme_similarity(source1, reference)
    selected = 0 if sim0 >= sim1 else 1
    tie = not reference or sim0 == sim1
    if not has0 and not has1:
        return WakeAnchorDecision(selected, 0, True, tie, sim0, sim1, "both_sources_empty_raw_wake")
    if tie and has0 and has1:
        return WakeAnchorDecision(
            selected,
            1 - selected,
            True,
            tie,
            sim0,
            sim1,
            "wake_text_phoneme_similarity",
        )
    return WakeAnchorDecision(
        selected,
        1 - selected,
        False,
        tie,
        sim0,
        sim1,
        "wake_text_phoneme_similarity",
    )


def should_switch_command(*, confidence: float, command_keyword_score: float, chars_per_sec: float) -> bool:
    """The frozen two-dimensional FireRed confidence switch from the best run."""

    if confidence <= 0.70:
        return command_keyword_score >= 0.5 or chars_per_sec >= 3.0 or confidence <= 0.57
    return confidence <= 1.01 and chars_per_sec >= 3.0 and command_keyword_score < 0.5


def normalized_edit_distance(left: str | None, right: str | None) -> float:
    """归一化编辑距离：0 = 完全一致，1 = 完全不同。"""

    a, b = str(left or ""), str(right or "")
    return levenshtein(a, b) / max(len(a), len(b), 1)


def should_switch_command_timbre(
    *,
    confidence: float,
    command_keyword_score: float,
    sv_diff: float,
    confidence_high: float = 0.70,
    confidence_low: float = 0.57,
    sv_diff_tau: float = 0.05,
) -> bool:
    """音色触发规则（szy_v2 当前采用）。

    sv_diff = max(cos(anchor, cmd_ms0), cos(anchor, cmd_ms1)) - cos(anchor, raw_cmd)，
    直接测量"目标嗓音在分离通道里是否比在原始混合里更突出"——切换收益的
    物理机制本身。三特征分工：conf 管必坏线（<=0.57 必切）、sv_diff 管主判决、
    kw 管高置信否决（自信转写但不像指令 + 音色说分离更好 => 转错了人）。
    DatasetA：CER 0.3058（文本规则 0.3092 / cps 规则 0.3181）；
    τ 盆地 [0.02, 0.08] 平坦，conf 切点在 [0.50,0.70]x[0.65,0.80] 内平坦。

    注意：上述默认值是在 ELM 融入主解码、confidence 被压到 ~0.60 量程时标定
    的。ELM 改为选择后重解码之后 confidence 回到 ~0.88 量程，切点必须重标定，
    否则规则会落在错误的位置（实测掉 0.0027）。重扫结果见配置文件里的
    firered_switch_* 三项。
    """

    if confidence <= confidence_high:
        return command_keyword_score >= 0.5 or sv_diff > sv_diff_tau or confidence <= confidence_low
    return sv_diff > sv_diff_tau and command_keyword_score < 0.5


def should_switch_command_text(*, confidence: float, command_keyword_score: float, d_sel: float) -> bool:
    """文本一致性触发规则（szy_v2 采用，无语速）。

    d_sel 是 raw 转写与 ft 余弦所选分离通道转写的归一化编辑距离，
    直接测量"raw 里装的是不是目标说话人的内容"，替代原规则的语速特征
    （语速是对"ASR 混流"的间接假设）。分档结构沿用原规则：
    DatasetA 上 CER 0.309 vs 原规则 0.318，且阈值在 [0.35, 0.70] 盆地里平坦。
    """

    if confidence <= 0.70:
        return command_keyword_score >= 0.5 or d_sel >= 0.5 or confidence <= 0.57
    return d_sel >= 0.5 and command_keyword_score < 0.5


def choose_command_candidate(
    *,
    should_switch: bool,
    source0_text: str | None,
    source1_text: str | None,
    source0_difference: float | None,
    source1_difference: float | None,
) -> CommandCandidateDecision:
    """Choose ``orig``/``ms0``/``ms1`` without changing the ASR waveform.

    An empty MossFormer command source is treated as a failed separation and
    forces the original FireRed output.  Difference ties resolve to ``ms0``;
    this mirrors the prior experiment's ``>=`` comparison.
    """

    if not should_switch:
        return CommandCandidateDecision("orig", False, False, "confidence_rule_not_triggered")
    if not str(source0_text or "").strip() or not str(source1_text or "").strip():
        return CommandCandidateDecision("orig", True, True, "command_source_empty")
    if source0_difference is None or source1_difference is None:
        raise ValueError("differential scores are required when a command switch is attempted")
    selected = "ms0" if float(source0_difference) >= float(source1_difference) else "ms1"
    return CommandCandidateDecision(selected, True, False, "same_domain_differential_score")


def as_float(mapping: dict[str, Any], field: str, identifier: str) -> float:
    try:
        return float(mapping[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{identifier} has no valid {field!r}") from error
