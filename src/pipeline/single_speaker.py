"""pipeline.single_speaker：V2 单人判别器，以及"双单人"原始音频余弦路由。

该模块只使用原始音频和 MossFormer2 的两路输出，不依赖 ASR 文本：

1. 分别判断唤醒音频、命令音频是否为单人（20 维声学特征 + 线性分类器）；
2. 只有两段音频都被高置信判为单人时，才进入独立的"双单人"分支；
3. 双单人分支使用微调 ERes2NetV2 的 ``cos(raw_kws, raw_cmd)`` 判断
   是否为同一说话人。其余场景仍交给融合门控流程处理。

为什么需要它？整条管线最贵的是 ASR 与声纹 embedding。如果唤醒和命令
两段音频从波形上就能断定"各自只有一个人在说话"，那分离通道不会带来
任何额外信息——直接用原始音频算一次余弦、解码一次原始命令即可，
跳过整套路由。这是一个"便宜特征先行短路贵流程"的典型工程模式。

20 维特征的设计要点（feature_mapping）：
  - 全部特征对"通道交换"不变（ms0/ms1 谁强谁弱不重要，分离器的通道
    顺序本来就是随机的）——强/弱通道一律按能量排序后再取统计量；
  - 分帧 RMS（20ms 帧）刻画"什么时候有几个人在说话"：单人录音里
    弱通道基本不活跃（weak_share 低、both_active_ratio 低）；
  - 重构误差刻画"两路输出叠回来像不像原混合"：分离质量好时
    raw ≈ ms0 + ms1（再允许一次最小二乘拟合系数）。

分类器是外部训练的线性 logistic（权重 JSON bundle，训练和阈值选择
均未使用 DatasetA）；本模块只负责特征提取与打分，不参与训练。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


FORMAT = "moss-single-speaker-router-v2"
# 20 维特征的固定顺序：与训练时喂给 logistic 的特征向量顺序一一对应，
# 加载 bundle 时逐一校验，顺序错位立即报错（同 FusionGate 的"隐形契约"）
FEATURE_NAMES = (
    "raw_log_rms",
    "strong_log_rms",
    "weak_log_rms",
    "abs_ms_energy_log_ratio",
    "raw_active_ratio",
    "strong_active_ratio_common",
    "weak_active_ratio_common",
    "active_ratio_gap",
    "both_active_ratio",
    "exactly_one_active_ratio",
    "weak_share_mean",
    "weak_share_p50",
    "weak_share_p90",
    "dominance_flip_ratio",
    "envelope_correlation",
    "mix_ms_reconstruction_error",
    "mix_ms_fitted_reconstruction_error",
    "ms_waveform_cosine",
    "raw_ms_cosine_max",
    "raw_ms_cosine_min",
)


def _wave(value: np.ndarray) -> np.ndarray:
    """Return a finite one-dimensional float waveform."""
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if not result.size or not np.isfinite(result).all():
        raise ValueError("waveform must be non-empty and finite")
    return result


def _rms(value: np.ndarray) -> float:
    """Calculate numerically stable root-mean-square amplitude."""
    return float(np.sqrt(np.mean(np.square(value), dtype=np.float64) + 1e-12))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate waveform cosine after truncating to the common length."""
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    left, right = left[:size], right[:size]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-8))


def _frame_rms(value: np.ndarray, frame_samples: int = 320) -> np.ndarray:
    """Return non-overlapping 20-ms frame RMS at 16 kHz."""
    usable = len(value) // frame_samples * frame_samples
    if usable == 0:
        return np.zeros(1, dtype=np.float32)
    frames = value[:usable].reshape(-1, frame_samples)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12).astype(np.float32)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation, using zero for constant sequences."""
    if len(left) < 2 or float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def feature_mapping(raw: np.ndarray, ms0: np.ndarray, ms1: np.ndarray) -> dict[str, float]:
    """Extract channel-swap-invariant global and frame-level acoustic features."""
    raw, ms0, ms1 = _wave(raw), _wave(ms0), _wave(ms1)
    size = min(len(raw), len(ms0), len(ms1))
    raw, ms0, ms1 = raw[:size], ms0[:size], ms1[:size]

    raw_rms, rms0, rms1 = _rms(raw), _rms(ms0), _rms(ms1)
    # 通道交换不变性的关键：不区分 ms0/ms1，只分"强通道/弱通道"
    strong_rms, weak_rms = max(rms0, rms1), min(rms0, rms1)
    raw_frames, frames0, frames1 = _frame_rms(raw), _frame_rms(ms0), _frame_rms(ms1)
    frame_count = min(len(raw_frames), len(frames0), len(frames1))
    raw_frames, frames0, frames1 = raw_frames[:frame_count], frames0[:frame_count], frames1[:frame_count]

    strong_frames = np.maximum(frames0, frames1)
    weak_frames = np.minimum(frames0, frames1)
    raw_threshold = max(float(raw_frames.max()) * 0.10, 1e-4)
    # A shared threshold prevents a very weak residual channel from declaring
    # itself active merely because its own peak is also weak.
    # （中文）活跃判定用"共享阈值"：若各通道按自己的峰值定阈值，极弱的
    # 残余通道也会"自我宣布活跃"，单人录音就被误判成双人。
    common_threshold = max(float(strong_frames.max()) * 0.05, raw_threshold * 0.25, 1e-4)
    active0, active1 = frames0 >= common_threshold, frames1 >= common_threshold
    raw_active = raw_frames >= raw_threshold
    active_strong, active_weak = strong_frames >= common_threshold, weak_frames >= common_threshold

    valid = raw_active | active_strong
    # weak_share：弱通道能量占两路之和的比例——单人时弱通道几乎没能量
    weak_share = weak_frames[valid] / np.maximum(strong_frames[valid] + weak_frames[valid], 1e-8)
    if not weak_share.size:
        weak_share = np.zeros(1, dtype=np.float32)

    # Ignore nearly tied/inaudible frames when measuring which output dominates.
    log_ratio = np.log((frames0 + 1e-6) / (frames1 + 1e-6))
    dominant = np.where(log_ratio > 0.20, 1, np.where(log_ratio < -0.20, -1, 0))
    dominant = dominant[valid & (dominant != 0)]
    flip_ratio = float(np.mean(dominant[1:] != dominant[:-1])) if len(dominant) > 1 else 0.0

    # 重构误差：raw - ms0 - ms1 的残余能量占比（分离保真度）
    reconstruction = _rms(raw - ms0 - ms1) / max(raw_rms, 1e-6)
    # Solve the two-coefficient least-squares projection explicitly.  This is
    # equivalent to lstsq for this tiny system but avoids an SVD per utterance.
    # （中文）再允许两个自由系数做最小二乘拟合（吸收分离器的整体增益差），
    # 2 元方程直接展开求解，省去每条语音一次 SVD 分解。
    a00 = float(np.dot(ms0, ms0)) + 1e-8
    a01 = float(np.dot(ms0, ms1))
    a11 = float(np.dot(ms1, ms1)) + 1e-8
    b0, b1 = float(np.dot(ms0, raw)), float(np.dot(ms1, raw))
    determinant = max(a00 * a11 - a01 * a01, 1e-8)
    coefficient0 = (b0 * a11 - b1 * a01) / determinant
    coefficient1 = (b1 * a00 - b0 * a01) / determinant
    fitted = coefficient0 * ms0 + coefficient1 * ms1
    fitted_reconstruction = _rms(raw - fitted) / max(raw_rms, 1e-6)
    raw_cosines = sorted((_cosine(raw, ms0), _cosine(raw, ms1)))

    values = {
        "raw_log_rms": float(np.log(raw_rms + 1e-6)),
        "strong_log_rms": float(np.log(strong_rms + 1e-6)),
        "weak_log_rms": float(np.log(weak_rms + 1e-6)),
        "abs_ms_energy_log_ratio": float(abs(np.log((rms0 + 1e-6) / (rms1 + 1e-6)))),
        "raw_active_ratio": float(np.mean(raw_active)),
        "strong_active_ratio_common": float(np.mean(active_strong)),
        "weak_active_ratio_common": float(np.mean(active_weak)),
        "active_ratio_gap": float(abs(np.mean(active0) - np.mean(active1))),
        "both_active_ratio": float(np.mean(active0 & active1)),
        "exactly_one_active_ratio": float(np.mean(active0 ^ active1)),
        "weak_share_mean": float(np.mean(weak_share)),
        "weak_share_p50": float(np.quantile(weak_share, 0.50)),
        "weak_share_p90": float(np.quantile(weak_share, 0.90)),
        "dominance_flip_ratio": flip_ratio,
        "envelope_correlation": _correlation(frames0, frames1),
        "mix_ms_reconstruction_error": reconstruction,
        "mix_ms_fitted_reconstruction_error": fitted_reconstruction,
        "ms_waveform_cosine": _cosine(ms0, ms1),
        "raw_ms_cosine_max": raw_cosines[1],
        "raw_ms_cosine_min": raw_cosines[0],
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError("V2 single-speaker features contain non-finite values")
    return values


def feature_vector(raw: np.ndarray, ms0: np.ndarray, ms1: np.ndarray) -> np.ndarray:
    """Return V2 features in the serialized classifier order."""
    values = feature_mapping(raw, ms0, ms1)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)


@dataclass(frozen=True)
class SingleSpeakerDecision:
    """一段混合音频的单人判定结果。"""

    probability: float
    is_single: bool
    features: dict[str, float]


@dataclass(frozen=True)
class BothSingleDecision:
    """唤醒和命令是否同时为单人的联合判定结果。"""

    wake: SingleSpeakerDecision
    command: SingleSpeakerDecision

    @property
    def is_both_single(self) -> bool:
        """只有唤醒和命令均为单人时返回真。"""
        return self.wake.is_single and self.command.is_single


class BothSingleRouter:
    """加载外部训练的 V2 线性分类器并识别"双单人"场景。

    分类器阈值保存在模型包中，训练和阈值选择均未使用 DatasetA。
    ``same_speaker_threshold`` 只用于双单人分支中的 SFT 原始余弦判断。
    """

    def __init__(self, bundle_path: str | Path, same_speaker_threshold: float = 0.29) -> None:
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        if bundle.get("format") != FORMAT:
            raise ValueError(f"{self.bundle_path} is not a {FORMAT} bundle")
        if tuple(bundle.get("features", ())) != FEATURE_NAMES:
            raise ValueError("V2 single-speaker feature schema does not match this code")
        self.mean = np.asarray(bundle["mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(bundle["std"], dtype=np.float32), 1e-6)
        self.weight = np.asarray(bundle["weight"], dtype=np.float32)
        self.bias = float(bundle["bias"])
        self.single_threshold = float(bundle["threshold"])
        self.same_speaker_threshold = float(same_speaker_threshold)
        expected = (len(FEATURE_NAMES),)
        if self.mean.shape != expected or self.std.shape != expected or self.weight.shape != expected:
            raise ValueError("invalid V2 single-speaker router parameter shape")
        if not 0.0 <= self.single_threshold <= 1.0:
            raise ValueError("single-speaker probability threshold must be in [0, 1]")
        if not -1.0 <= self.same_speaker_threshold <= 1.0:
            raise ValueError("same-speaker cosine threshold must be in [-1, 1]")
        self.metadata = dict(bundle.get("metadata", {}))

    def decide_single(self, raw: np.ndarray, ms0: np.ndarray, ms1: np.ndarray) -> SingleSpeakerDecision:
        """使用一组 ``raw/ms0/ms1`` 波形判断该录音是否为单人。"""
        values = feature_mapping(raw, ms0, ms1)
        vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
        # z-score 标准化 + 线性打分 + sigmoid 变概率；clip 防 exp 溢出
        logit = float(((vector - self.mean) / self.std) @ self.weight + self.bias)
        probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))
        return SingleSpeakerDecision(
            probability=probability,
            is_single=probability >= self.single_threshold,
            features=values,
        )

    def decide_both(
        self,
        raw_wake: np.ndarray,
        wake0: np.ndarray,
        wake1: np.ndarray,
        raw_command: np.ndarray,
        command0: np.ndarray,
        command1: np.ndarray,
    ) -> BothSingleDecision:
        """分别判断唤醒与命令，返回是否属于"双单人"场景。"""
        return BothSingleDecision(
            wake=self.decide_single(raw_wake, wake0, wake1),
            command=self.decide_single(raw_command, command0, command1),
        )

    def accept_same_speaker(self, cosine_score: float) -> bool:
        """按固定 SFT 原始余弦阈值判断双单人是否为同一说话人。"""
        return float(cosine_score) >= self.same_speaker_threshold
