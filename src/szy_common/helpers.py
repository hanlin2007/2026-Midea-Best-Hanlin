"""szy_common.helpers：各阶段管线共享的通用工具。

内容均为历史上在 szy_v1 / szy_v2 / scripts 中重复出现的实现，收敛后语义
与原实现逐字一致（含置信度非有限值回退 1.0、分离缓存文件命名规则等），
仅将私有下划线名改为公共名：

  - ``sha256_file``            文件分块 SHA-256（manifest 用）；
  - ``read_mono_16k``          严格 16k 单声道 float32 音频读取（分离器输入）；
  - ``duration_seconds``       音频时长（秒）；
  - ``command_features``       指令文本特征（字速 + 指令关键词命中数）；
  - ``asr_confidence``         FireRed 行级置信度解析（非法/非有限值回退 1.0）；
  - ``transcribe_firered_rows`` FireRed 批量转写并归一化文本 + 置信度；
  - ``separation_paths_for``   某样本唤醒/指令的分离缓存输出路径对。

依赖：midea_qiji（文本归一化、Sample/submission_id、静默上下文），
numpy、soundfile。不依赖 torch / onnxruntime / modelscope。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np
import soundfile as sf

from midea_qiji.audio import SAMPLE_RATE
from midea_qiji.infer import Sample, submission_id, suppress_model_output
from midea_qiji.text import normalize_transcript

CMD_KW = re.compile(
    r"打开|关闭|关掉|调到|空调|灯光?|窗帘|音乐|电视|风扇|加湿|净化|扫地|"
    r"温度|音量|模式|洗衣|烘干|冰箱|热水|新风|播放|暂停|继续|小声|大声|"
    r"亮度|风速|洗澡|叫醒|闹钟|出门|回家|晚安"
)


def sha256_file(path: Path) -> str:
    """返回文件的 SHA-256 十六进制摘要（1MB 分块读取）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mono_16k(path: Path) -> np.ndarray:
    """读取严格 16kHz 单声道音频为 float32，通道混叠为均值，校验有效性。

    两个分离器（szy_v1 的 ONNX 版与 szy_v4 的 torch 版）共用此读取器，
    保证输入口径一致。
    """
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz audio, got {rate}: {path}")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid audio waveform: {path}")
    return waveform


def duration_seconds(path: Path) -> float:
    """返回音频时长（秒），只读文件头不解码。"""
    info = sf.info(str(path))
    return float(info.frames) / max(int(info.samplerate), 1)


def command_features(text: str, duration: float) -> tuple[float, float]:
    """指令文本双特征：归一化后字速（字/秒）与 CMD_KW 指令关键词命中数。"""
    normalized = normalize_transcript(text)
    return float(len(normalized) / max(duration, 0.1)), float(len(CMD_KW.findall(normalized)))


def asr_confidence(row: dict[str, Any]) -> float:
    """从 FireRed 结果行解析置信度；非法或非有限值回退 1.0。"""
    value = row.get("confidence")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return value if np.isfinite(value) else 1.0


def transcribe_firered_rows(asr: Any, paths: list[Path], prefix: str) -> list[dict[str, Any]]:
    """FireRed 批量转写：返回 [{"text": 归一化文本, "confidence": float}]。

    ``asr`` 为 FireRedRecognizer / FireRedLLMRecognizer（二者统一暴露
    ``_model.transcribe``）；``prefix`` 为本次调用的 uttid 前缀（不同阶段
    使用不同前缀以区分日志来源）。与 szy_v1/szy_v2 历史实现同语义。
    """
    if not paths:
        return []
    utterance_ids = [f"{prefix}_{index}" for index in range(len(paths))]
    with suppress_model_output():
        rows = asr._model.transcribe(utterance_ids, [str(path) for path in paths])
    mapped = {str(row.get("uttid")): row for row in rows}
    if len(mapped) != len(paths):
        raise RuntimeError("FireRed returned an unexpected result count")
    output = []
    for identifier in utterance_ids:
        row = mapped[identifier]
        output.append({
            "text": normalize_transcript(str(row.get("text", ""))),
            "confidence": asr_confidence(row),
        })
    return output


def separation_paths_for(sample: Sample, id_mode: str, cache_dir: Path, separator: Any):
    """返回 (唤醒分离路径对, 指令分离路径对)，缓存命中时直接复用。

    落盘命名规则与历史一致：``{item_id}_wake_ms{0,1}.wav`` /
    ``{item_id}_command_ms{0,1}.wav``，item_id 中的 ``/`` 替换为 ``_``。
    """
    item_id = submission_id(sample, id_mode).replace("/", "_")
    wake_prefix = cache_dir / f"{item_id}_wake"
    command_prefix = cache_dir / f"{item_id}_command"
    return (
        separator.separate(sample.wake_audio, wake_prefix),
        separator.separate(sample.command_audio, command_prefix),
    )
