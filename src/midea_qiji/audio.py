"""midea_qiji.audio：音频常量与 WAV 读取工具。

统一全仓库的采样率常量（16k）与两种读取口径：

  - ``read_wav``               严格 16k 单声道 PCM16，无重采样（wave 模块）；
  - ``load_audio_mono_16k``    soundfile 读取 + 混叠为单声道，必要时重采样。

原 read_wav 定义于 gate.py、load_audio_mono_16k 定义于 infer.py，现收敛
至此；gate.py 仍通过导入保持 ``from midea_qiji.gate import read_wav`` 旧
路径可用。

依赖：numpy / soundfile；重采样分支才需要 torch/torchaudio（延迟导入）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def read_wav(path: str | Path) -> np.ndarray:
    """读取严格 16k 单声道 PCM16 wav 为 float32（值域 [-1, 1]）。"""
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ValueError(f"unsupported wav format: {path}")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load_audio_mono_16k(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """读取音频为单声道 float32，必要时重采样到目标采样率。

    与 sv_asr/train_asr.py 的 load_audio 口径一致（LoRA 条件识别后端用）。
    """
    import soundfile as sf

    waveform, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate != sample_rate:
        import torch
        import torchaudio

        tensor = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(tensor, source_rate, sample_rate).numpy()
    return waveform.astype(np.float32, copy=False)
