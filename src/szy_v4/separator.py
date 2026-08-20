"""Torch 版 MossFormer2_SS_16K 分离器（szy_v4 音频前端）。

与 szy_v1.pipeline.MossFormer2Separator（ONNX 版）同接口：整波推理、
peak->0.5 归一化、``_ms0``/``_ms1`` 落盘缓存。模型定义 vendor 在
``third_party/mossformer2/``，权重可以是：

  - ClearerVoice 预训练（ModelScope iic/ClearerVoice-Studio）；
  - 经 ``szy_v4.train_mossformer2`` 在 dataset_dasynth 上域适配微调的 .pt
    （生产用法：修复离线分离器在混响/掩盖样本上的崩塌）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIRD_PARTY = _REPO_ROOT / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from mossformer2.mossformer2 import MossFormer2_SS  # noqa: E402


def load_mossformer2_torch(checkpoint: str | Path, device: str):
    """构建 MossFormer2_SS_16K 并加载 .pt 权重（含/不含 'model' 键均可）。"""
    import torch

    ckpt = Path(checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"MossFormer2 torch checkpoint not found: {ckpt}")
    cfg = SimpleNamespace(
        num_spks=2, encoder_kernel_size=16, encoder_embedding_dim=512,
        mossformer_sequence_dim=512, num_mossformer_layer=24,
    )
    model = MossFormer2_SS(cfg).model
    payload = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)
    use = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return model.to(use).eval(), use


class MossFormer2TorchSeparator:
    """与 ONNX 分离器同接口的 torch 实现（v4 产线用）。"""

    def __init__(self, model_path: str | Path, device: str) -> None:
        self.model, self.device = load_mossformer2_torch(model_path, device)
        self.model_path = Path(model_path).expanduser().resolve()

    @staticmethod
    def _waveform(path: Path) -> np.ndarray:
        waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected 16 kHz audio, got {rate}: {path}")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"invalid audio waveform: {path}")
        return waveform

    def separate(self, source: Path, output_prefix: Path) -> tuple[Path, Path]:
        import torch

        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs = (output_prefix.with_name(output_prefix.name + "_ms0.wav"),
                   output_prefix.with_name(output_prefix.name + "_ms1.wav"))
        if all(path.is_file() for path in outputs):
            return outputs
        waveform = self._waveform(source)
        with torch.no_grad():
            out = self.model(torch.from_numpy(waveform).float().to(self.device)[None, :])
        for index, output_path in enumerate(outputs):
            signal = np.asarray(out[index].squeeze(0).float().cpu().numpy(), dtype=np.float32)
            peak = float(np.max(np.abs(signal))) if signal.size else 0.0
            if peak > 0.0:
                signal = signal / peak * 0.5
            sf.write(str(output_path), signal, SAMPLE_RATE, subtype="PCM_16")
        return outputs
