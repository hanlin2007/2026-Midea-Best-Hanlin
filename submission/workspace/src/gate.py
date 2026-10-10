"""双前端 MLP 声纹门控（ERes2NetV2 + CAM++ 特征 → MLP 裁判）。

特征定义与训练端（sv_backend/extract_features.py）严格一致：
整段余弦 + 滑窗（1.0s/0.2s）余弦统计 + 时长/能量/削波，每个模型 16 维，拼接成 32 维。
裁判为 32->32->16->1 的 MLP，权重与标准化参数来自 backend_dual.pt。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

SAMPLE_RATE = 16000

FEATURE_KEYS = [
    "cos_whole", "n_windows", "win_max", "win_mean", "win_std", "win_top3_mean",
    "win_count_ge_25", "win_count_ge_30", "win_count_ge_35", "win_frac_ge_30", "win_span_ge_30",
    "enroll_seconds", "test_seconds", "rms_ratio", "clip_ratio_enroll", "clip_ratio_test",
]


def read_wav(path: str | Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ValueError(f"unsupported wav format: {path}")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def sliding_windows(audio: np.ndarray, window_seconds: float = 1.0, shift_seconds: float = 0.2) -> list[np.ndarray]:
    window = int(window_seconds * SAMPLE_RATE)
    shift = int(shift_seconds * SAMPLE_RATE)
    if len(audio) <= window:
        padded = np.zeros(window, dtype=np.float32)
        padded[: len(audio)] = audio
        return [padded]
    segments = []
    for start in range(0, len(audio) - window + 1, shift):
        segments.append(audio[start : start + window])
    tail = len(audio) - (len(segments) - 1) * shift - window
    if tail >= int(0.4 * SAMPLE_RATE):
        segments.append(audio[len(audio) - window :])
    return segments


def _window_stats(win_scores: list[float]) -> list[float]:
    scores = np.asarray(win_scores, dtype=np.float64)
    n = len(scores)
    above = scores >= 0.30
    if above.any():
        indices = np.flatnonzero(above)
        span = (indices[-1] - indices[0] + 1) / n
    else:
        span = 0.0
    return [
        float(n), float(scores.max()), float(scores.mean()), float(scores.std()),
        float(np.sort(scores)[-3:].mean()),
        float((scores >= 0.25).sum()), float(above.sum()), float((scores >= 0.35).sum()),
        float(above.mean()), float(span),
    ]


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2) + 1e-12))


def _clip_ratio(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= 0.999))


class ModelScopeEmbedder:
    """对任意段波形提 L2 归一化 embedding（自动适配 ERes2NetV2 / CAM++）。"""

    def __init__(self, model: str | Path, device: str, batch_size: int = 64) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.batch_size = batch_size
        self.pipeline = pipeline(task=Tasks.speaker_verification, model=str(model), device=device)

    def extract_waveforms(self, waveforms: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not waveforms:
            return []
        import torch

        prepared = self.pipeline.preprocess(list(waveforms))
        model = self.pipeline.model
        extract_feature = None
        for attr in dir(model):
            if attr.endswith("__extract_feature"):
                extract_feature = getattr(model, attr)
                break
        if extract_feature is None:
            raise AttributeError(f"no __extract_feature on {type(model).__name__}")
        features = [extract_feature(w.unsqueeze(0) if w.ndim == 1 else w) for w in prepared]
        by_frame_count: dict[int, list[int]] = defaultdict(list)
        for index, feature in enumerate(features):
            by_frame_count[int(feature.shape[1])].append(index)
        embeddings: list[np.ndarray | None] = [None] * len(features)
        with torch.inference_mode():
            for indices in by_frame_count.values():
                for start in range(0, len(indices), self.batch_size):
                    batch_indices = indices[start : start + self.batch_size]
                    feature_batch = torch.cat([features[i] for i in batch_indices], dim=0).to(model.device)
                    batch = model.embedding_model(feature_batch).detach().cpu().numpy()
                    for index, embedding in zip(batch_indices, batch):
                        embeddings[index] = np.asarray(embedding, dtype=np.float32)
        normalized: list[np.ndarray] = []
        for embedding in embeddings:
            if embedding is None:
                raise ValueError("embedder did not return every embedding")
            norm = float(np.linalg.norm(embedding))
            normalized.append((embedding / max(norm, 1e-8)).astype(np.float32))
        return normalized


class DualMLPGate:
    """双前端 MLP 门控：score >= threshold 判为含目标说话人。"""

    def __init__(self, sv_model: str, cam_model: str, backend_path: str | Path,
                 threshold: float, device: str, model_root: str | Path | None = None) -> None:
        import torch
        from torch import nn

        from infer import resolve_model_path

        self.threshold = threshold
        self.eres_embedder = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
        self.cam_embedder = ModelScopeEmbedder(resolve_model_path(cam_model, model_root), device)

        bundle = torch.load(str(backend_path), map_location="cpu", weights_only=False)["dual"]
        self.mean = np.asarray(bundle["mean"], dtype=np.float32)
        self.std = np.asarray(bundle["std"], dtype=np.float32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        renamed = {k.replace("net.", ""): v for k, v in bundle["mlp"].items()}
        self.mlp.load_state_dict(renamed)
        self.mlp.eval()
        self.torch = torch

    def _model_features(self, embedder: ModelScopeEmbedder, enroll_wav, test_wav, windows) -> list[float]:
        embeddings = embedder.extract_waveforms([enroll_wav, test_wav, *windows])
        emb_enroll, emb_test = embeddings[0], embeddings[1]
        win_scores = [cosine_similarity(emb_enroll, w) for w in embeddings[2:]]
        return [
            cosine_similarity(emb_enroll, emb_test),
            *_window_stats(win_scores),
            len(enroll_wav) / SAMPLE_RATE,
            len(test_wav) / SAMPLE_RATE,
            _rms(test_wav) / max(_rms(enroll_wav), 1e-6),
            _clip_ratio(enroll_wav),
            _clip_ratio(test_wav),
        ]

    def score(self, wake_audio: Path, command_audio: Path) -> float:
        enroll_wav = read_wav(wake_audio)
        test_wav = read_wav(command_audio)
        windows = sliding_windows(test_wav)
        features = self._model_features(self.eres_embedder, enroll_wav, test_wav, windows)
        features += self._model_features(self.cam_embedder, enroll_wav, test_wav, windows)
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        with self.torch.no_grad():
            logit = self.mlp(self.torch.from_numpy(x).unsqueeze(0)).item()
        return float(1.0 / (1.0 + np.exp(-logit)))

    def accept(self, wake_audio: Path, command_audio: Path) -> tuple[bool, float]:
        score = self.score(wake_audio, command_audio)
        return score >= self.threshold, score
