"""End-to-end SZY v2 target-speaker ASR route.

Same overall flow as szy_v1 (gate -> raw FireRed -> switch trigger ->
MossFormer2 candidate selection).  Three changes relative to v1:

1. The primary gate is a linear fusion gate over two ERes2NetV2 frontends
   (pretrained + separation-domain fine-tuned, dataset_dasynth ep6),
   using six cosine features; threshold fixed on the dasynth valid split.
2. The command candidate is selected by the fine-tuned model's direct
   cosine (cos(anchor, ms0) vs cos(anchor, ms1)) instead of the eres+cam
   same-separation differential.  Measured statistically tied with the
   production rule on DatasetA (1021 vs 1013 chars of error over 368
   switch trials), but it is the same scoring principle as the gate and
   avoids the contrast-channel pollution failure mode of the
   differential form.  The ft per-channel cosines computed at gate time
   are reused, so candidate selection adds no extra embedding pass.
3. Command routing uses the acoustic single-speaker classifier directly.
   A single-speaker command is decoded only from Raw CMD; a two-speaker
   command is always decoded from cmd0/cmd1 and selected by fine-tuned
   ERes2NetV2 cosine.  Raw confidence, keyword score and sv_diff no longer
   participate in routing, so Raw and separated command ASR are never both
   executed for the same sample.

Everything else (wake anchor, raw fallbacks, output schema) is identical
to szy_v1.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from gate import ModelScopeEmbedder, read_wav
from moss_single_speaker_v2 import BothSingleRouter
from infer import (
    FireRedLLMRecognizer,
    FireRedRecognizer,
    Sample,
    levenshtein,
    normalize_transcript,
    read_many_jsonl,
    resolve_model_path,
    submission_id,
)
from firered_nbest import conservative_mbr_rerank
from moss_candidate import (
    WakeAnchorDecision,
    select_wake_anchor,
)
from szy_v1 import (
    MossFormer2Separator,
    MossFormer2TorchSeparator,
    SzyConfig,
    summarize,
    write_outputs,
)


FUSION_FEATURE_KEYS = [
    "pre_raw", "ft_raw", "pre_sep", "ft_sep", "ch_sim_pre", "ch_sim_ft",
]


@dataclass(frozen=True)
class SzyV2Config(SzyConfig):
    """szy_v2 配置：继承 v1 全部字段，门控相关换成融合器。"""

    fusion_backend: str = "../dataset_dasynth/out/gate_fusion.json"
    finetuned_sv_ckpt: str = "../dataset_dasynth/out/ep6_datasetA_best.pt"
    fusion_threshold: float = -0.6469
    separation_cache: str = "outputs/szy_v2_cache/separation"
    # False 时只保留进程内缓存（供同一条音频的 N-best 复用），不读写 asr_cache.json。
    asr_cache_persistent: bool = True
    # ASR 后端："aed" = FireRedASR2-AED（现状基线）；"llm" = FireRedASR2-LLM（Qwen2-7B 解码）
    firered_asr_type: str = "aed"
    firered_llm_batch_size: int = 4
    # Beam search configuration
    firered_beam_size: int = 1
    firered_beam_mode: str = "always"  # "always" or "conditional"
    firered_conditional_gate_margin: float = 0.15
    firered_conditional_switch_margin: float = 0.0
    firered_conditional_confidence_threshold: float = 0.7
    # N-best reranking configuration (对齐 zhy_workspace)
    firered_nbest_mode: str = "off"  # "off" or "conservative_mbr"
    firered_nbest_beam_size: int = 5
    firered_nbest_count: int = 3
    firered_nbest_min_confidence_gain: float = 0.03
    firered_nbest_min_mbr_risk_reduction: float = 0.05
    # ELM (External Language Model) configuration
    firered_elm_dir: str | None = None
    firered_elm_weight: float = 0.0
    # ELM 重解码使用的 beam。ELM 不再参与主解码，只在路由选定后对选中
    # 通道重解码一次，因此它的 beam 与主解码的 firered_beam_size 解耦。
    firered_elm_beam_size: int = 3
    # 音色切换规则的三个切点。默认值是 ELM 融入主解码时代标定的；ELM 后处理化
    # 之后 confidence 量程改变，生产配置里给的是在新量程上重扫的值。
    firered_switch_confidence_high: float = 0.70
    firered_switch_confidence_low: float = 0.57
    firered_switch_sv_diff_tau: float = 0.05
    # 命令通道切换策略：
    #   "timbre_rule" —— should_switch_command_timbre(confidence x keyword x sv_diff)
    #   "always"      —— 无条件采用分离通道（ft 余弦最高的一路），规则短路为恒真
    # 默认保持 timbre_rule，历史配置（无此键）可原样复现。
    command_switch_mode: str = "timbre_rule"
    # 双单人独立路由。关闭时可严格复现历史 V100；生产配置显式开启。
    both_single_mode: str = "off"  # "off" or "sft_raw_cosine"
    both_single_backend: str = "configs/moss_single_speaker_router_v2.json"
    both_single_sft_threshold: float = 0.29


def load_szy_v2_config(path: str | Path) -> SzyV2Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = SzyV2Config.__dataclass_fields__
    # Filter out unknown fields with warning instead of raising error
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        print(f"Warning: ignoring unknown config keys: {unknown}")
        payload = {k: v for k, v in payload.items() if k in fields}
    return SzyV2Config(**payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
     return float(np.dot(left, right))


class AsrCache:
    """Persistent cache for ASR transcription results, keyed by audio file path."""

    def __init__(self, cache_path: Path, batch_writes: int = 100, persistent: bool = True):
        self.cache_path = cache_path
        self.batch_writes = max(int(batch_writes), 1)
        self.persistent = bool(persistent)
        self.dirty_entries = 0
        self.cache: dict[str, dict] = {}
        if self.persistent and cache_path.exists():
            self.load()

    def load(self):
        """Load cache from disk."""
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            self.cache = json.load(f)

    def save(self):
        """Save cache to disk."""
        if not self.persistent:
            self.dirty_entries = 0
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        self.dirty_entries = 0

    def save_if_needed(self) -> None:
        """每累计指定数量的新转写结果后写盘，避免每次解码都重写整个 JSON。"""
        if self.dirty_entries >= self.batch_writes:
            self.save()

    def flush(self) -> None:
        """程序正常结束时强制保存最后不足一个批次的缓存。"""
        if self.dirty_entries:
            self.save()

    def get(self, path_str: str) -> dict | None:
        """Get cached result for a path."""
        return self.cache.get(path_str)

    def put(self, path_str: str, result: dict):
        """Store result for a path."""
        self.cache[path_str] = result
        self.dirty_entries += 1


class FusionGate:
    """双前端线性融合门控：预训练 + 分离域微调 ERes2NetV2 的 6 维余弦特征。"""

    def __init__(
        self,
        sv_model: str,
        finetuned_ckpt: str | Path,
        fusion_backend: str | Path,
        threshold: float,
        device: str,
        model_root: str | Path | None = None,
    ) -> None:
        import torch

        self.threshold = float(threshold)
        self.pretrained = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
        # 微调模型：同一基座结构，覆盖 encoder 权重
        self.finetuned = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
        checkpoint = torch.load(str(finetuned_ckpt), map_location="cpu", weights_only=False)
        self.finetuned.pipeline.model.embedding_model.load_state_dict(checkpoint["encoder"])

        bundle = json.loads(Path(fusion_backend).read_text(encoding="utf-8"))
        if list(bundle["features"]) != FUSION_FEATURE_KEYS:
            raise ValueError(f"fusion backend 特征顺序不符：{bundle['features']}")
        self.mean = np.asarray(bundle["mean"], dtype=np.float32)
        self.std = np.asarray(bundle["std"], dtype=np.float32)
        self.weight = np.asarray(bundle["weight"], dtype=np.float32)
        self.bias = float(bundle["bias"])

    def score(
        self,
        anchor_wave: np.ndarray,
        raw_wake_wave: np.ndarray,
        raw_command_wave: np.ndarray,
        command0_wave: np.ndarray,
        command1_wave: np.ndarray,
        command_source_empty: bool,
    ) -> tuple[bool, float, dict[str, float]]:
        waves = [anchor_wave, raw_wake_wave, raw_command_wave, command0_wave, command1_wave]
        # 两套 ERes2NetV2 的声学前端完全相同，只是 embedding_model 权重不同。
        # Fbank 只计算一次；两套模型仍串行前向，不增加并发激活显存。
        shared_features = self.pretrained.prepare_waveforms(waves)
        values: dict[str, float] = {}
        for tag, embedder in (("pre", self.pretrained), ("ft", self.finetuned)):
            e_anchor, e_raw_wake, e_raw_cmd, e_cmd0, e_cmd1 = embedder.extract_features(shared_features)
            candidates = [_cosine(e_anchor, e_cmd0), _cosine(e_anchor, e_cmd1)]
            if command_source_empty:
                candidates.append(_cosine(e_anchor, e_raw_cmd))
            values[f"{tag}_raw"] = _cosine(e_raw_wake, e_raw_cmd)
            values[f"{tag}_sep"] = max(candidates)
            values[f"ch_sim_{tag}"] = _cosine(e_cmd0, e_cmd1)
            # 候选选择（ft 直接余弦）复用，避免二次提 embedding
            values[f"{tag}_cos_cmd0"] = _cosine(e_anchor, e_cmd0)
            values[f"{tag}_cos_cmd1"] = _cosine(e_anchor, e_cmd1)
            # 音色触发器的分离收益量：cos(anchor, raw_cmd)
            values[f"{tag}_rawcmd"] = _cosine(e_anchor, e_raw_cmd)
        x = (np.asarray([values[key] for key in FUSION_FEATURE_KEYS], dtype=np.float32) - self.mean) / self.std
        score = float(x @ self.weight + self.bias)
        return score >= self.threshold, score, values

    def finetuned_raw_cosine(self, wake_wave: np.ndarray, command_wave: np.ndarray) -> float:
        """只用微调 ERes2NetV2 计算原始唤醒与原始命令的余弦。"""
        wake_embedding, command_embedding = self.finetuned.extract_waveforms(
            [wake_wave, command_wave]
        )
        return _cosine(wake_embedding, command_embedding)


class SzyV2Runner:
    """szy_v1 流程 + 融合门控。候选选择部分与 v1 逐行一致。"""

    def __init__(self, config: SzyV2Config, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.cache_dir = Path(config.separation_cache)
        if not self.cache_dir.is_absolute():
            self.cache_dir = (Path.cwd() / self.cache_dir).resolve()
        
        # 始终保留进程内缓存；是否跨进程落盘由配置控制。
        asr_cache_path = self.cache_dir.parent / "asr_cache.json"
        self.asr_cache = AsrCache(
            asr_cache_path, config.batch_writes, config.asr_cache_persistent,
        )
        
        print(f"[初始化] 加载 FusionGate (预训练 + 微调 ERes2NetV2)...", flush=True)
        self.gate = FusionGate(
            config.sv_model,
            config.finetuned_sv_ckpt,
            config.fusion_backend,
            config.fusion_threshold,
            config.sv_device or config.device,
            config.model_root,
        )
        print(f"[初始化] ✓ FusionGate 加载完成", flush=True)
        if config.both_single_mode not in {"off", "sft_raw_cosine"}:
            raise ValueError("both_single_mode must be off or sft_raw_cosine")
        self.both_single_router = (
            BothSingleRouter(config.both_single_backend, config.both_single_sft_threshold)
            if config.both_single_mode == "sft_raw_cosine" else None
        )
        if self.both_single_router is not None:
            print(
                "[初始化] ✓ 双单人独立路由已启用 "
                f"(single_p>={self.both_single_router.single_threshold:.6f}, "
                f"SFT cosine>={self.both_single_router.same_speaker_threshold:.3f})",
                flush=True,
            )
        
        # 候选选择复用门控的 ft 逐通道余弦（ft 直接余弦规则），无需 cam++ 前端
        print(f"[初始化] 加载 MossFormer2 分离器...", flush=True)
        if str(config.moss_model).endswith(".pt"):
            # dasynth 域适配微调的 torch 分离器（整波推理，同接口）
            self.separator = MossFormer2TorchSeparator(config.moss_model, config.device)
        else:
            self.separator = MossFormer2Separator(config.moss_model, config.device, config.moss_provider)
        print(f"[初始化] ✓ MossFormer2 分离器加载完成", flush=True)
        
        print(f"[初始化] 加载 FireRed ASR...", flush=True)
        if config.firered_asr_type == "llm":
            self.asr: Any = FireRedLLMRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
                batch_size=config.firered_llm_batch_size,
            )
            self.wake_asr = self.asr  # LLM mode: use same instance
        else:
            # 唤醒与命令通道共用一份 AED 权重。唤醒侧原本是独立的
            # FireRedRecognizer(beam=1, 无 ELM)，等价于在共享实例上临时把
            # beam 压到 1 且把 elm_weight 置 0（见 _transcribe_wake）。
            # 省掉整份 AED 权重的重复驻留，也消除两份模型同时在显存里
            # 带来的数值/内核选择抖动。
            self.asr = FireRedRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
                beam_size=config.firered_beam_size,
                nbest=config.firered_beam_size,
                elm_dir=config.firered_elm_dir,
                elm_weight=config.firered_elm_weight,
            )
            self.wake_asr = self.asr
            # ELM 与主解码共用同一份 AED 权重（Phase 1 已验证）。主解码一律
            # 用 temporary_elm_weight(0.0) 关闭 ELM，ELM 只作为选择后的重解码
            # 通道，因此不会污染 confidence，也就不会干扰切换规则。
            self.elm_asr = self.asr if config.firered_elm_dir else None
            print(f"[初始化] ✓ FireRed ASR 加载完成 (唤醒/命令共享同一份 AED 权重)", flush=True)
            print(f"[初始化]   - Wake ASR: beam=1, ELM off (临时切换)", flush=True)
            print(f"[初始化]   - Command ASR: beam={config.firered_beam_size} + ELM", flush=True)
            print(f"[初始化]   - N-best mode: {config.firered_nbest_mode}", flush=True)
            if config.firered_elm_dir:
                print(f"[初始化]   - ELM: enabled (weight={config.firered_elm_weight})", flush=True)
            else:
                print(f"[初始化]   - ELM: disabled", flush=True)

    def _transcribe_with_asr(
        self, paths: list[Path], asr_instance, *, cache_tag: str = "default",
    ) -> list[dict[str, Any]]:
        """Transcribe paths using a specific ASR instance.

        ``cache_tag`` identifies the decode configuration (beam + ELM weight).
        The same wav is now decoded under more than one configuration — the
        primary pass with ELM off and the post-selection pass with ELM on — so
        keying the cache on the path alone would silently return a transcript
        produced by the wrong configuration.
        """
        from infer import suppress_model_output

        if not paths:
            return []
        
        # Check cache first
        results = []
        uncached_paths = []
        uncached_indices = []
        
        for idx, path in enumerate(paths):
            path_str = str(path)
            cached = self.asr_cache.get(f"{path_str}|{cache_tag}")
            if cached is not None:
                results.append((idx, cached))
            else:
                uncached_paths.append(path)
                uncached_indices.append(idx)
        
        # If all cached, return early
        if not uncached_paths:
            return [r for _, r in sorted(results, key=lambda x: x[0])]
        
        # Transcribe uncached paths
        utterance_ids = [f"szy2_{index}" for index in range(len(uncached_paths))]
        with suppress_model_output():
            rows = asr_instance._model.transcribe(utterance_ids, [str(path) for path in uncached_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(uncached_paths):
            raise RuntimeError("FireRed returned an unexpected result count")
        
        # Process and cache results
        for idx, identifier in enumerate(utterance_ids):
            row = mapped[identifier]
            confidence = row.get("confidence")
            try:
                confidence = float(confidence)
                if not np.isfinite(confidence):
                    confidence = 1.0
            except (TypeError, ValueError):
                confidence = 1.0
            
            result = {
                "text": normalize_transcript(str(row.get("text", ""))),
                "confidence": confidence,
            }
            # FireRed 在主 beam=3 解码时已经生成了 N-best。把它与 top-1 一起
            # 保存，最终通道选定后即可直接做 MBR，无需对同一音频再次解码。
            normalized_nbest: list[dict[str, Any]] = []
            for hypothesis in row.get("nbest", []) or []:
                if not isinstance(hypothesis, dict):
                    continue
                hypothesis_confidence = hypothesis.get("confidence", 0.0)
                try:
                    hypothesis_confidence = float(hypothesis_confidence)
                    if not np.isfinite(hypothesis_confidence):
                        hypothesis_confidence = 0.0
                except (TypeError, ValueError):
                    hypothesis_confidence = 0.0
                normalized_nbest.append({
                    "text": normalize_transcript(str(hypothesis.get("text", ""))),
                    "confidence": hypothesis_confidence,
                })
            if normalized_nbest:
                result["nbest"] = normalized_nbest
            
            # Cache the result
            path_str = str(uncached_paths[idx])
            self.asr_cache.put(f"{path_str}|{cache_tag}", result)
            results.append((uncached_indices[idx], result))
        
        # 只在累计达到 batch_writes 后写盘；最后一个不足批次由 main 强制刷新。
        self.asr_cache.save_if_needed()
        
        # Return in original order
        return [r for _, r in sorted(results, key=lambda x: x[0])]

    def _primary_transcribe(
        self, paths: list[Path], *, beam_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Primary AED decode with ELM disabled.

        Every decision-carrying transcript (wake anchors, separated command
        channels, the raw command) comes from here.  ELM is held at weight 0
        so ``confidence`` keeps the scale the switch rule was calibrated on;
        ELM is applied later, once, to the selected channel only.
        """
        beam = self.config.firered_beam_size if beam_size is None else beam_size
        tag = f"b{beam}|elm0"
        if not hasattr(self.asr, "temporary_elm_weight"):
            return self._transcribe_with_asr(paths, self.asr, cache_tag=tag)
        with self.asr.temporary_elm_weight(0.0):
            if beam_size is None:
                return self._transcribe_with_asr(paths, self.asr, cache_tag=tag)
            with self.asr.temporary_beam_size(beam_size):
                return self._transcribe_with_asr(paths, self.asr, cache_tag=tag)

    def _transcribe(self, paths: list[Path]) -> list[dict[str, Any]]:
        """Transcribe the separated command channels (primary beam, no ELM)."""
        return self._primary_transcribe(paths)

    def _transcribe_wake(self, paths: list[Path]) -> list[dict[str, Any]]:
        """Transcribe wake channels at beam=1 with no ELM (stable anchors)."""
        return self._primary_transcribe(paths, beam_size=1)

    def _elm_redecode(self, audio: Path, baseline_text: str) -> tuple[str, bool, str | None]:
        """Re-decode the selected channel with ELM enabled.

        Runs after routing, so it can no longer alter the gate or the switch
        decision.  An empty baseline is left untouched: a language model must
        not invent a transcript where the acoustic model produced none.
        """
        if self.elm_asr is None or not baseline_text:
            return baseline_text, False, None
        beam = self.config.firered_elm_beam_size
        weight = self.config.firered_elm_weight
        tag = f"b{beam}|elm{weight:g}"
        with self.elm_asr.temporary_elm_weight(weight):
            with self.elm_asr.temporary_beam_size(beam):
                rows = self._transcribe_with_asr([audio], self.elm_asr, cache_tag=tag)
        elm_text = rows[0]["text"]
        if not elm_text:
            return baseline_text, False, elm_text
        return elm_text, True, elm_text

    def _nbest_redecode(
        self, audio: Path, baseline_text: str, baseline_confidence: float,
    ) -> dict[str, Any]:
        """Conservative MBR rerank of the selected channel."""
        disabled = {"text": baseline_text, "nbest_applied": False, "nbest_reason": "nbest_disabled"}
        if self.config.firered_nbest_mode == "off" or not baseline_text:
            return disabled
        # 主解码和这里使用相同的 beam、nbest 与 ELM=0 配置。优先复用主解码
        # 已经产生的候选；兼容旧缓存，若其中没有 nbest 字段则执行历史二次解码。
        primary_tag = f"b{self.config.firered_beam_size}|elm0"
        cached = self.asr_cache.get(f"{str(audio)}|{primary_tag}")
        hypotheses = list(cached.get("nbest", [])) if cached is not None else []
        if not hypotheses:
            try:
                with self.asr.temporary_elm_weight(0.0):
                    hypotheses = self.asr.transcribe_many_nbest([audio])[0]
            except Exception as error:
                return {**disabled, "nbest_reason": f"nbest_error_keep_baseline:{type(error).__name__}"}
        if not hypotheses:
            return {**disabled, "nbest_reason": "no_hypotheses"}
        baseline = {"text": baseline_text, "confidence": baseline_confidence}
        if len(hypotheses) == 1:
            return {**disabled, "nbest_reason": "single_hypothesis"}
        return conservative_mbr_rerank(
            baseline=baseline,
            nbest=[h for h in hypotheses if h["text"] != baseline_text],
            min_confidence_gain=self.config.firered_nbest_min_confidence_gain,
            min_mbr_risk_reduction=self.config.firered_nbest_min_mbr_risk_reduction,
        )

    def _postprocess_text(
        self, audio: Path, baseline_text: str, baseline_confidence: float,
    ) -> dict[str, Any]:
        """N-best rerank then ELM, applied to whichever channel routing chose.

        Previously N-best only ever ran on the raw command, so it was
        discarded whenever the switch rule picked a separated channel.
        """
        nbest = self._nbest_redecode(audio, baseline_text, baseline_confidence)
        final_text, elm_applied, elm_text = self._elm_redecode(audio, nbest["text"])
        return {
            "content": final_text,
            "baseline_content": baseline_text,
            "elm_applied": elm_applied,
            "elm_content": elm_text,
            "nbest_applied": nbest["text"] != baseline_text,
            "nbest_reason": nbest.get("nbest_reason", "unknown"),
        }

    def flush_caches(self) -> None:
        """保存最后不足 batch_writes 条的 ASR 缓存。"""
        self.asr_cache.flush()

    def _paths_for(self, sample: Sample) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
        item_id = submission_id(sample, self.config.id_mode).replace("/", "_")
        wake_prefix = self.cache_dir / f"{item_id}_wake"
        command_prefix = self.cache_dir / f"{item_id}_command"
        return (
            self.separator.separate(sample.wake_audio, wake_prefix),
            self.separator.separate(sample.command_audio, command_prefix),
        )

    def process(self, sample: Sample) -> dict[str, Any]:
        started = time.perf_counter()
        item_id = submission_id(sample, self.config.id_mode)
        result: dict[str, Any] = {
            "id": item_id,
            "content": "",
            "label": sample.label,
            "wake_text": sample.wake_text,
            "sample_type": "positive" if sample.label is not None else "negative",
            "sv_score": "",
            "accepted": False,
            "sv_accept_threshold": f"{self.config.fusion_threshold:.6f}",
            "route": "fusion_reject",
            "selected_candidate": "",
            "switch_triggered": False,
            "command_source_empty": None,
            "wake_selected_source": None,
            "wake_raw_fallback": None,
            "wake_text_tie": None,
            "raw_confidence": None,
            "chars_per_sec": None,
            "command_keyword_score": None,
            "error": None,
            "both_single": False,
            "both_single_wake_probability": None,
            "both_single_command_probability": None,
            "both_single_sft_raw_cosine": None,
            "command_is_single": None,
            "separated_command_source_empty": None,
        }
        try:
            (wake_paths, command_paths) = self._paths_for(sample)
            # 先从波形判断唤醒/命令是否均为单人。该判断不依赖 ASR 文本。
            raw_wake_wave = read_wav(sample.wake_audio)
            raw_command_wave = read_wav(sample.command_audio)
            wake0_wave, wake1_wave = read_wav(wake_paths[0]), read_wav(wake_paths[1])
            command0_wave, command1_wave = read_wav(command_paths[0]), read_wav(command_paths[1])
            both_single = (
                self.both_single_router.decide_both(
                    raw_wake_wave, wake0_wave, wake1_wave,
                    raw_command_wave, command0_wave, command1_wave,
                )
                if self.both_single_router is not None else None
            )
            if both_single is not None:
                result.update({
                    "both_single": bool(both_single.is_both_single),
                    "both_single_wake_probability": f"{both_single.wake.probability:.6f}",
                    "both_single_command_probability": f"{both_single.command.probability:.6f}",
                    "command_is_single": bool(both_single.command.is_single),
                })

            # 双单人是独立的短路分支：不再使用六维门控或分离通道文本。
            if both_single is not None and both_single.is_both_single:
                score = self.gate.finetuned_raw_cosine(raw_wake_wave, raw_command_wave)
                accepted = self.both_single_router.accept_same_speaker(score)
                result.update({
                    "sv_score": f"{score:.6f}",
                    "sv_accept_threshold": f"{self.both_single_router.same_speaker_threshold:.6f}",
                    "accepted": bool(accepted),
                    "both_single_sft_raw_cosine": f"{score:.6f}",
                    "route": "both_single_raw" if accepted else "both_single_reject",
                    "selected_candidate": "orig" if accepted else "",
                    "selection_reason": "both_single_sft_raw_cosine",
                    "switch_triggered": False,
                })
                if not accepted:
                    result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                    return result
                raw_command = self._primary_transcribe([sample.command_audio])[0]
                result.update({
                    **self._postprocess_text(
                        sample.command_audio, raw_command["text"], raw_command["confidence"],
                    ),
                    "raw_confidence": f"{raw_command['confidence']:.6f}",
                    "raw_text": raw_command["text"],
                })
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # Wake channels: beam=1 (stable anchor selection)
            wake0, wake1 = self._transcribe_wake([wake_paths[0], wake_paths[1]])
            # 命令路由完全由声学单双人分类器决定，不再依赖 Raw ASR 置信度、
            # 命令词或 sv_diff。单人命令只解码 Raw；双人命令只解码分离源。
            command_is_single = bool(
                both_single is not None and both_single.command.is_single
            )
            # 命令 ASR 延迟到门控接受、余弦选源完成之后。此处不解码 Raw、
            # cmd0 或 cmd1，因此被拒识的样本不会浪费任何命令 ASR 计算。
            command0: dict[str, Any] | None = None
            command1: dict[str, Any] | None = None
            # 门控唤醒锚点：生产同一规则（一路为空回退原始 kws）
            # 注：曾验证"非空通道直选"（PR/CER 更优），但在计分公式 (1−CER+RR)/2 下
            # RR 权重高，raw 回退版公式分更高（0.7977 vs 0.7939），故保持 raw 回退。
            gate_anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
            anchor_path = sample.wake_audio if gate_anchor.raw_wake_fallback else wake_paths[gate_anchor.selected_source]
            # 只有单人命令才允许门控把 Raw CMD 纳入候选；双人命令保持恒切。
            gate_use_raw_command = command_is_single

            anchor_wave = (
                raw_wake_wave if gate_anchor.raw_wake_fallback
                else (wake0_wave, wake1_wave)[gate_anchor.selected_source]
            )
            accepted, score, gate_values = self.gate.score(
                anchor_wave,
                raw_wake_wave,
                raw_command_wave,
                command0_wave,
                command1_wave,
                gate_use_raw_command,
            )
            result.update({
                "sv_score": f"{score:.6f}",
                "accepted": bool(accepted),
                "gate_features": {k: f"{gate_values[k]:.6f}" for k in FUSION_FEATURE_KEYS},
                "wake_selected_source": int(gate_anchor.selected_source),
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                "wake_raw_fallback": bool(gate_anchor.raw_wake_fallback),
                "wake_text_tie": bool(gate_anchor.text_tie),
                "wake_contrast_source": int(gate_anchor.contrast_source),
                "wake_reason": gate_anchor.reason,
                "wake_source0_sim": f"{gate_anchor.source0_sim:.6f}" if gate_anchor.source0_sim is not None else None,
                "wake_source1_sim": f"{gate_anchor.source1_sim:.6f}" if gate_anchor.source1_sim is not None else None,
                # ASR 尚未执行，分离源是否为空要在双人命令分支解码后才能确定。
                "command_source_empty": None,
                "command_is_single": bool(command_is_single),
                "separated_command_source_empty": None,
            })
            if not accepted:
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            if command_is_single:
                # 单人命令：门控接受后只解码一次 Raw CMD。
                raw_command = self._primary_transcribe([sample.command_audio])[0]
                result.update({
                    "content": raw_command["text"],
                    "route": "raw_command_single",
                    "selected_candidate": "orig",
                    "selection_reason": "acoustic_command_single",
                    "switch_triggered": False,
                    "switch_mode": "acoustic_single_double",
                    "raw_confidence": f"{raw_command['confidence']:.6f}",
                    "raw_text": raw_command["text"],
                    "command_source0_text": "",
                    "command_source1_text": "",
                    "ft_cos_ms0": f"{gate_values['ft_cos_cmd0']:.6f}",
                    "ft_cos_ms1": f"{gate_values['ft_cos_cmd1']:.6f}",
                })
                result.update(self._postprocess_text(
                    sample.command_audio, raw_command["text"], raw_command["confidence"],
                ))
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # 双人命令恒切：先按 FT 余弦选源，只对选中通道执行一次 ASR。
            # 仅当首选通道文本为空时，才补充解码另一通道；绝不解码 Raw CMD。
            anchor = gate_anchor
            preferred_index = (
                0 if gate_values["ft_cos_cmd0"] >= gate_values["ft_cos_cmd1"] else 1
            )
            selected_index = preferred_index
            selected_result = self._transcribe([command_paths[selected_index]])[0]
            if selected_index == 0:
                command0 = selected_result
            else:
                command1 = selected_result
            reason = "ft_direct_cosine_lazy_asr"

            if not selected_result["text"]:
                # 首选分离源识别为空时，才付出第二次 ASR 的开销并改选另一通道。
                selected_index = 1 - preferred_index
                selected_result = self._transcribe([command_paths[selected_index]])[0]
                if selected_index == 0:
                    command0 = selected_result
                else:
                    command1 = selected_result
                reason = "preferred_empty_fallback_other_separated"

            selected = f"ms{selected_index}"
            selected_audio = command_paths[selected_index]
            selected_confidence = selected_result["confidence"]
            selected_text = selected_result["text"]
            final_source_empty = not bool(selected_text)
            result.update({
                # 路由已定：N-best 重排 + ELM 只作用在选中的那一路上
                **self._postprocess_text(
                    selected_audio, selected_text, selected_confidence,
                ),
                "selected_candidate": selected,
                "selection_reason": reason,
                "route": "moss_candidate",
                "switch_triggered": True,
                "switch_mode": "acoustic_single_double",
                "wake_contrast_source": int(anchor.contrast_source),
                "wake_text_tie": bool(anchor.text_tie),
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                # 未执行 ASR 的通道保留 None，便于确认延迟解码确实生效。
                "command_source0_text": command0["text"] if command0 is not None else None,
                "command_source1_text": command1["text"] if command1 is not None else None,
                "command_source_empty": final_source_empty,
                "separated_command_source_empty": final_source_empty,
                "ft_cos_ms0": f"{gate_values['ft_cos_cmd0']:.6f}",
                "ft_cos_ms1": f"{gate_values['ft_cos_cmd1']:.6f}",
            })
        except Exception as error:  # preserve raw ASR as the operational fallback
            result["error"] = f"{type(error).__name__}: {error}"
            try:
                result["content"] = self.asr.transcribe_many([sample.command_audio])[0]
                result["route"] = "raw_fallback_error"
                result["selected_candidate"] = "orig"
            except Exception as raw_error:
                result["error"] += f"; raw_fallback={type(raw_error).__name__}: {raw_error}"
                result["route"] = "empty_fallback_error"
        result["item_duration"] = f"{time.perf_counter() - started:.6f}"
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SZY v2 fusion-gate MossFormer2 target-speaker ASR")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--config", default="configs/szy_v2_firered.json")
    parser.add_argument("--output", default="outputs/szy_v2_submission.json")
    parser.add_argument("--final-output", default="outputs/szy_v2_detail.json")
    parser.add_argument("--checkpoint-output", default="outputs/szy_v2_checkpoint.json")
    parser.add_argument("--manifest-output", default="outputs/szy_v2_manifest.json")
    parser.add_argument("--model-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_szy_v2_config(args.config)
    if args.model_root:
        config = SzyV2Config(**{**config.__dict__, "model_root": args.model_root})
    samples = read_many_jsonl(args.jsonl, args.data_root)
    if args.limit is not None:
        import random
        random.seed(42)
        pos = [s for s in samples if s.label is not None]
        neg = [s for s in samples if s.label is None]
        n_pos = round(args.limit * len(pos) / len(samples))
        n_neg = args.limit - n_pos
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))
        samples = random.sample(pos, n_pos) + random.sample(neg, n_neg)
        random.shuffle(samples)
        print(f"[limit] {args.limit} samples: {n_pos} pos + {n_neg} neg (ratio {len(pos)}/{len(pos)+len(neg)})")
    checkpoint_path = Path(args.checkpoint_output)
    results: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = list(checkpoint.get("results", []))
        elapsed_before = float(checkpoint.get("duration", 0.0))
        completed = {str(item["id"]) for item in results}
        samples = [sample for sample in samples if submission_id(sample, config.id_mode) not in completed]
    runner = SzyV2Runner(config, Path(args.output).parent)
    started = time.perf_counter()
    pending_checkpoint_rows = 0
    for sample in tqdm(samples, desc="szy_v2", unit="utt"):
        result = runner.process(sample)
        results.append(result)
        pending_checkpoint_rows += 1
        elapsed = elapsed_before + time.perf_counter() - started
        if pending_checkpoint_rows >= max(config.batch_writes, 1):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps({"version": "szy_v2", "duration": elapsed, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
            pending_checkpoint_rows = 0
    duration = elapsed_before + time.perf_counter() - started
    # 正常结束时保存尾部不足一个批次的结果；恰好落盘则不重复重写。
    if pending_checkpoint_rows or not checkpoint_path.is_file():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"version": "szy_v2", "duration": duration, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    runner.flush_caches()
    summary = summarize(results, duration)
    summary["route"] = "szy_v2"
    write_outputs(results, summary, Path(args.output), Path(args.final_output))
    manifest = {
        "version": "szy_v2",
        "config": config.__dict__,
        "data_root": str(Path(args.data_root).resolve()),
        "jsonl": [str(Path(path).resolve()) for path in args.jsonl],
        "sample_count": len(results),
        "moss_model_sha256": _sha256(Path(config.moss_model).resolve()),
        "fusion_backend_sha256": _sha256(Path(config.fusion_backend).resolve()),
        "finetuned_sv_ckpt_sha256": _sha256(Path(config.finetuned_sv_ckpt).resolve()),
        "output": str(Path(args.output).resolve()),
        "detail": str(Path(args.final_output).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
