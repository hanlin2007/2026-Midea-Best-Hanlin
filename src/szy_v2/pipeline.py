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
3. The switch trigger is the timbre rule (should_switch_command_timbre):
   confidence x keyword x sv_diff, where sv_diff = max cos(anchor, cmd_ms)
   - cos(anchor, raw_cmd) measures whether separation makes the target
   voice more prominent than in the raw mixture — the physical mechanism
   of switching itself.  Replaces v1's chars_per_sec proxy (absolute
   speaking-rate assumption) and the intermediate d_sel text-comparison
   variant.  DatasetA: CER 0.3058 vs 0.3092 (text) vs 0.3181 (cps);
   flat basins over tau in [0.02, 0.08] and conf cutoffs in
   [0.50, 0.70] x [0.65, 0.80].

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

from midea_qiji.gate import ModelScopeEmbedder, read_wav
from midea_qiji.infer import (
    FireRedLLMRecognizer,
    FireRedRecognizer,
    Sample,
    levenshtein,
    normalize_transcript,
    read_many_jsonl,
    resolve_model_path,
    submission_id,
)
from midea_qiji.moss_candidate import (
    WakeAnchorDecision,
    select_wake_anchor,
    should_switch_command_timbre,
)
from szy_v1.pipeline import (
    CMD_KW,
    MossFormer2Separator,
    MossFormer2TorchSeparator,
    SzyConfig,
    _command_features,
    _duration,
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
    # ASR 后端："aed" = FireRedASR2-AED（现状基线）；"llm" = FireRedASR2-LLM（Qwen2-7B 解码）
    firered_asr_type: str = "aed"
    firered_llm_batch_size: int = 4


def load_szy_v2_config(path: str | Path) -> SzyV2Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = SzyV2Config.__dataclass_fields__
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"unknown szy_v2 config keys: {unknown}")
    return SzyV2Config(**payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


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
        values: dict[str, float] = {}
        for tag, embedder in (("pre", self.pretrained), ("ft", self.finetuned)):
            e_anchor, e_raw_wake, e_raw_cmd, e_cmd0, e_cmd1 = embedder.extract_waveforms(waves)
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


class SzyV2Runner:
    """szy_v1 流程 + 融合门控。候选选择部分与 v1 逐行一致。"""

    def __init__(self, config: SzyV2Config, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.cache_dir = Path(config.separation_cache)
        if not self.cache_dir.is_absolute():
            self.cache_dir = (Path.cwd() / self.cache_dir).resolve()
        self.gate = FusionGate(
            config.sv_model,
            config.finetuned_sv_ckpt,
            config.fusion_backend,
            config.fusion_threshold,
            config.sv_device or config.device,
            config.model_root,
        )
        # 候选选择复用门控的 ft 逐通道余弦（ft 直接余弦规则），无需 cam++ 前端
        if str(config.moss_model).endswith(".pt"):
            # dasynth 域适配微调的 torch 分离器（整波推理，同接口）
            self.separator = MossFormer2TorchSeparator(config.moss_model, config.device)
        else:
            self.separator = MossFormer2Separator(config.moss_model, config.device, config.moss_provider)
        if config.firered_asr_type == "llm":
            self.asr: Any = FireRedLLMRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
                batch_size=config.firered_llm_batch_size,
            )
        else:
            self.asr = FireRedRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
            )

    @staticmethod
    def _transcribe(asr: Any, paths: list[Path]) -> list[dict[str, Any]]:
        from midea_qiji.infer import suppress_model_output

        if not paths:
            return []
        utterance_ids = [f"szy2_{index}" for index in range(len(paths))]
        with suppress_model_output():
            rows = asr._model.transcribe(utterance_ids, [str(path) for path in paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(paths):
            raise RuntimeError("FireRed returned an unexpected result count")
        output = []
        for identifier in utterance_ids:
            row = mapped[identifier]
            confidence = row.get("confidence")
            try:
                confidence = float(confidence)
                if not np.isfinite(confidence):
                    confidence = 1.0
            except (TypeError, ValueError):
                confidence = 1.0
            output.append({
                "text": normalize_transcript(str(row.get("text", ""))),
                "confidence": confidence,
            })
        return output

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
        }
        try:
            (wake_paths, command_paths) = self._paths_for(sample)
            wake0, wake1, command0, command1 = self._transcribe(
                self.asr,
                [wake_paths[0], wake_paths[1], command_paths[0], command_paths[1]],
            )
            # 门控唤醒锚点：生产同一规则（一路为空回退原始 kws）
            # 注：曾验证"非空通道直选"（PR/CER 更优），但在计分公式 (1−CER+RR)/2 下
            # RR 权重高，raw 回退版公式分更高（0.7977 vs 0.7939），故保持 raw 回退。
            gate_anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
            anchor_path = sample.wake_audio if gate_anchor.raw_wake_fallback else wake_paths[gate_anchor.selected_source]
            command_source_empty = not command0["text"] or not command1["text"]

            accepted, score, gate_values = self.gate.score(
                read_wav(anchor_path),
                read_wav(sample.wake_audio),
                read_wav(sample.command_audio),
                read_wav(command_paths[0]),
                read_wav(command_paths[1]),
                command_source_empty,
            )
            result.update({
                "sv_score": f"{score:.6f}",
                "accepted": bool(accepted),
                "gate_features": {k: f"{gate_values[k]:.6f}" for k in FUSION_FEATURE_KEYS},
                "wake_selected_source": int(gate_anchor.selected_source),
                "wake_raw_fallback": bool(gate_anchor.raw_wake_fallback),
                "command_source_empty": bool(command_source_empty),
            })
            if not accepted:
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # ---- 接受分支：raw ASR + 音色触发（与 v1 的语速规则不同） ----
            raw_command = self._transcribe(self.asr, [sample.command_audio])[0]
            chars_per_sec, command_keyword_score = _command_features(
                raw_command["text"], _duration(sample.command_audio)
            )
            # sv_diff：分离后目标嗓音浓度 − 原始混合中的浓度（复用门控已算余弦）
            sv_diff = max(gate_values["ft_cos_cmd0"], gate_values["ft_cos_cmd1"]) \
                - gate_values["ft_rawcmd"]
            switch = should_switch_command_timbre(
                confidence=raw_command["confidence"],
                command_keyword_score=command_keyword_score,
                sv_diff=sv_diff,
            )
            result.update({
                "content": raw_command["text"],
                "route": "moss_candidate" if switch else "raw",
                "selected_candidate": "orig",
                "selection_reason": "candidate_pending" if switch else "timbre_switch_rule_not_triggered",
                "switch_triggered": bool(switch),
                "raw_confidence": f"{raw_command['confidence']:.6f}",
                "chars_per_sec": f"{chars_per_sec:.6f}",
                "command_keyword_score": f"{command_keyword_score:.6f}",
                "sv_diff": f"{sv_diff:.6f}",
                # 供触发阈值离线重扫（与 switch 分支重复写入，值一致）
                "raw_text": raw_command["text"],
                "command_source0_text": command0["text"],
                "command_source1_text": command1["text"],
                "ft_cos_ms0": f"{gate_values['ft_cos_cmd0']:.6f}",
                "ft_cos_ms1": f"{gate_values['ft_cos_cmd1']:.6f}",
            })
            if not switch:
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # 候选选择：ft 直接余弦（与门控同一打分原理，复用门控已算的逐通道余弦）
            # 实测与生产 eres+cam 差分打平，但逻辑更一致，且避开对比通道污染。
            anchor = gate_anchor
            selected = "orig"
            reason = "candidate_pending"
            if command_source_empty:
                reason = "command_source_empty"
            else:
                selected = "ms0" if gate_values["ft_cos_cmd0"] >= gate_values["ft_cos_cmd1"] else "ms1"
                reason = "ft_direct_cosine"
            text_by_candidate = {
                "orig": raw_command["text"],
                "ms0": command0["text"],
                "ms1": command1["text"],
            }
            result.update({
                "content": text_by_candidate[selected] if selected != "orig" else raw_command["text"],
                "selected_candidate": selected,
                "selection_reason": reason,
                "wake_contrast_source": int(anchor.contrast_source),
                "wake_text_tie": bool(anchor.text_tie),
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                "command_source0_text": command0["text"],
                "command_source1_text": command1["text"],
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
        samples = samples[: args.limit]
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
    for sample in tqdm(samples, desc="szy_v2", unit="utt"):
        result = runner.process(sample)
        results.append(result)
        elapsed = elapsed_before + time.perf_counter() - started
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"version": "szy_v2", "duration": elapsed, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    duration = elapsed_before + time.perf_counter() - started
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
