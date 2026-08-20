"""End-to-end SZY v1 target-speaker ASR route.

The route keeps the production dual-MLP gate as the primary presence decision.
For every primary-gate accepted item it generates MossFormer2 candidates for
both wake and command audio, uses wake text to select the target anchor, and
uses same-separation differential SV evidence to select a command source only
when the raw FireRed result is considered unreliable.  Raw audio is retained
as the fallback for every failure or empty source.

This module deliberately runs from the DatasetA JSONL/audio inputs.  It does
not consume the historical ``exp_sep`` caches, so its output is suitable for
an end-to-end A/B run and includes a restartable checkpoint.

本模块只保留 v1 独有逻辑（SzyConfig、ONNX 分离器、主流程）；文件哈希 /
指令特征 / FireRed 转写 / 评测写出等共享工具已收敛到 szy_common，
语义与原实现逐字一致。注意：v1 依赖的私有资产（backend_dual.pt、
ONNX MossFormer2 模型）未随仓库分发，本路线处于休眠状态，仅作历史与
教学保留；现役路线请走 szy_v4。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from midea_qiji.audio import SAMPLE_RATE
from midea_qiji.gate import DualMLPGate, read_wav
from midea_qiji.infer import (
    Sample,
    read_many_jsonl,
    submission_id,
)
from midea_qiji.moss_candidate import select_wake_anchor, should_switch_command
from midea_qiji.recognizers import FireRedRecognizer
from szy_common.helpers import (
    command_features,
    duration_seconds,
    read_mono_16k,
    separation_paths_for,
    sha256_file,
    transcribe_firered_rows,
)
from szy_common.reporting import summarize, write_outputs  # noqa: F401  (re-export)


DEFAULT_MOSS_MODEL = (
    "../model/workspace/midea/models/modelscope/"
    "dengcunqin/speech_mossformer2_separation_temporal_16k/simple_model.onnx"
)


@dataclass(frozen=True)
class SzyConfig:
    device: str = "cuda:0"
    sv_device: str | None = None
    model_root: str | None = None
    sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    cam_model: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    dual_backend: str = "/home/hustdun/szy_workspace/sv_backend/backend_dual_v2_out/backend_dual.pt"
    dual_threshold: float = 0.4460311856655159
    firered_model: str = "../model/external/FireRedASR2-AED"
    firered_repo: str = "../model/external/FireRedASR2S"
    firered_vendor: str = "../model/runtime/vendor"
    moss_model: str = DEFAULT_MOSS_MODEL
    moss_provider: str = "auto"
    id_mode: str = "relative_stem"
    separation_cache: str = "outputs/szy_v1_cache/separation"
    batch_writes: int = 1


def load_szy_config(path: str | Path) -> SzyConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = SzyConfig.__dataclass_fields__
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"unknown szy_v1 config keys: {unknown}")
    return SzyConfig(**payload)


class MossFormer2Separator:
    """Small ONNX wrapper matching the historical MossFormer2 invocation."""

    def __init__(self, model_path: str | Path, device: str, provider: str = "auto") -> None:
        import onnxruntime as ort

        model = Path(model_path).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"MossFormer2 ONNX model not found: {model}")
        available = ort.get_available_providers()
        use_cuda = device.startswith("cuda") and "CUDAExecutionProvider" in available
        if provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is unavailable for MossFormer2")
        if provider == "cpu":
            providers = ["CPUExecutionProvider"]
        elif use_cuda or provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.model_path = model
        self.providers = providers
        self.session = ort.InferenceSession(str(model), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def separate(self, source: Path, output_prefix: Path) -> tuple[Path, Path]:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs = (output_prefix.with_name(output_prefix.name + "_ms0.wav"),
                   output_prefix.with_name(output_prefix.name + "_ms1.wav"))
        if all(path.is_file() for path in outputs):
            return outputs
        waveform = read_mono_16k(source)
        separated = self.session.run(None, {self.input_name: waveform[np.newaxis, :]})[0]
        array = np.asarray(separated)
        if array.ndim != 3 or array.shape[0] < 1 or array.shape[2] < 2:
            raise RuntimeError(f"unexpected MossFormer2 output shape {array.shape} for {source}")
        for index, output_path in enumerate(outputs):
            signal = np.asarray(array[0, :, index], dtype=np.float32)
            peak = float(np.max(np.abs(signal))) if signal.size else 0.0
            if peak > 0.0:
                signal = signal / peak * 0.5
            sf.write(str(output_path), signal, SAMPLE_RATE, subtype="PCM_16")
        return outputs


class SzyV1Runner:
    def __init__(self, config: SzyConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.cache_dir = Path(config.separation_cache)
        if not self.cache_dir.is_absolute():
            self.cache_dir = (Path.cwd() / self.cache_dir).resolve()
        self.gate = DualMLPGate(
            config.sv_model,
            config.cam_model,
            config.dual_backend,
            config.dual_threshold,
            config.sv_device or config.device,
            config.model_root,
        )
        self.separator = MossFormer2Separator(config.moss_model, config.device, config.moss_provider)
        self.asr = FireRedRecognizer(
            config.firered_model,
            config.device,
            config.firered_repo,
            config.firered_vendor,
        )

    def _paths_for(self, sample: Sample) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
        return separation_paths_for(sample, self.config.id_mode, self.cache_dir, self.separator)

    def process(self, sample: Sample) -> dict[str, Any]:
        started = time.perf_counter()
        item_id = submission_id(sample, self.config.id_mode)
        accepted, gate_score = self.gate.accept(sample.wake_audio, sample.command_audio)
        result: dict[str, Any] = {
            "id": item_id,
            "content": "",
            "label": sample.label,
            "sample_type": "positive" if sample.label is not None else "negative",
            "sv_score": f"{gate_score:.6f}",
            "accepted": bool(accepted),
            "sv_accept_threshold": f"{self.config.dual_threshold:.6f}",
            "route": "primary_reject",
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
        if not accepted:
            result["item_duration"] = f"{time.perf_counter() - started:.6f}"
            return result

        try:
            # The frozen switch only depends on raw FireRed output.  Delay
            # separation until it is actually needed; this is decision
            # equivalent to the historical cached pipeline but avoids four
            # unnecessary candidate transcriptions for raw-kept samples.
            raw_command = transcribe_firered_rows(self.asr, [sample.command_audio], "szy")[0]
            chars_per_sec, command_keyword_score = command_features(
                raw_command["text"], duration_seconds(sample.command_audio)
            )
            switch = should_switch_command(
                confidence=raw_command["confidence"],
                command_keyword_score=command_keyword_score,
                chars_per_sec=chars_per_sec,
            )
            result.update({
                "content": raw_command["text"],
                "route": "moss_candidate" if switch else "raw",
                "selected_candidate": "orig",
                "selection_reason": "candidate_pending" if switch else "confidence_rule_not_triggered",
                "switch_triggered": bool(switch),
                "raw_confidence": f"{raw_command['confidence']:.6f}",
                "chars_per_sec": f"{chars_per_sec:.6f}",
                "command_keyword_score": f"{command_keyword_score:.6f}",
            })
            if not switch:
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            (wake_paths, command_paths) = self._paths_for(sample)
            wake0, wake1, command0, command1 = transcribe_firered_rows(
                self.asr,
                [wake_paths[0], wake_paths[1], command_paths[0], command_paths[1]],
                "szy",
            )
            anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
            anchor_path = sample.wake_audio if anchor.raw_wake_fallback else wake_paths[anchor.selected_source]
            contrast_path = wake_paths[anchor.contrast_source]

            anchor_wave = read_wav(anchor_path)
            contrast_wave = read_wav(contrast_path)
            candidate_waves = [read_wav(command_paths[0]), read_wav(command_paths[1])]
            embeddings = {}
            for name, embedder in (("eres", self.gate.eres_embedder), ("cam", self.gate.cam_embedder)):
                values = embedder.extract_waveforms([anchor_wave, contrast_wave, *candidate_waves])
                anchor_embedding, contrast_embedding = values[:2]
                embeddings[name] = [
                    float(np.dot(anchor_embedding, candidate) - np.dot(contrast_embedding, candidate))
                    for candidate in values[2:]
                ]

            source_empty = not command0["text"] or not command1["text"]
            selected = "orig"
            reason = "candidate_pending"
            if source_empty:
                reason = "command_source_empty"
            else:
                selected = "ms0" if (
                    embeddings["eres"][0] + embeddings["cam"][0]
                ) >= (
                    embeddings["eres"][1] + embeddings["cam"][1]
                ) else "ms1"
                reason = "same_domain_differential_score"
            text_by_candidate = {
                "orig": raw_command["text"],
                "ms0": command0["text"],
                "ms1": command1["text"],
            }
            result.update({
                "content": text_by_candidate[selected] if selected != "orig" else raw_command["text"],
                "route": "moss_candidate" if switch else "raw",
                "selected_candidate": selected,
                "selection_reason": reason,
                "command_source_empty": bool(source_empty),
                "wake_selected_source": int(anchor.selected_source),
                "wake_contrast_source": int(anchor.contrast_source),
                "wake_raw_fallback": bool(anchor.raw_wake_fallback),
                "wake_text_tie": bool(anchor.text_tie),
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                "command_source0_text": command0["text"],
                "command_source1_text": command1["text"],
                "eres_diff_ms0": f"{embeddings['eres'][0]:.6f}",
                "eres_diff_ms1": f"{embeddings['eres'][1]:.6f}",
                "cam_diff_ms0": f"{embeddings['cam'][0]:.6f}",
                "cam_diff_ms1": f"{embeddings['cam'][1]:.6f}",
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
    parser = argparse.ArgumentParser(description="SZY v1 formal MossFormer2 target-speaker ASR")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--config", default="configs/szy_v1_firered.json")
    parser.add_argument("--output", default="outputs/szy_v1_submission.json")
    parser.add_argument("--final-output", default="outputs/szy_v1_detail.json")
    parser.add_argument("--checkpoint-output", default="outputs/szy_v1_checkpoint.json")
    parser.add_argument("--manifest-output", default="outputs/szy_v1_manifest.json")
    parser.add_argument("--model-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_szy_config(args.config)
    if args.model_root:
        config = SzyConfig(**{**config.__dict__, "model_root": args.model_root})
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
    runner = SzyV1Runner(config, Path(args.output).parent)
    started = time.perf_counter()
    for sample in tqdm(samples, desc="szy_v1", unit="utt"):
        result = runner.process(sample)
        results.append(result)
        elapsed = elapsed_before + time.perf_counter() - started
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"version": "szy_v1", "duration": elapsed, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    duration = elapsed_before + time.perf_counter() - started
    summary = summarize(results, duration)
    write_outputs(results, summary, Path(args.output), Path(args.final_output))
    manifest = {
        "version": "szy_v1",
        "config": config.__dict__,
        "data_root": str(Path(args.data_root).resolve()),
        "jsonl": [str(Path(path).resolve()) for path in args.jsonl],
        "sample_count": len(results),
        "moss_model_sha256": sha256_file(Path(config.moss_model).resolve()),
        "dual_backend_sha256": sha256_file(Path(config.dual_backend).resolve()),
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
