"""midea_qiji.infer：DatasetA 基线（ERes2NetV2 门控 + Qwen3-ASR）的编排与 CLI。

本模块是 midea_qiji 内核的"主程序"：负责配置加载、JSONL 样本读取、断点
续跑、识别后端/门控装配、批量 ASR 调度、指标汇总与提交物写出。算法部件
已按职责拆分到同包子模块：

  - ``text.py``         文本归一化 / 编辑距离 / CER；
  - ``audio.py``        音频常量与 WAV 读取；
  - ``paths.py``        模型路径常量与本地解析；
  - ``recognizers.py``  Qwen / FireRed AED / FireRed LLM / 条件 LoRA 后端；
  - ``gate.py``         双前端 MLP 门控与声纹 embedding。

为兼容既有代码与文档，本模块 re-export 了 text / paths 的公共名（如
``from midea_qiji.infer import levenshtein`` 仍可用）；新代码请直接从
``midea_qiji.text`` / ``midea_qiji.paths`` 导入。

用法（baseline 路线，SZY 系列请走 szy_v4 入口）：
  PYTHONPATH=src python -m midea_qiji.infer \\
    --data-root dataset/dataset_test/datasetA/datasetA \\
    --jsonl dataset/.../pos.jsonl --config configs/xxx.json

依赖：numpy / tqdm；torch / modelscope / FireRed 等重型依赖延迟到使用时。
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

# ---- re-export（兼容旧导入路径 midea_qiji.infer.<name>） ----
from midea_qiji.paths import (  # noqa: F401
    DEFAULT_FIRED_MODEL,
    DEFAULT_FIRED_REPO,
    DEFAULT_FIRED_VENDOR,
    PACKAGE_MODEL_ROOT,
    resolve_model_path,
)
from midea_qiji.text import cer, extract_asr_text, levenshtein, normalize_transcript  # noqa: F401


@dataclass(frozen=True)
class Config:
    device: str = "cuda:0"
    sv_device: str | None = None
    asr_model: str = "Qwen/Qwen3-ASR-0.6B"
    sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    sv_model_revision: str | None = "v1.0.0"
    sv_accept_threshold: float = 0.27
    id_mode: str = "relative_stem"
    # 门控模式："cos" = ERes2NetV2 余弦阈值（baseline）；"dual_mlp" = 双前端 MLP 裁判
    gate_mode: str = "cos"
    cam_model: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    dual_backend: str | None = None
    dual_threshold: float = 0.5334
    # ASR 后端："qwen" = Qwen3-ASR-0.6B；"firered" = FireRedASR2-AED；
    # "qwen_lora" = sv_asr 双音频条件 LoRA 微调权重（[kws, cmd] → 目标人文本）
    asr_backend: str = "qwen"
    firered_model: str = DEFAULT_FIRED_MODEL
    firered_repo: str = DEFAULT_FIRED_REPO
    qwen_lora_adapter: str | None = None


@dataclass(frozen=True)
class Sample:
    raw_id: str
    wake_audio: Path
    command_audio: Path
    command_relative_path: str
    label: str | None
    wake_text: str | None = None


def load_config(path: str | Path) -> Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = Config.__dataclass_fields__
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return Config(**payload)


class SpeakerVerifier:
    """ERes2NetV2 余弦门控（baseline 路线用；SZY 系列改用 DualMLPGate）。"""

    def __init__(self, model: str, device: str, model_root: str | Path | None) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=str(resolve_model_path(model, model_root)),
            device=device,
        )

    def score(self, enrollment_audio: Path, command_audio: Path) -> float:
        return extract_speaker_score(self.pipeline([str(enrollment_audio), str(command_audio)]))


def read_many_jsonl(paths: Iterable[str | Path], data_root: str | Path) -> list[Sample]:
    root = Path(data_root)
    samples: list[Sample] = []
    for jsonl_path in paths:
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    wake_relative_path = str(row["唤醒音频"])
                    command_relative_path = str(row["识别音频"])
                except KeyError as error:
                    raise KeyError(f"{jsonl_path}:{line_number} missing field {error}") from error
                samples.append(
                    Sample(
                        raw_id=str(row.get("id", "")),
                        wake_audio=root / wake_relative_path,
                        command_audio=root / command_relative_path,
                        command_relative_path=command_relative_path,
                        label=row.get("识别文本"),
                        wake_text=row.get("唤醒文本") or row.get("wake_text"),
                    )
                )
    return samples


def submission_id(sample: Sample, mode: str) -> str:
    if mode == "raw":
        return sample.raw_id
    if mode == "name":
        return sample.command_audio.name
    if mode == "path":
        return sample.command_audio.as_posix()
    if mode == "relative":
        return sample.command_relative_path
    if mode == "relative_stem":
        path = Path(sample.command_relative_path)
        return (path.parent / path.stem).as_posix()
    if mode == "stem":
        return sample.command_audio.stem
    raise ValueError(f"unsupported id mode: {mode}")


def extract_speaker_score(result: Any) -> float:
    if isinstance(result, dict):
        for key in ("score", "scores", "similarity", "output", "outputs"):
            if key in result:
                try:
                    return extract_speaker_score(result[key])
                except ValueError:
                    continue
        raise ValueError(f"speaker verification output has no score: {list(result)}")
    if isinstance(result, (list, tuple)):
        for value in result:
            try:
                return extract_speaker_score(value)
            except ValueError:
                continue
        raise ValueError("speaker verification output was empty")
    if isinstance(result, np.ndarray):
        if result.size == 0:
            raise ValueError("speaker verification score array was empty")
        return float(result.reshape(-1)[0])
    try:
        return float(result)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot parse speaker score from {result!r}") from error


@contextlib.contextmanager
def suppress_model_output():
    """静默模型加载/推理期间的 stdout+stderr（FireRed/Qwen 打印噪音）。"""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def write_reports(results: list[dict[str, Any]], duration: float, output: Path, final_output: Path) -> dict[str, float]:
    total_errors = 0
    total_characters = 0
    positive_nonempty = 0
    negative_total = 0
    negative_rejects = 0
    submission_results = []
    final_results = []
    for item in results:
        content = str(item["content"] or "")
        label = item["label"]
        positive = label is not None and label != ""
        item_cer = cer(content, label) if positive else None
        if positive:
            total_errors += levenshtein(content, str(label))
            total_characters += len(str(label))
            positive_nonempty += int(content != "")
        else:
            negative_total += 1
            negative_rejects += int(content == "")
        submission_results.append({"id": item["id"], "content": content, "label": label, "cer": "" if item_cer is None else f"{item_cer:.6f}"})
        final_item = {
            "id": item["id"],
            "content": content,
            "label": label,
            "sample_type": "positive" if positive else "negative",
            "cer": item_cer,
            "pr": float(content != "") if positive else None,
            "rr": None if positive else float(content == ""),
        }
        for key in (
            "sv_score",
            "accepted",
            "kept_segments",
            "wake_localized",
            "wake_candidate_count",
            "wake_transcript",
            "wake_text_score",
            "wake_selected_source",
            "wake_text_confirmed",
            "sv_accept_threshold",
            "item_duration",
        ):
            if key in item:
                final_item[key] = item[key]
        final_results.append(final_item)
    summary = {
        "final_cer": total_errors / total_characters if total_characters else 0.0,
        "final_pr": positive_nonempty / (len(results) - negative_total) if len(results) > negative_total else 0.0,
        "final_rr": negative_rejects / negative_total if negative_total else 0.0,
        "positive_count": len(results) - negative_total,
        "positive_nonempty": positive_nonempty,
        "negative_count": negative_total,
        "negative_rejects": negative_rejects,
        "duration": duration,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"result": {"results": submission_results, "final_cer": f"{summary['final_cer']:.6f}", "duration": f"{duration:.6f}"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    final_output.write_text(json.dumps({"summary": summary, "results": final_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone ERes2NetV2 plus Qwen3-ASR DatasetA baseline")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--config", default="configs/eres2netv2_qwen3_raw_command.json")
    parser.add_argument("--model-root", help="ModelScope cache root containing iic/ and Qwen/")
    parser.add_argument("--output", default="outputs/submission.json")
    parser.add_argument("--final-output", default="outputs/detail.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--asr-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.asr_batch_size < 1:
        raise ValueError("--asr-batch-size must be positive")
    config = load_config(args.config)
    samples = read_many_jsonl(args.jsonl, args.data_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else None
    results: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires --checkpoint-output from a prior run")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = checkpoint["results"]
        elapsed_before = float(checkpoint["duration"])
        completed = {item["id"] for item in results}
        samples = [sample for sample in samples if submission_id(sample, config.id_mode) not in completed]
    gate = None
    verifier = None
    if config.gate_mode == "dual_mlp":
        from midea_qiji.gate import DualMLPGate

        if config.dual_backend is None:
            raise ValueError("gate_mode=dual_mlp requires dual_backend in config")
        print(f"Loading dual MLP gate (ERes2NetV2 + CAM++) on {config.sv_device or config.device}...", flush=True)
        gate = DualMLPGate(
            config.sv_model, config.cam_model, config.dual_backend,
            config.dual_threshold, config.sv_device or config.device, args.model_root,
        )
    else:
        print(f"Loading ERes2NetV2 on {config.sv_device or config.device}...", flush=True)
        verifier = SpeakerVerifier(config.sv_model, config.sv_device or config.device, args.model_root)
    if config.asr_backend == "firered":
        from midea_qiji.recognizers import FireRedRecognizer

        print(f"Loading FireRedASR2-AED on {config.device}...", flush=True)
        recognizer = FireRedRecognizer(config.firered_model, config.device, config.firered_repo)
    elif config.asr_backend == "qwen_lora":
        from midea_qiji.recognizers import ConditionedQwenRecognizer

        if config.qwen_lora_adapter is None:
            raise ValueError("asr_backend=qwen_lora requires qwen_lora_adapter in config")
        print(f"Loading Qwen3-ASR + LoRA adapter ({config.qwen_lora_adapter}) on {config.device}...", flush=True)
        recognizer = ConditionedQwenRecognizer(config.asr_model, config.qwen_lora_adapter, config.device, args.model_root)
    else:
        from midea_qiji.recognizers import QwenRecognizer

        print(f"Loading Qwen3-ASR on {config.device}...", flush=True)
        recognizer = QwenRecognizer(config.asr_model, config.device, args.model_root, args.asr_batch_size)
    started = time.perf_counter()
    pending_pairs: list[tuple[Path, Path]] = []
    pending_indices: list[int] = []

    def save_checkpoint() -> None:
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps({"duration": elapsed_before + time.perf_counter() - started, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    def flush_asr() -> None:
        if not pending_pairs:
            return
        if getattr(recognizer, "conditioned", False):
            transcripts = recognizer.transcribe_pairs(pending_pairs)
        else:
            transcripts = recognizer.transcribe_many([command for _, command in pending_pairs])
        if len(transcripts) != len(pending_indices):
            raise RuntimeError("ASR returned an unexpected transcript count")
        completed_at = time.perf_counter()
        for index, transcript in zip(pending_indices, transcripts):
            results[index]["content"] = transcript
            results[index]["item_duration"] = f"{completed_at - results[index].pop('_started'):.6f}"
        pending_pairs.clear()
        pending_indices.clear()
        save_checkpoint()

    for sample in tqdm(samples, desc="infer", unit="utt"):
        item_started = time.perf_counter()
        item_id = submission_id(sample, config.id_mode)
        if not sample.wake_audio.is_file() or not sample.command_audio.is_file():
            raise FileNotFoundError(f"missing audio for {item_id}")
        if gate is not None:
            accepted, speaker_score = gate.accept(sample.wake_audio, sample.command_audio)
            active_threshold = config.dual_threshold
        else:
            speaker_score = verifier.score(sample.wake_audio, sample.command_audio)
            accepted = speaker_score >= config.sv_accept_threshold
            active_threshold = config.sv_accept_threshold
        item = {"id": item_id, "content": "", "label": sample.label, "sv_score": f"{speaker_score:.6f}", "accepted": accepted, "kept_segments": 0, "wake_localized": False, "wake_candidate_count": 0, "wake_transcript": "", "wake_text_score": "0.000000", "wake_selected_source": None, "wake_text_confirmed": True, "sv_accept_threshold": f"{active_threshold:.6f}", "item_duration": ""}
        results.append(item)
        if accepted:
            item["_started"] = item_started
            pending_pairs.append((sample.wake_audio, sample.command_audio))
            pending_indices.append(len(results) - 1)
            if len(pending_pairs) >= args.asr_batch_size:
                flush_asr()
        else:
            item["item_duration"] = f"{time.perf_counter() - item_started:.6f}"
            if not pending_pairs:
                save_checkpoint()
    flush_asr()
    for item in results:
        item.pop("_started", None)
    duration = elapsed_before + time.perf_counter() - started
    summary = write_reports(results, duration, Path(args.output), Path(args.final_output))
    print(f"samples={len(results)} final_cer={summary['final_cer']:.6f} final_pr={summary['final_pr']:.6f} final_rr={summary['final_rr']:.6f} duration={duration:.3f}s", flush=True)


if __name__ == "__main__":
    main()
