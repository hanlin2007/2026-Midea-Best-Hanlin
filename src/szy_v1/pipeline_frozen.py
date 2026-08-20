"""SZY v1 冻结门控版（消融实验用）：跳过 MLP 主门控，只放行 pos 样本进入下游链路。

================================================================================
【这个文件与 pipeline.py 的差异（只改一处，其余逐字对齐）】

  pipeline.py（正式版）：
      ① DualMLPGate.accept() 判决 → 分数低于阈值就拒识（输出空）
      ② 放行样本走 raw FireRed → 切换规则 → MossFormer2 → 差分声纹选源

  pipeline_frozen.py（本文件，消融版）：
      ① 门控被"冻结"——完全不调用 DualMLPGate.accept()，忽略 MLP 分数
      ② 手动筛选：label 是 pos（识别文本非空）→ 放行进入下游；
         label 是 neg（识别文本为 null/空）→ 直接拒识（输出空文本，不进下游）
      ③ 下游链路（raw → 切换 → Moss → 差分声纹）与正式版逐字一致

  消融目的：把"门控"这个因子从结果里剔除，单独测量 MossFormer2 + 差分声纹
  候选链路对正样本的 CER 理论上限（门控为满分时的上限）。
  注意：这样得到的 PR/RR 是失真的（PR 会被推到 100%，RR 也会被推到 100%），
  只用于看 CER 上限，不用于正式提交口径。

  为什么还要保留 DualMLPGate 对象？——下游差分声纹选源需要它的两个嵌入器
  eres_embedder / cam_embedder（第 388 行 self.gate.eres_embedder 等），
  所以我们照常构造门控、照常加载模型，只是不调用 accept()（跳过 MLP 头）。
================================================================================
"""
from __future__ import annotations  # 延迟注解求值

import argparse  # 命令行参数
import json
import time  # 计时
from pathlib import Path
from typing import Any

import numpy as np

from midea_qiji.gate import read_wav
from midea_qiji.infer import Sample, read_many_jsonl, submission_id
from midea_qiji.moss_candidate import select_wake_anchor, should_switch_command

# 复用正式版 pipeline 的全部基础设施：配置、Moss 分离器、转写、汇总、写盘。
# 只"换掉"运行器里 process() 的判决头（子类覆盖），其余逻辑原样继承。
from .pipeline import (
    SzyConfig,
    load_szy_config,
    SzyV1Runner,
    _command_features,
    _duration,
    _sha256,
    summarize,
    write_outputs,
)


class FrozenSzyV1Runner(SzyV1Runner):
    """冻结门控版运行器：跳过 MLP 主门控，只放行 pos 样本进入下游。

    继承 SzyV1Runner 拿到三个重型组件（gate 的嵌入器 / Moss 分离器 / FireRed），
    以及 _transcribe()、_paths_for() 两个下游辅助方法；只覆盖 process() 这一步。
    """

    def process(self, sample: Sample) -> dict[str, Any]:
        """处理一条样本（冻结门控版）。与 pipeline.SzyV1Runner.process 逐字对齐，
        唯一差异：第 ① 步从"MLP 判决"换成"label 是不是 pos"。"""
        started = time.perf_counter()
        item_id = submission_id(sample, self.config.id_mode)
        # ① 冻结门控：不看 MLP 分数，只看 label。pos=放行，neg=拒识。
        is_pos = sample.label is not None and str(sample.label).strip() != ""
        # 结果骨架：字段与正式版一致（保证 summarize/write_outputs 能直接消费）。
        result: dict[str, Any] = {
            "id": item_id,
            "content": "",                                   # 最终输出文本（默认空）
            "label": sample.label,
            "sample_type": "positive" if is_pos else "negative",
            "sv_score": "frozen",                            # 门控被冻结，没有分数
            "accepted": is_pos,
            "sv_accept_threshold": "frozen",                 # 阈值也被冻结
            "route": "frozen_neg_reject" if not is_pos else "raw",
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
        if not is_pos:  # ② neg → 冻结门控下等价于"完美拒识"，直接输出空文本
            result["item_duration"] = f"{time.perf_counter() - started:.6f}"
            return result

        # ③④⑤⑥⑦ 与正式版逐字一致（raw → 切换 → Moss → 差分声纹选源）
        try:
            # ③ raw 命令音频 FireRed 转写（冻结的切换决策只依赖 raw 输出）
            raw_command = self._transcribe(self.asr, [sample.command_audio])[0]
            # 算两个命令特征（语速、关键词数）
            chars_per_sec, command_keyword_score = _command_features(
                raw_command["text"], _duration(sample.command_audio)
            )
            # ④ 冻结的二维切换规则
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
            if not switch:  # 不触发候选 → 保留 raw 文本，收工
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # ⑤ 分离唤醒音频 + 命令音频（共 4 个 wav）
            (wake_paths, command_paths) = self._paths_for(sample)
            # 把 4 路分离结果一次性批转写（wake0/wake1/cmd0/cmd1）
            wake0, wake1, command0, command1 = self._transcribe(
                self.asr,
                [wake_paths[0], wake_paths[1], command_paths[0], command_paths[1]],
            )
            # ⑥ 用唤醒文本在两路唤醒源里选锚点（返回 WakeAnchorDecision）
            anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
            anchor_path = sample.wake_audio if anchor.raw_wake_fallback else wake_paths[anchor.selected_source]
            contrast_path = wake_paths[anchor.contrast_source]

            # ⑦ 差分声纹选择命令源：读 4 段波形（锚点、对照、cmd0、cmd1）
            anchor_wave = read_wav(anchor_path)
            contrast_wave = read_wav(contrast_path)
            candidate_waves = [read_wav(command_paths[0]), read_wav(command_paths[1])]
            embeddings = {}
            # 两个声纹前端（eres/cam）各算一次"差分"：
            #   diff(c) = cos(anchor, c) - cos(contrast, c)
            for name, embedder in (("eres", self.gate.eres_embedder), ("cam", self.gate.cam_embedder)):
                values = embedder.extract_waveforms([anchor_wave, contrast_wave, *candidate_waves])
                anchor_embedding, contrast_embedding = values[:2]
                embeddings[name] = [
                    float(np.dot(anchor_embedding, candidate) - np.dot(contrast_embedding, candidate))
                    for candidate in values[2:]
                ]

            # 命令源选择：任一命令源为空 → 分离失败，回退 orig（不赌单路）
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
        except Exception as error:  # 任何异常 → 回退 raw ASR（保留原始结果的控制权）
            result["error"] = f"{type(error).__name__}: {error}"
            try:
                result["content"] = self.asr.transcribe_many([sample.command_audio])[0]
                result["route"] = "raw_fallback_error"
                result["selected_candidate"] = "orig"
            except Exception as raw_error:  # raw 也失败 → 只能输出空
                result["error"] += f"; raw_fallback={type(raw_error).__name__}: {raw_error}"
                result["route"] = "empty_fallback_error"
        result["item_duration"] = f"{time.perf_counter() - started:.6f}"
        return result


def build_parser() -> argparse.ArgumentParser:
    """冻结版命令行参数（与 pipeline.build_parser 同款，仅输出文件名默认值不同）。"""
    parser = argparse.ArgumentParser(description="SZY v1 冻结门控消融（跳过 MLP 主门控，只放行 pos）")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jsonl", nargs="*", help="JSONL 清单；不传则自动在 data-root 下找 pos.jsonl + neg.jsonl")
    parser.add_argument("--config", default="configs/szy_v1_firered.json")
    parser.add_argument("--output", default="outputs/szy_v1_frozen_submission.json")
    parser.add_argument("--final-output", default="outputs/szy_v1_frozen_detail.json")
    parser.add_argument("--checkpoint-output", default="outputs/szy_v1_frozen_checkpoint.json")
    parser.add_argument("--manifest-output", default="outputs/szy_v1_frozen_manifest.json")
    parser.add_argument("--model-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    """冻结版主入口：加载配置 → 读样本 → 逐条 process → 汇总写盘（同 pipeline.main）。"""
    args = build_parser().parse_args()
    config = load_szy_config(args.config)
    if args.model_root:
        config = SzyConfig(**{**config.__dict__, "model_root": args.model_root})
    if not args.jsonl:
        args.jsonl = []
        root = Path(args.data_root)
        for name in ("pos.jsonl", "neg.jsonl"):
            candidate = root / name
            if candidate.is_file():
                args.jsonl.append(str(candidate))
        if not args.jsonl:
            print("[ERROR] --jsonl 未提供，且 data-root 下找不到 pos.jsonl 或 neg.jsonl", flush=True)
            raise SystemExit(1)
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
    runner = FrozenSzyV1Runner(config, Path(args.output).parent)
    started = time.perf_counter()
    for sample in samples:  # 冻结版循环（进度条可选，避免刷屏）
        result = runner.process(sample)
        results.append(result)
        elapsed = elapsed_before + time.perf_counter() - started
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"version": "szy_v1_frozen", "duration": elapsed, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    duration = elapsed_before + time.perf_counter() - started
    summary = summarize(results, duration)
    write_outputs(results, summary, Path(args.output), Path(args.final_output))
    manifest = {
        "version": "szy_v1_frozen",
        "config": config.__dict__,
        "data_root": str(Path(args.data_root).resolve()),
        "jsonl": [str(Path(path).resolve()) for path in args.jsonl],
        "sample_count": len(results),
        "moss_model_sha256": _sha256(Path(config.moss_model).resolve()),
        "dual_backend_sha256": _sha256(Path(config.dual_backend).resolve()),
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
