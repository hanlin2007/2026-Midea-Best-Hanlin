#!/usr/bin/env python3
"""构建 v4 全链路重放所需的一次性资产（FireRed ASR）。

part=dataseta：为 sweep-ep6 全量运行中**被门控拒绝**的样本补 transcribe
  （raw cmd 文本+置信度、cmd 两分离通道文本）——接受样本的文本已在
  detail.json 里。产物 outputs/sweep_v4/ep6_replay_texts.json。
part=valid：为 dataset_dasynth_v4sep（新分离器）的 valid mos0/mos1 通道
  转写文本，产物 dataset_dasynth_v4sep/metadata/valid_moss_text.json。

用法（yuyin，仓库根目录）：
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/build_v4_replay_assets.py --part dataseta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WS = _REPO.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_WS / "model" / "runtime" / "vendor"))

from midea_qiji.recognizers import FireRedRecognizer  # noqa: E402
from szy_common.helpers import transcribe_firered_rows  # noqa: E402

DATASETA = _WS / "dataset" / "dataset_test" / "datasetA" / "datasetA"
DETAIL = _REPO / "outputs" / "sweep_v4" / "ep6_detail.json"
CACHE = _REPO / "outputs" / "szy_v4_cache_sweep_best" / "separation"
OUT_A = _REPO / "outputs" / "sweep_v4" / "ep6_replay_texts.json"
V4SEP = _WS / "dataset_dasynth_v4sep"
OUT_B = V4SEP / "metadata" / "valid_moss_text.json"

FIRERED_MODEL = _WS / "model" / "external" / "FireRedASR2-AED"
FIRERED_REPO = _WS / "model" / "external" / "FireRedASR2S"
FIRERED_VENDOR = _WS / "model" / "runtime" / "vendor"


def part_dataseta(asr: FireRedRecognizer, batch: int) -> None:
    detail = json.load(DETAIL.open(encoding="utf-8"))["results"]
    assets: dict[str, dict] = {}
    todo: list[tuple[str, list[Path]]] = []
    for r in detail:
        uid = r["id"]
        item = uid.replace("/", "_")
        if r.get("raw_text") is not None:
            assets[uid] = {
                "raw_text": r["raw_text"],
                "raw_confidence": float(r["raw_confidence"]),
                "cmd0_text": r.get("command_source0_text", ""),
                "cmd1_text": r.get("command_source1_text", ""),
            }
        else:
            sample_dir = DATASETA / uid.split("/")[0]
            raw_cmd = sample_dir / f"{uid.split('/')[1]}.wav"
            todo.append((uid, [
                raw_cmd,
                CACHE / f"{item}_command_ms0.wav",
                CACHE / f"{item}_command_ms1.wav",
            ]))
    print(f"已有文本 {len(assets)}，待补 {len(todo)}", flush=True)
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        paths = [p for _, ps in chunk for p in ps]
        outs = transcribe_firered_rows(asr, paths, "rp")
        for i, (uid, _) in enumerate(chunk):
            raw, c0, c1 = outs[3 * i : 3 * i + 3]
            assets[uid] = {
                "raw_text": raw["text"], "raw_confidence": raw["confidence"],
                "cmd0_text": c0["text"], "cmd1_text": c1["text"],
            }
        if (start + batch) % (batch * 10) == 0 or start + batch >= len(todo):
            print(f"asr {min(start + batch, len(todo))}/{len(todo)}", flush=True)
    json.dump(assets, OUT_A.open("w", encoding="utf-8"), ensure_ascii=False)
    print(f"written {OUT_A} ({len(assets)})", flush=True)


def part_valid(asr: FireRedRecognizer, batch: int) -> None:
    ids = sorted(p.stem for p in (V4SEP / "valid" / "mos0").glob("*.wav"))
    print(f"valid 通道转写 {len(ids)} 条", flush=True)
    out: dict[str, dict] = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        paths = [V4SEP / "valid" / "mos0" / f"{i}.wav" for i in chunk] + \
                [V4SEP / "valid" / "mos1" / f"{i}.wav" for i in chunk]
        outs = transcribe_firered_rows(asr, paths, "rp")
        n = len(chunk)
        for i, uid in enumerate(chunk):
            out[uid] = {"ms0": outs[i]["text"], "ms1": outs[n + i]["text"]}
        if (start + batch) % (batch * 10) == 0 or start + batch >= len(ids):
            print(f"asr {min(start + batch, len(ids))}/{len(ids)}", flush=True)
    OUT_B.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, OUT_B.open("w", encoding="utf-8"), ensure_ascii=False)
    print(f"written {OUT_B} ({len(out)})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dataseta", "valid"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    asr = FireRedRecognizer(FIRERED_MODEL, args.device, FIRERED_REPO, FIRERED_VENDOR)
    if args.part == "dataseta":
        part_dataseta(asr, args.batch)
    else:
        part_valid(asr, args.batch)


if __name__ == "__main__":
    main()
