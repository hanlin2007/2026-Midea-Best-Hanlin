#!/usr/bin/env python3
"""测量 q：被门控拒绝的正样本，如果接受了，CER 到底是多少？

背景
----
现在的 pipeline 对 fusion_reject 的样本直接输出空、不跑 ASR，所以
detail.json 里没有它们的转写。任何"放松门控"的离线评估都只能假设一个 q，
而结论对 q 极其敏感（q=0.08 → +0.0064，q=0.8 → -0.0035）。

本脚本对这些样本补跑 ASR（分离音频已在 cache 里，只花 ASR 的时间），
给出真实的 q。命令通道 ASR 配置与 pipeline 一致：beam=3 + ELM。

用法
----
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:model/runtime/vendor \
  python scripts/experiments/measure_q.py --run <NAME>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 仓库根（本文件在 scripts/experiments/ 下）
sys.path.insert(0, str(ROOT / "src"))       # 允许不设 PYTHONPATH 直接运行

from midea_qiji.recognizers import FireRedRecognizer  # noqa: E402
from midea_qiji.text import levenshtein  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/ 下的运行名（读取 <run>_detail.json 与 cache/）")
    ap.add_argument("--config", default="configs/pipeline_firered.json")
    ap.add_argument("--data-root", default="dataset/dataset_test/datasetA/datasetA",
                    help="音频根目录（样本 id 即其下的相对路径）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    detail = ROOT / f"outputs/{args.run}/{args.run}_detail.json"
    sep_dir = ROOT / f"outputs/{args.run}/cache/separation"
    data_root = ROOT / args.data_root
    out = Path(args.out) if args.out else ROOT / f"outputs/{args.run}/measured_q.json"

    results = json.loads(detail.read_text(encoding="utf-8"))["results"]
    targets = [r for r in results
               if r.get("sample_type") == "positive" and r.get("label")
               and not (r.get("accepted") and (r.get("content") or ""))]
    print(f"被拒正样本 {len(targets)} 条", flush=True)

    print("[加载] FireRedASR2-AED (beam=3 + ELM)...", flush=True)
    t0 = time.perf_counter()
    asr = FireRedRecognizer(
        ROOT / cfg["firered_model"], "cuda:0", ROOT / cfg["firered_repo"],
        vendor_root=ROOT / cfg["firered_vendor"],
        beam_size=cfg.get("firered_nbest_beam_size", 3),
        nbest=1,
        elm_dir=cfg.get("firered_elm_dir"),
        elm_weight=cfg.get("firered_elm_weight", 0.0),
    )
    print(f"[加载] 完成 {time.perf_counter()-t0:.1f}s", flush=True)

    rows = []
    for i, r in enumerate(targets):
        sid = r["id"]                      # e.g. pos/cmd_37
        stem = sid.replace("/", "_")       # pos_cmd_37
        raw = data_root / f"{sid}.wav"
        paths = [raw,
                 sep_dir / f"{stem}_command_ms0.wav",
                 sep_dir / f"{stem}_command_ms1.wav"]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[跳过] {sid}: 缺 {[p.name for p in missing]}", flush=True)
            continue
        got = asr.transcribe_many_with_confidence(paths)
        texts = [g.get("text", "") for g in got]
        label = r["label"]
        rows.append({
            "id": sid, "label": label,
            "raw": texts[0], "ms0": texts[1], "ms1": texts[2],
            "conf_raw": got[0].get("confidence"),
            "err_raw": levenshtein(texts[0], label),
            "err_ms0": levenshtein(texts[1], label),
            "err_ms1": levenshtein(texts[2], label),
            "ref_len": len(label),
            "sv_score": r.get("sv_score"),
        })
        if (i + 1) % 20 == 0:
            print(f"[进度] {i+1}/{len(targets)}  {time.perf_counter()-t0:.0f}s", flush=True)

    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {out}")

    tot = sum(x["ref_len"] for x in rows)
    print("\n" + "=" * 62)
    print(f"被拒正样本 {len(rows)} 条，参考字符 {tot}")
    print("=" * 62)
    for nm, key in (("强制 raw", "err_raw"), ("强制 ms0", "err_ms0"), ("强制 ms1", "err_ms1")):
        e = sum(x[key] for x in rows)
        print(f"  {nm:10s} 错字 {e:5d}  →  q = {e/tot:.4f}")
    e_or = sum(min(x["err_raw"], x["err_ms0"], x["err_ms1"]) for x in rows)
    print(f"  {'oracle三选一':10s} 错字 {e_or:5d}  →  q = {e_or/tot:.4f}")
    print("\n  对照：全部计错（当前行为）    q = 1.0000")

    # 直接给出"若全部接受"的 Score
    all_pos = [r for r in results if r.get("sample_type") == "positive" and r.get("label")]
    all_ch = sum(len(r["label"]) for r in all_pos)
    acc_err = sum(levenshtein(r.get("content") or "", r["label"]) for r in all_pos
                  if r.get("accepted") and (r.get("content") or ""))
    neg = [r for r in results if r.get("sample_type") == "negative"]
    rr = sum(1 for r in neg if not (r.get("accepted") and (r.get("content") or ""))) / len(neg)
    cur_cer = (acc_err + sum(x["ref_len"] for x in rows)) / all_ch
    print(f"\n  当前            CER={cur_cer:.4f} RR={rr:.4f} Score={(1-cur_cer+rr)/2:.4f}")
    for nm, key in (("全部接受(raw)", "err_raw"),):
        new_cer = (acc_err + sum(x[key] for x in rows)) / all_ch
        print(f"  {nm:14s} CER={new_cer:.4f} RR={rr:.4f} Score={(1-new_cer+rr)/2:.4f} "
              f"(Δ{(1-new_cer+rr)/2-(1-cur_cer+rr)/2:+.4f})")


if __name__ == "__main__":
    main()
