"""szy_common.reporting：SZY 系列的评测汇总与提交物写出。

``summarize`` 按 label 区分正负样本，计算 final_cer / final_pr / final_rr、
路由分布、切换次数与分离错误数；``write_outputs`` 写出竞赛格式的
submission.json（result.results + avg_cer/avg_rr/duration）与全字段
detail.json（summary + results）。原实现位于 szy_v1.pipeline，v2/v4 复用，
现收敛到此共享层，语义与输出 schema 不变。

依赖：midea_qiji.text（levenshtein / normalize_transcript）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from midea_qiji.text import levenshtein, normalize_transcript


def summarize(results: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """汇总一行结果列表为指标字典（CER 按归一化文本编辑距离口径）。"""
    positive = [row for row in results if row.get("label") is not None and row.get("label") != ""]
    negative = [row for row in results if row.get("label") is None or row.get("label") == ""]
    errors = sum(
        levenshtein(
            normalize_transcript(str(row.get("content") or "")),
            normalize_transcript(str(row["label"])),
        )
        for row in positive
    )
    characters = sum(len(normalize_transcript(str(row["label"]))) for row in positive)
    nonempty = sum(bool(str(row.get("content") or "")) for row in positive)
    rejects = sum(not str(row.get("content") or "") for row in negative)
    routes: dict[str, int] = {}
    for row in results:
        routes[row.get("route", "unknown")] = routes.get(row.get("route", "unknown"), 0) + 1
    return {
        "route": "szy_v1",
        "sample_count": len(results),
        "positive_count": len(positive),
        "positive_nonempty": nonempty,
        "final_pr": nonempty / len(positive) if positive else 0.0,
        "negative_count": len(negative),
        "negative_rejects": rejects,
        "final_rr": rejects / len(negative) if negative else 0.0,
        "errors": errors,
        "characters": characters,
        "final_cer": errors / characters if characters else 0.0,
        "duration": duration,
        "routes": routes,
        "switches": sum(bool(row.get("switch_triggered")) for row in results),
        "separation_errors": sum(bool(row.get("error")) for row in results),
    }


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], output: Path, detail: Path) -> None:
    """写出竞赛提交物 submission.json 与全字段 detail.json。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.parent.mkdir(parents=True, exist_ok=True)
    submission = []
    details = []
    for row in results:
        label = row.get("label")
        if label is not None and label != "":
            reference = normalize_transcript(str(label))
            hypothesis = normalize_transcript(str(row.get("content") or ""))
            item_cer = (
                levenshtein(hypothesis, reference) / len(reference)
                if reference else float(hypothesis != "")
            )
        else:
            item_cer = None
        submission.append({
            "id": row["id"],
            "content": str(row.get("content") or ""),
            "label": label,
            "cer": item_cer,
        })
        item = dict(row)
        item["cer"] = item_cer
        details.append(item)
    output.write_text(
        json.dumps(
            {
                "result": {
                    "results": submission,
                    "avg_cer": float(summary["final_cer"]),
                    "avg_rr": float(summary["final_rr"]),
                    "duration": round(float(summary["duration"]) * 1000.0, 3),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    detail.write_text(json.dumps({"summary": summary, "results": details}, ensure_ascii=False, indent=2), encoding="utf-8")
