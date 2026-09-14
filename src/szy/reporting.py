"""szy.reporting：评测汇总与提交物写出。

``summarize`` 按 label 区分正负样本，计算 final_cer / final_pr / final_rr、
路由分布、切换次数与分离错误数；``write_outputs`` 写出竞赛格式的
submission.json（result.results + avg_cer/avg_rr/duration）与全字段
detail.json（summary + results）。

三个核心指标的口径（竞赛计分基于它们）：
  - CER（Character Error Rate，字错率）：正样本转写与标注的编辑距离 /
    标注总字数，越低越好；
  - PR（Positive Rate，正样本产出率）：正样本里"转写非空"的比例——空
    转写整条 0 分，所以 PR 与 CER 共同决定正样本得分；
  - RR（Reject Rate，负样本拒绝率）：负样本里"输出空"的比例——负样本
    输出任何文字都整条 0 分。公式分 = (1−CER+RR)/2 正是在三者间权衡。

依赖：midea_qiji.text（levenshtein / normalize_transcript）。
（本模块同样不依赖 torch：汇总可以在任何轻量环境离线重算，甚至只拿
detail.json 就能重新统计，不必重新推理。）
"""
from __future__ import annotations

import json                                     # 提交物序列化
from pathlib import Path                        # 输出路径
from typing import Any                           # 结果行的宽松类型

from midea_qiji.text import levenshtein, normalize_transcript


def summarize(results: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """汇总一行结果列表为指标字典（CER 按归一化文本编辑距离口径）。

    label 非空 = 正样本（有标注指令文本）；label 空 = 负样本（应当拒绝）。
    划分依据是数据集的标注：正样本的 label 是真实指令文本，负样本的
    label 缺失（数据集里根本没有"应该转写什么"），此时系统的正确行为
    是输出空字符串（拒绝）。
    """
    # 列表推导式按 label 切分正负样本；get("label") 用 None 兜底防 KeyError
    positive = [row for row in results if row.get("label") is not None and row.get("label") != ""]
    negative = [row for row in results if row.get("label") is None or row.get("label") == ""]
    # 编辑距离（levenshtein）：把转写改成标注需要的最少"增/删/改字符"数。
    # 先各自归一化再比，保证口径一致（标点差异不算错）
    errors = sum(
        levenshtein(
            normalize_transcript(str(row.get("content") or "")),
            normalize_transcript(str(row["label"])),
        )
        for row in positive
    )
    # 分母：全部正样本标注的总字数（CER 的分母，与 errors 同一口径）
    characters = sum(len(normalize_transcript(str(row["label"]))) for row in positive)
    # nonempty：正样本里转写非空的条数（bool 转 int 求和是 Python 计数惯用法）
    nonempty = sum(bool(str(row.get("content") or "")) for row in positive)
    # rejects：负样本里正确"拒绝"（输出为空）的条数
    rejects = sum(not str(row.get("content") or "") for row in negative)
    # 路由分布：每个 route 值出现多少次（如 fusion_reject/raw/moss_candidate），
    # 用来观察"系统实际走了哪条路"，是调参与排查的第一手信息
    routes: dict[str, int] = {}
    for row in results:
        routes[row.get("route", "unknown")] = routes.get(row.get("route", "unknown"), 0) + 1
    return {
        "route": "szy",
        "sample_count": len(results),
        "positive_count": len(positive),
        "positive_nonempty": nonempty,
        # 三处"分母为 0 时给 0.0"的守卫：空数据集不崩、且 0.0 是中性值
        # （若没有正样本，PR 无意义，报 0 比报 NaN/除零异常更利于下游脚本）
        "final_pr": nonempty / len(positive) if positive else 0.0,
        "negative_count": len(negative),
        "negative_rejects": rejects,
        "final_rr": rejects / len(negative) if negative else 0.0,
        "errors": errors,
        "characters": characters,
        "final_cer": errors / characters if characters else 0.0,
        "duration": duration,
        "routes": routes,
        # 过程性统计：切换发生次数、异常兜底次数——它们不进竞赛分，
        # 但直接反映管线健康度（switches 太高说明频繁换通道，separation_errors
        # 高说明分离/ASR 环节在崩）
        "switches": sum(bool(row.get("switch_triggered")) for row in results),
        "separation_errors": sum(bool(row.get("error")) for row in results),
    }


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], output: Path, detail: Path) -> None:
    """写出竞赛提交物 submission.json 与全字段 detail.json。

    两个产物的定位不同：
      - submission.json 是"交卷"：只含竞赛要求的字段，格式严格；
      - detail.json 是"病历"：每样本全字段（含门控特征、耗时、路由），
        供事后离线重扫阈值、排查单条样本问题。二者同一批结果写出，
        保证口径一致。
    """
    # mkdir(parents=True, exist_ok=True)：递归创建父目录且已存在不报错——
    # "先保证目录存在再写文件"是写出类代码的标准动作（直接写不存在的
    # 路径会 FileNotFoundError）
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.parent.mkdir(parents=True, exist_ok=True)
    submission = []
    details = []
    for row in results:
        label = row.get("label")
        if label is not None and label != "":
            # 正样本：算该条样本自己的 CER（归一化后编辑距离 / 标注长度）
            reference = normalize_transcript(str(label))
            hypothesis = normalize_transcript(str(row.get("content") or ""))
            item_cer = (
                levenshtein(hypothesis, reference) / len(reference)
                if reference else float(hypothesis != "")
            )
        else:
            # 负样本没有"正确答案"，无 CER 可言，置 None（JSON 里写作 null）
            item_cer = None
        submission.append({
            "id": row["id"],
            "content": str(row.get("content") or ""),
            "label": label,
            "cer": item_cer,
        })
        # detail 行 = 原结果行的浅拷贝 + cer 字段。
        # 用 dict(row) 复制而不是直接改 row：不污染调用方持有的列表
        #（调用方可能还要拿原结果做别的统计）
        item = dict(row)
        item["cer"] = item_cer
        details.append(item)
    # json.dumps 的三个关键参数：
    #   ensure_ascii=False  中文按原样写入（默认会把中文转成 \uXXXX 转义，
    #                       文件里全是乱码一样的十六进制，人没法读）；
    #   indent=2            缩进 2 空格——牺牲一点体积换取可读性，
    #                       提交物体积不大，可读性更值钱
    output.write_text(
        json.dumps(
            {
                "result": {
                    "results": submission,
                    "avg_cer": float(summary["final_cer"]),
                    "avg_rr": float(summary["final_rr"]),
                    # 竞赛格式要求 duration 以毫秒为单位、保留 3 位小数
                    "duration": round(float(summary["duration"]) * 1000.0, 3),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    detail.write_text(json.dumps({"summary": summary, "results": details}, ensure_ascii=False, indent=2), encoding="utf-8")
