"""pipeline.reporting：评测汇总与提交物写出。

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

依赖：core.text（levenshtein / normalize_transcript）。
（本模块同样不依赖 torch：汇总可以在任何轻量环境离线重算，甚至只拿
detail.json 就能重新统计，不必重新推理。）
"""
from __future__ import annotations

import json                                     # 提交物序列化
from pathlib import Path                        # 输出路径
from typing import Any                           # 结果行的宽松类型

from core.text import levenshtein, normalize_transcript


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


# ----------------------------------------------------------------------
# 人类可读报告（每次管线运行后自动生成，供逐样本排查）
# ----------------------------------------------------------------------

def _fmt(val, digits=4):
    """把 detail.json 里的字段格式化成短字符串；空值显示为 "-"。"""
    if val is None or val == "":
        return "-"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def _tag(route, accepted, selected_candidate):
    """给每条样本一个一眼可读的路由标签。"""
    if not accepted:
        return "REJECTED"
    if route in ("raw", "raw_command_single", "both_single_raw"):
        return "RAW (kept)"
    if route == "moss_candidate":
        ch = selected_candidate if selected_candidate else "ms0"
        return f"SEPARATED ({ch} chosen)"
    if "error" in route or "fallback" in route:
        return f"ERROR ({route})"
    return route


def generate_report(detail_path: str | Path) -> str:
    """从 detail.json 生成人类可读的纯文本报告（聚合指标 + 逐样本明细）。

    报告的用途是"逐样本病历"：每条样本给出标注 vs 输出 vs CER、唤醒
    通道转写与锚点标记、逐通道余弦、门控分数与路由决策，排查掉分样本
    时不必在 JSON 里翻字段。管线每次运行结束自动调用（见 pipeline.cli），
    也可离线单独生成：``python scripts/generate_report.py <detail.json>``。
    """
    data = json.loads(Path(detail_path).read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    results = data.get("results", [])

    lines: list[str] = []

    pr = summary.get("final_pr", 0)
    rr = summary.get("final_rr", 0)
    cer = summary.get("final_cer", 0)
    score = (1 - cer + rr) / 2 if cer is not None else 0
    dur = summary.get("duration", 0)
    n = summary.get("sample_count", len(results))
    n_pos = summary.get("positive_count", 0)
    n_neg = summary.get("negative_count", 0)
    n_acc_pos = summary.get("positive_nonempty", 0)
    n_rej_neg = summary.get("negative_rejects", 0)
    switches = summary.get("switches", 0)
    sep_errors = summary.get("separation_errors", 0)
    total_chars = summary.get("characters", 0)
    total_errors = summary.get("errors", 0)
    routes = summary.get("routes", {})

    lines.append("=" * 72)
    lines.append("  SZY Pipeline Report")
    lines.append(f"  Generated from: {Path(detail_path).resolve()}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("--- Aggregate Metrics ---")
    lines.append(f"  Samples       : {n}  (positive {n_pos}, negative {n_neg})")
    lines.append(f"  PR            : {pr:.4f}  ({n_acc_pos}/{n_pos} accepted with non-empty output)")
    lines.append(f"  RR            : {rr:.4f}  ({n_rej_neg}/{n_neg} negatives correctly rejected)")
    lines.append(f"  CER           : {cer:.4f}  ({total_errors}/{total_chars} characters wrong)")
    lines.append(f"  Score         : {score:.4f}  = (1 - CER + RR) / 2")
    lines.append(f"  Duration      : {dur:.1f}s")
    lines.append(f"  Switches      : {switches} triggered out of {n}")
    lines.append(f"  Sep Errors    : {sep_errors}")
    lines.append("")
    lines.append("--- Route Distribution ---")
    for route_name, count in sorted(routes.items(), key=lambda x: -x[1]):
        lines.append(f"  {route_name:30s} : {count:5d}  ({count/n*100:.1f}%)" if n else f"  {route_name}: {count}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("  Per-Sample Results")
    lines.append("=" * 72)

    for idx, row in enumerate(results, start=1):
        sid = row.get("id", f"sample_{idx}")
        sample_type = row.get("sample_type", "unknown")
        label = row.get("label")
        content = row.get("content", "")
        cer_val = row.get("cer")
        sv_score = row.get("sv_score")
        accepted = row.get("accepted", False)
        route = row.get("route", "")
        selected = row.get("selected_candidate", "")
        switch = row.get("switch_triggered", False)
        error = row.get("error")
        raw_conf = row.get("raw_confidence")
        sv_diff = row.get("sv_diff")
        duration = row.get("item_duration")

        raw_text = row.get("raw_text")
        cmd0_text = row.get("command_source0_text")
        cmd1_text = row.get("command_source1_text")
        wake0_text = row.get("wake_source0_text")
        wake1_text = row.get("wake_source1_text")
        wake_sel = row.get("wake_selected_source")
        wake_fallback = row.get("wake_raw_fallback")
        wake_tie = row.get("wake_text_tie")
        wake_contrast = row.get("wake_contrast_source")
        wake_reason = row.get("wake_reason")
        wake_sim0 = row.get("wake_source0_sim")
        wake_sim1 = row.get("wake_source1_sim")
        ft_cos_cmd0 = row.get("ft_cos_ms0")
        ft_cos_cmd1 = row.get("ft_cos_ms1")

        is_pos = sample_type == "positive"
        tag = _tag(route, accepted, selected)
        is_correct = (cer_val is not None and float(cer_val) == 0.0) if is_pos else (not content and not label)
        correctness = "CORRECT" if is_correct else ("ERROR" if is_pos and cer_val and float(cer_val) > 0 else "REJECTED" if not is_pos else "")

        lines.append("")
        lines.append(f"  [{idx:>5d}] {sid}")
        lines.append(f"         Type: {sample_type.upper()}  |  Tag: {tag}  |  {correctness}")
        lines.append(f"      {'':->64s}")

        if is_pos:
            lines.append(f"         Ground Truth : {label}")
            lines.append(f"         Final Output : {content}")
            lines.append(f"         Item CER     : {_fmt(cer_val)}")
        else:
            lines.append(f"         Final Output : '{content}' (should be empty)")

        # ---- 唤醒通道：转写 + 锚点选择依据 ----
        if wake0_text is not None or wake1_text is not None:
            wake_text = row.get("wake_text") or "(not provided)"
            lines.append("")
            lines.append("         --- Wake Audio Channels ---")
            lines.append(f"           Ground truth wake text : {wake_text}")

            ws0 = "" if wake_sel != 0 else " <-- ANCHOR (adopted)"
            ws1 = "" if wake_sel != 1 else " <-- ANCHOR (adopted)"
            ws0_c = "" if wake_contrast != 0 else " <-- contrast"
            ws1_c = "" if wake_contrast != 1 else " <-- contrast"
            lines.append(f"           ms0 transcription : {wake0_text}{ws0}{ws0_c}")
            lines.append(f"           ms1 transcription : {wake1_text}{ws1}{ws1_c}")
            lines.append(f"           Text tie          : {'YES' if wake_tie else 'no'}")
            lines.append("           Anchor Selection:")
            if wake_fallback:
                lines.append("             -> Used RAW wake audio as anchor (fallback)")
            else:
                lines.append(f"             -> Anchor = ms{wake_sel} (by {wake_reason})")

            if wake_sim0 is not None or wake_sim1 is not None:
                lines.append("           Phoneme Similarity vs wake text:")
                lines.append(f"             sim(ms0, ref)  : {_fmt(wake_sim0, 4)}")
                lines.append(f"             sim(ms1, ref)  : {_fmt(wake_sim1, 4)}")

        # ---- 指令通道：各候选转写 + 选路依据 ----
        if raw_text or cmd0_text or cmd1_text:
            lines.append("")
            lines.append("         --- Command Audio Channels ---")
            if raw_text is not None:
                raw_mark = " <-- CHOSEN (kept raw)" if selected == "orig" and route in ("raw", "raw_command_single", "both_single_raw") else (" <-- KEPT RAW" if selected == "orig" else "")
                lines.append(f"             raw (direct)       : {raw_text}{raw_mark}")
            if cmd0_text is not None:
                sel_mark0 = " <-- CHOSEN" if selected == "ms0" else ""
                lines.append(f"             cmd src0 (ms0)     : {cmd0_text}{sel_mark0}")
            if cmd1_text is not None:
                sel_mark1 = " <-- CHOSEN" if selected == "ms1" else ""
                lines.append(f"             cmd src1 (ms1)     : {cmd1_text}{sel_mark1}")
            if selected not in ("orig", "", None) and raw_text:
                lines.append(f"             -> Switched from raw to separated {selected}")
            elif selected == "orig":
                lines.append("             -> Kept raw output")

            lines.append("           Cosine Similarity vs Anchor (finetuned):")
            lines.append(f"             cos(anchor, cmd_ms0): {_fmt(ft_cos_cmd0):>7}")
            lines.append(f"             cos(anchor, cmd_ms1): {_fmt(ft_cos_cmd1):>7}")
            if ft_cos_cmd0 and ft_cos_cmd1:
                try:
                    c0 = float(ft_cos_cmd0)
                    c1 = float(ft_cos_cmd1)
                    winner = "ms0" if c0 >= c1 else "ms1"
                    lines.append(f"             ft winner             : {winner} (cos={max(c0,c1):.4f} vs {min(c0,c1):.4f}, delta={abs(c0-c1):.4f})")
                except (TypeError, ValueError):
                    pass

        # ---- 门控分数与后处理痕迹 ----
        lines.append("")
        lines.append("         --- Gate Scores ---")
        lines.append(f"             SV score        : {_fmt(sv_score, 4)}  (threshold: {row.get('sv_accept_threshold', '-')})")
        lines.append(f"             Accepted        : {'YES' if accepted else 'no'}")
        lines.append(f"             Raw confidence  : {_fmt(raw_conf, 4)}")
        lines.append(f"             SV diff         : {_fmt(sv_diff, 4)}")
        lines.append(f"             Switch triggered: {'YES' if switch else 'no'}")
        if row.get("nbest_applied") or row.get("elm_applied"):
            lines.append(f"             N-best applied  : {'YES' if row.get('nbest_applied') else 'no'}  ({row.get('nbest_reason', '-')})")
            lines.append(f"             ELM applied     : {'YES' if row.get('elm_applied') else 'no'}")
            if row.get("baseline_content") != content:
                lines.append(f"             Baseline text   : {row.get('baseline_content')}")
        if duration:
            lines.append(f"         Item duration     : {duration}s")
        if error:
            lines.append(f"         ERROR: {error}")

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  End of Report ({n} samples)")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)
