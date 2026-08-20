"""midea_qiji.text：文本归一化、编辑距离与 CER 指标。

全仓库的文本口径统一由本模块负责：数字/百分号转中文、去标点空白、小写
（``normalize_transcript``），字符级编辑距离（``levenshtein``），CER
（``cer``），以及从各类 ASR 输出结构中抽取文本（``extract_asr_text``）。

依赖：仅标准库（re / typing）。原实现在 infer.py，拆分后 infer.py 通过
re-export 保持 ``from midea_qiji.infer import normalize_transcript`` 等
旧导入路径可用。
"""
from __future__ import annotations

import re
from typing import Any


def extract_asr_text(result: Any) -> str:
    """从 Qwen3-ASR 等后端返回结构中递归抽取文本字段。"""
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "sentence", "raw_text"):
            if key in result:
                return str(result[key])
        for value in result.values():
            try:
                return extract_asr_text(value)
            except ValueError:
                continue
    if isinstance(result, list):
        parts = []
        for value in result:
            try:
                parts.append(extract_asr_text(value))
            except ValueError:
                continue
        if parts:
            return "".join(parts)
    raise ValueError(f"cannot extract ASR text from {result!r}")


def normalize_transcript(text: str) -> str:
    """转写文本口径归一化：数字转中文、百分号展开、去标点/空白、小写。

    与官方评测口径一致（CER 计算前必须调用）。冻结语义，改动会破坏
    CER/PR/RR 复现。
    """
    digits = "零一二三四五六七八九"

    def number_to_chinese(value: int) -> str:
        if value < 10:
            return digits[value]
        if value < 20:
            return "十" + (digits[value % 10] if value % 10 else "")
        if value < 100:
            return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
        if value < 1000:
            remainder = value % 100
            prefix = digits[value // 100] + "百"
            if remainder == 0:
                return prefix
            if remainder < 10:
                return prefix + "零" + digits[remainder]
            return prefix + number_to_chinese(remainder)
        return str(value)

    def convert_number(match: re.Match[str]) -> str:
        return number_to_chinese(int(match.group(0)))

    normalized = re.sub(r"<\|[^|]+?\|>", "", str(text))
    normalized = re.sub(r"(\d+)\s*%", r"百分之\1", normalized)
    normalized = re.sub(r"百分之\s*(\d+)", lambda match: "百分之" + number_to_chinese(int(match.group(1))), normalized)
    normalized = re.sub(r"\d+", convert_number, normalized)
    return re.sub(r"[\s,，。.!！?？:：;；、】【、\"'“”‘’（）()\[\]<>《》|/\\_-]+", "", normalized.lower())


def levenshtein(left: str, right: str) -> int:
    """字符级编辑距离（O(len(left)*len(right)) 空间 O(len(right))）。"""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_character != right_character)))
        previous = current
    return previous[-1]


def cer(hypothesis: str, label: str | None) -> float | None:
    """字符错误率；label 为空串时按 hypothesis 是否非空计 0/1，None 返回 None。"""
    if label is None:
        return None
    return levenshtein(hypothesis, label) / len(label) if label else float(hypothesis != "")
