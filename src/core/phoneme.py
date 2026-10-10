"""core.phoneme：拼音音素相似度与唤醒词别名表。

为什么需要"拼音相似度"而不是直接用字符编辑距离？
中文语音识别最常见的错误是"同音字"：参考文本是"你好科慕"，ASR 可能
输出"你好科目"——读音完全一样，字不同。字符级编辑距离会罚 1 个字，
但在"哪一路分离通道念了唤醒词"这个判断里，读音一致就已经是足够强的
证据。把文本转成拼音再算编辑距离，同音字自动归 0：

    "你好科慕" → "nihaokemu"
    "你好科目" → "nihaokemu"   # 距离 0，字符距离却是 1

这正是本模块在生产管线里的唯一用途：为
``moss_candidate.select_wake_anchor`` 提供"哪一路更像唤醒词"的打分
（分数 = 负的拼音编辑距离，越大越像）。

别名表（WAKE_WORD_ALIASES）解决另一个问题：同一个唤醒词有多个合法
写法/读法——英文 "hi colmo" 的拼音化是 "hicolmo"，中文"嗨科慕"的
拼音是 "haikemu"。ASR 输出其中任一个都算对。做法是把参考文本展开成
"所有已知变体"，对每个变体各算一次相似度，取最高分。

依赖：pypinyin（中文 → 拼音）。pypinyin 缺失时退化为小写原文比较
（功能降级但不崩，保证轻量环境可 import 本模块）。
"""
from __future__ import annotations

import re


# 唤醒词别名表（双向）：key 和 value 都是"去声调、小写、无空格"的拼音
# 字串。新增变体时两个方向都要能查到——这里显式写两条（A→B、B→A），
# 而不是代码里自动反转，是为了让"有哪些别名"一眼可见、增删不易出错。
WAKE_WORD_ALIASES = {
    "hicolmo": ["haikemu"],
    "haikemu": ["hicolmo"],
}


def _to_pinyin_str(text: str) -> str:
    """把文本转成紧凑拼音串（小写、去声调数字、无分隔）。

    例："你好科慕" → "nihaokemu"。

    Style.TONE3 把声调标成音节尾的数字（"ni3 hao3"），随后用正则把
    数字删掉——同音不同调的字在唤醒词匹配里不应有区别（ASR 并不保证
    声调正确）。pypinyin 未安装时退化为小写原文：锚点选择退回到
    "字符级"口径，分数语义不同但仍可用（防御性降级）。
    """
    try:
        from pypinyin import Style, lazy_pinyin
        # lazy_pinyin 返回音节列表：["ni3", "hao3", "ke4", "mu4"]
        syllables = lazy_pinyin(text or "", style=Style.TONE3)
        joined = "".join(syllables).lower()
        return re.sub(r"[0-9]", "", joined)
    except ImportError:
        return (text or "").lower()


def _levenshtein(a: str, b: str) -> int:
    """字符级编辑距离（滚动两行数组，空间 O(n)）。

    与 core.text.levenshtein 语义相同；这里单独留一份实现，是因为
    拼音串是 ASCII，本模块希望"只依赖标准库 + pypinyin（可选）"，
    不向上引用 text 模块，保持可以被任意层 import 的轻量性。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def phoneme_similarity(text_a: str, text_b: str) -> float:
    """拼音空间相似度：返回负的拼音编辑距离（0 = 读音完全一致，越大越像）。

    若 ``text_b`` 命中唤醒词别名表，先把它展开成所有已知变体，逐变体
    计算后取最高分——"只要像任一变体就算像"。

    任一输入转拼音后为空（如纯标点/空串）时返回一个极大的负值：
    空转写不该赢得任何锚点竞争，但又不能抛异常（分离通道经常转写为空，
    这属于正常输入而非错误）。
    """
    py_a = _to_pinyin_str(text_a)
    py_b = _to_pinyin_str(text_b)
    if not py_a or not py_b:
        return -1000000.0

    # 展开 text_b（通常是参考唤醒文本）的全部别名，逐个别名比一遍取最优
    aliases = [py_b]
    if py_b in WAKE_WORD_ALIASES:
        aliases.extend(WAKE_WORD_ALIASES[py_b])

    best_score = float("-inf")
    for alias in aliases:
        if not alias:
            continue
        # 距离越小越好 → 取负后越大越好；max 留下"最像的那个变体"
        best_score = max(best_score, -_levenshtein(py_a, alias))
    return best_score
