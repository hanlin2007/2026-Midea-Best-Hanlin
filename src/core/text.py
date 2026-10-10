"""core.text：文本归一化、编辑距离与 CER 指标。

全仓库的文本口径统一由本模块负责：数字/百分号转中文、去标点空白、小写
（``normalize_transcript``），字符级编辑距离（``levenshtein``），CER
（``cer``），以及从各类 ASR 输出结构中抽取文本（``extract_asr_text``）。

为什么"比对文本前必须先归一化"：ASR 输出的文本在格式上极不稳定——同一
句话可能被转写成阿拉伯数字（"25度"）或汉字（"二十五度"），可能带大小写、
标点、空白，甚至带模型专属的特殊 token（如 Qwen 系列的 ``<|...|>``）。
如果直接比较原始字符串，编辑距离会被这些"与内容无关"的格式差异污染，
CER 就不再反映真实识别错误。归一化把所有文本统一到同一口径（全中文、
无标点、小写），编辑距离于是只度量"内容差异"。

本模块同时是冻结语义重灾区：normalize_transcript 与官方评测口径一致，
levenshtein 是 moss_candidate 决策规则的基础。这俩函数任何一行改动都会
改变下游数字（CER/PR/RR 以及唤醒锚点选择），改动前先跑
``tests/test_infer.py`` 与 ``tests/test_moss_candidate.py``。

依赖：仅标准库（re / typing）。原实现在 infer.py，拆分后 infer.py 通过
re-export 保持 ``from core.samples import normalize_transcript`` 等
旧导入路径可用。
"""
from __future__ import annotations  # 延迟求值注解：str | None 等新式注解存成字符串，不产生运行时开销

import re  # 正则：归一化管线的四步替换全靠它
from typing import Any  # Any = "来者不拒"：extract_asr_text 的入参来自不同后端，结构未知


def extract_asr_text(result: Any) -> str:
    """从 Qwen3-ASR 等后端返回结构中递归抽取文本字段。

    不同 ASR 后端返回结构五花八门：有的返回带 ``.text`` 属性的对象
    （Qwen3-ASR 的结果类型），有的直接返回字符串，有的把文本包在
    dict / list 里甚至多层嵌套。与其让每个调用方各写一套 if-else
    探测（必然口径漂移、有的写错），不如集中在这里做一次防御式递归
    解析。

    两个设计要点：
      - 递归：list 的每个元素、dict 的每个 value 都可能再藏一层结构，
        所以探测本身要能自我调用；
      - 探测失败显式抛 ValueError，而不是静默返回空串——空串会污染
        下游 CER（看起来像"识别出空"，实际是"没解析出来"），显式失败
        让问题在最早的位置暴露。
    """
    text = getattr(result, "text", None)  # 优先取对象属性（Qwen3-ASR 的 result 对象带 .text）
    if text is not None:
        return str(text)
    if isinstance(result, str):  # 本身就是裸字符串，直接收下
        return result
    if isinstance(result, dict):
        for key in ("text", "sentence", "raw_text"):  # 常见 dict 键名，按出现概率排序
            if key in result:
                return str(result[key])
        for value in result.values():  # 没有认识的键 -> 逐个值递归探测
            try:
                return extract_asr_text(value)
            except ValueError:  # 该子树里没有文本，换下一个值继续
                continue
    if isinstance(result, list):
        parts = []
        for value in result:
            try:
                parts.append(extract_asr_text(value))
            except ValueError:  # 列表里夹着纯元数据元素时跳过
                continue
        if parts:
            return "".join(parts)  # 多段文本按原顺序拼接成一段
    raise ValueError(f"cannot extract ASR text from {result!r}")


def normalize_transcript(text: str) -> str:
    """转写文本口径归一化：数字转中文、百分号展开、去标点/空白、小写。

    与官方评测口径一致（CER 计算前必须调用）。冻结语义，改动会破坏
    CER/PR/RR 复现。

    处理步骤是精心排序的，每步依赖前一步的结果（不能乱序）：
      1. 删除 Qwen 系列模型的特殊 token（``<|...|>`` 形式）；
      2. "数字+百分号"（如 "30%"）先改写成 "百分之30"；
      3. 再把 "百分之+数字" 整体转成 "百分之三十"；
      4. 把剩余的所有数字串转成中文读法（"25" -> "二十五"）；
      5. 最后删除全部标点/空白并转小写。
    第 2、3 步单独处理百分号，是因为官方评测把 "30%" 视为 "百分之三
    十"：若直接走第 4 步会得到半中文半数字的 "百分之30"，与人工标注
    差 3 个字符，白白抬高 CER。
    """
    digits = "零一二三四五六七八九"  # 下标 0-9 恰好对应数值 0-9，天然充当映射表

    def number_to_chinese(value: int) -> str:
        # 阿拉伯数字 -> 中文读法。规则刻意从简（这是"口径约定"，不是
        # 语言学完备实现）：
        #   - 十位数省略开头的"一"：读"十五"而不是"一十五"；
        #   - 百位数中间的"零"不能省：否则"105"和"150"都会读成"一百五"，
        #     无法区分；
        #   - 1000 及以上直接返回原数字串：语音指令场景几乎不会出现长
        #     数字，而大数中文读法（万/亿进位）规则复杂，保持简单反而
        #     稳定、可预期。
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
            return prefix + number_to_chinese(remainder)  # 十位/个位复用上面的规则（递归）
        return str(value)

    def convert_number(match: re.Match[str]) -> str:
        # re.sub 的回调形式：每匹配到一个数字串，就交给 number_to_chinese。
        return number_to_chinese(int(match.group(0)))

    normalized = re.sub(r"<\|[^|]+?\|>", "", str(text))  # 步骤 1：剥掉模型特殊 token（如 <|endoftext|>）
    normalized = re.sub(r"(\d+)\s*%", r"百分之\1", normalized)  # 步骤 2："30%" -> "百分之30"
    normalized = re.sub(r"百分之\s*(\d+)", lambda match: "百分之" + number_to_chinese(int(match.group(1))), normalized)  # 步骤 3：-> "百分之三十"
    normalized = re.sub(r"\d+", convert_number, normalized)  # 步骤 4：剩余数字串全部转中文
    return re.sub(r"[\s,，。.!！?？:：;；、】【、\"'“”‘’（）()\[\]<>《》|/\\_-]+", "", normalized.lower())  # 步骤 5：删标点/空白 + 小写


def levenshtein(left: str, right: str) -> int:
    """字符级编辑距离（O(len(left)*len(right)) 空间 O(len(right))）。

    编辑距离 = 把 left 变成 right 所需的最少单字符编辑次数（插入、删除、
    替换各计 1 次）。时间复杂度 O(len(left)*len(right))、空间复杂度
    O(len(right))，做法见下。经典解法是动态规划，定义：

        dp[i][j] = left 前 i 个字符变成 right 前 j 个字符的最小编辑次数

    转移方程（三种操作的代价取最小）：

        dp[i][j] = min(
            dp[i][j-1] + 1,                  # 在 left 末尾插入 right 的第 j 个字符
            dp[i-1][j] + 1,                  # 删除 left 的第 i 个字符
            dp[i-1][j-1] + (两字符不同则 1)   # 替换 / 字符相同则不花代价
        )

    空间优化：每个格子只依赖"上一行 + 本行左侧"两个邻居，因此不必保存
    整个二维表，滚动两行数组即可，空间从 O(m*n) 降到 O(n)。开头的三个
    边界短路（相等 / 空串）既是性能优化，也顺便处理了 dp 表的第 0 行
    第 0 列。基于字符而非词，中文场景天然适用（每个汉字一个字符）。
    """
    if left == right:  # 完全一致：0 步（最常见的快速路径）
        return 0
    if not left:  # left 为空：全是对 right 的插入
        return len(right)
    if not right:  # right 为空：全是删除
        return len(left)
    previous = list(range(len(right) + 1))  # dp 第 0 行：空 left 变成 right 前 j 个字符需要 j 次插入
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]  # 本行第 0 列：right 为空时，删除 left 前 i 个字符需要 i 次
        for right_index, right_character in enumerate(right, start=1):
            # 三种操作取最小：插入（当前行左邻 +1）/ 删除（上一行同列 +1）/
            # 替换（上一行左邻 + 字符是否不同）。
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_character != right_character)))
        previous = current  # 滚动：处理下一行时，本行就是"上一行"
    return previous[-1]  # 表右下角 = 完整两个字符串的编辑距离


def cer(hypothesis: str, label: str | None) -> float | None:
    """字符错误率；label 为空串时按 hypothesis 是否非空计 0/1，None 返回 None。

    CER = 编辑距离 / 标注长度。除以标注长度而不是假设长度，让"长句错
    几个字"占比更低——以标注为基准才公平（否则说长了反而占便宜）。
    两个边界约定：
      - label 是空串：分母为 0 无意义，约定按"识别结果是否也为空"记
        0 或 1（对上了算全对，多说了算全错）；
      - label 是 None：该样本没有标注（负样本 / 被拒绝样本），返回
        None 表示"不参与统计"，由上层汇总时跳过。
    """
    if label is None:
        return None
    return levenshtein(hypothesis, label) / len(label) if label else float(hypothesis != "")
