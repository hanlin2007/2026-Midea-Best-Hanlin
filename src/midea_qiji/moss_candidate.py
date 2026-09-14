"""Shared decision rules for the cached MossFormer candidate experiment.

This module intentionally contains only the two decisions that produced the
historical offline result:

* wake source selection is conditioned on the supplied wake text;
* command replacement is enabled only when the original FireRed hypothesis is
  unreliable, then uses same-separation differential speaker evidence.

MossFormer itself is a blind candidate generator.  It is not a replacement
for the primary dual-MLP gate and this module must not be enabled in online
inference until a separate end-to-end evaluation passes.

（中文补充）本模块 = "冻结的决策规则层"：不碰任何模型、音频或文件 IO，
只做"给定特征 -> 输出决策"的纯计算。所谓"冻结"，指这些规则与阈值是历史
离线实验定下的、产出过被记录的数字（如 DatasetA CER 0.3058），任何一处
改动都会让旧产物无法复现——tests/test_moss_candidate.py 就是专门锁死
这些边界的语义锁测试。

背景：MossFormer2 是盲源分离器（不依赖声纹信息，直接把混合音频拆成两路
"可能属于不同人"的通道 ms0/ms1）。它只是"候选生成器"：分离结果不可靠时
要退回 FireRed 对原始混合音频的转写（orig），只有证据充分时才改用分离
通道的转写。本模块的所有函数就是"用不用候选、用哪个候选"的裁决规则。

这类规则代码的工程要求（写规则/策略代码的好习惯）：
  - 纯函数：只读参数、不写全局、不做 IO、不打印。同样的输入永远给出
    同样的输出——单测可以直接断言，离线重放可以逐条复现；
  - 关键字实参（参数列表里的 ``*``）：规则函数常有一堆同类型浮点特征，
    调用处若不强制写名字（confidence=...），顺序传错根本不会报错，
    只会在评测里悄悄掉点。强制命名把这类事故挡在写代码阶段；
  - 结果用 frozen dataclass 返回（见下）：决策记录不可被事后篡改。

关键导出：``select_wake_anchor``、``should_switch_command``（及其
timbre/text 变体）、``normalized_edit_distance``、``choose_command_candidate``、
``as_float``。上层（szy 管线）从这里 import 规则，而不是自己复制粘贴。
"""
from __future__ import annotations  # 延迟求值注解（int | None 等），无运行时开销

from dataclasses import dataclass  # 免手写 __init__/__repr__；frozen=True 提供冻结语义
from typing import Any  # as_float 的缓存字典值类型未知，用 Any 表达"来者不拒"

from .text import levenshtein, normalize_transcript  # 相对导入：内核层内部引用，与包名解耦


# 决策规则 schema 的版本号：规则结构或阈值一旦变动，必须同步改这个号。
# 它被写进离线缓存/报告的元数据，用于区分"这份产物是用哪版规则算的"；
# 不换版本号就改规则，会让新旧规则的结果混在同一批缓存里，复现无从谈起。
DECISION_SCHEMA_VERSION = "moss-candidate-reproduction-v1"


# frozen=True 的意义（工程上很关键）：
#   - 冻结：构造完成后字段不可再赋值（赋值抛 FrozenInstanceError）。
#     决策记录是实验的"物证"——缓存重放时这些对象会被反复传递，任何
#     一处误改（比如顺手把 selected_source 改成 0）都会让复现结果静默
#     失真；冻结让这类 bug 在发生的第一时间就抛异常；
#   - 附带收益：可哈希（能放进 set / 当 dict 的键），相等比较按字段值；
#   - dataclass 本身免去手写 __init__/__repr__，repr 打印出的字段全貌
#     正好可以直接写进日志和报告。
@dataclass(frozen=True)
class WakeAnchorDecision:
    """Wake-source decision before differential command scoring.

    （中文）唤醒锚点决策的记录：唤醒音频被分离成两路（source0/source1）
    后，哪一路被认定为"目标说话人的唤醒语音"，作为后续门控/声纹注册的
    锚点。字段含义：
      - selected_source / contrast_source：选中哪一路做锚点、哪一路做
        "同分离域对比项"（同一批分离产物里的另一路，用来构造差分证据）；
      - raw_wake_fallback：是否退回"整段原始唤醒音频"当锚点（分离产物
        不可用时走这条路）；
      - text_tie：两路文本与唤醒文本的编辑距离是否打平；
      - source0_edit / source1_edit：两路各自的编辑距离；None 表示该侧
        没有转写、无从比较（对应注解里的 int | None）；
      - reason：人类可读的决策路径标识（写进报告，排查时一眼定位走了
        哪个分支）。
    """

    selected_source: int
    contrast_source: int
    raw_wake_fallback: bool
    text_tie: bool
    source0_edit: int | None
    source1_edit: int | None
    reason: str


# 与 WakeAnchorDecision 同为 frozen dataclass：冻结语义 / 可哈希 / 自带
# repr，理由同上——决策记录不允许被事后改动。
@dataclass(frozen=True)
class CommandCandidateDecision:
    """Which transcript is used for one primary-gate accepted positive.

    （中文）指令转写候选决策：门控"接受"的样本，最终提交的 content 用
    哪一份转写。字段含义：
      - selected_candidate：三选一——"orig"（FireRed 对原始混合音频的
        转写）、"ms0"/"ms1"（MossFormer 分离通道的转写）；
      - attempted_switch：是否"尝试过"切换（规则被触发即为 True，即便
        最后因分离失败退回 orig）——报告里必须能区分"根本没尝试"与
        "尝试了但失败"；
      - command_source_empty：分离出的指令通道是否为空（空 = 分离失败）；
      - reason：决策路径标识。
    """

    selected_candidate: str
    attempted_switch: bool
    command_source_empty: bool
    reason: str


def select_wake_anchor(
    wake_text: str | None,
    source0_text: str | None,
    source1_text: str | None,
) -> WakeAnchorDecision:
    """Select the source that uttered ``wake_text`` using legacy-compatible rules.

    A missing transcript marks the corresponding separated source as unusable
    for enrollment.  The full wake waveform is then the anchor, while the
    silent source remains a same-separation contrast term.  When both sources
    contain speech, unnormalized character edit distance preserves the exact
    historical selection behavior.

    （中文）裁决流程（三分支）：
      1. 只有一路有转写：selected_source 记有声那路，但
         raw_wake_fallback=True 表示实际锚点退回整段原始唤醒音频
         （静默路没有成对信息可用），静默路仅作同分离域的对比项；
      2. 两路都有转写：谁与唤醒文本的编辑距离小选谁——编辑距离度量
         "哪路的内容更像目标人说的唤醒词"（见 text.levenshtein）；
      3. 两路都空（退化情形）：沿用历史管线的行为——选 source0、锚点
         用原始唤醒音频、对比项也设成 0。这个分支几乎不会出现，但必须
         与历史逐字一致：缓存重放要求确定性，"顺手改成更合理的写法"
         都会让重放结果对不上历史产物。
    历史兼容说明：唤醒文本（人工标注）走 normalize_transcript 归一化；
    而 source0/source1 的转写只做 strip()——因为历史缓存写盘时它们已经
    是 FireRed 归一化后的字符串，重放直接用原串比对编辑距离，才能逐字
    复现当年的决策（见下方英文注释）。
    """

    reference = normalize_transcript(wake_text or "")  # 唤醒文本统一口径后再比对（None 兜底成空串）
    # Candidate transcripts were already normalized by FireRed when the
    # historical cache was written.  Replay uses those cached strings directly
    # for emptiness and edit distance rather than normalizing them again.
    # （中文）这段注释是"复现优先"原则的典型例子：即使"再归一化一次"看
    # 似更正确，也会让重放结果偏离当年写盘时的原串——保持与历史逐字一致
    # 比"逻辑上更漂亮"重要。
    source0 = str(source0_text or "").strip()  # 只去首尾空白：缓存里的字符串已是归一化文本
    source1 = str(source1_text or "").strip()
    has0, has1 = bool(source0), bool(source1)  # "该路是否有内容"：有内容才能参与编辑距离比较

    if has0 and not has1:  # 分支 1a：只有 source0 有声 -> 它当锚点，静默的 source1 当对比项
        return WakeAnchorDecision(0, 1, True, False, None, None, "source1_empty_raw_wake")
    if has1 and not has0:  # 分支 1b：对称情形
        return WakeAnchorDecision(1, 0, True, False, None, None, "source0_empty_raw_wake")

    edit0 = levenshtein(source0, reference)  # 两路都有转写：各自与唤醒文本比编辑距离
    edit1 = levenshtein(source1, reference)
    selected = 0 if edit0 <= edit1 else 1  # 距离小者胜；平手取 source0（<= 而非 <，复刻历史比较）
    tie = not reference or edit0 == edit1  # reference 为空时编辑距离无意义，一律视为打平
    if not has0 and not has1:
        # The old pipeline used source 0 as a contrast term in this degenerate
        # case.  Keeping it makes cached replay deterministic.
        # （中文）双空退化分支：保留历史行为以保证缓存重放确定性。
        return WakeAnchorDecision(selected, 0, True, tie, edit0, edit1, "both_sources_empty_raw_wake")
    return WakeAnchorDecision(
        selected,
        1 - selected,  # 对比项 = 没被选中的那一路（同一批分离产物里的另一路）
        False,
        tie,
        edit0,
        edit1,
        "wake_text_edit_distance",
    )


def should_switch_command(*, confidence: float, command_keyword_score: float, chars_per_sec: float) -> bool:
    """The frozen two-dimensional FireRed confidence switch from the best run.

    （中文）文本触发规则（历史最佳运行版；现役管线已改用音色规则，但本
    函数保留且冻结，供对比与重放）。输入三个特征：
      - confidence：FireRed 对原始混合指令音频的转写置信度；
      - command_keyword_score：指令文本里关键词的得分（"像不像指令"）；
      - chars_per_sec：每秒字符数 = 语速。
    结构解读——置信度把决策空间切成两半：
      - confidence <= 0.70（低置信区）：模型自己都不确定，规则宽松——
        关键词像指令、或语速过快（ASR 对混流"胡说"时语速会异常高）、
        或置信度极低（<= 0.57 必坏线）任一成立即切换；
      - confidence > 0.70（高置信区）：只有"语速快且内容不像指令"才切
        ——转写可信但内容可疑，怀疑转错了人。
    所有阈值（0.70 / 0.57 / 0.5 / 3.0 / 1.01）都是历史实验定下的冻结值，
    tests/test_moss_candidate.py 对这些边界逐一断言（如 0.57 必切、
    1.01+3.0 切、1.01+3.0+kw0.5 不切），改动任何数字都会挂测试。
    注意参数前的 ``*``：强制调用处写关键字实参——三个同类型浮点参数
    顺序极易传错且不报错，强制命名杜绝此类事故。
    """

    if confidence <= 0.70:  # 低置信区：三个"或"条件，任一命中即切换
        return command_keyword_score >= 0.5 or chars_per_sec >= 3.0 or confidence <= 0.57
    return confidence <= 1.01 and chars_per_sec >= 3.0 and command_keyword_score < 0.5  # 高置信区：三个"与"条件从严


def normalized_edit_distance(left: str | None, right: str | None) -> float:
    """归一化编辑距离：0 = 完全一致，1 = 完全不同。

    为什么归一化：原始编辑距离是"绝对错字数"，与文本长度成正比——
    "开灯"错 1 字和"把卧室空调调到二十六度"错 1 字的严重程度完全不同。
    除以两者较长者的长度，把距离压缩到 [0, 1]，不同长度的文本才能放在
    一起比较、才能与固定阈值（如 0.5）对比。
    分母 max(len(a), len(b), 1) 里的 1 是防除零：两个空串比较时长度都
    是 0，此时编辑距离也是 0，0 / 1 = 0，语义正好是"完全一致"。
    """

    a, b = str(left or ""), str(right or "")  # None 兜底成空串：允许调用方直接传可能为 None 的转写
    return levenshtein(a, b) / max(len(a), len(b), 1)


def should_switch_command_timbre(*, confidence: float, command_keyword_score: float, sv_diff: float) -> bool:
    """音色触发规则（SZY 生产管线当前采用）。

    sv_diff = max(cos(anchor, cmd_ms0), cos(anchor, cmd_ms1)) - cos(anchor, raw_cmd)，
    直接测量"目标嗓音在分离通道里是否比在原始混合里更突出"——切换收益的
    物理机制本身。三特征分工：conf 管必坏线（<=0.57 必切）、sv_diff 管主判决、
    kw 管高置信否决（自信转写但不像指令 + 音色说分离更好 => 转错了人）。
    DatasetA：CER 0.3058（文本规则 0.3092 / cps 规则 0.3181）；
    τ 盆地 [0.02, 0.08] 平坦，conf 切点在 [0.50,0.70]x[0.65,0.80] 内平坦。

    （中文补充）与 should_switch_command 同构的两段式结构，只是把语速
    特征换成音色差 sv_diff（声纹余弦相似度的差值）：低置信区宽松放行
    （kw 像指令 / sv_diff 超阈值 / 必坏线），高置信区从严（只有 sv_diff
    强且内容不像指令才切）。"盆地平坦"是阈值稳健性的证据：阈值在区间内
    挪一点结果几乎不变，说明规则没有过拟合到某个幸运数字——做阈值类
    调参时，选盆地中心而不是悬崖边，是通用经验。
    """

    if confidence <= 0.70:  # 低置信区：三个"或"条件，任一命中即切换
        return command_keyword_score >= 0.5 or sv_diff > 0.05 or confidence <= 0.57
    return sv_diff > 0.05 and command_keyword_score < 0.5  # 高置信区：音色证据强且内容不像指令才切


def should_switch_command_text(*, confidence: float, command_keyword_score: float, d_sel: float) -> bool:
    """文本一致性触发规则（历史路线采用，无语速）。

    d_sel 是 raw 转写与 ft 余弦所选分离通道转写的归一化编辑距离，
    直接测量"raw 里装的是不是目标说话人的内容"，替代原规则的语速特征
    （语速是对"ASR 混流"的间接假设）。分档结构沿用原规则：
    DatasetA 上 CER 0.309 vs 原规则 0.318，且阈值在 [0.35, 0.70] 盆地里平坦。

    （中文补充）与上面两个规则同构：低置信区三选一放行，高置信区"文本
    不一致且内容不像指令"才切。三个规则函数结构完全一致、只有第三个
    特征不同，正是"纯函数 + 同构接口"的好处——上层调用方想换触发规则，
    只改一处 import 即可，其余代码不动。
    """

    if confidence <= 0.70:  # 低置信区：三个"或"条件，任一命中即切换
        return command_keyword_score >= 0.5 or d_sel >= 0.5 or confidence <= 0.57
    return d_sel >= 0.5 and command_keyword_score < 0.5  # 高置信区：文本不一致且不像指令才切


def choose_command_candidate(
    *,
    should_switch: bool,
    source0_text: str | None,
    source1_text: str | None,
    source0_difference: float | None,
    source1_difference: float | None,
) -> CommandCandidateDecision:
    """Choose ``orig``/``ms0``/``ms1`` without changing the ASR waveform.

    An empty MossFormer command source is treated as a failed separation and
    forces the original FireRed output.  Difference ties resolve to ``ms0``;
    this mirrors the prior experiment's ``>=`` comparison.

    （中文）"切换已触发"之后的最后一步：从三个候选中挑一个作为最终提交
    的 content。注意函数只换"取谁的转写文本"，不换 ASR 输入的波形——
    声学证据保持原样，这是可复现性的要求。分支逻辑：
      1. 规则没触发 -> 直接用 orig（原始 FireRed 转写），attempted_switch
         记 False；
      2. 触发但任一分离通道转写为空 -> 分离失败，宁可信原始转写也不拿
         半个候选充数；attempted_switch=True 但 command_source_empty=True，
         报告里能看到"尝试过、被分离质量拦住"；
      3. 差分为 None -> 抛 ValueError：调用方漏传了选源必需的证据。
         显式失败优于猜一个默认值——漏传证据是编程错误，不是数据问题；
      4. 正常选源：差分值大者胜（更大 = 该分离通道与目标嗓音更匹配）；
         平手取 ms0，复刻历史实验的 >= 比较（改动会破坏复现）。
    """

    if not should_switch:  # 分支 1：规则没触发，直接信原始转写
        return CommandCandidateDecision("orig", False, False, "confidence_rule_not_triggered")
    if not str(source0_text or "").strip() or not str(source1_text or "").strip():  # 分支 2：分离产物为空
        return CommandCandidateDecision("orig", True, True, "command_source_empty")
    if source0_difference is None or source1_difference is None:  # 分支 3：证据缺失是编程错误
        raise ValueError("differential scores are required when a command switch is attempted")
    selected = "ms0" if float(source0_difference) >= float(source1_difference) else "ms1"  # 分支 4：差分大者胜；float() 兼容缓存里存成字符串的数值；平手取 ms0（历史口径）
    return CommandCandidateDecision(selected, True, False, "same_domain_differential_score")


def as_float(mapping: dict[str, Any], field: str, identifier: str) -> float:
    """从缓存 dict 里取一个字段并转 float，失败时给出带样本标识的错误。

    这是典型的"防御式解析"助手：缓存文件（jsonl）里的字段可能缺失、
    类型不对、或根本不是数字。三个细节值得学：
      - 一次捕获 KeyError / TypeError / ValueError 三种失败：键不存在 /
        类型没法转 / 值不是合法数字；
      - ``from error`` 异常链：新异常保留原始异常的堆栈（__cause__），
        排查时能看到"到底因为什么失败的"，而不是只剩一句笼统的报错；
      - identifier 参数把样本 id 织进错误消息：批量处理大量缓存记录时，
        报错能直接定位是哪条记录坏了。
    """
    try:
        return float(mapping[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{identifier} has no valid {field!r}") from error
