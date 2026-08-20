"""MossFormer 候选路线的"共享决策规则"（纯逻辑，无模型、无 IO）。

================================================================================
【这个文件在整条流水线里做什么？】

它不加载任何模型，也不读任何文件——只装着 SZY v1 候选路线里最关键的
几个"决策函数"。把这些决策单独抽成一个文件，好处是：
  1. 它们是"纯函数"（输入确定 → 输出确定），极易单元测试（tests/test_moss_candidate.py 测的就是它们）；
  2. 与沉重的模型代码隔离，可以独立审查、独立复现；
  3. 决策边界被"冻结"在这里，后人改动一眼可见。

文件里只有两个真正用于生产的决策：
  * select_wake_anchor()：唤醒分离后，用"唤醒文本"判断两路分离源里
    哪一路是目标人说的唤醒词（选出来当声纹锚点 anchor）。
  * should_switch_command()：根据 raw FireRed 的置信度/文本特征，
    判断"要不要进入 MossFormer2 候选路线"。

另外 choose_command_candidate() 是历史缓存回放工具用的（离线评估用），
主流程 pipeline.py 里用"内联的等价逻辑"实现，不走这个函数——详见注释。

重要纪律（来自队长文档）：
  MossFormer2 本身只是"盲分离"的候选生成器——它把混合语音分成两路，
  但分出来的两路谁是谁，模型自己并不知道。所以它【不能】替代主门控，
  只能在主门控通过后、且 raw 结果不可靠时，作为"候选替换"使用。
================================================================================
"""
from __future__ import annotations  # 延迟注解求值

from dataclasses import dataclass  # 数据类：把"一个决策的结果"打包成一个结构化对象
from typing import Any

from .infer import levenshtein, normalize_transcript  # 相对导入：从同包 infer.py 借两个函数
# .infer 里的点号 = "当前包（midea_qiji）里的 infer 模块"（相对导入，见课程第 1 课）


DECISION_SCHEMA_VERSION = "moss-candidate-reproduction-v1"  # 决策结构版本号（可审计性）


@dataclass(frozen=True)
class WakeAnchorDecision:
    """唤醒锚点选择的结果（frozen=True：创建后只读，防止中途被篡改）。

    字段含义：
      selected_source   : 选中的分离源编号（0 或 1）
      contrast_source   : 作为"对照"的另一路分离源编号
      raw_wake_fallback : True 表示没有可信分离源，直接用【原始完整唤醒音频】当锚点
      text_tie          : 两路编辑距离打平（或参考文本为空）
      source0_edit      : source0 文本与唤醒文本的编辑距离（None=该路为空没算）
      source1_edit      : source1 文本与唤醒文本的编辑距离
      reason            : 决策理由（机器可读字符串，写进 detail.json 供审计）
    """

    selected_source: int
    contrast_source: int
    raw_wake_fallback: bool
    text_tie: bool
    source0_edit: int | None
    source1_edit: int | None
    reason: str


@dataclass(frozen=True)
class CommandCandidateDecision:
    """命令候选替换的结果（历史回放工具用）。

    selected_candidate: "orig"（保留原始）/ "ms0" / "ms1"
    attempted_switch  : 是否真的尝试过替换
    command_source_empty: 是否有任一分离命令源为空（空 = 分离失败）
    reason            : 决策理由
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
    """用"唤醒文本"从两路分离源里选出目标人说话的锚点。

    背景：唤醒音频（kws.wav）是多人重叠的混合音。MossFormer2 把它盲分离成
    两路 source0 / source1。但我们不知道哪一路是"说唤醒词的人"。怎么办？
    用已知的唤醒文本（如"你好科慕"）去比对两路的 ASR 转写：
    哪一路转写更接近唤醒文本，哪一路就是目标人 → 作为声纹锚点（anchor）。

    规则（与历史行为逐字一致，不能改）：
      - 只有 source0 非空 → 选 source0，对照 source1，但锚点用【原始完整唤醒音频】
      - 只有 source1 非空 → 选 source1，对照 source0，锚点用【原始完整唤醒音频】
      - 两路都非空       → 编辑距离更小的一路当选，另一路当对照
      - 两路都为空       → 锚点用【原始完整唤醒音频】，对照固定 source0（保证确定性）
      - 编辑距离相等     → 固定选 source0（保证可复现，不引入随机性）

    为什么某一路为空时要用"原始完整唤醒音频"当锚点？
      因为空的那路是分离失败的产物（可能全是静音/噪声），而"原始完整唤醒音"
      虽然含噪声，但【完整保留了目标人的全部声纹信息】——短唤醒、强重叠、
      分离失败时，原始音反而比残缺的分离源更可靠。这是关键的 raw fallback 设计。
    """
    # 唤醒文本也过一遍归一化（与转写同一口径），保证比较公平
    reference = normalize_transcript(wake_text or "")
    # 候选转写在历史缓存写入时已经归一化过了，回放时直接用缓存字符串
    # 判断空与非空、算编辑距离，不重复归一化（重复归一化可能改历史行为）
    source0 = str(source0_text or "").strip()
    source1 = str(source1_text or "").strip()
    has0, has1 = bool(source0), bool(source1)  # 元组同时赋值：两个布尔标志

    if has0 and not has1:  # 只有 source0 有内容
        return WakeAnchorDecision(0, 1, True, False, None, None, "source1_empty_raw_wake")
    if has1 and not has0:  # 只有 source1 有内容
        return WakeAnchorDecision(1, 0, True, False, None, None, "source0_empty_raw_wake")

    # 两路都非空 → 算编辑距离（谁离唤醒文本"更近"谁当选）
    edit0 = levenshtein(source0, reference)
    edit1 = levenshtein(source1, reference)
    selected = 0 if edit0 <= edit1 else 1  # <=：距离更小或相等都选 source0（确定性）
    tie = not reference or edit0 == edit1  # 打平：参考为空，或两路距离相同
    if not has0 and not has1:  # 两路都为空（退化情况）
        # 历史管线在这种退化情形下固定用 source0 作对照，保留它保证缓存回放确定
        return WakeAnchorDecision(selected, 0, True, tie, edit0, edit1, "both_sources_empty_raw_wake")
    return WakeAnchorDecision(
        selected,
        1 - selected,  # 对照源 = 另一个（0→1，1→0 的巧妙写法）
        False,         # 两路都非空 → 不需要退回原始唤醒音
        tie,
        edit0,
        edit1,
        "wake_text_edit_distance",
    )


def should_switch_command(*, confidence: float, command_keyword_score: float, chars_per_sec: float) -> bool:
    """二维置信度切换规则（队长冻结的边界，改动即破坏复现）。

    输入三个特征，全部来自 raw FireRed 对原始命令音频的转写：
      confidence           : FireRed 自身给出的置信度（0~1 左右，不是目标人概率！）
      command_keyword_score: 转写文本里"命令关键词"的个数（打开/关闭/空调/音量……）
      chars_per_sec        : 转写文本长度 / 音频时长（字符每秒，衡量"语速"）

    规则（与文档第 7 节、测试用例逐字对应）：
      若 confidence <= 0.70：
          关键词数 >= 0.5  或  语速 >= 3.0  或  置信度 <= 0.57 → 切换
      若 confidence > 0.70：
          置信度 <= 1.01 且 语速 >= 3.0 且 关键词数 < 0.5 → 切换
      其余 → 不切换（保留 raw 文本）

    注意：这个函数只回答"要不要做候选复核"，它【不是】拒识器——切换后
    候选失败照样回退 raw 文本，不会因为切换就清空输出。
    """
    # 关键字参数用 * 强制：调用者必须显式写参数名（confidence=...），
    # 防止三个 float 参数顺序写错导致静默 bug
    if confidence <= 0.70:
        return command_keyword_score >= 0.5 or chars_per_sec >= 3.0 or confidence <= 0.57
    return confidence <= 1.01 and chars_per_sec >= 3.0 and command_keyword_score < 0.5


def choose_command_candidate(
    *,
    should_switch: bool,
    source0_text: str | None,
    source1_text: str | None,
    source0_difference: float | None,
    source1_difference: float | None,
) -> CommandCandidateDecision:
    """选择 "orig"/"ms0"/"ms1"（不改变 ASR 波形，只挑用哪路转写文本）。

    注意：这是【历史缓存回放工具】用的决策函数。主流程 pipeline.py 里的
    SzyV1Runner.process() 用的是内联等价逻辑（因为主流程要顺带记录更多
    审计字段），两者的判定口径完全一致。

    规则：
      - 不切换            → 保留 orig
      - 任一命令源为空    → 分离失败，保留 orig（不从单一路候选推断目标人）
      - 两路都非空        → 差分分数大的一路；相等固定 ms0（>= 的历史行为）
    """
    if not should_switch:
        return CommandCandidateDecision("orig", False, False, "confidence_rule_not_triggered")
    if not str(source0_text or "").strip() or not str(source1_text or "").strip():
        return CommandCandidateDecision("orig", True, True, "command_source_empty")
    if source0_difference is None or source1_difference is None:
        # 要切换却没给差分分数 → 这是调用方的 bug，必须报错（fail-fast）
        raise ValueError("differential scores are required when a command switch is attempted")
    selected = "ms0" if float(source0_difference) >= float(source1_difference) else "ms1"
    return CommandCandidateDecision(selected, True, False, "same_domain_differential_score")


def as_float(mapping: dict[str, Any], field: str, identifier: str) -> float:
    """从 dict 里取一个字段并强制转 float（历史回放工具读缓存用的辅助函数）。"""
    try:
        return float(mapping[field])
    except (KeyError, TypeError, ValueError) as error:
        # 三种失败：键不存在 / 值是 None / 值无法转数字，统一包成带上下文的错误
        raise ValueError(f"{identifier} has no valid {field!r}") from error
