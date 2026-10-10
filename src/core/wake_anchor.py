"""core.wake_anchor：唤醒锚点选择的冻结决策规则（纯函数，无模型/IO）。

本模块只做一件事：唤醒音频被盲分离成两路（source0/source1）后，裁决
"哪一路是目标说话人念的唤醒词"，作为后续门控/声纹注册的锚点
（anchor）。锚点越纯，门控判决越准——注册进错了人，后面全错。

裁决依据是**拼音音素相似度**（core.phoneme）：把两路的 ASR 转写与
唤醒文本都转成拼音再比编辑距离。同音字（ASR 最高频的错误，如
"科慕" vs "科目"）在拼音空间里距离为 0，不会冤枉好通道——相对旧的
字符编辑距离版本，这是全系统在评测集上最大的单项护城河之一。

这类规则代码的工程要求（写规则/策略代码的好习惯）：
  - 纯函数：只读参数、不写全局、不做 IO、不打印。同样的输入永远给出
    同样的输出——单测可以直接断言，离线重放可以逐条复现；
  - 结果用 frozen dataclass 返回：决策记录不可被事后篡改；
  - 边界行为由 tests/test_wake_anchor.py 逐点锁死（"语义锁"）。

依赖：core.phoneme / core.text（均为轻量模块）。
"""
from __future__ import annotations  # 延迟求值注解（int | None 等），无运行时开销

from dataclasses import dataclass  # 免手写 __init__/__repr__；frozen=True 提供冻结语义

from .phoneme import phoneme_similarity              # 拼音空间相似度（锚点选择打分）
from .text import normalize_transcript               # 唤醒文本统一口径


# frozen=True 的意义（工程上很关键）：
#   - 冻结：构造完成后字段不可再赋值（赋值抛 FrozenInstanceError）。
#     决策记录是实验的"物证"——缓存重放时这些对象会被反复传递，任何
#     一处误改都会让复现结果静默失真；冻结让这类 bug 在发生的第一时间
#     就抛异常；
#   - 附带收益：可哈希（能放进 set / 当 dict 的键），相等比较按字段值；
#   - dataclass 本身免去手写 __init__/__repr__，repr 打印出的字段全貌
#     正好可以直接写进日志和报告。
@dataclass(frozen=True)
class WakeAnchorDecision:
    """唤醒锚点决策的记录。

    字段含义：
      - selected_source / contrast_source：选中哪一路做锚点、哪一路做
        "同分离域对比项"（同一批分离产物里的另一路）；
      - raw_wake_fallback：是否退回"整段原始唤醒音频"当锚点（分离产物
        不可用、或两路拼音相似度打平无法裁决时走这条路）；
      - text_tie：两路与唤醒文本的拼音相似度是否打平；
      - source0_sim / source1_sim：两路各自的拼音相似度（负的拼音编辑
        距离，0 = 读音完全一致，越大越像）；None 表示该侧没有转写、
        无从比较（对应注解里的 float | None）；
      - reason：人类可读的决策路径标识（写进报告，排查时一眼定位走了
        哪个分支）。
    """

    selected_source: int
    contrast_source: int
    raw_wake_fallback: bool
    text_tie: bool
    source0_sim: float | None
    source1_sim: float | None
    reason: str


def select_wake_anchor(
    wake_text: str | None,
    source0_text: str | None,
    source1_text: str | None,
) -> WakeAnchorDecision:
    """Select the source that uttered ``wake_text`` using phoneme similarity.

    A missing transcript marks the corresponding separated source as unusable
    for enrollment.  The full wake waveform is then the anchor, while the
    silent source remains a same-separation contrast term.  When both sources
    contain speech, phoneme (pinyin) similarity selects the best match, with
    wake-word aliases expanded; a similarity tie falls back to the raw wake.

    （中文）裁决流程（四分支）：
      1. 只有一路有转写：selected_source 记有声那路，但
         raw_wake_fallback=True 表示实际锚点退回整段原始唤醒音频
         （静默路没有成对信息可用），静默路仅作同分离域的对比项；
      2. 两路都有转写：谁与唤醒文本的**拼音相似度**高选谁——拼音相似度
         度量"哪路的读音更像唤醒词"，同音字错误不受惩罚（见
         core/phoneme.py 的模块文档）；参考文本命中别名表时会展开所有
         变体取最高分；
      3. 两路拼音相似度打平：无法可靠裁决，锚点退回原始唤醒音频
         （raw_wake_fallback=True）。打平说明两路"听起来一样像"，选错
         一路的代价高于用混合音频的代价，保守回退；
      4. 两路都空（退化情形）：沿用历史行为——选 source0、锚点用原始
         唤醒音频、对比项也设成 0，保证缓存重放确定性。
    """

    reference = normalize_transcript(wake_text or "")  # 唤醒文本统一口径后再比对（None 兜底成空串）
    # 分离通道转写只做 strip()：它们在上游已被 FireRed 归一化，重放直接用
    # 原串比相似度，才能逐字复现当时的决策（"复现优先"原则）。
    source0 = str(source0_text or "").strip()  # 只去首尾空白：缓存里的字符串已是归一化文本
    source1 = str(source1_text or "").strip()
    has0, has1 = bool(source0), bool(source1)  # "该路是否有内容"：有内容才能参与相似度比较

    if has0 and not has1:  # 分支 1a：只有 source0 有声 -> 它当锚点，静默的 source1 当对比项
        return WakeAnchorDecision(0, 1, True, False, None, None, "source1_empty_raw_wake")
    if has1 and not has0:  # 分支 1b：对称情形
        return WakeAnchorDecision(1, 0, True, False, None, None, "source0_empty_raw_wake")

    sim0 = phoneme_similarity(source0, reference)  # 两路都有转写：各自与唤醒文本比拼音相似度
    sim1 = phoneme_similarity(source1, reference)
    selected = 0 if sim0 >= sim1 else 1  # 相似度高者胜；平手先记 source0（>= 而非 >，复刻历史比较）
    tie = not reference or sim0 == sim1  # reference 为空时相似度无意义，一律视为打平
    if not has0 and not has1:
        # 双空退化分支：保留历史行为（对比项钉在 source0）以保证缓存重放
        # 确定性。
        return WakeAnchorDecision(selected, 0, True, tie, sim0, sim1, "both_sources_empty_raw_wake")
    if tie:
        # 拼音打平：两路"听起来一样像"，无法可靠裁决，锚点退回原始唤醒
        # 音频（保守方向：混合音频里至少有完整的目标人嗓音）。
        return WakeAnchorDecision(
            selected,
            1 - selected,  # 对比项 = 没被选中的那一路（同一批分离产物里的另一路）
            True,
            tie,
            sim0,
            sim1,
            "wake_text_phoneme_similarity",
        )
    return WakeAnchorDecision(
        selected,
        1 - selected,
        False,
        tie,
        sim0,
        sim1,
        "wake_text_phoneme_similarity",
    )
