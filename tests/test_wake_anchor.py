"""唤醒锚点选择规则的单元测试（numpy-only，无需模型/GPU）。

这些测试锁死 select_wake_anchor 的决策边界：拼音相似度"像者胜"、
打平/单路缺失一律回退原始唤醒音频。任何改动导致失败都意味着复现
被破坏——锚点选择是全系统在评测集上最大的单项护城河（51 条锚点
选择差异带来 81 字优势），边界漂移会直接改变生产输出。

"测试即文档"：这组测试同时充当决策规则的规格说明——测试名 + 断言值
完整描述了"什么输入得到什么决策"，比注释更可信（注释会过期，测试
不会说谎）。

拼音相关用例在 pypinyin 缺失时自动跳过（此时 phoneme 模块退化为小写
字符比较，语义不同，不测）。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest tests.test_wake_anchor -v
"""
from __future__ import annotations

import unittest

from core.wake_anchor import select_wake_anchor


def _has_pypinyin() -> bool:
    try:
        import pypinyin  # noqa: F401
        return True
    except ImportError:
        return False


class WakeAnchorPolicyTest(unittest.TestCase):
    def test_wake_text_selects_nonempty_source(self) -> None:
        # 唤醒文本"你好小美"与 source0 的转写完全一致：拼音相似度更高，
        # 选 source0 作唤醒锚点（selected_source=0），另一个通道作为
        # 对比项（contrast_source=1）。相似度明确分出胜负时不需要
        # 回退到原始唤醒音频（raw_wake_fallback=False）。
        decision = select_wake_anchor("你好小美", "你好小美", "打开电视")
        self.assertEqual(decision.selected_source, 0)
        self.assertEqual(decision.contrast_source, 1)
        self.assertFalse(decision.raw_wake_fallback)

    def test_empty_wake_source_uses_full_wake_as_anchor(self) -> None:
        # source0 转写为空（分离失败/静音）：该通道不能用于注册，
        # 锚点回退到整段原始唤醒音频（raw_wake_fallback=True），
        # 有内容的 source1 当选，空通道作对比项。
        # 这条锁死的是"分离失败时的降级路径"——历史管线正是这样
        # 处理的，改掉会让缓存重放的锚点选择对不上。
        decision = select_wake_anchor("你好小美", "", "打开电视")
        self.assertEqual(decision.selected_source, 1)
        self.assertEqual(decision.contrast_source, 0)
        self.assertTrue(decision.raw_wake_fallback)

    def test_phoneme_tie_uses_full_wake_as_anchor(self) -> None:
        # 同音字情形："你好科慕"与"你好科目"读音完全一致（拼音相同），
        # 两路相似度打平 → 无法可靠裁决 → 锚点保守回退原始唤醒音频。
        # 这是拼音版锚点选择与旧字符编辑距离版的核心行为差异：
        # 同音字不再制造"假胜负"。
        if not _has_pypinyin():
            self.skipTest("pypinyin not installed")
        decision = select_wake_anchor("你好科慕", "你好科慕", "你好科目")
        self.assertTrue(decision.text_tie)
        self.assertTrue(decision.raw_wake_fallback)
        self.assertEqual(decision.reason, "wake_text_phoneme_similarity")
        # 打平时 selected 仍按 >= 记 source0（复刻历史比较方向）
        self.assertEqual(decision.selected_source, 0)
        # 两条通道的拼音相似度都被记录（0 = 读音完全一致）
        self.assertEqual(decision.source0_sim, 0.0)
        self.assertEqual(decision.source1_sim, 0.0)

    def test_phoneme_similarity_picks_closer_source(self) -> None:
        # 拼音相似度"像者胜"：source1 与唤醒文本同音（科慕/科目），
        # source0 是无关指令文本 → 选 source1，且明确分出胜负不回退。
        if not _has_pypinyin():
            self.skipTest("pypinyin not installed")
        decision = select_wake_anchor("你好科慕", "打开空调", "你好科目")
        self.assertEqual(decision.selected_source, 1)
        self.assertEqual(decision.contrast_source, 0)
        self.assertFalse(decision.raw_wake_fallback)
        # 相似度是"负的拼音编辑距离"：同音的 source1 得 0，距离越远越负
        self.assertEqual(decision.source1_sim, 0.0)
        self.assertLess(decision.source0_sim, 0.0)


if __name__ == "__main__":
    unittest.main()
