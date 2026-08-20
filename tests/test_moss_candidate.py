"""moss_candidate 决策规则的单元测试。

这些测试是"决策边界"的守护者：should_switch_command 的阈值边界、
select_wake_anchor 的选路/回退分支、choose_command_candidate 的平局规则
都被冻结在这里。任何人改动了决策逻辑，这里的断言会立刻红掉。

运行方式（在仓库根目录）：
    PYTHONPATH=src python -m unittest tests.test_moss_candidate -v
"""
from __future__ import annotations

import unittest

from midea_qiji.moss_candidate import choose_command_candidate, select_wake_anchor, should_switch_command


class MossCandidatePolicyTest(unittest.TestCase):

    def test_wake_text_selects_nonempty_source(self) -> None:
        """唤醒文本选中转写正确的那一路分离源。"""
        # "你好小美" 与 source0 完全一致 → 选 source0，对照 source1，且无需 raw 回退
        decision = select_wake_anchor("你好小美", "你好小美", "打开电视")
        self.assertEqual(decision.selected_source, 0)
        self.assertEqual(decision.contrast_source, 1)
        self.assertFalse(decision.raw_wake_fallback)

    def test_empty_wake_source_uses_full_wake_as_anchor(self) -> None:
        """某一路唤醒源为空 → 锚点退回原始完整唤醒音频。"""
        decision = select_wake_anchor("你好小美", "", "打开电视")
        self.assertEqual(decision.selected_source, 1)   # 非空的一路
        self.assertEqual(decision.contrast_source, 0)   # 空的一路作对照
        self.assertTrue(decision.raw_wake_fallback)     # 关键：raw 回退标志

    def test_confidence_rule_matches_frozen_boundaries(self) -> None:
        """should_switch_command 的冻结边界逐点验证（改这里=破坏复现）。"""
        # confidence<=0.57 无条件触发
        self.assertTrue(should_switch_command(confidence=0.57, command_keyword_score=0.0, chars_per_sec=0.0))
        # confidence<=0.70 且关键词>=0.5 触发
        self.assertTrue(should_switch_command(confidence=0.70, command_keyword_score=0.5, chars_per_sec=0.0))
        # confidence<=0.70 但语速 2.9<3.0、无关键词 → 不触发（差一点就是差一点）
        self.assertFalse(should_switch_command(confidence=0.70, command_keyword_score=0.0, chars_per_sec=2.9))
        # confidence>0.70 且 <=1.01 且语速>=3.0 且关键词<0.5 → 触发
        self.assertTrue(should_switch_command(confidence=1.01, command_keyword_score=0.0, chars_per_sec=3.0))
        # 同样条件但有关键词 → 不触发（规则里的 <0.5 条件）
        self.assertFalse(should_switch_command(confidence=1.01, command_keyword_score=0.5, chars_per_sec=3.0))

    def test_empty_command_source_forces_original_transcript(self) -> None:
        """任一命令分离源为空 → 必须回退原始转写（不赌单路）。"""
        decision = choose_command_candidate(
            should_switch=True,
            source0_text="开灯",
            source1_text="",       # 空 → 分离失败
            source0_difference=0.2,
            source1_difference=0.3,
        )
        self.assertEqual(decision.selected_candidate, "orig")
        self.assertTrue(decision.command_source_empty)

    def test_differential_tie_prefers_source_zero(self) -> None:
        """差分分数平局时固定选 ms0（确定性复现）。"""
        decision = choose_command_candidate(
            should_switch=True,
            source0_text="开灯",
            source1_text="关灯",
            source0_difference=0.2,
            source1_difference=0.2,  # 平局
        )
        self.assertEqual(decision.selected_candidate, "ms0")


if __name__ == "__main__":
    unittest.main()  # 直接 python tests/test_moss_candidate.py 也能跑
