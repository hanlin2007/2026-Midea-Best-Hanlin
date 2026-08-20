from __future__ import annotations

import unittest

from midea_qiji.moss_candidate import choose_command_candidate, select_wake_anchor, should_switch_command


class MossCandidatePolicyTest(unittest.TestCase):
    def test_wake_text_selects_nonempty_source(self) -> None:
        decision = select_wake_anchor("你好小美", "你好小美", "打开电视")
        self.assertEqual(decision.selected_source, 0)
        self.assertEqual(decision.contrast_source, 1)
        self.assertFalse(decision.raw_wake_fallback)

    def test_empty_wake_source_uses_full_wake_as_anchor(self) -> None:
        decision = select_wake_anchor("你好小美", "", "打开电视")
        self.assertEqual(decision.selected_source, 1)
        self.assertEqual(decision.contrast_source, 0)
        self.assertTrue(decision.raw_wake_fallback)

    def test_confidence_rule_matches_frozen_boundaries(self) -> None:
        self.assertTrue(should_switch_command(confidence=0.57, command_keyword_score=0.0, chars_per_sec=0.0))
        self.assertTrue(should_switch_command(confidence=0.70, command_keyword_score=0.5, chars_per_sec=0.0))
        self.assertFalse(should_switch_command(confidence=0.70, command_keyword_score=0.0, chars_per_sec=2.9))
        self.assertTrue(should_switch_command(confidence=1.01, command_keyword_score=0.0, chars_per_sec=3.0))
        self.assertFalse(should_switch_command(confidence=1.01, command_keyword_score=0.5, chars_per_sec=3.0))

    def test_empty_command_source_forces_original_transcript(self) -> None:
        decision = choose_command_candidate(
            should_switch=True,
            source0_text="开灯",
            source1_text="",
            source0_difference=0.2,
            source1_difference=0.3,
        )
        self.assertEqual(decision.selected_candidate, "orig")
        self.assertTrue(decision.command_source_empty)

    def test_differential_tie_prefers_source_zero(self) -> None:
        decision = choose_command_candidate(
            should_switch=True,
            source0_text="开灯",
            source1_text="关灯",
            source0_difference=0.2,
            source1_difference=0.2,
        )
        self.assertEqual(decision.selected_candidate, "ms0")


if __name__ == "__main__":
    unittest.main()
