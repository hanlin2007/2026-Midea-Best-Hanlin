"""冻结决策规则（唤醒锚点选择 / 候选切换）的单元测试（numpy-only）。

这些测试锁死历史离线结果的决策边界，任何改动导致失败都意味着复现被破坏。

"冻结语义"为什么是测试的重要使命：竞赛/研究项目里有一类代码的价值
不在"运行得快"，而在"复现历史结果"。moss_candidate 里的决策规则
（唤醒锚点怎么选、什么时候切换候选）是历史离线实验用过的规则，那些
实验的数字（CER 0.3058 等）全部建立在这些规则上。如果后来有人
"顺手优化"了规则（比如把 <= 改成 <、把平局判给 ms1），新代码跑得
更好也没用——历史结果再也复现不出来，之前所有对比实验作废。所以
这类代码要用测试把边界逐点锁死：测试里出现的每个数字（0.57、0.70、
1.01、3.0、0.5）都对应实现里的一条分支边界，改实现必然改挂测试。

"测试即文档"：这组测试同时充当决策规则的规格说明——五个测试名 +
 断言值完整描述了"什么输入得到什么决策"，比注释更可信（注释会过期，
 测试不会说谎）。接手的同学读这 50 行测试，比读实现更快懂规则。

为什么无需模型/GPU：被测函数是纯规则函数（字符串 + 浮点比较），
测试直接喂边界输入、断言输出，全程只碰标准库。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest tests.test_moss_candidate -v
"""
from __future__ import annotations

import unittest

from midea_qiji.moss_candidate import choose_command_candidate, select_wake_anchor, should_switch_command


class MossCandidatePolicyTest(unittest.TestCase):
    def test_wake_text_selects_nonempty_source(self) -> None:
        # 唤醒文本"你好小美"与 source0 的转写完全一致：编辑距离更小，
        # 选 source0 作唤醒锚点（selected_source=0），另一个通道作为
        # 对比项（contrast_source=1）。两个分离通道都有内容时不需要
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

    def test_confidence_rule_matches_frozen_boundaries(self) -> None:
        # 逐点锁死 should_switch_command 的两段式置信度规则的每个分支边界。
        # 规则结构（对应实现的 if/else）：
        #   低置信区（<=0.70）：关键词命中 or 语速高 or 置信度 <=0.57 -> 切换；
        #   高置信区（>0.70）：语速高 且 关键词未命中 且 <=1.01 -> 切换。
        # 下面 5 个断言各踩一个分支的临界点：
        # 1) 0.57 触发"置信度 <=0.57"必切线（即使关键词 0、语速 0）；
        self.assertTrue(should_switch_command(confidence=0.57, command_keyword_score=0.0, chars_per_sec=0.0))
        # 2) 0.70 落在低置信区，关键词 0.5 达到命中线 -> 切换；
        self.assertTrue(should_switch_command(confidence=0.70, command_keyword_score=0.5, chars_per_sec=0.0))
        # 3) 低置信区，关键词 0、语速 2.9（差 0.1 不到 3.0 线）-> 不切换；
        self.assertFalse(should_switch_command(confidence=0.70, command_keyword_score=0.0, chars_per_sec=2.9))
        # 4) 1.01 在高置信区，语速 3.0 过线、关键词 0 -> 切换；
        self.assertTrue(should_switch_command(confidence=1.01, command_keyword_score=0.0, chars_per_sec=3.0))
        # 5) 同样 1.01/3.0，但关键词 0.5 命中 -> 高置信区的"关键词否决"
        #    生效，不切换（自信转写且像指令，不该动）。
        self.assertFalse(should_switch_command(confidence=1.01, command_keyword_score=0.5, chars_per_sec=3.0))

    def test_empty_command_source_forces_original_transcript(self) -> None:
        # 切换被触发、但 source1 转写为空：分离失败，不能换到坏通道，
        # 强制用原始转写（selected_candidate="orig"），并把
        # command_source_empty 标记为 True（下游依赖这个标记做回退）。
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
        # 两个分离通道的差分分数打平（都是 0.2）时，历史实现用 >= 比较，
        # 所以平局判给 ms0。这条锁死的是"平局裁决"——把 >= 改成 > 或者
        # 平局判 ms1 都会让历史重放选错通道、内容对不上。
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
