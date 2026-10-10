"""拼音音素相似度与 N-best MBR 重排序的单元测试（numpy-only）。

覆盖两个 LSZ v4.5 新增的内核模块：

  - ``core.phoneme``：拼音空间相似度与唤醒词别名表——
    同音字不罚分（"科慕"vs"科目"相似度 0）、别名展开取最优；
  - ``core.firered_nbest``：保守 MBR 重排序——只有"置信度增益
    与风险下降同时过闸"才允许换掉 beam 搜索的 top-1。

这两组测试同样是"语义锁"：拼音相似度是 DatasetA 上全系统最大的单项
护城河（51 条锚点选择差异带来 81 字优势），MBR 的两道闸决定了
"什么时候允许改动 ASR 结果"，边界漂移会直接改变生产输出。

拼音相关用例在 pypinyin 缺失时自动跳过（此时 phoneme 退化为小写
字符比较，语义不同，不测）。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest tests.test_phoneme -v
"""
from __future__ import annotations

import unittest

from core.nbest import conservative_mbr_rerank, normalized_edit_distance
from core.phoneme import WAKE_WORD_ALIASES, phoneme_similarity


def _has_pypinyin() -> bool:
    try:
        import pypinyin  # noqa: F401
        return True
    except ImportError:
        return False


class PhonemeSimilarityTest(unittest.TestCase):
    def test_identical_text_scores_zero(self) -> None:
        # 完全相同的文本：拼音距离 0 → 相似度 0（最大值）。
        self.assertEqual(phoneme_similarity("打开空调", "打开空调"), 0.0)

    def test_empty_input_scores_very_low(self) -> None:
        # 空输入返回极大负值：空转写不该赢得任何锚点竞争。
        self.assertLess(phoneme_similarity("", "你好"), -1e5)
        self.assertLess(phoneme_similarity("你好", ""), -1e5)

    def test_homophone_scores_zero(self) -> None:
        # 核心语义锁：同音字不罚分。"你好科慕"与"你好科目"字不同但
        # 拼音完全相同 → 相似度 0（= 完全一致）；而语义无关的
        # "打开电视"相似度严格小于 0。这正是拼音相似度优于字符编辑
        # 距离的地方：ASR 的同音字错误不该影响锚点裁决。
        if not _has_pypinyin():
            self.skipTest("pypinyin not installed")
        self.assertEqual(phoneme_similarity("你好科慕", "你好科目"), 0.0)
        self.assertLess(phoneme_similarity("打开电视", "你好科目"), 0.0)

    def test_wake_word_aliases_expand_both_ways(self) -> None:
        # 别名表双向："hi colmo" 的拼音化（hicolmo）与中文"嗨科慕"的
        # 拼音（haikemu）互为别名。英文写法 vs 中文读音，展开别名后
        # 距离 0。别名表缺了哪个方向都会在这里挂掉。
        if not _has_pypinyin():
            self.skipTest("pypinyin not installed")
        self.assertIn("hicolmo", WAKE_WORD_ALIASES)
        self.assertIn("haikemu", WAKE_WORD_ALIASES["hicolmo"])
        self.assertEqual(phoneme_similarity("haikemu", "hicolmo"), 0.0)
        self.assertEqual(phoneme_similarity("hicolmo", "haikemu"), 0.0)

    def test_closer_reading_beats_distant_one(self) -> None:
        # 排序语义：与参考读音更近的文本必须拿到更高（不那么负）的分。
        if not _has_pypinyin():
            self.skipTest("pypinyin not installed")
        close = phoneme_similarity("你好科慕", "你好科目")   # 同音 → 0
        far = phoneme_similarity("你好科慕", "打开空调")     # 无关 → 很负
        self.assertGreater(close, far)


class MbrRerankTest(unittest.TestCase):
    def test_normalized_edit_distance_bounds(self) -> None:
        # 归一化编辑距离的三个锚点：全同 0、全不同 1、双空 0。
        self.assertEqual(normalized_edit_distance("abc", "abc"), 0.0)
        self.assertEqual(normalized_edit_distance("abc", "xyz"), 1.0)
        self.assertEqual(normalized_edit_distance("", ""), 0.0)

    def test_baseline_kept_when_it_has_minimum_risk(self) -> None:
        # 基线与其他候选几乎一致时，基线风险最低 → 不换。
        result = conservative_mbr_rerank(
            baseline={"text": "打开空调", "confidence": 0.90},
            nbest=[{"text": "打开空调呀", "confidence": 0.80}],
        )
        self.assertFalse(result["nbest_applied"])
        self.assertEqual(result["text"], "打开空调")
        self.assertEqual(result["nbest_reason"], "baseline_has_minimum_risk")

    def test_switch_requires_both_gates(self) -> None:
        # 两道闸缺一不可：构造一个"风险更低但置信度增益不足"的候选，
        # 不许换人。这是"保守"二字的语义锁——宁可不换，不可换错。
        result = conservative_mbr_rerank(
            baseline={"text": "打开空调", "confidence": 0.90},
            nbest=[{"text": "关掉空调", "confidence": 0.905},  # 增益 0.005 < 0.03
                   {"text": "关掉空调", "confidence": 0.91}],   # 与上一候选同文，拉低其风险
            min_confidence_gain=0.03,
            min_mbr_risk_reduction=0.0,
        )
        self.assertFalse(result["nbest_applied"])
        self.assertIn("insufficient_confidence_gain", result["nbest_reason"])

    def test_switch_when_clearly_dominant(self) -> None:
        # 明显占优时允许换人：两个高置信候选文本一致（"关掉空调"），
        # 基线孤立且置信度低 → MBR 最优换到多数派，且增益/风险都过闸。
        result = conservative_mbr_rerank(
            baseline={"text": "打开空调", "confidence": 0.50},
            nbest=[{"text": "关掉空调", "confidence": 0.95},
                   {"text": "关掉空调", "confidence": 0.90}],
            min_confidence_gain=0.03,
            min_mbr_risk_reduction=0.05,
        )
        self.assertTrue(result["nbest_applied"])
        self.assertEqual(result["text"], "关掉空调")
        # 换人的时候基线被记录下来（审计口径：事后能还原"原来是什么"）
        self.assertEqual(result["nbest_baseline"]["text"], "打开空调")


if __name__ == "__main__":
    unittest.main()
