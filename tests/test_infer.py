"""midea_qiji.infer 工具层的单元测试。

运行方式（在仓库根目录）：
    PYTHONPATH=src python -m unittest tests.test_infer -v
    # 或：PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'

这些测试只依赖 numpy（不需要 GPU、不需要下载模型），是理解
文本归一化 / 稳定 ID / 报告结构 这几个纯函数行为的最佳入口。
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile  # 创建临时目录（用完自动删除，测试不污染仓库）
import unittest  # Python 标准库测试框架

# 从被测模块导入要测的函数/类（PYTHONPATH=src 让解释器能找到 midea_qiji 包）
from midea_qiji.infer import Sample, normalize_transcript, read_many_jsonl, submission_id, write_reports


class InferTest(unittest.TestCase):
    """每个 test_ 开头的方法会被 unittest 自动当成一条用例执行。"""

    def test_normalization_matches_baseline_rules(self) -> None:
        """文本归一化必须与基线口径一致（数字转中文 + 百分比 + 去标点）。"""
        self.assertEqual(normalize_transcript("空调开到25度，风量30%"), "空调开到二十五度风量百分之三十")
        self.assertEqual(normalize_transcript("设置101分钟"), "设置一百零一分钟")

    def test_relative_stem_id_is_split_safe(self) -> None:
        """relative_stem 模式要从相对路径派生 ID，去掉扩展名、跨平台用 / 分隔。"""
        sample = Sample("1", Path("wake.wav"), Path("cmd.wav"), "pos/cmd_1.wav", None)
        self.assertEqual(submission_id(sample, "relative_stem"), "pos/cmd_1")

    def test_manifest_preserves_chinese_wake_text(self) -> None:
        """读 JSONL 时要正确保留中文唤醒文本（回归测试：防止编码/字段名改坏）。"""
        with tempfile.TemporaryDirectory() as directory:  # with 块结束自动删临时目录
            root = Path(directory)
            manifest = root / "items.jsonl"
            # 写一行 JSONL（ensure_ascii=False 让中文原样写盘；末尾加换行符合 JSONL 规范）
            manifest.write_text(json.dumps({"id": 1, "唤醒音频": "pos/kws_1.wav", "唤醒文本": "你好科慕", "识别音频": "pos/cmd_1.wav", "识别文本": "开灯"}, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = read_many_jsonl([manifest], root)
        self.assertEqual(samples[0].wake_text, "你好科慕")

    def test_report_metrics_and_schema(self) -> None:
        """write_reports 的 CER/PR/RR 计算与输出结构要对。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_reports(
                [
                    {"id": "pos/a", "content": "开灯", "label": "开灯", "sv_score": "0.5"},   # 正样本，全对
                    {"id": "neg/a", "content": "", "label": None, "sv_score": "0.1"},        # 负样本，正确拒识
                ],
                2.0,
                root / "submission.json",
                root / "detail.json",
            )
            payload = json.loads((root / "detail.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["final_cer"], 0.0)   # 全对 → CER=0
        self.assertEqual(summary["final_pr"], 1.0)    # 正样本全非空 → PR=1
        self.assertEqual(summary["final_rr"], 1.0)    # 负样本全拒识 → RR=1
        self.assertEqual(payload["results"][0]["sample_type"], "positive")  # 明细里要标正负
        self.assertEqual(payload["results"][1]["rr"], 1.0)                  # 负样本 rr 字段=1.0
