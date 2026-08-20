"""midea_qiji 基线数据/文本/报告口径的单元测试（numpy-only，无需模型）。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest tests.test_infer -v
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from midea_qiji.infer import Sample, normalize_transcript, read_many_jsonl, submission_id, write_reports
from szy_common.reporting import summarize, write_outputs


class InferTest(unittest.TestCase):
    def test_normalization_matches_baseline_rules(self) -> None:
        self.assertEqual(normalize_transcript("空调开到25度，风量30%"), "空调开到二十五度风量百分之三十")
        self.assertEqual(normalize_transcript("设置101分钟"), "设置一百零一分钟")

    def test_relative_stem_id_is_split_safe(self) -> None:
        sample = Sample("1", Path("wake.wav"), Path("cmd.wav"), "pos/cmd_1.wav", None)
        self.assertEqual(submission_id(sample, "relative_stem"), "pos/cmd_1")

    def test_manifest_preserves_chinese_wake_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "items.jsonl"
            manifest.write_text(json.dumps({"id": 1, "唤醒音频": "pos/kws_1.wav", "唤醒文本": "你好科慕", "识别音频": "pos/cmd_1.wav", "识别文本": "开灯"}, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = read_many_jsonl([manifest], root)
        self.assertEqual(samples[0].wake_text, "你好科慕")

    def test_report_metrics_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_reports(
                [
                    {"id": "pos/a", "content": "开灯", "label": "开灯", "sv_score": "0.5"},
                    {"id": "neg/a", "content": "", "label": None, "sv_score": "0.1"},
                ],
                2.0,
                root / "submission.json",
                root / "detail.json",
            )
            payload = json.loads((root / "detail.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["final_cer"], 0.0)
        self.assertEqual(summary["final_pr"], 1.0)
        self.assertEqual(summary["final_rr"], 1.0)
        self.assertEqual(payload["results"][0]["sample_type"], "positive")
        self.assertEqual(payload["results"][1]["rr"], 1.0)

    def test_szy_submission_matches_competition_schema(self) -> None:
        rows = [
            {"id": "1", "content": "开灯", "label": "开灯", "route": "raw"},
            {"id": "2", "content": "", "label": None, "route": "fusion_reject"},
        ]
        summary = summarize(rows, 2.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "submission.json"
            write_outputs(rows, summary, output, root / "detail.json")
            result = json.loads(output.read_text(encoding="utf-8"))["result"]

        self.assertEqual(set(result), {"results", "avg_cer", "avg_rr", "duration"})
        self.assertEqual(result["avg_cer"], 0.0)
        self.assertEqual(result["avg_rr"], 1.0)
        self.assertEqual(result["duration"], 2500.0)
        self.assertEqual(result["results"][0]["cer"], 0.0)
        self.assertIsNone(result["results"][1]["cer"])
