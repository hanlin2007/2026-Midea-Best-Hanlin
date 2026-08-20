"""szy_common 共享工具（文件哈希/指令特征/置信度/音频读取）的单元测试。

覆盖重构中从 szy_v1/szy_v2/scripts 收敛到 szy_common.helpers 的公共函数，
保证共享实现与历史语义一致（numpy-only，无需模型/GPU）。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest tests.test_szy_common -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from szy_common.helpers import (
    asr_confidence,
    command_features,
    duration_seconds,
    read_mono_16k,
    sha256_file,
)


class SzyCommonHelpersTest(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"midea_qiji" * 1000)
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_command_features_counts_keywords_and_rate(self) -> None:
        chars_per_sec, keywords = command_features("打开空调", 1.0)
        # 归一化后 "打开空调" 共 4 字，1 秒 => 4 字/秒
        self.assertAlmostEqual(chars_per_sec, 4.0)
        # "打开" 与 "空调" 各命中一次 CMD_KW
        self.assertEqual(keywords, 2.0)

    def test_command_features_short_clip_floor(self) -> None:
        # 时长下限 0.1s，避免除零
        chars_per_sec, _ = command_features("打开", 0.01)
        self.assertAlmostEqual(chars_per_sec, 20.0)

    def test_asr_confidence_fallbacks(self) -> None:
        self.assertEqual(asr_confidence({"confidence": "0.7"}), 0.7)
        self.assertEqual(asr_confidence({"confidence": None}), 1.0)
        self.assertEqual(asr_confidence({}), 1.0)
        self.assertEqual(asr_confidence({"confidence": float("nan")}), 1.0)
        self.assertEqual(asr_confidence({"confidence": float("inf")}), 1.0)

    def test_read_mono_16k_rejects_wrong_rate_and_accepts_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mono = root / "mono.wav"
            sf.write(str(mono), np.zeros(1600, dtype=np.float32), 16000)
            self.assertEqual(read_mono_16k(mono).shape, (1600,))

            stereo = root / "stereo.wav"
            sf.write(str(stereo), np.zeros((1600, 2), dtype=np.float32), 16000)
            self.assertEqual(read_mono_16k(stereo).shape, (1600,))

            bad = root / "bad_rate.wav"
            sf.write(str(bad), np.zeros(800, dtype=np.float32), 8000)
            with self.assertRaises(ValueError):
                read_mono_16k(bad)

    def test_duration_seconds_reads_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one_sec.wav"
            sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
            self.assertAlmostEqual(duration_seconds(path), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
