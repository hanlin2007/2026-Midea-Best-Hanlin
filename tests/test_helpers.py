"""pipeline 共享工具（文件哈希 / 严格音频读取）的单元测试（numpy-only）。

覆盖收敛到 pipeline.helpers 的公共函数，保证共享实现与历史语义一致
（numpy-only，无需模型/GPU）。

测试策略是"对照 oracle"：
- sha256 直接和标准库 hashlib 的整读结果比对（标准库就是真理）；
- 音频测试用 soundfile 现场合成真实的 wav 文件再喂给被测函数
  （tempfile.TemporaryDirectory：系统临时目录 + 自动清理，测试完
  不留任何痕迹）。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest tests.test_helpers -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.helpers import read_mono_16k, sha256_file


class HelpersTest(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self) -> None:
        # 对照测试：我们的分块实现（sha256_file）必须与标准库 hashlib
        # 一次性读入的结果逐位一致。hashlib 是"标准答案"（oracle），
        # 这条测试保证分块读写的优化没有引入任何偏差。
        # 局部 import hashlib：只有一个测试用到它，就地导入让依赖关系
        # 更局部（也顺便演示：Python 允许在函数内 import）。
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            # 构造一个跨越多次 1MB 读循环的测试文件。
            path.write_bytes(b"target-speaker" * 1000)
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_read_mono_16k_rejects_wrong_rate_and_accepts_stereo(self) -> None:
        # 音频读取的三条契约：
        # 1) 合法单声道 16k 原样读出（1600 个采样点）；
        # 2) 立体声（两声道）自动混成单声道（取均值），形状压回一维；
        # 3) 采样率不对（8k）直接 ValueError——分离器输入口径必须是
        #    16k，错误采样率会静默改变时间尺度，属于"必须炸出来"的错误。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # sf.write 现场合成：np.zeros 全零静音波形即可满足测试。
            mono = root / "mono.wav"
            sf.write(str(mono), np.zeros(1600, dtype=np.float32), 16000)
            self.assertEqual(read_mono_16k(mono).shape, (1600,))

            # 立体声：形状 (1600, 2) = 1600 帧 × 2 声道。
            stereo = root / "stereo.wav"
            sf.write(str(stereo), np.zeros((1600, 2), dtype=np.float32), 16000)
            self.assertEqual(read_mono_16k(stereo).shape, (1600,))

            bad = root / "bad_rate.wav"
            sf.write(str(bad), np.zeros(800, dtype=np.float32), 8000)
            # assertRaises 作上下文管理器：with 块内的代码必须抛出指定
            # 异常才算通过；没抛或抛了别的异常都算测试失败。
            with self.assertRaises(ValueError):
                read_mono_16k(bad)


if __name__ == "__main__":
    unittest.main()
