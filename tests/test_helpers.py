"""szy 共享工具（文件哈希/指令特征/置信度/音频读取）的单元测试。

覆盖重构中从 szy_v1/szy_v2/scripts 收敛到 pipeline.helpers 的公共函数，
保证共享实现与历史语义一致（numpy-only，无需模型/GPU）。

这组测试的背景：重构把散落在三个版本里的重复实现（sha256、指令特征、
置信度解析、音频读取）收敛成了 pipeline.helpers 里的唯一实现。"收敛"的
风险在于：新实现只要有一处与某个历史版本语义不同，使用它的管线就会
静默改变行为。所以每个函数都配了测试，而且测试策略是"对照 oracle"：
- sha256 直接和标准库 hashlib 的整读结果比对（标准库就是真理）；
- 音频测试用 soundfile 现场合成真实的 wav 文件再喂给被测函数；
- 数值断言用 assertAlmostEqual（浮点不能比 ==，见下）。

测试数据构造（初学可对照正文）：
  - tempfile.TemporaryDirectory：系统临时目录 + 自动清理（见 test_infer
    的注释），音频文件在里面现场合成，测试完不留任何痕迹；
  - numpy + soundfile 合成 wav：sf.write 写一个全零浮点波形就是一段
    "静音"音频——测读取/时长逻辑完全够用，不需要真实录音。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest discover -s tests
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.helpers import (
    asr_confidence,
    command_features,
    duration_seconds,
    read_mono_16k,
    sha256_file,
)


class SzyHelpersTest(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self) -> None:
        # 对照测试：我们的分块实现（sha256_file）必须与标准库 hashlib
        # 一次性读入的结果逐位一致。hashlib 是"标准答案"（oracle），
        # 这条测试保证分块读写的优化没有引入任何偏差。
        # 局部 import hashlib：只有一个测试用到它，就地导入让依赖关系
        # 更局部（也顺便演示：Python 允许在函数内 import）。
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            # 构造一个比单个哈希块（1MB）更大的测试文件：
            # "core" 是 10 字节，×1000 = 10000 字节，足够跨越
            # 多次 read 循环，能真正测到"多块拼接"的路径。
            path.write_bytes(b"core" * 1000)
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_command_features_counts_keywords_and_rate(self) -> None:
        # 指令文本双特征的"正常值"口径：
        chars_per_sec, keywords = command_features("打开空调", 1.0)
        # 归一化后 "打开空调" 共 4 字，1 秒 => 4 字/秒
        # assertAlmostEqual：浮点比较不能用 ==（浮点表示有舍入误差），
        # 它允许在默认 7 位小数精度内相等即可。
        self.assertAlmostEqual(chars_per_sec, 4.0)
        # "打开" 与 "空调" 各命中一次 CMD_KW
        # （CMD_KW 是 pipeline.helpers 里的指令关键词正则表）。
        self.assertEqual(keywords, 2.0)

    def test_command_features_short_clip_floor(self) -> None:
        # 时长下限 0.1s，避免除零
        # 极端短音频（0.01s）测试：如果没有 0.1 的时长下限，2 字 / 0.01s
        # = 200 字/秒，会把语速特征炸上天、误触发切换规则。
        # 有下限后 2 / 0.1 = 20.0，行为稳定。锁死这个防御性边界。
        chars_per_sec, _ = command_features("打开", 0.01)
        self.assertAlmostEqual(chars_per_sec, 20.0)

    def test_asr_confidence_fallbacks(self) -> None:
        # 置信度解析的回退语义：什么情况下回退到 1.0（视为"完全可信"）。
        # 回退到 1.0 而不是 0 的原因：置信度不可用时，宁可相信 ASR 结果
        # 是可信的（保守地不做切换），也不要用 0 置信度去触发一堆切换。
        # 字符串数字正常解析；None/缺键/NaN/Inf 全部回退 1.0。
        self.assertEqual(asr_confidence({"confidence": "0.7"}), 0.7)
        self.assertEqual(asr_confidence({"confidence": None}), 1.0)
        self.assertEqual(asr_confidence({}), 1.0)
        # float("nan") / float("inf") 是合法的 float 但不是有限值，
        # 属于"非法输入"类，同样回退（NaN 的另一个麻烦：NaN != NaN，
        # 任何与 NaN 的比较都是 False，会污染下游决策）。
        self.assertEqual(asr_confidence({"confidence": float("nan")}), 1.0)
        self.assertEqual(asr_confidence({"confidence": float("inf")}), 1.0)

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

    def test_duration_seconds_reads_header(self) -> None:
        # 时长读取：16000 采样 @ 16kHz = 1.0 秒。
        # places=3 把浮点比较精度放宽到小数点后 3 位——wav 头里的
        # 帧数字段是整数，理论上精确，但跨库浮点换算允许微小误差。
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one_sec.wav"
            sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
            self.assertAlmostEqual(duration_seconds(path), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
