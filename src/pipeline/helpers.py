"""pipeline.helpers：管线共享的通用工具。

这些函数历史上分散在多个阶段与脚本里重复实现，已收敛为唯一实现：

  - ``sha256_file``            文件分块 SHA-256（manifest 资产指纹用）；
  - ``read_mono_16k``          严格 16k 单声道 float32 音频读取（分离器输入）；
  - ``separation_paths_for``   某样本唤醒/指令的分离缓存输出路径对。

"收敛为唯一实现"的意义：同一段逻辑如果在三处各抄一份，修 bug 时很容易
漏改其中一处，于是不同入口行为不一致、极难排查。现在全局只有一个版本，
行为变更只改这里一处。

依赖：core（采样率常量、Sample/submission_id），numpy、soundfile。
不依赖 torch / modelscope。
（不依赖 torch 是刻意设计：本模块会被 engine/cli 顶层 import，
只要它不碰 torch，"无 torch 也能 --help / 单测"的承诺才成立。）
"""
from __future__ import annotations

import hashlib                                # 文件摘要（资产指纹）
from pathlib import Path                      # 面向对象路径（跨平台，替代 os.path 字符串拼接）
from typing import Any                         # 类型标注：分离器实例的宽松类型

import numpy as np                            # 音频数组与向量化数值计算
import soundfile as sf                        # 音频文件读写（libsndfile 的 Python 绑定）

from core.audio import SAMPLE_RATE            # 全系统统一的采样率常量（16000 Hz）
from core.samples import Sample, submission_id


def sha256_file(path: Path) -> str:
    """返回文件的 SHA-256 十六进制摘要（1MB 分块读取，大文件不撑爆内存）。

    manifest（资产指纹清单）用它记录"这次评测用的是哪一份权重/配置"：
    权重文件动辄几百 MB，若整文件读进内存再算摘要会瞬间吃光内存，
    所以按 1MB 一块流式喂给哈希对象——摘要算法本身支持增量更新。
    """
    digest = hashlib.sha256()                # 哈希对象：多次 update 等价于一次算完
    with path.open("rb") as handle:          # "rb" 二进制读：哈希是对字节算的，与文本编码无关
        # iter(可调用对象, 终止值)：反复调用 lambda 读 1MB，读到 b""（EOF）为止。
        # 这种写法把"循环读文件"压成一行，且避免先 read 全部再切块
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()                # 摘要字节转成 64 位十六进制字符串


def read_mono_16k(path: Path) -> np.ndarray:
    """读取严格 16kHz 单声道音频为 float32（分离器唯一输入口径）。

    校验采样率必须是 16k；多声道混成均值；波形为空或含 NaN/Inf 时报错。

    为什么要"严格"？声纹/分离模型是在 16k 采样率上训练的，喂 8k 或 44.1k
    的音频，模型听到的"音高/语速"全都不对（相当于用不同采样率放同一段
    录音），输出必然失真。入口处直接报错，好过让错误数据流入推理。
    """
    # sf.read 返回 (样本数组, 采样率)：
    #   dtype="float32"     读进来直接是浮点（默认是 float64，显式指定省一半内存）；
    #   always_2d=False     单声道给一维数组、多声道给二维（否则总是 (N, 1) 形状）
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        # 采样率不符立刻 ValueError：错误出现在"数据入口"，排查成本最低
        raise ValueError(f"expected 16 kHz audio, got {rate}: {path}")
    if waveform.ndim == 2:
        # 多声道（如立体声）沿"声道轴"（axis=1）取均值混成单声道。
        # 注意：不是丢一路，是平均——两路信息都保留一部分
        waveform = waveform.mean(axis=1)
    # np.asarray 保证类型为 float32 且是"数组"语义（混音后仍是数组，
    # 这里主要兜底 ndarray 子类/非连续内存等情况）
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        # 空音频（0 采样点）或含 NaN/Inf：后续 FFT/模型推理会算出一堆 NaN，
        # 且错误会传播很远才暴露，所以入口直接挡掉
        raise ValueError(f"invalid audio waveform: {path}")
    return waveform


def separation_paths_for(sample: Sample, id_mode: str, cache_dir: Path, separator: Any):
    """返回 (唤醒分离路径对, 指令分离路径对)，缓存命中时直接复用。

    落盘命名规则：``{item_id}_wake_ms{0,1}.wav`` / ``{item_id}_command_ms{0,1}.wav``，
    item_id 中的 ``/`` 替换为 ``_``。

    缓存是"路径即键"的设计：文件名由样本 id 唯一决定，separator 内部检查
    两个文件都已存在就直接返回路径、不再推理。重复跑同一批数据（调参时
    常跑几十遍）只有第一次真正花时间做分离，之后全部秒回。
    """
    # submission_id 按竞赛口径生成样本唯一 id；relative_stem 模式下 id 形如
    # "子目录/文件名"，含路径分隔符，直接拼进文件名会多出层级，所以替换成 _
    item_id = submission_id(sample, id_mode).replace("/", "_")
    # cache_dir / "..."：pathlib 的 / 运算符拼接路径，跨平台自动用正确分隔符
    wake_prefix = cache_dir / f"{item_id}_wake"
    command_prefix = cache_dir / f"{item_id}_command"
    # 注意：这里会真实触发两次分离（唤醒/指令各一次），或命中缓存直接返回。
    # 返回的是四元组 ((wake0, wake1), (cmd0, cmd1))，即每个位置的完整路径
    return (
        separator.separate(sample.wake_audio, wake_prefix),
        separator.separate(sample.command_audio, command_prefix),
    )
