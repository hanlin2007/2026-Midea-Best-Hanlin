"""core.audio：音频常量与 WAV 读取工具。

统一全仓库的采样率常量（16k）与唯一的读取口径：

  - ``read_wav``    严格 16k 单声道 PCM16，无重采样（wave 模块）。

关键导出：``SAMPLE_RATE`` / ``read_wav``。

为什么要把"读音频"收敛到内核层：音频是下游所有模型（声纹 / ASR / 分离）
的共同输入，采样率、声道数、dtype 任何一个对不上都会让模型静默掉点——
而且这类问题极难排查（没有报错，只有变差的分数）。各调用方各写一套读取
逻辑必然口径漂移（有的忘校验、有的忘混声道），集中到一处实现并锁死语义，
才能保证全仓库喂给模型的都是同一种波形。

依赖：numpy（wave 为标准库，函数体内延迟导入）。
"""
from __future__ import annotations  # 延迟求值注解（str | Path 等），无运行时开销

from pathlib import Path  # 路径参数统一用 Path 注解，同时接受 str（Path 构造兼容）

import numpy as np

# 全仓库唯一的采样率常量。为什么锁死 16k：
#   - 16k 是语音识别 / 说话人验证领域的事实标准采样率，本仓库所有预训练
#     模型（FireRed、ERes2NetV2 声纹、MossFormer2 分离器）都只在 16k 数据上
#     训练过，输入其他采样率的波形会直接失配；
#   - 把数值收敛到一处常量，而不是让 16000 这个魔法数字散落各文件，
#     想查"系统采样率是多少"只需看这一个地方（事实上这里是冻结语义，
#     不允许改）。
SAMPLE_RATE = 16000


def read_wav(path: str | Path) -> np.ndarray:
    """读取严格 16k 单声道 PCM16 wav 为 float32（值域 [-1, 1]）。

    "严格"体现在：不重采样、不混声道。一旦发现采样率不是 16k 或声道数
    不是 1，直接抛 ValueError 而不是默默转换——重采样是有损操作，会
    改变波形内容；与其让下游拿着被悄悄改过的音频算出不可信的分数，不如
    显式失败，让调用方立刻知道这份音频不符合约定（比如拿错了文件）。

    为什么函数体内才 ``import wave``：wave 是标准库，专读 WAV、零第三方
    依赖；放进函数体内把"读 WAV 的依赖"限定在本函数，模块 import 时零
    额外开销——与 torch 等重型依赖的延迟导入是同一思路。
    """
    import wave

    with wave.open(str(path), "rb") as handle:  # with 保证文件句柄异常时也会被关闭
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ValueError(f"unsupported wav format: {path}")
        raw = handle.readframes(handle.getnframes())  # 一次性读完全部帧，得到原始字节串
    # PCM16：每个采样点 2 字节有符号整数。frombuffer 把字节串按 int16
    # 重新解释成数组（零拷贝"视图"），astype 再转 float32（复制并转换）。
    # 除以 32768 把 int16 值域 [-32768, 32767] 归一化到约 [-1.0, 1.0)：
    # 浮点波形是下游前处理（FFT / 模型输入）的统一约定，范围稳定、
    # dtype 统一比绝对值本身更重要。
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
