"""pipeline.helpers：管线共享的通用工具。

这些函数历史上分散在多个阶段与脚本里重复实现，已收敛为唯一实现，
语义与原实现逐字一致（含置信度非有限值回退 1.0、分离缓存文件命名规则）：

  - ``sha256_file``            文件分块 SHA-256（manifest 资产指纹用）；
  - ``read_mono_16k``          严格 16k 单声道 float32 音频读取（分离器输入）；
  - ``duration_seconds``       音频时长（秒）；
  - ``command_features``       指令文本特征（字速 + 指令关键词命中数）；
  - ``asr_confidence``         FireRed 行级置信度解析（非法/非有限值回退 1.0）；
  - ``transcribe_firered_rows`` FireRed 批量转写并归一化文本 + 置信度；
  - ``separation_paths_for``   某样本唤醒/指令的分离缓存输出路径对。

"收敛为唯一实现"的意义：同一段逻辑如果在三处各抄一份，修 bug 时很容易
漏改其中一处，于是不同入口行为不一致、极难排查。现在全局只有一个版本，
行为变更只改这里一处。

依赖：core（文本归一化、Sample/submission_id、静默上下文），
numpy、soundfile。不依赖 torch / modelscope。
（不依赖 torch 是刻意设计：本模块会被 pipeline/cli 顶层 import，
只要它不碰 torch，"无 torch 也能 --help / 单测"的承诺才成立。）
"""
from __future__ import annotations

import hashlib                                # 文件摘要（资产指纹）
from pathlib import Path                      # 面向对象路径（跨平台，替代 os.path 字符串拼接）
import re                                     # 指令关键词正则
from typing import Any                         # 类型标注：不确定具体类型时用 Any 占位

import numpy as np                            # 音频数组与向量化数值计算
import soundfile as sf                        # 音频文件读写（libsndfile 的 Python 绑定）

from core.audio import SAMPLE_RATE      # 全系统统一的采样率常量（16000 Hz）
from core.samples import Sample, submission_id, suppress_model_output
from core.text import normalize_transcript

# 指令关键词表：触发切换的前提之一（转写里命中这些词才算"像指令"）。
# 正则要点：r"" 原始字符串——反斜杠不转义，写正则不用 \\ 双重转义；
# | 是"或"；"灯光?" 里的 ? 表示"灯"字可有可无（"灯"或"灯光"都命中）。
# 相邻字符串字面量会自动拼接（隐式拼接），这里拆成三行仅为可读性。
CMD_KW = re.compile(
    r"打开|关闭|关掉|调到|空调|灯光?|窗帘|音乐|电视|风扇|加湿|净化|扫地|"
    r"温度|音量|模式|洗衣|烘干|冰箱|热水|新风|播放|暂停|继续|小声|大声|"
    r"亮度|风速|洗澡|叫醒|闹钟|出门|回家|晚安"
)


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


def duration_seconds(path: Path) -> float:
    """返回音频时长（秒），只读文件头不解码（快）。

    sf.info 只解析 WAV 头部的元信息，不解码整段波形——对几十分钟的长音频
    也能瞬间返回；时长 = 总采样点数 / 采样率。
    """
    info = sf.info(str(path))
    # max(..., 1) 防除零：损坏文件可能报采样率 0，此时返回 0 秒而不是抛异常
    return float(info.frames) / max(int(info.samplerate), 1)


def command_features(text: str, duration: float) -> tuple[float, float]:
    """指令文本双特征：归一化后字速（字/秒）与 CMD_KW 指令关键词命中数。

    这两个量刻画"转写像不像一条真实指令"：真实指令语速正常（约 2~8 字/秒）
    且必含关键词（"打开空调"）；噪声/错转写的文本往往又短、又快得离谱、
    且一个关键词都没有。特征进入音色触发规则（见 pipeline.process）。
    """
    # 先归一化再统计：标点、大小写、繁简等噪声会被 normalize_transcript 清掉，
    # 否则"打开空调！"与"打开空调"会算成不同文本
    normalized = normalize_transcript(text)
    # 字速 = 字数 / 秒数；max(duration, 0.1) 是防除零护栏（0 秒音频理论上
    # 不会出现，但"宁可给个合理值也不抛异常"，这里 0.1s 是历史上沿用的下限）
    return float(len(normalized) / max(duration, 0.1)), float(len(CMD_KW.findall(normalized)))


def asr_confidence(row: dict[str, Any]) -> float:
    """从 FireRed 结果行解析置信度；非法或非有限值回退 1.0。

    为什么回退到 1.0（最自信）而不是 0.0？本系统的置信度只用于"怀疑转写
    不可靠时才考虑切分离通道"——回退 1.0 等于"不怀疑"，即保持原始转写，
    这是保守且安全的方向：宁可少切换，不可因一个解析 bug 把好结果换掉。
    """
    value = row.get("confidence")            # dict.get：键不存在返回 None，不抛 KeyError
    try:
        value = float(value)                 # 可能是字符串（如 "0.85"），统一转浮点
    except (TypeError, ValueError):
        # 键缺失（None）或内容不是数字：无法解析 → 回退 1.0
        return 1.0
    # np.isfinite 挡 NaN/Inf：模型偶尔会输出这些"非有限值"，它们会污染
    # 后续所有算术（NaN 传染），必须在入口处拦截
    return value if np.isfinite(value) else 1.0


def transcribe_firered_rows(asr: Any, paths: list[Path], prefix: str) -> list[dict[str, Any]]:
    """FireRed 批量转写：返回 [{"text": 归一化文本, "confidence": float}]。

    ``asr`` 为 FireRedRecognizer / FireRedLLMRecognizer（二者统一暴露
    ``_model.transcribe``）；``prefix`` 为本次调用的 uttid 前缀（不同调用
    使用不同前缀以区分日志来源）。

    为什么批量一次转 4 段（唤醒两通道 + 指令两通道）而不是逐段调用？
    ASR 每次调用都有固定开销（模型状态准备、日志、结果对象组装），
    批量调用摊薄这些成本；同时本函数统一了"转写+归一化+置信度解析"
    三件事，调用方只拿干净的 dict，不用重复写解析代码。
    """
    if not paths:
        # 空输入直接返回空列表：下游按索引取值前通常会先判空，
        # 这里提前返回避免给 ASR 发一个空请求（某些后端会报错）
        return []
    # uttid（utterance id）是 FireRed 要求的每条音频唯一标识；同一进程里
    # 不同调用用不同 prefix，日志里就能区分"这次是门控转写还是指令转写"
    utterance_ids = [f"{prefix}_{index}" for index in range(len(paths))]
    # suppress_model_output：FireRed 加载/推理会往 stdout 刷大量进度噪音，
    # 这个上下文管理器把它们重定向到 /dev/null，保持命令行输出干净
    with suppress_model_output():
        rows = asr._model.transcribe(utterance_ids, [str(path) for path in paths])
    # FireRed 返回的行带 uttid 字段，但顺序可能与请求不一致；
    # 先建成 {uttid: row} 字典，再按我们发出的顺序取，保证一一对应
    mapped = {str(row.get("uttid")): row for row in rows}
    if len(mapped) != len(paths):
        # 返回数量对不上（漏转/重复）说明后端状态异常，宁可报错也不
        # 静默错位——错位会把 A 的文本安到 B 头上，CER 全错且极难发现
        raise RuntimeError("FireRed returned an unexpected result count")
    output = []
    for identifier in utterance_ids:
        row = mapped[identifier]
        output.append({
            # 统一口径：文本一律归一化、置信度一律经 asr_confidence 解析，
            # 下游（门控/触发规则）不需要关心后端差异
            "text": normalize_transcript(str(row.get("text", ""))),
            "confidence": asr_confidence(row),
        })
    return output


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
