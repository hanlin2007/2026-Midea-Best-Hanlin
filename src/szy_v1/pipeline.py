"""SZY v1 端到端目标说话人 ASR 主流程（当前最佳路线的编排器）。

================================================================================
【这个文件在整条流水线里做什么？】

它是"总指挥"：把其它文件里的组件（门控 gate、ASR infer、决策 moss_candidate）
按固定顺序编排成一条完整的端到端流水线：

  1. 读取 DatasetA 的 pos.jsonl / neg.jsonl + WAV  → Sample 列表
  2. ERes2NetV2 + CAM++ 双前端 MLP 主门控          → 拒识（输出空）或放行
  3. 放行样本：raw 命令音频 → FireRed 转写 + 置信度
  4. 二维置信度/文本规则判断是否触发候选路线       → 不触发则保留 raw
  5. 触发：MossFormer2 盲分离唤醒/命令音频成两路   → wake0/wake1, cmd0/cmd1
  6. 唤醒文本选锚点（编辑距离）+ 差分声纹选命令源
  7. 任何异常 → 回退 raw FireRed → raw 也失败才输出空文本

  
设计要点（工程上值得学的）：

  - 分离候选【只可改善、不可夺权】：raw 文本永远保留为兜底
  - 每处理一条就写 checkpoint，支持 --resume 断点续跑
  - manifest 记录模型 SHA256，保证"哪次结果用了哪份模型"可追溯
================================================================================
"""
from __future__ import annotations  # 延迟注解求值

import argparse  # 命令行参数
from dataclasses import dataclass  # 数据类
import hashlib  # 哈希（SHA256 校验模型文件）
import json
from pathlib import Path
import re  # 正则（命令关键词列表 CMD_KW）
import time  # 计时
from typing import Any

import numpy as np
import soundfile as sf  # 音频读写（Moss 分离结果写 wav、读时长）
from tqdm.auto import tqdm  # 进度条

# 从"工具层"和"决策层"导入组件（相对包名导入：midea_qiji 是顶层包）
from midea_qiji.gate import DualMLPGate, read_wav
from midea_qiji.infer import (
    FireRedRecognizer,
    Sample,
    levenshtein,
    normalize_transcript,
    read_many_jsonl,
    submission_id,
    suppress_model_output,
)
from midea_qiji.moss_candidate import select_wake_anchor, should_switch_command


SAMPLE_RATE = 16000  # 16 kHz，与全项目一致
# MossFormer2 ONNX 模型的默认路径（仓库内 model/ 目录；配置里的 moss_model 会覆盖它）
DEFAULT_MOSS_MODEL = (
    "../model/workspace/midea/models/modelscope/"
    "dengcunqin/speech_mossformer2_separation_temporal_16k/simple_model.onnx"
)
# 命令关键词正则：用于统计转写文本里出现了多少个"家电指令关键词"。
# 为什么要这个特征？——正常命令（"打开空调"）含关键词；ASR 出现幻觉/乱码时
# 往往不含关键词或语速异常，这些信号可辅助判断"转写结果可不可靠"。
# re.compile 把正则"预编译"成对象，重复匹配时更快；r"..." 是原始字符串（\ 不转义）。
CMD_KW = re.compile(
    r"打开|关闭|关掉|调到|空调|灯光?|窗帘|音乐|电视|风扇|加湿|净化|扫地|"
    r"温度|音量|模式|洗衣|烘干|冰箱|热水|新风|播放|暂停|继续|小声|大声|"
    r"亮度|风速|洗澡|叫醒|闹钟|出门|回家|晚安"
)


@dataclass(frozen=True)
class SzyConfig:
    """SZY v1 的冻结配置数据类（字段与 configs/szy_v1_firered.json 一一对应）。

    frozen=True：配置实例创建后不可修改——冻结配置是实验纪律的一部分。
    每个字段的默认值 = 不读配置文件时的兜底（正式运行都会用 JSON 覆盖）。
    """

    device: str = "cuda:0"            # 主计算设备
    sv_device: str | None = None      # 声纹模型单独设备；None = 跟随 device
    model_root: str | None = None     # 模型根目录（把模型 ID 解析成路径用）
    sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"   # ERes2NetV2
    cam_model: str = "iic/speech_campplus_sv_zh-cn_16k-common"    # CAM++
    # 双 MLP 裁判权重。原队长工作区是绝对路径，这里改成仓库内相对路径，
    # 使整个仓库可整体搬迁（见 SETUP 指南的"改动说明"）。正式运行时由 JSON 覆盖。
    dual_backend: str = "model/workspace/sv_backend/backend_dual_v2_out/backend_dual.pt"
    dual_threshold: float = 0.4460311856655159  # ★ 冻结的主门控阈值（训练+校准得来）
    firered_model: str = "model/external/FireRedASR2-AED"   # FireRed AED 模型目录
    firered_repo: str = "model/external/FireRedASR2S"       # FireRed 推理源码目录
    firered_vendor: str = "model/runtime/vendor"            # FireRed 依赖（kaldi_native_fbank）
    moss_model: str = DEFAULT_MOSS_MODEL  # MossFormer2 ONNX
    moss_provider: str = "auto"           # ONNX 执行提供者：auto=有 CUDA 用 CUDA 否则 CPU
    id_mode: str = "relative_stem"        # 稳定 ID 生成方式（见 infer.submission_id）
    separation_cache: str = "outputs/szy_v1_cache/separation"  # 分离缓存目录
    batch_writes: int = 1                 # 历史遗留字段：本包当前逻辑未消费（保留以兼容配置格式）


def load_szy_config(path: str | Path) -> SzyConfig:
    """加载配置 JSON，带"未知键拒绝"校验（与 infer.load_config 同款 fail-closed）。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = SzyConfig.__dataclass_fields__
    unknown = sorted(set(payload) - set(fields))
    if unknown:  # 配置里出现了 SzyConfig 没有的键 → 报错，防止"打错字被静默忽略"
        raise ValueError(f"unknown szy_v1 config keys: {unknown}")
    return SzyConfig(**payload)


def _sha256(path: Path) -> str:
    """计算文件的 SHA-256 哈希（十六进制字符串）。

    用途：写进 manifest，作为"本次运行用的模型文件指纹"。
    为什么分块读？——模型文件可能几百 MB，一次性读进内存会爆，
    这里每次读 1MB 分块喂给哈希对象。
    """
    digest = hashlib.sha256()  # 创建哈希对象（可多次 update 累加数据）
    with path.open("rb") as handle:  # rb = 二进制读
        # iter(lambda: handle.read(1024*1024), b"") 的技巧：
        #   iter(可调用对象, 哨兵值) 会反复调用该对象，直到返回值等于哨兵值
        #   即：反复 read 1MB，直到读到空字节 b""（文件末尾）才停
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)  # 把这 1MB 喂给哈希
    return digest.hexdigest()  # 输出 64 位十六进制指纹


class MossFormer2Separator:
    """MossFormer2 盲分离器的 ONNX 封装（小包装，适配历史调用方式）。

    MossFormer2 是一个"语音分离"模型：输入混合语音，输出两路分离后的语音
    （源 0 / 源 1），至于哪一路是目标人，模型不知道——这正是后面要用
    "唤醒文本 + 差分声纹"再做判定的原因。
    """

    def __init__(self, model_path: str | Path, device: str, provider: str = "auto") -> None:
        import onnxruntime as ort  # lazy import：ONNX Runtime（跨平台推理引擎）

        model = Path(model_path).expanduser().resolve()  # expanduser 展开 ~，resolve 转绝对路径
        if not model.is_file():  # 模型文件必须存在（fail-fast）
            raise FileNotFoundError(f"MossFormer2 ONNX model not found: {model}")
        # 选择"执行提供者"（Execution Provider）：
        #   CUDAExecutionProvider = GPU 版算子；CPUExecutionProvider = CPU 版
        # auto 策略：设备是 cuda 且环境里有 CUDA 提供者 → 用 GPU，否则 CPU
        available = ort.get_available_providers()
        use_cuda = device.startswith("cuda") and "CUDAExecutionProvider" in available
        if provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is unavailable for MossFormer2")
        if provider == "cpu":
            providers = ["CPUExecutionProvider"]
        elif use_cuda or provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # GPU 优先，CPU 兜底
        else:
            providers = ["CPUExecutionProvider"]
        self.model_path = model
        self.providers = providers
        # 创建 ONNX 推理会话（加载模型图 + 权重）
        self.session = ort.InferenceSession(str(model), providers=providers)
        # 记住输入张量的名字（不同 ONNX 模型输入名不同，不能写死）
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _waveform(path: Path) -> np.ndarray:
        """读 16k 单声道 float32 波形，带严格校验（staticmethod：不依赖实例的辅助函数）。"""
        waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if rate != SAMPLE_RATE:  # 采样率不符 → 报错（模型按 16k 训练，喂错采样率会全错）
            raise ValueError(f"expected 16 kHz audio, got {rate}: {path}")
        if waveform.ndim == 2:  # 双声道 → 平均成单声道
            waveform = waveform.mean(axis=1)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size == 0 or not np.isfinite(waveform).all():  # 空波形或含 NaN/Inf → 报错
            raise ValueError(f"invalid audio waveform: {path}")
        return waveform

    def separate(self, source: Path, output_prefix: Path) -> tuple[Path, Path]:
        """把一条混合音频分离成两路 wav，返回 (ms0 路径, ms1 路径)。

        带"缓存复用"：如果两路结果已经存在，直接返回不重算（长任务中断重跑时省时）。
        """
        output_prefix.parent.mkdir(parents=True, exist_ok=True)  # 建缓存目录
        # 输出文件命名：<前缀>_ms0.wav / <前缀>_ms1.wav（ms = Moss source）
        outputs = (output_prefix.with_name(output_prefix.name + "_ms0.wav"),
                   output_prefix.with_name(output_prefix.name + "_ms1.wav"))
        if all(path.is_file() for path in outputs):  # 缓存命中 → 直接返回
            return outputs
        waveform = self._waveform(source)
        # ONNX 推理：session.run(None, {输入名: 张量})。
        #   np.newaxis 加一维：把 (T,) 变成 (1, T)（批维度 = 1 条）
        #   返回 [0] 取第一个输出（分离结果张量），形状 (1, T, 2)（两路）
        separated = self.session.run(None, {self.input_name: waveform[np.newaxis, :]})[0]
        array = np.asarray(separated)
        # 形状校验：必须 3 维、至少 1 批、最后一维 >= 2（两路）
        if array.ndim != 3 or array.shape[0] < 1 or array.shape[2] < 2:
            raise RuntimeError(f"unexpected MossFormer2 output shape {array.shape} for {source}")
        for index, output_path in enumerate(outputs):
            signal = np.asarray(array[0, :, index], dtype=np.float32)  # 取第 index 路
            # 峰归一化：信号 / 峰值 * 0.5。
            #   把最大振幅压到 0.5，留出余量，避免写回 PCM16 时削波（±1 极限）；
            #   两路统一峰值刻度，后续声纹比较才公平
            peak = float(np.max(np.abs(signal))) if signal.size else 0.0
            if peak > 0.0:
                signal = signal / peak * 0.5
            # 写回 wav：PCM_16 = 16 位整数（与竞赛音频格式一致）
            sf.write(str(output_path), signal, SAMPLE_RATE, subtype="PCM_16")
        return outputs


def _duration(path: Path) -> float:
    """用 soundfile 读音频时长（帧数 / 采样率 = 秒），不必把波形整个读进内存。"""
    info = sf.info(str(path))
    return float(info.frames) / max(int(info.samplerate), 1)  # max(...,1) 防除零


def _command_features(text: str, duration: float) -> tuple[float, float]:
    """从转写文本 + 时长算出两个"命令特征"。

    返回 (chars_per_sec 语速, keyword_count 关键词数)。
    chars_per_sec：归一化文本长度 / 时长（分母至少 0.1 秒，防止超短音频数值爆炸）
    """
    normalized = normalize_transcript(text)
    # CMD_KW.findall：找出文本里所有命令关键词匹配（返回列表），len 得到个数
    return float(len(normalized) / max(duration, 0.1)), float(len(CMD_KW.findall(normalized)))


def _asr_confidence(row: dict[str, Any]) -> float:
    """从 FireRed 返回行里安全地取置信度；取不到就默认 1.0（视为"可信"）。"""
    value = row.get("confidence")
    try:
        value = float(value)
    except (TypeError, ValueError):  # None 或非数字 → 兜底 1.0
        return 1.0
    return value if np.isfinite(value) else 1.0  # NaN/Inf 也兜底 1.0


class SzyV1Runner:
    """SZY v1 的运行器：持有所有重型组件（门控、分离器、ASR），逐样本 process。"""

    def __init__(self, config: SzyConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.cache_dir = Path(config.separation_cache)
        if not self.cache_dir.is_absolute():  # 相对路径 → 转成基于当前工作目录的绝对路径
            self.cache_dir = (Path.cwd() / self.cache_dir).resolve()
        # 组装三个重型组件（构造时就加载模型，耗时但只做一次）：
        self.gate = DualMLPGate(  # ① 主门控
            config.sv_model,
            config.cam_model,
            config.dual_backend,
            config.dual_threshold,
            config.sv_device or config.device,  # sv_device 为空则用 device
            config.model_root,
        )
        self.separator = MossFormer2Separator(config.moss_model, config.device, config.moss_provider)  # ② 分离器
        self.asr = FireRedRecognizer(  # ③ ASR
            config.firered_model,
            config.device,
            config.firered_repo,
            config.firered_vendor,
        )

    @staticmethod
    def _transcribe(asr: FireRedRecognizer, paths: list[Path]) -> list[dict[str, Any]]:
        """转写一批音频，返回 [{"text":..., "confidence":...}, ...]（保持顺序）。

        staticmethod：不需要访问实例字段，纯粹借用 asr 参数。
        """
        if not paths:
            return []
        # FireRed 接口要求每条音频一个 ID；这里用 szy_0/szy_1/... 现造
        utterance_ids = [f"szy_{index}" for index in range(len(paths))]
        with suppress_model_output():  # 屏蔽模型内部打印
            # 直接调底层模型对象的 transcribe（返回行 dict 列表，带 confidence）
            rows = asr._model.transcribe(utterance_ids, [str(path) for path in paths])
        # 用 uttid 对齐，防止模型乱序
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(paths):  # 数量守卫
            raise RuntimeError("FireRed returned an unexpected result count")
        output = []
        for identifier in utterance_ids:
            row = mapped[identifier]
            output.append({
                "text": normalize_transcript(str(row.get("text", ""))),  # 归一化文本
                "confidence": _asr_confidence(row),                      # 置信度
            })
        return output

    def _paths_for(self, sample: Sample) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
        """为一条样本规划分离输出路径：唤醒/命令各两路，共 4 个缓存文件。

        返回 ((wake_ms0, wake_ms1), (command_ms0, command_ms1))。
        缓存键用稳定 ID（submission_id），并把 "/" 换成 "_" 防止出现嵌套目录。
        """
        item_id = submission_id(sample, self.config.id_mode).replace("/", "_")
        wake_prefix = self.cache_dir / f"{item_id}_wake"      # 例：.../pos_cmd_13_wake
        command_prefix = self.cache_dir / f"{item_id}_command"
        return (
            self.separator.separate(sample.wake_audio, wake_prefix),
            self.separator.separate(sample.command_audio, command_prefix),
        )

    def process(self, sample: Sample) -> dict[str, Any]:
        """处理一条样本，返回该样本的完整结果 dict（含大量审计字段）。

        这是整个文件最核心的方法，流程见模块 docstring 的 7 步。
        """
        started = time.perf_counter()
        item_id = submission_id(sample, self.config.id_mode)
        # ① 主门控判决
        accepted, gate_score = self.gate.accept(sample.wake_audio, sample.command_audio)
        # 结果骨架：先把所有字段填好默认值（保证无论走哪条路、是否异常，
        # 每条样本都有结构一致的记录，方便下游统计与审计）
        result: dict[str, Any] = {
            "id": item_id,
            "content": "",                                   # 最终输出文本（默认空）
            "label": sample.label,
            "sample_type": "positive" if sample.label is not None else "negative",
            "sv_score": f"{gate_score:.6f}",                 # 主门控分数（6 位小数）
            "accepted": bool(accepted),
            "sv_accept_threshold": f"{self.config.dual_threshold:.6f}",
            "route": "primary_reject",                       # 路由标签（审计核心字段）
            "selected_candidate": "",
            "switch_triggered": False,
            "command_source_empty": None,
            "wake_selected_source": None,
            "wake_raw_fallback": None,
            "wake_text_tie": None,
            "raw_confidence": None,
            "chars_per_sec": None,
            "command_keyword_score": None,
            "error": None,
        }
        if not accepted:  # ② 拒绝 → 直接返回空文本结果（不跑 ASR）
            result["item_duration"] = f"{time.perf_counter() - started:.6f}"
            return result

        # ③④⑤⑥⑦ 都包在 try 里：任何一步出问题都走回退，保证"候选失败不夺权"
        try:
            # ③ raw 命令音频 FireRed 转写（冻结的切换决策只依赖 raw 输出）
            #   为什么先转写 raw 再决定要不要分离？——大部分样本 raw 是可靠的，
            #   先判断能省下 4 次分离+转写（省大量算力），且决策结果与历史一致
            raw_command = self._transcribe(self.asr, [sample.command_audio])[0]
            # 算两个命令特征（语速、关键词数）
            chars_per_sec, command_keyword_score = _command_features(
                raw_command["text"], _duration(sample.command_audio)
            )
            # ④ 冻结的二维切换规则
            switch = should_switch_command(
                confidence=raw_command["confidence"],
                command_keyword_score=command_keyword_score,
                chars_per_sec=chars_per_sec,
            )
            result.update({  # dict.update：批量合并键值对
                "content": raw_command["text"],
                "route": "moss_candidate" if switch else "raw",
                "selected_candidate": "orig",
                "selection_reason": "candidate_pending" if switch else "confidence_rule_not_triggered",
                "switch_triggered": bool(switch),
                "raw_confidence": f"{raw_command['confidence']:.6f}",
                "chars_per_sec": f"{chars_per_sec:.6f}",
                "command_keyword_score": f"{command_keyword_score:.6f}",
            })
            if not switch:  # 不触发候选 → 保留 raw 文本，收工
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # ⑤ 分离唤醒音频 + 命令音频（共 4 个 wav）
            (wake_paths, command_paths) = self._paths_for(sample)
            # 把 4 路分离结果一次性批转写（wake0/wake1/cmd0/cmd1），
            # 列表解包同时赋给四个变量
            wake0, wake1, command0, command1 = self._transcribe(
                self.asr,
                [wake_paths[0], wake_paths[1], command_paths[0], command_paths[1]],
            )
            # ⑥ 用唤醒文本在两路唤醒源里选锚点（返回 WakeAnchorDecision）
            anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
            # 锚点波形：若需要 raw fallback 就用原始唤醒音频，否则用选中的分离源
            anchor_path = sample.wake_audio if anchor.raw_wake_fallback else wake_paths[anchor.selected_source]
            contrast_path = wake_paths[anchor.contrast_source]  # 对照源波形

            # ⑦ 差分声纹选择命令源：读 4 段波形（锚点、对照、cmd0、cmd1）
            anchor_wave = read_wav(anchor_path)
            contrast_wave = read_wav(contrast_path)
            candidate_waves = [read_wav(command_paths[0]), read_wav(command_paths[1])]
            embeddings = {}
            # 两个声纹前端（eres/cam）各算一次"差分"：
            #   对每个命令候选 c ∈ {cmd0, cmd1}：
            #     diff(c) = cos(anchor, c) - cos(contrast, c)
            #   含义：候选与"目标锚点"的相似度，减去它与"对照（非目标）"的相似度。
            #   若 c 是目标人 → 第一项大、第二项小 → diff 大；反之 diff 小甚至为负。
            #   用"差分"而不是"绝对相似度"的好处：抵消掉声道/噪声等共性干扰。
            for name, embedder in (("eres", self.gate.eres_embedder), ("cam", self.gate.cam_embedder)):
                # 批量提取 4 段波形的 embedding（一次 GPU 前向）
                values = embedder.extract_waveforms([anchor_wave, contrast_wave, *candidate_waves])
                anchor_embedding, contrast_embedding = values[:2]  # 前两个是锚点/对照
                # 后两个是命令候选；embedding 已 L2 归一化，np.dot 即余弦相似度
                embeddings[name] = [
                    float(np.dot(anchor_embedding, candidate) - np.dot(contrast_embedding, candidate))
                    for candidate in values[2:]
                ]

            # 命令源选择：任一命令源为空 → 分离失败，回退 orig（不赌单路）
            source_empty = not command0["text"] or not command1["text"]
            selected = "orig"
            reason = "candidate_pending"
            if source_empty:
                reason = "command_source_empty"
            else:
                # 双前端差分分数相加后比较（>= 时选 ms0，保证确定性）
                selected = "ms0" if (
                    embeddings["eres"][0] + embeddings["cam"][0]
                ) >= (
                    embeddings["eres"][1] + embeddings["cam"][1]
                ) else "ms1"
                reason = "same_domain_differential_score"
            # 三选一：orig / ms0 / ms1 的文本
            text_by_candidate = {
                "orig": raw_command["text"],
                "ms0": command0["text"],
                "ms1": command1["text"],
            }
            result.update({
                "content": text_by_candidate[selected] if selected != "orig" else raw_command["text"],
                "route": "moss_candidate" if switch else "raw",
                "selected_candidate": selected,
                "selection_reason": reason,
                "command_source_empty": bool(source_empty),
                "wake_selected_source": int(anchor.selected_source),
                "wake_contrast_source": int(anchor.contrast_source),
                "wake_raw_fallback": bool(anchor.raw_wake_fallback),
                "wake_text_tie": bool(anchor.text_tie),
                # 审计字段：把两路唤醒/命令的转写、四个差分分数全部记录下来
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                "command_source0_text": command0["text"],
                "command_source1_text": command1["text"],
                "eres_diff_ms0": f"{embeddings['eres'][0]:.6f}",
                "eres_diff_ms1": f"{embeddings['eres'][1]:.6f}",
                "cam_diff_ms0": f"{embeddings['cam'][0]:.6f}",
                "cam_diff_ms1": f"{embeddings['cam'][1]:.6f}",
            })
        except Exception as error:  # 任何异常 → 回退 raw ASR（保留原始结果的控制权）
            result["error"] = f"{type(error).__name__}: {error}"  # 记录异常类型+消息
            try:
                # 回退：重新对原始命令音频转写（asr.transcribe_many 是 FireRed 的另一个接口）
                result["content"] = self.asr.transcribe_many([sample.command_audio])[0]
                result["route"] = "raw_fallback_error"
                result["selected_candidate"] = "orig"
            except Exception as raw_error:  # raw 也失败 → 只能输出空
                result["error"] += f"; raw_fallback={type(raw_error).__name__}: {raw_error}"
                result["route"] = "empty_fallback_error"
        result["item_duration"] = f"{time.perf_counter() - started:.6f}"
        return result


def summarize(results: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """聚合所有样本结果 → 计算 CER / PR / RR 与路由分布（SZY v1 口径）。"""
    # 正样本 = label 非空；负样本 = label 为空或 None
    positive = [row for row in results if row.get("label") is not None and row.get("label") != ""]
    negative = [row for row in results if row.get("label") is None or row.get("label") == ""]
    # CER 分子：所有正样本 输出 vs 标签 的编辑距离累加（generator 表达式喂给 sum）
    errors = sum(
        levenshtein(
            normalize_transcript(str(row.get("content") or "")),
            normalize_transcript(str(row["label"])),
        )
        for row in positive
    )
    # CER 分母：所有正样本参考字符数累加
    characters = sum(len(normalize_transcript(str(row["label"]))) for row in positive)
    nonempty = sum(bool(str(row.get("content") or "")) for row in positive)  # 正样本非空输出数
    rejects = sum(not str(row.get("content") or "") for row in negative)    # 负样本被拒识数
    routes: dict[str, int] = {}  # 统计每种路由出现了多少次（如 raw:870, moss_candidate:408）
    for row in results:
        key = row.get("route", "unknown")
        routes[key] = routes.get(key, 0) + 1  # dict.get 带默认值 0，实现计数器
    return {
        "route": "szy_v1",
        "sample_count": len(results),
        "positive_count": len(positive),
        "positive_nonempty": nonempty,
        "final_pr": nonempty / len(positive) if positive else 0.0,  # PR = 非空/正样本总数
        "negative_count": len(negative),
        "negative_rejects": rejects,
        "final_rr": rejects / len(negative) if negative else 0.0,   # RR = 拒识/负样本总数
        "errors": errors,
        "characters": characters,
        "final_cer": errors / characters if characters else 0.0,    # CER = 编辑距离/字符数
        "duration": duration,
        "routes": routes,
        "switches": sum(bool(row.get("switch_triggered")) for row in results),  # 触发候选路线的条数
        "separation_errors": sum(bool(row.get("error")) for row in results),    # 出错条数
    }


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], output: Path, detail: Path) -> None:
    """写 submission（提交格式）与 detail（审计明细）两个 JSON。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.parent.mkdir(parents=True, exist_ok=True)
    submission = []  # 提交文件内容
    details = []     # 明细文件内容
    for row in results:
        label = row.get("label")
        if label is not None and label != "":  # 正样本：算单条 CER
            reference = normalize_transcript(str(label))
            hypothesis = normalize_transcript(str(row.get("content") or ""))
            item_cer = (
                levenshtein(hypothesis, reference) / len(reference)
                if reference else float(hypothesis != "")
            )
        else:  # 负样本：无 CER
            item_cer = None
        submission.append({
            "id": row["id"],
            "content": str(row.get("content") or ""),
            "label": label,
            "cer": "" if item_cer is None else f"{item_cer:.6f}",  # None 写成空串
        })
        item = dict(row)   # 复制一份行数据，避免污染原结果
        item["cer"] = item_cer
        details.append(item)
    # ensure_ascii=False：保留中文；indent=2：缩进可读
    output.write_text(json.dumps({"result": {"results": submission, "summary": summary}}, ensure_ascii=False, indent=2), encoding="utf-8")
    detail.write_text(json.dumps({"summary": summary, "results": details}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """SZY v1 命令行参数定义。"""
    parser = argparse.ArgumentParser(description="SZY v1 formal MossFormer2 target-speaker ASR")
    parser.add_argument("--data-root", required=True)             # 数据根目录
    parser.add_argument("--jsonl", nargs="*", help="JSONL 清单；不传则自动在 data-root 下找 pos.jsonl + neg.jsonl")      # 清单文件（pos + neg）
    parser.add_argument("--config", default="configs/szy_v1_firered.json")  # 冻结配置
    parser.add_argument("--output", default="outputs/szy_v1_submission.json")
    parser.add_argument("--final-output", default="outputs/szy_v1_detail.json")
    parser.add_argument("--checkpoint-output", default="outputs/szy_v1_checkpoint.json")
    parser.add_argument("--manifest-output", default="outputs/szy_v1_manifest.json")
    parser.add_argument("--model-root")                          # 可选：覆盖配置里的模型根目录
    parser.add_argument("--limit", type=int)                     # 只跑前 N 条（冒烟）
    parser.add_argument("--resume", action="store_true")         # 断点续跑开关
    return parser


def main() -> None:
    """SZY v1 主入口：加载配置 → 读样本 → 逐条 process → 汇总写盘。"""
    args = build_parser().parse_args()  # 解析命令行参数
    config = load_szy_config(args.config)  # 读配置
    if args.model_root:  # 命令行 --model-root 优先于配置文件
        # {**config.__dict__, ...}：把 dataclass 的字段字典复制一份并覆盖 model_root，
        # 再构造一个新 SzyConfig（frozen 对象不能原地改，只能新建）
        config = SzyConfig(**{**config.__dict__, "model_root": args.model_root})
    # --jsonl 不传时，自动在 data-root 下找 pos.jsonl + neg.jsonl
    if not args.jsonl:
        args.jsonl = []
        root = Path(args.data_root)
        for name in ("pos.jsonl", "neg.jsonl"):
            candidate = root / name
            if candidate.is_file():
                args.jsonl.append(str(candidate))
        if not args.jsonl:
            print("[ERROR] --jsonl 未提供，且 data-root 下找不到 pos.jsonl 或 neg.jsonl", flush=True)
            raise SystemExit(1)
    samples = read_many_jsonl(args.jsonl, args.data_root)  # 读样本
    if args.limit is not None:
        samples = samples[: args.limit]
    # ---- 断点续跑 ----
    checkpoint_path = Path(args.checkpoint_output)
    results: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = list(checkpoint.get("results", []))
        elapsed_before = float(checkpoint.get("duration", 0.0))
        completed = {str(item["id"]) for item in results}  # 已完成 ID 集合
        # 过滤：只保留没处理过的样本（用稳定 ID 去重）
        samples = [sample for sample in samples if submission_id(sample, config.id_mode) not in completed]
    runner = SzyV1Runner(config, Path(args.output).parent)  # 组装重型组件（加载模型）
    started = time.perf_counter()
    for sample in tqdm(samples, desc="szy_v1", unit="utt"):  # 带进度条主循环
        result = runner.process(sample)
        results.append(result)
        elapsed = elapsed_before + time.perf_counter() - started
        # 每处理完一条就写 checkpoint（原子性要求不高，但保证中断后损失≤1条）
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"version": "szy_v1", "duration": elapsed, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    duration = elapsed_before + time.perf_counter() - started
    summary = summarize(results, duration)  # 汇总指标
    write_outputs(results, summary, Path(args.output), Path(args.final_output))  # 写两个 JSON
    # ---- manifest：记录"这次运行用了什么"（可追溯性）----
    manifest = {
        "version": "szy_v1",
        "config": config.__dict__,  # 配置快照（含所有路径/阈值）
        "data_root": str(Path(args.data_root).resolve()),
        "jsonl": [str(Path(path).resolve()) for path in args.jsonl],
        "sample_count": len(results),
        "moss_model_sha256": _sha256(Path(config.moss_model).resolve()),    # 分离模型指纹
        "dual_backend_sha256": _sha256(Path(config.dual_backend).resolve()),  # 门控权重指纹
        "output": str(Path(args.output).resolve()),
        "detail": str(Path(args.final_output).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)  # 打印汇总（flush 立即刷出）


if __name__ == "__main__":
    main()  # python -m szy_v1.pipeline 时的入口