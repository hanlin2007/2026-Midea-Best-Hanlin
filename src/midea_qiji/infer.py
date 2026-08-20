"""midea_qiji 工具层：样本解析、音频读取、ASR 后端封装、文本归一化、指标计算。

================================================================================
【这个文件在整条流水线里做什么？】

它是"公共工具箱"，被 SZY v1 流水线（szy_v1/pipeline.py）复用，也自带一条
独立的旧版基线入口（main()，ERes2NetV2 余弦阈值 + Qwen3-ASR）。

对外提供的关键内容：
  - Sample 数据类           : 一条竞赛样本（唤醒/识别音频路径 + 文本标签）的载体
  - read_many_jsonl()       : 把竞赛的 JSONL 清单解析成 Sample 列表
  - submission_id()         : 给样本生成"稳定 ID"（断点续跑、防覆盖的关键）
  - normalize_transcript()  : 文本归一化（数字转中文、去标点），CER 的统计口径
  - levenshtein() / cer()   : 编辑距离 / 字符错误率
  - FireRedRecognizer       : FireRedASR2-AED 语音识别后端的封装（SZY v1 主力 ASR）
  - QwenRecognizer          : Qwen3-ASR 后端（旧基线用）
  - ConditionedQwenRecognizer : 双音频条件 LoRA 后端（实验分支用，SZY v1 不用）
  - SpeakerVerifier         : 单模型余弦阈值门控（旧基线用，SZY v1 用 gate.py 的 DualMLPGate）
  - write_reports()         : 把逐条结果汇总成 CER/PR/RR 并写出 submission/detail JSON

注意区分两条路线（课程第 3~5 课会反复强调）：
  - 旧基线（infer.main()）: gate_mode="cos"，单 ERes2NetV2，阈值 0.27，Qwen3-ASR
  - SZY v1（szy_v1/pipeline.py）: DualMLPGate，阈值 0.446031...，FireRed ASR + MossFormer2 候选
================================================================================
"""
from __future__ import annotations  # 延迟注解求值：允许在类型标注里写"字符串形式的向前引用"

import argparse  # 命令行参数解析（python xx.py --data-root ... 背后就是它）
import contextlib  # 上下文管理工具（本文件用 redirect_stdout 屏蔽模型打印）
from dataclasses import dataclass  # 数据类：一行声明一个"只装数据"的类
import json
import os
from pathlib import Path
import re  # 正则表达式：文本归一化的主力工具
import time  # 计时（item_duration、总耗时）
from typing import Any, Iterable  # Any=任意类型；Iterable=可迭代对象

import numpy as np
from tqdm.auto import tqdm  # 进度条（tqdm.auto 会自动适配 Jupyter/控制台）


# 模型总目录的默认位置。
# Path(__file__).resolve().parents[2] = src/midea_qiji/infer.py 往上两级的 src/ 的
# 上一级 = 包根目录（仓库根目录）。原队长工作区把模型放在仓库的"同级目录 ../model"，
# 本仓库为了"一个文件夹可整体搬迁"，改为仓库内的 model/ 目录（见 SETUP 指南）。
# 注意：SZY v1 的配置显式给出了所有模型路径，这里的默认值只在配置缺省时兜底。
PACKAGE_MODEL_ROOT = Path(__file__).resolve().parents[2] / "model"
DEFAULT_FIRED_MODEL = str(PACKAGE_MODEL_ROOT / "external" / "FireRedASR2-AED")
DEFAULT_FIRED_REPO = str(PACKAGE_MODEL_ROOT / "external" / "FireRedASR2S")
DEFAULT_FIRED_VENDOR = PACKAGE_MODEL_ROOT / "runtime" / "vendor"


@dataclass(frozen=True)
class Config:
    """旧基线（infer.main）的配置数据类。

    frozen=True 表示实例创建后不可修改（不可变对象）——配置一旦加载就不该
    被中途偷改，这能防止"某处代码悄悄改了阈值"这类隐蔽 bug。
    """

    device: str = "cuda:0"                          # 主计算设备（GPU 编号或 cpu）
    sv_device: str | None = None                    # 声纹模型单独的设备；None = 跟 device 走
    asr_model: str = "Qwen/Qwen3-ASR-0.6B"          # Qwen ASR 的模型 ID
    sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"  # ERes2NetV2 声纹模型
    sv_model_revision: str | None = "v1.0.0"        # 模型版本号
    sv_accept_threshold: float = 0.27               # 旧基线余弦门控阈值（≠ SZY v1 的 0.446031！）
    id_mode: str = "relative_stem"                  # 样本 ID 生成方式（见 submission_id）
    # 门控模式："cos" = ERes2NetV2 余弦阈值（baseline）；"dual_mlp" = 双前端 MLP 裁判
    gate_mode: str = "cos"
    cam_model: str = "iic/speech_campplus_sv_zh-cn_16k-common"   # CAM++ 声纹模型
    dual_backend: str | None = None                 # 双 MLP 裁判权重路径（gate_mode=dual_mlp 时必需）
    dual_threshold: float = 0.5334                  # 双 MLP 默认阈值（旧值；SZY v1 用配置里的 0.446031）
    # ASR 后端："qwen" = Qwen3-ASR-0.6B；"firered" = FireRedASR2-AED；
    # "qwen_lora" = sv_asr 双音频条件 LoRA 微调权重（[kws, cmd] → 目标人文本）
    asr_backend: str = "qwen"
    firered_model: str = DEFAULT_FIRED_MODEL        # FireRed AED 模型目录
    firered_repo: str = DEFAULT_FIRED_REPO          # FireRedASR2S 推理源码目录
    qwen_lora_adapter: str | None = None            # LoRA 适配器路径（qwen_lora 后端用）
    # Optional cache-driven second-stage target-presence gate.  It is disabled
    # for the production baseline until an external model is trained and
    # calibrated against the required PR/RR constraints.
    # （二级存在性门控：默认关闭。SZY v1 主链路也不启用它）
    second_stage_enabled: bool = False
    second_stage_model: str | None = None
    second_stage_features: str | None = None
    second_stage_presence_threshold: float | None = None


@dataclass(frozen=True)
class Sample:
    """一条竞赛样本的载体（纯数据，不含任何处理逻辑）。

    raw_id 是 JSONL 里原始的 "id" 字段；注意竞赛数据里不同文件的 id 可能
    重复（pos 和 neg 各自从 0 开始编号），所以提交时要用 submission_id()
    从路径派生稳定 ID，而不是直接用 raw_id。
    """

    raw_id: str
    wake_audio: Path            # 唤醒音频的绝对路径（kws_xxx.wav）
    command_audio: Path         # 识别音频的绝对路径（cmd_xxx.wav）
    command_relative_path: str  # 识别音频的相对路径（如 "pos/cmd_13.wav"，派生稳定 ID 用）
    label: str | None           # 识别文本标签；负样本为 None（"识别文本": null）
    wake_text: str | None = None  # 唤醒文本（如"你好科慕"）；JSONL 里可能缺失


def load_config(path: str | Path) -> Config:
    """从 JSON 文件加载旧基线配置，带"未知键拒绝"（fail-closed）校验。

    设计思想：配置文件里打错一个字（如把 dual_threshold 写成 dual_thresholds）
    会静默变成"用了默认值"，这是实验事故的常见来源。所以：任何 JSON 里出现了
    Config 没有的键 → 直接报错，而不是假装没看见。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = Config.__dataclass_fields__  # dataclass 自带的"字段名 → 字段对象"字典
    unknown = sorted(set(payload) - set(fields))  # 集合差：JSON 的键 - 合法的键
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return Config(**payload)  # **payload：把字典的键值对展开成关键字参数传给构造函数


def resolve_model_path(model: str | Path, model_root: str | Path | None) -> Path:
    """把模型 ID（如 "iic/speech_eres2netv2_sv_zh-cn_16k-common"）解析成磁盘路径。

    查找顺序（第一个存在即返回）：
      1. 原样当作路径直接存在？（支持直接传 /path/to/model）
      2. model_root / 模型ID（配置的模型根目录下）
      3. 环境变量 MODELSCOPE_CACHE 下
      4. 默认的 models/modelscope、~/.cache/modelscope/hub 下
    注意 ModelScope 缓存会把 ID 里的 "." 换成 "___" 作目录名，所以两种拼法都试。
    """
    direct = Path(model)
    if direct.exists():  # 本身就是个存在的路径（绝对/相对都行）
        return direct.resolve()
    roots = []  # 候选根目录列表，按优先级排列
    if model_root is not None:
        roots.append(Path(model_root))
    if os.environ.get("MODELSCOPE_CACHE"):  # 环境变量优先级次之
        roots.append(Path(os.environ["MODELSCOPE_CACHE"]))
    roots.extend([Path("models/modelscope"), Path.home() / ".cache" / "modelscope" / "hub"])
    for root in roots:
        for candidate in (
            root / str(model),                       # 正常拼法
            root / str(model).replace(".", "___"),   # ModelScope 的目录名转义
            root / "models" / str(model),            # 有的缓存多一层 models/
            root / "models" / str(model).replace(".", "___"),
        ):
            if candidate.exists():
                return candidate.resolve()
    # 全都没找到 → 报错并提示补救办法（设置 --model-root 或 MODELSCOPE_CACHE）
    raise FileNotFoundError(
        f"local model {model!r} was not found; set --model-root or MODELSCOPE_CACHE"
    )


class SpeakerVerifier:
    """旧基线的单模型余弦门控：只用一个 ERes2NetV2 给（唤醒, 命令）打相似度分。

    SZY v1 不用它（用的是 gate.py 的 DualMLPGate 双前端 MLP），保留它是为了
    旧基线入口 main() 在 gate_mode="cos" 时可用。
    """

    def __init__(self, model: str, device: str, model_root: str | Path | None) -> None:
        # lazy import：modelscope 很重，用到才导入（加快纯文本工具函数的导入速度）
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            task=Tasks.speaker_verification,  # 说话人确认任务
            model=str(resolve_model_path(model, model_root)),
            device=device,
        )

    def score(self, enrollment_audio: Path, command_audio: Path) -> float:
        # pipeline 的输入是一对音频路径 [注册音频, 测试音频]，输出是相似度分数
        return extract_speaker_score(self.pipeline([str(enrollment_audio), str(command_audio)]))


class QwenRecognizer:
    """Qwen3-ASR-0.6B 识别后端（旧基线的默认 ASR）。"""

    def __init__(
        self,
        model: str,
        device: str,
        model_root: str | Path | None,
        batch_size: int,
    ) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel  # 官方推理包（pip install qwen-asr）

        self.model = Qwen3ASRModel.from_pretrained(
            str(resolve_model_path(model, model_root)),
            dtype=torch.bfloat16,  # 半精度浮点：省一半显存，速度也更快（要求较新 GPU）
            device_map="cuda:0" if device == "cuda" else device,  # 自动映射到 GPU
            max_inference_batch_size=max(1, batch_size),  # 一次最多识别几条
            max_new_tokens=256,  # 最多生成的 token 数（中文命令文本 256 足够）
        )

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        with suppress_model_output():  # 屏蔽模型加载/推理时刷屏的日志（见文件底部）
            results = self.model.transcribe(
                audio=[str(path) for path in audio_paths],
                language=["Chinese"] * len(audio_paths),  # 显式指定中文
            )
        # 每个结果抽出文本并做归一化（口径与 SZY v1 完全一致）
        return [normalize_transcript(extract_asr_text(result)) for result in results]


class FireRedRecognizer:
    """FireRedASR2-AED 后端（zjm v2 实测 CER 比 Qwen3-ASR 好约 3.7pt）。

    FireRedASR 是小红书开源的 ASR 系统，AED = Attention Encoder-Decoder
    （编码器-注意力-解码器）。SZY v1 全链路都用它，因为它对重叠语音的
    原始转写质量最好（课程第 4 课详解）。

    依赖：kaldi_native_fbank（装在 vendor 目录，不污染共享 conda）+
    FireRedASR2S 源码仓库（examples_infer/asr 下的 fireredasr2 包）。
    """

    def __init__(self, model_dir: str | Path, device: str, repo_root: str | Path,
                 vendor_root: str | Path = DEFAULT_FIRED_VENDOR) -> None:
        import sys

        # 把两个目录临时塞进 sys.path（模块搜索路径），这样 import fireredasr2 才能找到：
        #   1) vendor 目录：装着 kaldi_native_fbank 等第三方依赖
        #   2) FireRedASR2S 源码仓库的 examples_infer/asr：装着 fireredasr2 包本体
        # sys.path 是 Python 找模块时逐条查找的目录列表（类似系统 PATH）
        for path in (str(vendor_root), str(Path(repo_root) / "examples_infer" / "asr")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

        # 从本地目录加载 AED 模型（"aed" = 选择 AED 架构变体，区别于 LLM 变体）
        # 下面这些解码参数是队长冻结的（改动会改变识别结果，不能随便调）：
        self._model = FireRedAsr2.from_pretrained(
            "aed",
            str(Path(model_dir).resolve()),
            FireRedAsr2Config(
                use_gpu=device.startswith("cuda"),  # 是否用 GPU
                use_half=device.startswith("cuda"), # GPU 上用半精度加速
                beam_size=1,            # 束搜索宽度 1 = 贪心解码（每次取概率最大的 token）
                nbest=1,                # 只保留最好的 1 条假设
                decode_max_len=0,       # 0 = 不限制解码长度（按模型自定上限）
                softmax_smoothing=1.25, # 输出概率平滑，防过度自信（>1 拉平分布）
                aed_length_penalty=0.6, # 长度惩罚（<1 倾向短句，抑制重复）
                eos_penalty=1.0,        # 结束符惩罚（1.0 = 不惩罚）
                return_timestamp=False, # 不要时间戳（竞赛只比文本）
                decode_min_len=0,       # 最短输出长度不限制
                repetition_penalty=1.0, # 重复惩罚 1.0 = 关闭
                llm_length_penalty=0.0, # LLM 分支的长度惩罚（AED 不用）
                temperature=1.0,        # 采样温度 1.0 = 不改变分布
                elm_dir="",             # 错误语言模型目录（空 = 不用）
                elm_weight=0.0,         # 错误语言模型权重 0 = 关闭
            ),
        )

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        """批量转写。返回与输入等长的文本 list（已经过 normalize_transcript）。"""
        if not audio_paths:  # 空输入直接返回（避免调用模型时越界）
            return []
        # FireRed 的接口要求每条音频带一个唯一 ID，这里用 utt_0/utt_1/... 现造
        utterance_ids = [f"utt_{index}" for index in range(len(audio_paths))]
        with suppress_model_output():  # 屏蔽模型内部打印
            rows = self._model.transcribe(utterance_ids, [str(path) for path in audio_paths])
        # 返回值是一行一个 dict，用 uttid 对齐，防止模型内部乱序
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(audio_paths):  # 数量对不上 = 有音频转写失败
            raise RuntimeError("FireRed returned an unexpected result count")
        return [normalize_transcript(str(mapped[uid].get("text", ""))) for uid in utterance_ids]


def load_audio_mono_16k(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """与 sv_asr/train_asr.py load_audio 一致：单声道 float32，必要时重采样。

    训练端与推理端的音频读取必须口径一致（同样的归一化、同样的重采样方式），
    否则模型听到的声音和训练时不一样，效果会莫名变差。
    """
    import soundfile as sf  # lazy import：通用音频库，支持 wav/flac/ogg 等

    waveform, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:  # 双声道 → 取两声道平均转单声道
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate != sample_rate:  # 采样率不匹配 → 用 torchaudio 重采样
        import torch
        import torchaudio

        tensor = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(tensor, source_rate, sample_rate).numpy()
    return waveform.astype(np.float32, copy=False)  # copy=False：能原地转就不复制，省内存


class ConditionedQwenRecognizer:
    """sv_asr 双音频条件 LoRA 后端：输入 [kws(目标人参考), cmd] → 目标人文本。

    基座 Qwen3-ASR-0.6B + LoRA adapter（text decoder LoRA + audio projector
    全量，sv_asr/train_asr.py 训练，只学转写不学拒识，拒识仍由门控负责）。
    prompt / 解码 / max_new_tokens 与 train_asr.py 的 dev 评估完全一致。
    依赖：peft（vendor 目录）+ midea/src3 的 qwen_loader（只读复用）。

    注意：这是实验分支后端（asr_backend="qwen_lora"），SZY v1 冻结路线不使用。
    """

    conditioned = True  # 标志位：infer.main 的 flush_asr 靠它判断该用"成对转写"接口

    def __init__(
        self,
        model: str,
        adapter: str | Path,
        device: str,
        model_root: str | Path | None,
        vendor_root: str | Path = "/home/hustdun/szy_workspace/vendor",   # 队长工作区路径（实验分支）
        src3_root: str | Path = "/home/hustdun/szy_workspace/midea/src3",  # 队长工作区路径（实验分支）
    ) -> None:
        import sys

        import torch

        # 把 vendor（peft 等）和 midea/src3（qwen_loader）加入模块搜索路径
        for path in (str(vendor_root), str(src3_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from peft import PeftModel  # HuggingFace 的参数高效微调库
        from qwen_loader import load_base_model, load_processor

        model_path = resolve_model_path(model, model_root)
        self.processor = load_processor(model_path)  # 音频/文本预处理（与训练端同款）
        self.device = "cuda:0" if device == "cuda" else device
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._torch = torch
        # 加载基座模型（音频编码器 + LLM），整模型放到指定设备
        self.model = load_base_model(
            model_path, dtype=self.dtype, device_map={"": self.device}
        )
        # 把 LoRA 适配器挂到"思考器"（LLM 部分）上；adapters 很小（几十 MB），
        # 相当于给大模型打补丁，不用重新训练全部参数
        self.model.thinker = PeftModel.from_pretrained(
            self.model.thinker, str(Path(adapter).resolve())
        )
        self.model.eval()  # 评估模式（关闭 Dropout 等训练行为）
        self.model.thinker.config.use_cache = True  # 开 KV 缓存加速自回归解码
        self.max_new_tokens = 128

    def _build_prompt(self) -> str:
        """与 sv_asr/train_asr.py build_prompt 逐字一致。

        构造一个"双音频"提示词：告诉模型"第一段是目标人参考音，
        只转写第二段里目标人说的话"。prompt 必须与训练时完全一致，
        否则模型行为会漂移。
        """
        messages = [
            {
                "role": "system",  # 系统角色：设定任务
                "content": (
                    "你是目标说话人语音识别器。第一段音频是目标说话人的参考语音。"
                    "第二段音频中可能有多个人同时说话和背景噪声，"
                    "请只转写第二段音频中目标说话人"
                    "（与第一段参考语音是同一人）所说的内容，"
                    "忽略其他说话人。不要输出解释。"
                ),
            },
            {
                "role": "user",  # 用户角色：提供两个音频槽位
                "content": [
                    {"type": "audio", "audio": ""},  # 槽位 1：参考音频（后面用真实波形填充）
                    {"type": "audio", "audio": ""},  # 槽位 2：待识别音频
                ],
            },
        ]
        # apply_chat_template：把对话消息套上模型的聊天模板（特殊 token、格式）
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return prompt + "language Chinese<asr_text>"  # 追加语言与输出标记（训练时同款）

    def transcribe_pairs(self, pairs: list[tuple[Path, Path]]) -> list[str]:
        """逐对转写 [(唤醒音频, 识别音频), ...] → 目标人文本列表。"""
        torch = self._torch
        outputs: list[str] = []
        for wake_path, command_path in pairs:  # 逐对处理（双音频条件模型没法简单批量化）
            enroll_audio = load_audio_mono_16k(wake_path)      # 参考音频 → 波形
            command_audio = load_audio_mono_16k(command_path)  # 识别音频 → 波形
            # processor 把文本 prompt + 两段音频打包成模型输入张量
            inputs = dict(
                self.processor(
                    text=[self._build_prompt()],
                    audio=[enroll_audio, command_audio],
                    return_tensors="pt",  # 返回 PyTorch 张量
                    padding=True,         # 自动补齐不同长度
                )
            )
            # 把浮点张量搬到设备并转精度；非浮点（如 input_ids 整数）只搬设备；
            # 不是张量的（如 attention_mask 之外的普通对象）原样保留
            inputs = {
                key: (
                    value.to(device=self.device, dtype=self.dtype)
                    if hasattr(value, "is_floating_point") and value.is_floating_point()
                    else value.to(device=self.device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode():  # 无梯度推理
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )
            # 截掉输入部分，只留下新生成的 token，再解码回文本
            sequence = generated.sequences[0, inputs["input_ids"].shape[1] :]
            text = self.processor.decode(
                sequence,
                skip_special_tokens=True,          # 去掉 <|endoftext|> 等特殊 token
                clean_up_tokenization_spaces=False, # 保留原始空格（归一化交给 normalize_transcript）
            ).strip()
            outputs.append(normalize_transcript(text))
        return outputs


def read_many_jsonl(paths: Iterable[str | Path], data_root: str | Path) -> list[Sample]:
    """读取竞赛 JSONL 清单（pos.jsonl / neg.jsonl），解析成 Sample 列表。

    JSONL = JSON Lines：每行一个独立 JSON 对象（不是一个大 JSON 数组），
    适合大数据流式处理——一行一行读，内存占用小，断点续读也容易。

    一行示例（pos.jsonl）:
      {"id": 0, "唤醒音频": "pos/kws_0.wav", "唤醒文本": "你好科慕",
       "识别音频": "pos/cmd_0.wav", "识别文本": "空调开到制热调到二十五度风量调到百分之三十"}
    负样本的 "识别文本" 是 null（表示"目标人没有说话"，无标签）。
    """
    root = Path(data_root)  # 数据根目录：JSONL 里的相对路径都相对于它拼接
    samples: list[Sample] = []
    for jsonl_path in paths:  # 支持一次传多个清单文件（pos + neg）
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:  # 必须显式 utf-8（含中文）
            # enumerate(handle, start=1)：边读行边编号（从 1 开始），
            # 出错时能把"第几行"报给用户，方便定位坏数据
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():  # 跳过空行（strip 去掉首尾空白后为空 = 空行）
                    continue
                row = json.loads(line)  # 这一行文本 → dict
                try:
                    # 取字段；str() 保证类型（JSON 里的数字 id 也会被转成字符串存进 Sample.raw_id）
                    wake_relative_path = str(row["唤醒音频"])
                    command_relative_path = str(row["识别音频"])
                except KeyError as error:
                    # 缺字段 → 报错并带上"哪个文件的第几行"，而不是抛晦涩的裸 KeyError
                    raise KeyError(f"{jsonl_path}:{line_number} missing field {error}") from error
                samples.append(
                    Sample(
                        raw_id=str(row.get("id", "")),  # .get 带默认值：id 缺失也不崩
                        wake_audio=root / wake_relative_path,   # Path 的 / 运算 = 拼路径
                        command_audio=root / command_relative_path,
                        command_relative_path=command_relative_path,
                        label=row.get("识别文本"),  # 负样本这里是 None
                        wake_text=row.get("唤醒文本") or row.get("wake_text"),  # 兼容两种字段名
                    )
                )
    return samples


def submission_id(sample: Sample, mode: str) -> str:
    """生成样本的"稳定提交 ID"。

    为什么需要它？——JSONL 里的原始 id 在 pos/neg 两个文件里会重复（都从 0 编起），
    直接用会互相覆盖；而且断点续跑时需要一个"无论怎么处理都不变"的键来区分
    "这条样本处理过没有"。从路径派生 ID 是最稳妥的做法。

    mode 选项：
      raw           → 用 JSONL 的原始 id（有重复风险）
      name          → 只用文件名（cmd_13.wav）
      path          → 完整路径
      relative      → 相对路径（pos/cmd_13.wav）
      relative_stem → 相对路径去掉扩展名（pos/cmd_13）★ SZY v1 冻结使用这个
      stem          → 文件名去掉扩展名（cmd_13）
    """
    if mode == "raw":
        return sample.raw_id
    if mode == "name":
        return sample.command_audio.name  # Path.name = 最后一段（含扩展名）
    if mode == "path":
        return sample.command_audio.as_posix()  # 路径转成正斜杠字符串（跨平台统一）
    if mode == "relative":
        return sample.command_relative_path
    if mode == "relative_stem":
        path = Path(sample.command_relative_path)
        # path.parent = 目录部分；path.stem = 文件名去掉最后一个扩展名
        return (path.parent / path.stem).as_posix()
    if mode == "stem":
        return sample.command_audio.stem
    raise ValueError(f"unsupported id mode: {mode}")  # 非法模式直接报错（fail-fast）


def extract_speaker_score(result: Any) -> float:
    """从声纹 pipeline 的返回值里"挖"出相似度分数。

    不同版本的 ModelScope pipeline 返回格式五花八门（dict 套 dict、list、
    numpy 数组、纯数字……），这个函数用递归统一处理：一层层剥，直到剥出数字。
    这种"容忍多种格式"的工具函数叫适配器（adapter）。
    """
    if isinstance(result, dict):  # 是字典 → 按常见键名逐个试
        for key in ("score", "scores", "similarity", "output", "outputs"):
            if key in result:
                try:
                    return extract_speaker_score(result[key])  # 递归：剥开一层继续挖
                except ValueError:
                    continue  # 这个键里没有数字，换下一个键
        raise ValueError(f"speaker verification output has no score: {list(result)}")
    if isinstance(result, (list, tuple)):  # 是列表/元组 → 逐个元素试
        for value in result:
            try:
                return extract_speaker_score(value)
            except ValueError:
                continue
        raise ValueError("speaker verification output was empty")
    if isinstance(result, np.ndarray):  # 是 numpy 数组 → 摊平取第一个
        if result.size == 0:
            raise ValueError("speaker verification score array was empty")
        return float(result.reshape(-1)[0])
    try:
        return float(result)  # 是数字（或能转成数字的字符串）→ 直接返回
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot parse speaker score from {result!r}") from error


def extract_asr_text(result: Any) -> str:
    """从 ASR 返回结果里"挖"出文本（同 extract_speaker_score 的适配器思路）。

    Qwen3-ASR 的官方包返回对象带 .text 属性；FireRed 返回 dict；也可能有
    其他封装返回 str/list。这里全部兼容。
    """
    text = getattr(result, "text", None)  # 有 .text 属性（对象）→ 直接拿
    if text is not None:
        return str(text)
    if isinstance(result, str):  # 本身是字符串 → 直接返回
        return result
    if isinstance(result, dict):  # 字典 → 按常见键名找，找不到就递归翻值
        for key in ("text", "sentence", "raw_text"):
            if key in result:
                return str(result[key])
        for value in result.values():
            try:
                return extract_asr_text(value)
            except ValueError:
                continue
    if isinstance(result, list):  # 列表 → 把每个元素的文本拼起来（有的模型分句输出）
        parts = []
        for value in result:
            try:
                parts.append(extract_asr_text(value))
            except ValueError:
                continue
        if parts:
            return "".join(parts)
    raise ValueError(f"cannot extract ASR text from {result!r}")


def normalize_transcript(text: str) -> str:
    """文本归一化：把 ASR 输出和标签统一成同一个"口径"，CER 才有可比性。

    做四件事（顺序很重要）：
      1. 删除特殊 token（<|...|> 形如 <|zh|>、<|endoftext|>）
      2. "30%" → "百分之三十"（先抓百分比再转中文数字）
      3. 阿拉伯数字 → 中文数字（25 → 二十五）
      4. 删除所有空白与标点（中文的，。、英文的 ,.!? 等）

    例：'空调开到25度，风量30%' → '空调开到二十五度风量百分之三十'
    """
    # 中文数字表：下标即数字（digits[0]='零' ... digits[9]='九'），字符串可以按下标取值
    digits = "零一二三四五六七八九"

    def number_to_chinese(value: int) -> str:
        """整数 → 中文读法（0~999 精确转换，>=1000 放弃直接用阿拉伯数字）。"""
        if value < 10:
            return digits[value]  # 0~9：直接查表
        if value < 20:
            # 10~19：'十' + 个位（10='十' 而不是 '一十'；15='十五'）
            return "十" + (digits[value % 10] if value % 10 else "")
        if value < 100:
            # 20~99：十位 + '十' + 个位（25 → 二十五；30 → 三十）
            return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
        if value < 1000:
            # 100~999：百位 + '百' + 余数，余数中间要补'零'（105 → 一百零五）
            remainder = value % 100
            prefix = digits[value // 100] + "百"
            if remainder == 0:  # 整百（300 → 三百）
                return prefix
            if remainder < 10:  # 中间有零（305 → 三百零五）
                return prefix + "零" + digits[remainder]
            return prefix + number_to_chinese(remainder)  # 其余递归转换（315 → 三百一十五）
        return str(value)  # >=1000 放弃（标签里基本不会出现）

    def convert_number(match: re.Match[str]) -> str:
        """re.sub 的替换回调：每匹配到一串数字就调用一次。"""
        return number_to_chinese(int(match.group(0)))  # group(0) = 整个匹配到的数字串

    normalized = re.sub(r"<\|[^|]+?\|>", "", str(text))  # 删 <|任意内容|>（特殊 token）
    # 百分比两步走：'30%' → '百分之30' → '百分之三十'
    # \d+ 匹配数字串；\s* 允许'30 %'这种带空格写法
    normalized = re.sub(r"(\d+)\s*%", r"百分之\1", normalized)
    # 把'百分之30'里的 30 换成中文：lambda 回调里调 number_to_chinese
    normalized = re.sub(r"百分之\s*(\d+)", lambda match: "百分之" + number_to_chinese(int(match.group(1))), normalized)
    # 所有剩余阿拉伯数字 → 中文
    normalized = re.sub(r"\d+", convert_number, normalized)
    # 删空白与标点：字符类 [\s,，。.!！?？:：;；、】【、\"'“”‘’（）()\[\]<>《》|/\\_-]+
    #   \s = 空白；其余是常见中英文标点；+ 表示连续的一串一起删；lower() 顺手转小写
    return re.sub(r"[\s,，。.!！?？:：;；、】【、\"'“”‘’（）()\[\]<>《》|/\\_-]+", "", normalized.lower())


def levenshtein(left: str, right: str) -> int:
    """编辑距离（Levenshtein 距离）：把 left 变成 right 最少要几步编辑？

    允许的编辑操作：插入一个字符、删除一个字符、替换一个字符。
    例：'打开' → '开灯' 需要 1 步（'打' 替换成 '灯'）。
    这是 CER（字符错误率）的分子，也是唤醒锚点选择的核心度量。

    算法：动态规划（Wagner-Fischer）。
      dp[i][j] = left 前 i 个字符 变成 right 前 j 个字符 的最小编辑数。
      递推：dp[i][j] = min(
               dp[i-1][j] + 1,          # 删掉 left 的第 i 个字符
               dp[i][j-1] + 1,          # 插入 right 的第 j 个字符
               dp[i-1][j-1] + (不同?)   # 替换（字符相同则代价 0）
            )
    空间优化：不需要存整张 dp 表，只保留"上一行"和"当前行"两个列表即可
    （current 依赖 previous 和 current[-1]，见课程第 4 课的示例表）。
    """
    if left == right:  # 完全相同 → 0 步（快速通道，最常见的情况）
        return 0
    if not left:  # 一个为空 → 全部靠插入，步数 = 另一个的长度
        return len(right)
    if not right:
        return len(left)
    # previous = dp 表的第 0 行：空串 → right 的前 j 个字符 = j 次插入
    previous = list(range(len(right) + 1))
    # enumerate(left, start=1)：i 从 1 开始（第 0 行已初始化）
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]  # 每行第 0 列 = i 次删除
        for right_index, right_character in enumerate(right, start=1):
            # 三个候选取最小：
            #   current[-1] + 1           = dp[i][j-1] + 1（插入）
            #   previous[right_index] + 1 = dp[i-1][j] + 1（删除）
            #   previous[right_index-1] + (字符不同 ? 1 : 0)（替换/相同）
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_character != right_character)))
        previous = current  # 滚动：当前行变成下一轮的"上一行"
    return previous[-1]  # 最后一行最后一列 = 完整两串的编辑距离


def cer(hypothesis: str, label: str | None) -> float | None:
    """单条样本的 CER（字符错误率）：编辑距离 / 参考文本长度。负样本返回 None。"""
    if label is None:
        return None
    # 标签为空串的退化情况：输出非空算 1.0（完全错），输出空算 0.0（float(False)=0.0）
    return levenshtein(hypothesis, label) / len(label) if label else float(hypothesis != "")


@contextlib.contextmanager
def suppress_model_output():
    """上下文管理器：把进入这段代码时的 stdout/stderr 全部丢进"黑洞"。

    模型库（ModelScope、FireRed 等）加载时喜欢狂打印日志，会淹没我们自己的
    输出。用法：
        with suppress_model_output():
            noisy_call()
    实现：把标准输出/错误重定向到 os.devnull（操作系统的"黑洞设备"），
    离开 with 块时自动恢复（contextmanager 装饰器帮你管理进出两端的动作）。
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield  # 执行到这里暂停，把控制权交给 with 块里的代码


def write_reports(results: list[dict[str, Any]], duration: float, output: Path, final_output: Path) -> dict[str, float]:
    """汇总逐条结果 → 计算 CER/PR/RR → 写出 submission 与 detail 两个 JSON。

    指标口径（与竞赛官方一致）：
      CER = 所有正样本的编辑距离之和 / 所有正样本参考字符数之和
      PR  = 正样本中输出非空的比例（Positive Recall，召回率）
      RR  = 负样本中输出为空的比例（Reject Rate，拒识率）
    """
    total_errors = 0        # 分子：编辑距离累加
    total_characters = 0    # 分母：参考字符数累加
    positive_nonempty = 0   # 正样本输出非空的条数
    negative_total = 0      # 负样本总数
    negative_rejects = 0    # 负样本被正确拒识（输出空）的条数
    submission_results = []  # 提交文件的内容（id/content/label/cer）
    final_results = []       # 明细文件的内容（额外带 pr/rr/声纹分等审计字段）
    for item in results:
        content = str(item["content"] or "")  # or ""：None 也当空串
        label = item["label"]
        positive = label is not None and label != ""  # 判定正负样本：标签存在且非空
        item_cer = cer(content, label) if positive else None
        if positive:
            total_errors += levenshtein(content, str(label))
            total_characters += len(str(label))
            positive_nonempty += int(content != "")  # bool 转 int：True=1
        else:
            negative_total += 1
            negative_rejects += int(content == "")
        submission_results.append({"id": item["id"], "content": content, "label": label, "cer": "" if item_cer is None else f"{item_cer:.6f}"})
        final_item = {
            "id": item["id"],
            "content": content,
            "label": label,
            "sample_type": "positive" if positive else "negative",
            "cer": item_cer,
            "pr": float(content != "") if positive else None,   # 该条是否被"召回"
            "rr": None if positive else float(content == ""),   # 该条是否被正确拒识
        }
        # 把逐条结果里已有的审计字段原样带进明细（sv_score/accepted/路由信息等）
        for key in (
            "sv_score",
            "accepted",
            "kept_segments",
            "wake_localized",
            "wake_candidate_count",
            "wake_transcript",
            "wake_text_score",
            "wake_selected_source",
            "wake_text_confirmed",
            "sv_accept_threshold",
            "item_duration",
            "second_stage_enabled",
            "second_stage_probability",
            "second_stage_accepted",
            "second_stage_reason",
        ):
            if key in item:
                final_item[key] = item[key]
        final_results.append(final_item)
    summary = {
        "final_cer": total_errors / total_characters if total_characters else 0.0,
        "final_pr": positive_nonempty / (len(results) - negative_total) if len(results) > negative_total else 0.0,
        "final_rr": negative_rejects / negative_total if negative_total else 0.0,
        "positive_count": len(results) - negative_total,
        "positive_nonempty": positive_nonempty,
        "negative_count": negative_total,
        "negative_rejects": negative_rejects,
        "duration": duration,
    }
    # mkdir(parents=True, exist_ok=True)：目录不存在就建（含多级父目录），存在也不报错
    output.parent.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False：JSON 里保留中文原文（否则会变成 \uXXXX 转义，人没法读）
    # indent=2：缩进 2 空格，方便人工审计
    output.write_text(json.dumps({"result": {"results": submission_results, "final_cer": f"{summary['final_cer']:.6f}", "duration": f"{duration:.6f}"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    final_output.write_text(json.dumps({"summary": summary, "results": final_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """旧基线（infer.main）的命令行参数定义。"""
    parser = argparse.ArgumentParser(description="Standalone ERes2NetV2 plus Qwen3-ASR DatasetA baseline")
    parser.add_argument("--data-root", required=True)            # 数据根目录
    parser.add_argument("--jsonl", nargs="+", required=True)     # 清单文件（可多个）
    parser.add_argument("--config", default="configs/eres2netv2_qwen3_raw_command.json")  # 旧基线配置
    parser.add_argument("--model-root", help="ModelScope cache root containing iic/ and Qwen/")
    parser.add_argument("--output", default="outputs/submission.json")
    parser.add_argument("--final-output", default="outputs/detail.json")
    parser.add_argument("--limit", type=int)                     # 只跑前 N 条（冒烟测试）
    parser.add_argument("--asr-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-output")                   # 断点文件路径
    parser.add_argument("--resume", action="store_true")         # 是否续跑
    return parser


def main() -> None:
    """旧基线入口：单声纹门控（或双 MLP 门控）→ ASR 批量转写 → 写报告。

    注意：SZY v1 的正式入口是 szy_v1/pipeline.py 的 main()，不是这里。
    保留本入口是因为它还承载着 asr_backend/gate_mode/second_stage 等
    实验开关的"插座"功能（课程第 3 课会讲两入口的关系）。
    """
    args = build_parser().parse_args()
    if args.asr_batch_size < 1:  # 参数合法性守卫
        raise ValueError("--asr-batch-size must be positive")
    config = load_config(args.config)
    samples = read_many_jsonl(args.jsonl, args.data_root)
    if args.limit is not None:
        samples = samples[: args.limit]  # 切片：只取前 limit 条
    # ---- 断点续跑：读旧 checkpoint，跳过已完成的样本 ----
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else None
    results: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires --checkpoint-output from a prior run")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = checkpoint["results"]
        elapsed_before = float(checkpoint["duration"])
        completed = {item["id"] for item in results}  # 集合推导式：已完成 ID 的集合
        samples = [sample for sample in samples if submission_id(sample, config.id_mode) not in completed]
    # ---- 组装门控 ----
    gate = None
    verifier = None
    second_stage = None
    if config.second_stage_enabled:  # 二级门控：需要额外配置齐全，缺一不可
        if config.gate_mode != "dual_mlp":
            raise ValueError("second_stage_enabled requires gate_mode=dual_mlp")
        if not config.second_stage_model or not config.second_stage_features:
            raise ValueError(
                "second_stage_enabled requires second_stage_model and second_stage_features"
            )
        from midea_qiji.moss_second_stage import CachedPresenceStage  # 实验模块（本仓库未随附）

        second_stage = CachedPresenceStage(
            config.second_stage_model,
            config.second_stage_features,
            threshold=config.second_stage_presence_threshold,
        )
    if config.gate_mode == "dual_mlp":  # 双 MLP 门控（SZY v1 同款，但阈值配置不同）
        from midea_qiji.gate import DualMLPGate

        if config.dual_backend is None:
            raise ValueError("gate_mode=dual_mlp requires dual_backend in config")
        print(f"Loading dual MLP gate (ERes2NetV2 + CAM++) on {config.sv_device or config.device}...", flush=True)
        gate = DualMLPGate(
            config.sv_model, config.cam_model, config.dual_backend,
            config.dual_threshold, config.sv_device or config.device, args.model_root,
        )
        if second_stage is not None and second_stage.primary_threshold is not None:
            # 级联阈值必须成对使用：不能拿 A 实验的二级门控配 B 实验的主门控阈值
            if not np.isclose(config.dual_threshold, second_stage.primary_threshold, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "dual_threshold must equal the primary_threshold stored in the "
                    "second-stage artifact; jointly calibrated cascade thresholds "
                    "cannot be mixed with another primary threshold"
                )
    else:  # 旧基线：单 ERes2NetV2 余弦阈值门控
        print(f"Loading ERes2NetV2 on {config.sv_device or config.device}...", flush=True)
        verifier = SpeakerVerifier(config.sv_model, config.sv_device or config.device, args.model_root)
    # ---- 组装 ASR ----
    if config.asr_backend == "firered":
        print(f"Loading FireRedASR2-AED on {config.device}...", flush=True)
        recognizer = FireRedRecognizer(config.firered_model, config.device, config.firered_repo)
    elif config.asr_backend == "qwen_lora":
        if config.qwen_lora_adapter is None:
            raise ValueError("asr_backend=qwen_lora requires qwen_lora_adapter in config")
        print(f"Loading Qwen3-ASR + LoRA adapter ({config.qwen_lora_adapter}) on {config.device}...", flush=True)
        recognizer = ConditionedQwenRecognizer(config.asr_model, config.qwen_lora_adapter, config.device, args.model_root)
    else:
        print(f"Loading Qwen3-ASR on {config.device}...", flush=True)
        recognizer = QwenRecognizer(config.asr_model, config.device, args.model_root, args.asr_batch_size)
    # ---- 主循环：先批量做门控，再把"通过"的样本攒批交给 ASR ----
    started = time.perf_counter()  # 高精度计时（比 time.time 稳，不受系统改时间影响）
    pending_pairs: list[tuple[Path, Path]] = []  # 等待 ASR 的（唤醒, 命令）音频对
    pending_indices: list[int] = []              # 对应的 results 下标

    def save_checkpoint() -> None:
        """把当前进度写盘（嵌套函数：能直接访问外层 main 的变量）。"""
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps({"duration": elapsed_before + time.perf_counter() - started, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    def flush_asr() -> None:
        """把攒的批次一次喂给 ASR，回填结果并保存断点。"""
        if not pending_pairs:  # 没有待处理的 → 直接返回
            return
        if getattr(recognizer, "conditioned", False):  # 双音频条件模型走成对接口
            transcripts = recognizer.transcribe_pairs(pending_pairs)
        else:
            transcripts = recognizer.transcribe_many([command for _, command in pending_pairs])
        if len(transcripts) != len(pending_indices):  # 数量守卫
            raise RuntimeError("ASR returned an unexpected transcript count")
        completed_at = time.perf_counter()
        for index, transcript in zip(pending_indices, transcripts):  # zip 并行遍历两个列表
            results[index]["content"] = transcript
            if second_stage is not None:  # 二级门控：可能把内容清空（拒识）
                decision = second_stage.decide(results[index]["id"])
                results[index].update(
                    {
                        "second_stage_enabled": True,
                        "second_stage_probability": (
                            None if decision.probability is None else f"{decision.probability:.6f}"
                        ),
                        "second_stage_accepted": decision.accepted,
                        "second_stage_reason": decision.reason,
                    }
                )
                if not decision.accepted:
                    results[index]["content"] = ""
            # 记录耗时：results[index].pop('_started') 取出并删除该条的开始时间戳
            results[index]["item_duration"] = f"{completed_at - results[index].pop('_started'):.6f}"
        pending_pairs.clear()  # 清空批次缓存
        pending_indices.clear()
        save_checkpoint()

    for sample in tqdm(samples, desc="infer", unit="utt"):  # 带进度条的主循环
        item_started = time.perf_counter()
        item_id = submission_id(sample, config.id_mode)
        # 音频文件必须真实存在，否则后面的模型会在奇怪的地方崩掉
        if not sample.wake_audio.is_file() or not sample.command_audio.is_file():
            raise FileNotFoundError(f"missing audio for {item_id}")
        if gate is not None:  # 双 MLP 门控路径
            accepted, speaker_score = gate.accept(sample.wake_audio, sample.command_audio)
            active_threshold = config.dual_threshold
        else:  # 旧基线余弦路径
            speaker_score = verifier.score(sample.wake_audio, sample.command_audio)
            accepted = speaker_score >= config.sv_accept_threshold
            active_threshold = config.sv_accept_threshold
        # 每条样本先登记一个"骨架"结果（审计字段齐全），后面再回填 content
        item = {"id": item_id, "content": "", "label": sample.label, "sv_score": f"{speaker_score:.6f}", "accepted": accepted, "kept_segments": 0, "wake_localized": False, "wake_candidate_count": 0, "wake_transcript": "", "wake_text_score": "0.000000", "wake_selected_source": None, "wake_text_confirmed": True, "sv_accept_threshold": f"{active_threshold:.6f}", "item_duration": "", "second_stage_enabled": bool(second_stage is not None), "second_stage_probability": None, "second_stage_accepted": None, "second_stage_reason": "not_run_primary_reject" if second_stage is not None and not accepted else None}
        results.append(item)
        if accepted:  # 门控通过 → 攒进批次，等满 batch_size 一起转写
            item["_started"] = item_started  # 下划线开头 = 临时内部字段（写盘前会删掉）
            pending_pairs.append((sample.wake_audio, sample.command_audio))
            pending_indices.append(len(results) - 1)  # 记住这条在 results 里的位置
            if len(pending_pairs) >= args.asr_batch_size:
                flush_asr()
        else:  # 门控拒绝 → 直接空文本，连 ASR 都不跑（省算力、防止误输出）
            item["item_duration"] = f"{time.perf_counter() - item_started:.6f}"
            if not pending_pairs:
                save_checkpoint()
    flush_asr()  # 尾巴：最后不足一批的也要处理
    for item in results:
        item.pop("_started", None)  # 清理临时字段（pop 带默认值，没有也不报错）
    duration = elapsed_before + time.perf_counter() - started
    summary = write_reports(results, duration, Path(args.output), Path(args.final_output))
    print(f"samples={len(results)} final_cer={summary['final_cer']:.6f} final_pr={summary['final_pr']:.6f} final_rr={summary['final_rr']:.6f} duration={duration:.3f}s", flush=True)


if __name__ == "__main__":
    main()  # python -m midea_qiji.infer 或直接运行本文件时的入口