"""core.infer：DatasetA 基线（ERes2NetV2 门控 + Qwen3-ASR）的编排与 CLI。

本模块是 core 内核的"主程序"：负责配置加载、JSONL 样本读取、断点
续跑、识别后端/门控装配、批量 ASR 调度、指标汇总与提交物写出。算法部件
已按职责拆分到同包子模块：

  - ``text.py``         文本归一化 / 编辑距离 / CER；
  - ``audio.py``        音频常量与 WAV 读取；
  - ``paths.py``        模型路径常量与本地解析；
  - ``recognizers.py``  Qwen / FireRed AED / FireRed LLM / 条件 LoRA 后端；
  - ``gate.py``         双前端 MLP 门控与声纹 embedding。

为兼容既有代码与文档，本模块 re-export 了 text / paths 的公共名（如
``from core.samples import levenshtein`` 仍可用）；新代码请直接从
``core.text`` / ``core.paths`` 导入。

re-export 说明：重构前这些函数就活在 infer.py 里，历史脚本/文档按
core.infer.<name> 引用它们；重构后函数搬家了，re-export 让旧路径
继续可用（兼容性护栏），# noqa: F401 则是告诉 linter"这是有意为之"。
这也解释了为什么 recognizers.py 能从 .infer 顶部 import——infer 对
recognizers/gate 的 import 全部放在 main() 内部（按需装配），不会成环。

用法（baseline 路线，SZY 系列请走 szy 系统入口（python -m pipeline.cli））：
  PYTHONPATH=src python -m core.infer \\
    --data-root dataset/dataset_test/datasetA/datasetA \\
    --jsonl dataset/.../pos.jsonl --config configs/xxx.json

关键导出：
  - Config / Sample             配置与样本的数据类（@dataclass）；
  - load_config / read_many_jsonl / submission_id   配置与数据入口；
  - SpeakerVerifier             baseline 单模型余弦门控；
  - write_reports               指标汇总与双份产物写出；
  - suppress_model_output       模型噪音静默的上下文管理器；
  - main / build_parser         CLI 编排（模块作为脚本运行时的入口）。

依赖：numpy / tqdm；torch / modelscope / FireRed 等重型依赖延迟到使用时。
（延迟导入的收益：模块顶层 import 不碰它们，--help 与 numpy-only 单测都能跑。）
"""
from __future__ import annotations

import argparse       # 命令行参数解析（build_parser）
import contextlib     # redirect_stdout/stderr 等上下文工具（suppress_model_output）
from dataclasses import dataclass
import json           # 配置与 JSONL 读取、产物写出
import os             # os.devnull：平台无关的空设备路径
from pathlib import Path
import time           # perf_counter：高精度单调计时
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm   # 进度条（auto 版自动选择 notebook/终端渲染）

# ---- re-export（兼容旧导入路径 core.infer.<name>） ----
from core.model_paths import (  # noqa: F401
    DEFAULT_FIRED_MODEL,
    DEFAULT_FIRED_REPO,
    DEFAULT_FIRED_VENDOR,
    PACKAGE_MODEL_ROOT,
    resolve_model_path,
)
from core.text import cer, extract_asr_text, levenshtein, normalize_transcript  # noqa: F401


@dataclass(frozen=True)
class Config:
    """运行配置的数据类：JSON 配置文件 -> Config 对象（load_config）。

    @dataclass 替我们生成 __init__（按字段顺序收参），省掉一堆样板赋值；
    frozen=True 让实例不可变：配置一旦加载就不能被中途篡改，避免"跑着
    跑着配置被人改了"这类隐蔽 bug（改配置必须显式新建对象）。
    每个字段的默认值即"无配置文件时的历史 baseline 口径"，复现结果时
    不要动它们；需要改值时在 JSON 里覆盖对应字段即可。
    """

    device: str = "cuda:0"                  # ASR 推理设备
    sv_device: str | None = None            # 声纹门控设备（None = 跟随 device）
    asr_model: str = "Qwen/Qwen3-ASR-0.6B"  # ASR 基座模型（ModelScope id）
    sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"  # 声纹模型
    sv_model_revision: str | None = "v1.0.0"  # 声纹模型的 ModelScope 版本号
    sv_accept_threshold: float = 0.27       # baseline 余弦门控的接受阈值（历史调参值）
    id_mode: str = "relative_stem"          # 提交文件 id 的生成方式（见 submission_id）
    # 门控模式："cos" = ERes2NetV2 余弦阈值（baseline）；"dual_mlp" = 双前端 MLP 裁判
    gate_mode: str = "cos"
    cam_model: str = "iic/speech_campplus_sv_zh-cn_16k-common"  # dual_mlp 的辅助前端
    dual_backend: str | None = None         # dual_mlp 的 MLP 权重路径（backend_dual.pt）
    dual_threshold: float = 0.5334          # dual_mlp 的接受阈值（历史调参值）
    # ASR 后端："qwen" = Qwen3-ASR-0.6B；"firered" = FireRedASR2-AED；
    # "qwen_lora" = sv_asr 双音频条件 LoRA 微调权重（[kws, cmd] → 目标人文本）
    asr_backend: str = "qwen"
    firered_model: str = DEFAULT_FIRED_MODEL   # FireRed 模型目录
    firered_repo: str = DEFAULT_FIRED_REPO     # FireRed 源码仓库目录
    qwen_lora_adapter: str | None = None       # 条件 LoRA adapter 目录（qwen_lora 后端用）


@dataclass(frozen=True)
class Sample:
    """一条评测样本：JSONL 一行 -> Sample（read_many_jsonl 的解析产物）。

    wake_audio 是目标说话人的注册/唤醒音频；command_audio 是待判定并
    转写的指令音频；label 为 None 表示负样本（指令里没有目标人，期望
    输出为空=拒识）；wake_text 留给需要转写唤醒词的任务（可选字段）。
    """

    raw_id: str                    # JSONL 里的原始 id（可能为空字符串）
    wake_audio: Path               # 唤醒音频绝对路径（data_root 拼接后）
    command_audio: Path            # 指令音频绝对路径
    command_relative_path: str     # 指令音频相对路径（提交文件 id 的来源）
    label: str | None              # 期望转写文本；None = 负样本
    wake_text: str | None = None   # 唤醒词文本（可选，兼容两种 JSONL 字段名）


def load_config(path: str | Path) -> Config:
    """读 JSON 配置文件并构造 Config；出现未知字段直接报错。

    严格校验的用意：JSON 里拼错一个键（如 asr_backend 写成 asr_backendd）
    若不报错就会被静默忽略、跑出默认行为——这种 bug 极难排查，宁可
    启动时就失败。`Config(**payload)` 用 dict 展开传参构造数据类。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # __dataclass_fields__ 是 @dataclass 自动维护的"字段名 -> 定义"字典，
    # 用它当白名单与 JSON 键比对。
    fields = Config.__dataclass_fields__
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return Config(**payload)


class SpeakerVerifier:
    """ERes2NetV2 余弦门控（baseline 路线用；SZY 系列改用 DualMLPGate）。

    单模型 + 人工阈值：pipeline 对 (注册音频, 指令音频) 输出相似度分数，
    主循环里与 sv_accept_threshold（默认 0.27）比较决定放行与否。
    DualMLPGate 出现后它仍是默认门控（gate_mode="cos"），保持 baseline
    结果可复现。
    """

    def __init__(self, model: str, device: str, model_root: str | Path | None) -> None:
        # 延迟导入：modelscope 只在真正构造门控时才加载（--help/单测不需要它）。
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        # pipeline 一次性封装"加载模型 + 预处理 + 推理 + 打分"全流程
        # （模式说明见 gate.ModelScopeEmbedder）。speaker_verification
        # 任务即"判断两段音频是否同一说话人"。
        self.pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=str(resolve_model_path(model, model_root)),
            device=device,
        )

    def score(self, enrollment_audio: Path, command_audio: Path) -> float:
        # pipeline 接收一对音频路径，返回相似度结果（结构随版本变化），
        # extract_speaker_score 负责从各种可能的结构里挖出标量分数。
        return extract_speaker_score(self.pipeline([str(enrollment_audio), str(command_audio)]))


def read_many_jsonl(paths: Iterable[str | Path], data_root: str | Path) -> list[Sample]:
    """把多个 JSONL 清单文件解析成 Sample 列表。

    JSONL = 每行一个 JSON 对象的文本格式，适合大规模样本流式读取
    （不必像单个大 JSON 那样一次性全部载入内存）。这里按行流式解析，
    边读边建 Sample。
    """
    root = Path(data_root)
    samples: list[Sample] = []
    for jsonl_path in paths:
        # with 语句保证文件句柄用完即关（即使中途异常也会关闭）。
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            # enumerate(start=1)：行号从 1 开始，报错时人类可读。
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue          # 容忍空行（手工编辑清单时常出现）
                row = json.loads(line)
                try:
                    # 数据集字段名是中文（官方 JSONL 的约定）。
                    wake_relative_path = str(row["唤醒音频"])
                    command_relative_path = str(row["识别音频"])
                except KeyError as error:
                    # 把出错的文件与行号带进异常信息（排错体验）；
                    # ``from error`` 保留原始异常链（exception chaining）。
                    raise KeyError(f"{jsonl_path}:{line_number} missing field {error}") from error
                samples.append(
                    Sample(
                        raw_id=str(row.get("id", "")),
                        # 音频路径是相对 data_root 的：样本集整体搬迁时
                        # 只改 --data-root 即可，清单不用动。
                        wake_audio=root / wake_relative_path,
                        command_audio=root / command_relative_path,
                        command_relative_path=command_relative_path,
                        label=row.get("识别文本"),          # 可能为 None（负样本）
                        wake_text=row.get("唤醒文本") or row.get("wake_text"),  # 兼容两种字段名
                    )
                )
    return samples


def submission_id(sample: Sample, mode: str) -> str:
    """按 id_mode 生成提交文件里使用的样本 id（官方按 id 对答案）。

    relative_stem（默认）：相对路径去掉 .wav 扩展名——官方提交格式按
    "相对路径的 stem"作为 id 对齐参考答案。as_posix() 把 Windows 风格
    反斜杠统一成正斜杠，保证跨平台输出一致。
    """
    if mode == "raw":
        return sample.raw_id
    if mode == "name":
        return sample.command_audio.name   # 仅文件名（含扩展名）
    if mode == "path":
        return sample.command_audio.as_posix()
    if mode == "relative":
        return sample.command_relative_path
    if mode == "relative_stem":
        path = Path(sample.command_relative_path)
        return (path.parent / path.stem).as_posix()   # 去扩展名 + 统一斜杠
    if mode == "stem":
        return sample.command_audio.stem    # 仅文件名（不含扩展名）
    # 未知模式：显式报错优于静默返回奇怪 id。
    raise ValueError(f"unsupported id mode: {mode}")


def extract_speaker_score(result: Any) -> float:
    """从 ModelScope 声纹管线的输出里"挖"出相似度标量。

    为什么写这么防御：speaker_verification pipeline 的输出结构随
    modelscope 版本变化（有时是 {'score': 0.8}，有时是嵌套 list/ndarray），
    写死一层取值会在升级后静默拿错或崩溃。这里递归试探常见键名与容器
    类型，直到拿到一个可转 float 的标量。try/except ValueError 在这里
    被当作"这条路径挖不到，换下一条"的控制流使用。
    """
    if isinstance(result, dict):
        # 按可能性从高到低试常见键名。
        for key in ("score", "scores", "similarity", "output", "outputs"):
            if key in result:
                try:
                    return extract_speaker_score(result[key])  # 递归下钻
                except ValueError:
                    continue    # 这个键下面没有标量，试下一个键
        raise ValueError(f"speaker verification output has no score: {list(result)}")
    if isinstance(result, (list, tuple)):
        for value in result:    # 逐个元素试探
            try:
                return extract_speaker_score(value)
            except ValueError:
                continue
        raise ValueError("speaker verification output was empty")
    if isinstance(result, np.ndarray):
        if result.size == 0:
            raise ValueError("speaker verification score array was empty")
        return float(result.reshape(-1)[0])   # 展平后取第一个元素
    try:
        return float(result)    # 已经是标量（int/float/单元素张量）
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot parse speaker score from {result!r}") from error


@contextlib.contextmanager
def suppress_model_output():
    """静默模型加载/推理期间的 stdout+stderr（FireRed/Qwen 打印噪音）。

    @contextmanager 把含 yield 的生成器函数变成上下文管理器：yield 之前
    是 __enter__（进入 with 时执行），yield 之后是 __exit__（退出 with
    时执行，且无论 with 块内是否抛异常都会执行——所以 stdout/stderr 一定
    会被恢复）。os.devnull 是平台无关的"空设备"，写进去的内容直接被丢弃。
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        # redirect_stdout/stderr 临时替换全局输出流；两层 with 并列写在
        # 一个 with 语句里（Python 语法糖，等价于嵌套）。
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def write_reports(results: list[dict[str, Any]], duration: float, output: Path, final_output: Path) -> dict[str, float]:
    """汇总指标并写出两份产物：官方提交文件 + 详细分析文件。

    返回 summary dict（主循环最后打印的那行指标）。两个 JSON 文件都写
    ensure_ascii=False（中文不转义）与 indent=2（人类可读）。
    """
    # 累加器：CER 分子/分母 + 正负样本统计。
    total_errors = 0
    total_characters = 0
    positive_nonempty = 0
    negative_total = 0
    negative_rejects = 0
    submission_results = []
    final_results = []
    for item in results:
        content = str(item["content"] or "")
        label = item["label"]
        # 正样本 = label 存在且非空（有参考答案要转写）。
        positive = label is not None and label != ""
        item_cer = cer(content, label) if positive else None
        if positive:
            total_errors += levenshtein(content, str(label))   # 编辑距离 = 错误字符数
            total_characters += len(str(label))
            positive_nonempty += int(content != "")            # 转写出非空文本的条数
        else:
            negative_total += 1
            negative_rejects += int(content == "")             # 正确拒识（输出空）的条数
        # 提交文件行：官方格式，cer 以 6 位小数字符串呈现（保持既有输出格式）。
        submission_results.append({"id": item["id"], "content": content, "label": label, "cer": "" if item_cer is None else f"{item_cer:.6f}"})
        final_item = {
            "id": item["id"],
            "content": content,
            "label": label,
            "sample_type": "positive" if positive else "negative",
            "cer": item_cer,
            # pr = pass rate（正样本转写出非空）；rr = reject rate（负样本被拒识）。
            "pr": float(content != "") if positive else None,
            "rr": None if positive else float(content == ""),
        }
        # 把门控/后端写入 item 的附加诊断字段统一带进 detail.json
        # （不同门控写的字段不同，用"存在即复制"的方式兼容）。
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
        ):
            if key in item:
                final_item[key] = item[key]
        final_results.append(final_item)
    summary = {
        # 三件套指标：字符错误率 / 正样本通过率 / 负样本拒识率。
        # 分母为 0 时给 0.0，避免除零（例如没有正样本的清单）。
        "final_cer": total_errors / total_characters if total_characters else 0.0,
        "final_pr": positive_nonempty / (len(results) - negative_total) if len(results) > negative_total else 0.0,
        "final_rr": negative_rejects / negative_total if negative_total else 0.0,
        "positive_count": len(results) - negative_total,
        "positive_nonempty": positive_nonempty,
        "negative_count": negative_total,
        "negative_rejects": negative_rejects,
        "duration": duration,
    }
    # mkdir(parents=True, exist_ok=True)：目标目录不存在就递归创建，
    # 已存在也不报错（幂等，重跑安全）。
    output.parent.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"result": {"results": submission_results, "final_cer": f"{summary['final_cer']:.6f}", "duration": f"{duration:.6f}"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    final_output.write_text(json.dumps({"summary": summary, "results": final_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器（与 main 分离，便于测试与复用）。

    argparse 把命令行字符串解析成带属性的 args 对象，并自动生成 --help
    说明。默认值即 baseline 历史口径，勿动。
    """
    parser = argparse.ArgumentParser(description="Standalone ERes2NetV2 plus Qwen3-ASR DatasetA baseline")
    parser.add_argument("--data-root", required=True)              # 音频根目录（相对路径据此拼接）
    parser.add_argument("--jsonl", nargs="+", required=True)       # 清单文件，可多个（+ = 至少一个）
    parser.add_argument("--config", default="configs/eres2netv2_qwen3_raw_command.json")
    parser.add_argument("--model-root", help="ModelScope cache root containing iic/ and Qwen/")
    parser.add_argument("--output", default="outputs/submission.json")       # 官方提交文件
    parser.add_argument("--final-output", default="outputs/detail.json")     # 逐条明细文件
    parser.add_argument("--limit", type=int)                       # 只跑前 N 条（冒烟测试用）
    parser.add_argument("--asr-batch-size", type=int, default=1)   # ASR 批量大小
    parser.add_argument("--checkpoint-output")                     # 断点文件路径（配 --resume 使用）
    parser.add_argument("--resume", action="store_true")           # 断点续跑开关（布尔旗标）
    return parser


def main() -> None:
    """CLI 主流程：装配 -> 逐样本门控 -> 批量 ASR -> 汇总写出。"""
    args = build_parser().parse_args()     # argparse 解析 sys.argv
    if args.asr_batch_size < 1:
        # 参数合法性校验前置：早失败、错误信息清晰，优于在深处莫名崩溃。
        raise ValueError("--asr-batch-size must be positive")
    config = load_config(args.config)
    samples = read_many_jsonl(args.jsonl, args.data_root)
    if args.limit is not None:
        samples = samples[: args.limit]    # --limit：切前 N 条，快速验证整条链路
    # checkpoint_path 为 None 表示"不写断点"（默认）。results 是贯穿全程
    # 的逐条结果列表，最终喂给 write_reports。
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else None
    results: list[dict[str, Any]] = []
    elapsed_before = 0.0    # 续跑时：之前运行累计的耗时（用于总时长统计）
    if args.resume:
        # 断点续跑：必须先有上次的 checkpoint 文件，否则无从续起。
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires --checkpoint-output from a prior run")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = checkpoint["results"]            # 恢复已算完的结果
        elapsed_before = float(checkpoint["duration"])
        completed = {item["id"] for item in results}   # 已完成的样本 id 集合
        # 按 id 过滤掉已完成样本，只重跑剩余部分（id 用 config.id_mode
        # 生成，与 checkpoint 里存的 id 口径一致）。
        samples = [sample for sample in samples if submission_id(sample, config.id_mode) not in completed]
    gate = None       # DualMLPGate（dual_mlp 模式）或 None
    verifier = None   # SpeakerVerifier（cos 模式）或 None
    # 门控装配：按 gate_mode 二选一。import 放在分支内部 = 只加载选中
    # 那条路径的重型依赖（选了 baseline 就不碰 gate.py 的双模型加载；
    # 反之亦然）。
    if config.gate_mode == "dual_mlp":
        from core.embeddings import DualMLPGate

        # 前置校验：缺 dual_backend 权重路径时立即报错，别等跑到一半才发现。
        if config.dual_backend is None:
            raise ValueError("gate_mode=dual_mlp requires dual_backend in config")
        # flush=True：print 默认有缓冲，重定向/管道下可能迟迟不显示；
        # 加载大模型耗时久，进度提示要立即可见。
        print(f"Loading dual MLP gate (ERes2NetV2 + CAM++) on {config.sv_device or config.device}...", flush=True)
        # sv_device 未配置时回退 device（config.sv_device or config.device
        # 利用了"None 为假值"的短路求值）。
        gate = DualMLPGate(
            config.sv_model, config.cam_model, config.dual_backend,
            config.dual_threshold, config.sv_device or config.device, args.model_root,
        )
    else:
        print(f"Loading ERes2NetV2 on {config.sv_device or config.device}...", flush=True)
        verifier = SpeakerVerifier(config.sv_model, config.sv_device or config.device, args.model_root)
    # ASR 后端装配：同样的"按需导入 + 分支内装配"模式。
    if config.asr_backend == "firered":
        from core.firered import FireRedRecognizer

        print(f"Loading FireRedASR2-AED on {config.device}...", flush=True)
        recognizer = FireRedRecognizer(config.firered_model, config.device, config.firered_repo)
    elif config.asr_backend == "qwen_lora":
        from core.firered import ConditionedQwenRecognizer

        if config.qwen_lora_adapter is None:
            raise ValueError("asr_backend=qwen_lora requires qwen_lora_adapter in config")
        print(f"Loading Qwen3-ASR + LoRA adapter ({config.qwen_lora_adapter}) on {config.device}...", flush=True)
        recognizer = ConditionedQwenRecognizer(config.asr_model, config.qwen_lora_adapter, config.device, args.model_root)
    else:
        # qwen 是默认分支（baseline 口径）。
        from core.firered import QwenRecognizer

        print(f"Loading Qwen3-ASR on {config.device}...", flush=True)
        recognizer = QwenRecognizer(config.asr_model, config.device, args.model_root, args.asr_batch_size)
    started = time.perf_counter()   # 本次运行计时的起点（续跑时与 elapsed_before 相加）
    # 批量调度缓冲区：门控放行的样本先进缓冲区，攒够 asr_batch_size 才
    # 一次性送给 ASR——GPU 批量推理的吞吐远高于逐条调用。
    pending_pairs: list[tuple[Path, Path]] = []   # (wake, cmd) 音频对
    pending_indices: list[int] = []               # 对应 results 里的下标（用于回填）

    def save_checkpoint() -> None:
        """把当前进度写进断点文件（闭包：直接读写外层 main 的局部变量）。"""
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            # duration 是"总耗时"（含续跑前的历史耗时），保证 resume 后
            # 时长统计仍然连贯。
            checkpoint_path.write_text(json.dumps({"duration": elapsed_before + time.perf_counter() - started, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    def flush_asr() -> None:
        """把缓冲区内攒够的样本送去 ASR，并把转写结果回填到 results。"""
        if not pending_pairs:
            return          # 缓冲区为空（例如批大小 1 且刚 flush 过）
        # 鸭子类型分派：条件后端有 transcribe_pairs（需要 wake+cmd 成对），
        # 普通后端只有 transcribe_many（只要 cmd 音频）。
        if getattr(recognizer, "conditioned", False):
            transcripts = recognizer.transcribe_pairs(pending_pairs)
        else:
            transcripts = recognizer.transcribe_many([command for _, command in pending_pairs])
        # 数量校验：ASR 少返回/多返回都会让 zip 静默错位，必须先拦下来。
        if len(transcripts) != len(pending_indices):
            raise RuntimeError("ASR returned an unexpected transcript count")
        completed_at = time.perf_counter()
        for index, transcript in zip(pending_indices, transcripts):
            results[index]["content"] = transcript
            # _started 是暂存于结果里的"该样本开始处理时间"（下划线前缀 =
            # 内部临时字段），pop 取出的同时删掉它，item_duration 即该条
            # 样本从门控通过到 ASR 完成的耗时。
            results[index]["item_duration"] = f"{completed_at - results[index].pop('_started'):.6f}"
        pending_pairs.clear()
        pending_indices.clear()
        save_checkpoint()   # 每 flush 一次就落一次盘：崩溃最多丢一个 batch

    # 主循环：逐样本"门控 -> 放行/拒识 -> 攒批 -> flush"。tqdm 包一层即得进度条。
    for sample in tqdm(samples, desc="infer", unit="utt"):
        item_started = time.perf_counter()
        item_id = submission_id(sample, config.id_mode)
        # 音频缺失直接失败：继续跑只会产出脏数据，fail fast 更好定位。
        if not sample.wake_audio.is_file() or not sample.command_audio.is_file():
            raise FileNotFoundError(f"missing audio for {item_id}")
        # 门控打分：dual_mlp 与 baseline 两条路径共享 accept 语义
        # （accepted=是否放行 ASR），但阈值来源不同。
        if gate is not None:
            accepted, speaker_score = gate.accept(sample.wake_audio, sample.command_audio)
            active_threshold = config.dual_threshold
        else:
            speaker_score = verifier.score(sample.wake_audio, sample.command_audio)
            accepted = speaker_score >= config.sv_accept_threshold
            active_threshold = config.sv_accept_threshold
        # 每条样本的明细记录：所有字段一次性建好（包括未实现功能的占位
        # 默认值），保证 detail.json 结构稳定、后续分析代码不用判空。
        # 数值类诊断字段统一格式化成定长字符串（保持既有输出格式）。
        item = {"id": item_id, "content": "", "label": sample.label, "sv_score": f"{speaker_score:.6f}", "accepted": accepted, "kept_segments": 0, "wake_localized": False, "wake_candidate_count": 0, "wake_transcript": "", "wake_text_score": "0.000000", "wake_selected_source": None, "wake_text_confirmed": True, "sv_accept_threshold": f"{active_threshold:.6f}", "item_duration": ""}
        results.append(item)
        if accepted:
            # 放行：记录开始时间（_started 临时字段），进缓冲区等攒批。
            item["_started"] = item_started
            pending_pairs.append((sample.wake_audio, sample.command_audio))
            pending_indices.append(len(results) - 1)
            if len(pending_pairs) >= args.asr_batch_size:
                flush_asr()     # 攒够了就发车
        else:
            # 拒识：content 保持 ""（即"拒绝输出"），本条耗时直接结算。
            item["item_duration"] = f"{time.perf_counter() - item_started:.6f}"
            # 缓冲区空着才写 checkpoint，避免把"半批"状态落盘（resume 时
            # 半批样本会重跑，一致性更好）。
            if not pending_pairs:
                save_checkpoint()
    flush_asr()     # 收尾：把最后不足一批的余量也发出去
    # 清掉所有临时字段，保证写出的 results 干净（pop 带默认值 None，
    # 没有该键也不报错）。
    for item in results:
        item.pop("_started", None)
    # 总耗时 = 历史（续跑）+ 本次。
    duration = elapsed_before + time.perf_counter() - started
    summary = write_reports(results, duration, Path(args.output), Path(args.final_output))
    # 汇总行 flush=True：确保在输出重定向/管道下也能及时看到最终指标。
    print(f"samples={len(results)} final_cer={summary['final_cer']:.6f} final_pr={summary['final_pr']:.6f} final_rr={summary['final_rr']:.6f} duration={duration:.3f}s", flush=True)


# "模块即脚本"的分水岭：当文件被 import（如 recognizers.py 顶部
# ``from .samples import suppress_model_output``，或测试套件）时，__name__
# 是 "core.infer"，main() 不会执行——只导入定义，无副作用；
# 当文件作为脚本运行（python -m core.infer）时，__name__ 才是
# "__main__"，此时才真正执行 CLI 主流程。正是这个约定让"既是被引用的
# 库，又是可执行的命令行工具"得以共存。
if __name__ == "__main__":
    main()
