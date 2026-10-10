#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛推理入口：逐条运行当前 lsz_v4_5/V100 语音指令识别流程。

本文件兼容主办方示例定义的两个接口：

* ``load_model(model_path, device)``：一次性加载分离、声纹、门控和 ASR 模型；
* ``infer_model(model, device, sample)``：接收一条测试样本并返回最终文本。

命令行输出严格采用 ``{"result": {"results": [...]}}`` 格式。模型和依赖均从
当前交付目录读取，运行时不会访问网络。
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
VENDOR_ROOT = WORKSPACE_ROOT / "vendor"
for _path in (SRC_ROOT, VENDOR_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 禁止 ModelScope/Hugging Face 在评测时尝试联网。
os.environ.setdefault("MODELSCOPE_CACHE", str(WORKSPACE_ROOT / "model" / "modelscope"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


CONFIG: dict[str, Any] = {}


def log(message: str) -> None:
    """立即打印一条运行日志。"""

    print(message, flush=True)


def _resolve_under_workspace(value: str | None) -> str | None:
    """把配置中的相对路径解析到交付目录，绝对路径保持不变。"""

    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve())


def _set_reproducible_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的推理随机种子。"""

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_model(model_path: str, device: str):
    """加载完整推理链路并返回可被 ``infer_model`` 复用的上下文。

    Args:
        model_path: 交付物的模型根目录。标准值为 ``/workspace/model``。
        device: ``cpu``、``cuda`` 或 ``cuda:N``。当前生产流程建议使用 CUDA。

    Returns:
        已加载的 ``SzyV2Runner``；所有样本共享它，模型不会逐条重复初始化。
    """

    from szy_v2 import SzyV2Config, SzyV2Runner, load_szy_v2_config

    model_root = Path(model_path).expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"模型根目录不存在: {model_root}")

    config_path = Path(CONFIG.get("pipeline_config", WORKSPACE_ROOT / "config" / "inference.json"))
    if not config_path.is_absolute():
        config_path = (WORKSPACE_ROOT / config_path).resolve()
    base = load_szy_v2_config(config_path)

    # 路径全部绑定到交付目录；这样从任意工作目录启动都不会引用原服务器工作区。
    overrides = {
        "device": device,
        "sv_device": device,
        "model_root": str(model_root / "modelscope"),
        "fusion_backend": _resolve_under_workspace(base.fusion_backend),
        "finetuned_sv_ckpt": _resolve_under_workspace(base.finetuned_sv_ckpt),
        "firered_model": _resolve_under_workspace(base.firered_model),
        "firered_repo": _resolve_under_workspace(base.firered_repo),
        "firered_vendor": _resolve_under_workspace(base.firered_vendor),
        "moss_model": _resolve_under_workspace(base.moss_model),
        "separation_cache": _resolve_under_workspace(base.separation_cache),
        "firered_elm_dir": _resolve_under_workspace(base.firered_elm_dir),
        "both_single_backend": _resolve_under_workspace(base.both_single_backend),
    }
    config = SzyV2Config(**{**base.__dict__, **overrides})
    _set_reproducible_seed(int(CONFIG.get("seed", 2026)))

    runner = SzyV2Runner(config, WORKSPACE_ROOT / "outputs")
    atexit.register(runner.flush_caches)
    return runner


def infer_model(model, device: str, sample: dict[str, Any]) -> str:
    """对一条唤醒—命令样本推理并返回最终识别文本。

    输入至少包含 ``唤醒音频``、``唤醒文本`` 和 ``识别音频``。音频字段由命令行
    框架预先转换成绝对路径。负样本不需要提供 ``识别文本``。
    """

    del device  # 模型初始化后设备已经固定，无需逐样本再次传递。
    from infer import Sample

    required = ("唤醒音频", "识别音频")
    missing = [key for key in required if not sample.get(key)]
    if missing:
        raise KeyError(f"样本缺少必需字段: {', '.join(missing)}")

    wake_audio = Path(str(sample["唤醒音频"])).expanduser().resolve()
    command_audio = Path(str(sample["识别音频"])).expanduser().resolve()
    for name, path in (("唤醒音频", wake_audio), ("识别音频", command_audio)):
        if not path.is_file():
            raise FileNotFoundError(f"{name}不存在: {path}")

    internal = Sample(
        raw_id=str(sample.get("id", "")),
        wake_audio=wake_audio,
        command_audio=command_audio,
        command_relative_path=str(sample["识别音频"]),
        label=sample.get("识别文本"),
        wake_text=sample.get("唤醒文本") or sample.get("wake_text"),
    )
    detail = model.process(internal)
    if detail.get("error"):
        log(f"[!] 样本 {sample.get('id')} 触发内部回退: {detail['error']}")
    return str(detail.get("content", ""))


def load_config(path: str | None) -> None:
    """读取顶层运行配置；流水线参数由其中的 ``pipeline_config`` 指定。"""

    global CONFIG
    CONFIG = {}
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            CONFIG = json.load(handle)
        log(f"[*] 已加载运行配置: {path}")


def load_test_json(path: str) -> list[dict[str, Any]]:
    """读取 JSON 数组或每行一个对象的 JSONL 测试清单。"""

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise TypeError("测试 JSON 的顶层必须是数组")
        return payload
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} 第 {line_number} 行解析失败: {error}") from error
    return rows


def resolve_audio_paths(sample: dict[str, Any], data_root: str) -> dict[str, Any]:
    """将样本中以 ``.wav`` 结尾的相对路径解析为绝对路径。"""

    root = Path(data_root).expanduser().resolve()
    output = dict(sample)
    for key, value in sample.items():
        if isinstance(value, str) and value.lower().endswith(".wav"):
            path = Path(value).expanduser()
            output[key] = str(path if path.is_absolute() else (root / path).resolve())
    return output


def _rss_mb() -> float | None:
    """读取当前进程常驻内存，单位 MB。"""

    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        return None


def _gpu_mem_mb(device: str) -> float | None:
    """读取 PyTorch 在目标 GPU 上已分配的显存，单位 MB。"""

    try:
        import torch

        if not device.startswith("cuda") or not torch.cuda.is_available():
            return None
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return torch.cuda.memory_allocated(index) / 1024**2
    except Exception:
        return None


def run(args: argparse.Namespace) -> None:
    """执行完整测试集并写出主办方规定的结果 JSON。"""

    load_config(args.config)
    samples = load_test_json(args.test_json)
    log(f"[*] 读取 {len(samples)} 条样本: {args.test_json}")
    if not samples:
        raise ValueError("测试清单为空")

    log(f"[*] 加载完整模型（device={args.device}）")
    model = load_model(args.model_path, args.device)
    cpu_mem = _rss_mb() if args.profile else None
    gpu_mem = _gpu_mem_mb(args.device) if args.profile else None

    results: list[dict[str, Any]] = []
    total_infer = 0.0
    for index, sample in enumerate(samples, start=1):
        absolute_sample = resolve_audio_paths(sample, args.data_root)
        started = time.perf_counter()
        try:
            content = infer_model(model, args.device, absolute_sample)
        except Exception as error:
            # 单条失败不终止整批评测，输出空串并把错误写到标准错误流。
            print(f"[!] 样本 {sample.get('id')} 推理失败: {error}", file=sys.stderr, flush=True)
            content = ""
        total_infer += time.perf_counter() - started
        results.append({"id": sample.get("id"), "content": content})
        if index % 100 == 0 or index == len(samples):
            log(f"[*] 进度 {index}/{len(samples)}")

    model.flush_caches()
    result_object: dict[str, Any] = {"result": {"results": results}}
    if args.profile:
        result_object["result"].update(
            {
                "avg_dur": round(total_infer / len(results), 6),
                "cpu_mem": round(cpu_mem, 4) if cpu_mem is not None else None,
                "gpu_mem": round(gpu_mem, 4) if gpu_mem is not None else None,
            }
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_object, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[*] 已写出: {output}（{len(results)} 条）")


def build_parser() -> argparse.ArgumentParser:
    """建立与主办方示例一致的命令行参数。"""

    parser = argparse.ArgumentParser(description="抗干扰语音指令识别推理")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config" / "runtime.json"))
    parser.add_argument("--data-root", required=True, help="测试音频根目录")
    parser.add_argument("--test-json", required=True, help="JSON/JSONL 测试清单")
    parser.add_argument("--model-path", default=str(WORKSPACE_ROOT / "model"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-o", "--output", default=str(WORKSPACE_ROOT / "outputs" / "result.json"))
    parser.add_argument("--profile", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
