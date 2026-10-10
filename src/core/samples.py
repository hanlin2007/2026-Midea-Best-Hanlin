"""core.samples：样本数据模型与 JSONL 清单读取（数据入口层）。

本模块定义管线与外部数据之间的唯一接口：

  - ``Sample``                一条评测样本的冻结数据模型（JSONL 一行的解析产物）；
  - ``read_many_jsonl``       把多份 JSONL 清单解析成 Sample 列表；
  - ``submission_id``         按竞赛口径生成提交文件里的样本 id；
  - ``suppress_model_output`` 静默底层模型打印噪音的上下文管理器。

"唯一接口"的意义：清单字段名（官方中文字段）、相对路径拼接规则、
id 生成口径全部收敛在这一个文件里。上层（pipeline 引擎、CLI、工具脚本）
只面对 Sample，不关心 JSONL 长什么样——数据格式变化时只改这里。

依赖：仅标准库（dataclasses / json / pathlib / contextlib）。
"""
from __future__ import annotations

import contextlib     # redirect_stdout/stderr 等上下文工具（suppress_model_output）
from dataclasses import dataclass
import json           # JSONL 读取
import os             # os.devnull：平台无关的空设备路径
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Sample:
    """一条评测样本：JSONL 一行 -> Sample（read_many_jsonl 的解析产物）。

    wake_audio 是目标说话人的注册/唤醒音频；command_audio 是待判定并
    转写的指令音频；label 为 None 表示负样本（指令里没有目标人，期望
    输出为空=拒识）；wake_text 留给需要转写唤醒词的任务（可选字段）。
    frozen 的意义与配置相同：样本是"证据"，处理途中不允许被改写。
    """

    raw_id: str                    # JSONL 里的原始 id（可能为空字符串）
    wake_audio: Path               # 唤醒音频绝对路径（data_root 拼接后）
    command_audio: Path            # 指令音频绝对路径
    command_relative_path: str     # 指令音频相对路径（提交文件 id 的来源）
    label: str | None              # 期望转写文本；None = 负样本
    wake_text: str | None = None   # 唤醒词文本（可选，兼容两种 JSONL 字段名）


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


@contextlib.contextmanager
def suppress_model_output():
    """把 with 块内的 stdout/stderr 全部重定向到空设备（丢弃）。

    模型加载与推理（FireRed / ModelScope / torch）会往终端刷大量进度
    噪音，淹没我们自己的进度条和结果输出。os.devnull 是平台无关的
    "空设备"路径（Unix 的 /dev/null，Windows 的 NUL），写进去的内容
    直接丢弃。@contextlib.contextmanager 把这个"开两个重定向 → yield
    → 自动还原"的过程变成一个 with 语句可用的上下文管理器。
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
