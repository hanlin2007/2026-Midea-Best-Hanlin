"""core.embeddings：声纹 embedding 提取（ModelScope 前端封装）。

本模块只有一个主角：``ModelScopeEmbedder``——封装"模型 + 设备 + 批处理"，
把任意段波形变成 L2 归一化的声纹 embedding（自动适配 ERes2NetV2 / CAM++
等 ModelScope 说话人验证模型）。余弦相似度在归一化 embedding 上退化为
点积，这是全系统所有"像不像同一个人"判断的原子操作。

设计要点：
- 换模型不改调用代码：构造时传入模型路径，调用方只面对
  prepare/extract 两步接口——这正是面向对象封装的价值：把"哪个模型、
  在哪台设备"这些变化关进一个类；
- 提取拆成两步（``prepare_waveforms`` / ``extract_features``）：
  预训练与微调两个前端的声学前端完全相同，融合门控对同一份波形只算
  一次 Fbank，两个前端各自前向 embedding_model，计算量省一半；
- 批量按"帧数"分组：embedding 模型要求同一批 feature 在时间维等长
  才能 cat 拼接，而不同时长音频的帧数不同，故先按帧数分组再分小批；
- 重型依赖延迟导入：modelscope / torch 只在构造函数与提取函数内部
  import，保证 ``import core.embeddings`` 本身不需要安装这些包
  （--help、numpy-only 单元测试都能跑）。

依赖：numpy；modelscope / torch 延迟导入。
"""
from __future__ import annotations  # 延迟注解求值：list[np.ndarray] 这类现代注解在运行时不被真正求值

from collections import defaultdict  # "键不存在时自动建默认值"的字典，用于按帧数分组
from pathlib import Path
from typing import Sequence

import numpy as np


class ModelScopeEmbedder:
    """对任意段波形提 L2 归一化 embedding（自动适配 ERes2NetV2 / CAM++）。

    使用方式（两步）：

        shared = embedder.prepare_waveforms(waves)   # Fbank 声学特征（可共享）
        vectors = embedder.extract_features(shared)   # 前向 embedding_model

    只有单个前端时可直接用便捷入口 ``extract_waveforms``（= 两步的组合）。
    """

    def __init__(self, model: str | Path, device: str, batch_size: int = 64) -> None:
        # 延迟导入（重要工程惯例）：modelscope 是重型依赖（背后要装
        # torch、可能触发模型下载），如果放在模块顶部，任何 import 本模块
        # 的代码（哪怕是打印 --help）都会被强制要求装齐全部重依赖。
        # 放进构造函数后，只有真正"要用模型"的代码路径才付出这份成本。
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.batch_size = batch_size
        # pipeline 是 ModelScope 的"一站式"高层抽象：给定任务类型 + 模型名，
        # 自动完成模型加载、预处理、推理、后处理全套流程。
        # Tasks.speaker_verification（说话人确认：判断两段音频是否同一人）
        # 告诉 pipeline 该挂哪套预处理/后处理逻辑。
        self.pipeline = pipeline(task=Tasks.speaker_verification, model=str(model), device=device)

    def prepare_waveforms(self, waveforms: Sequence[np.ndarray]) -> list:
        """批量提取声学特征（Fbank），输出顺序与输入一一对应。

        拆分出本方法的原因：预训练与微调两个前端的**声学前端完全相同**
        （同一 ERes2NetV2 基座，只是 embedding_model 权重不同），融合门控
        对 5 段波形各提一次特征后，两个前端共享这份特征各自前向
        embedding_model——Fbank 计算量直接省一半，且不增加并发激活显存。
        返回的是 torch 张量列表（具体类型属于 pipeline 内部实现，故不标注）。
        """
        if not waveforms:
            return []
        # pipeline.preprocess 把原始波形（float 数组）转成模型要吃的
        # feature 张量（分帧、加窗、fbank 等声学前端处理）。
        prepared = self.pipeline.preprocess(list(waveforms))
        model = self.pipeline.model
        # ERes2NetV2 与 CAM++ 的"提特征"方法名都叫 __extract_feature，
        # 但类结构不同；用 dir() + 后缀匹配动态找到它，两个模型就能
        # 走同一份代码（反射/鸭子类型技巧）。找不到就抛出带类名的明确错误。
        extract_feature = None
        for attr in dir(model):
            if attr.endswith("__extract_feature"):
                extract_feature = getattr(model, attr)
                break
        if extract_feature is None:
            raise AttributeError(f"no __extract_feature on {type(model).__name__}")
        # 逐条提声学特征；一维波形补一个 batch 维（unsqueeze(0)），
        # 模型内部约定输入形如 (batch, frames, feat)。
        return [extract_feature(w.unsqueeze(0) if w.ndim == 1 else w) for w in prepared]

    def extract_features(self, features: list) -> list[np.ndarray]:
        """对已提取的声学特征批量前向 embedding_model，输出 L2 归一化 embedding。

        与 :meth:`prepare_waveforms` 配套：调用方先 prepare 一次，再把同一份
        特征分别喂给两个前端的 extract_features（见 pipeline.engine.FusionGate）。
        """
        if not features:
            return []
        import torch          # torch 同样延迟导入（原因见 __init__ 注释）

        model = self.pipeline.model
        # 按"帧数"（时间维长度）分组：同批 cat 拼接要求该维等长；
        # defaultdict(list) 免去"先判断 key 是否存在"的样板代码。
        by_frame_count: dict[int, list[int]] = defaultdict(list)
        for index, feature in enumerate(features):
            by_frame_count[int(feature.shape[1])].append(index)
        # 占位列表：按原下标回填，保证输出顺序与输入一一对应。
        embeddings: list[np.ndarray | None] = [None] * len(features)
        with torch.inference_mode():       # 推理上下文：不建图、不存激活，省时省显存
            for indices in by_frame_count.values():
                for start in range(0, len(indices), self.batch_size):
                    # 帧数相同的一组再按 batch_size 切小批，控制显存峰值。
                    batch_indices = indices[start : start + self.batch_size]
                    # 沿 batch 维拼接后搬到模型所在设备（GPU/CPU 统一处理）。
                    feature_batch = torch.cat([features[i] for i in batch_indices], dim=0).to(model.device)
                    # embedding_model 把声学特征映射成说话人向量；
                    # detach() 脱离计算图，cpu() 搬回内存，numpy() 转 numpy。
                    batch = model.embedding_model(feature_batch).detach().cpu().numpy()
                    for index, embedding in zip(batch_indices, batch):
                        embeddings[index] = np.asarray(embedding, dtype=np.float32)
        normalized: list[np.ndarray] = []
        for embedding in embeddings:
            if embedding is None:
                # 防御性检查：任何一条没被回填都说明批处理逻辑出错。
                raise ValueError("embedder did not return every embedding")
            norm = float(np.linalg.norm(embedding))
            # L2 归一化：除以自身模长，得到单位向量；max(norm, 1e-8)
            # 防止零向量触发除零。
            normalized.append((embedding / max(norm, 1e-8)).astype(np.float32))
        return normalized

    def extract_waveforms(self, waveforms: Sequence[np.ndarray]) -> list[np.ndarray]:
        """批量提取 L2 归一化声纹 embedding，输出顺序与输入一一对应。

        归一化的意义：把每个 embedding 缩放到单位长度，之后余弦相似度
        退化为点积，且消除绝对尺度（响度/模型输出量级）差异。
        本方法 = prepare_waveforms + extract_features 的组合便捷入口；
        需要两个前端共享特征时（融合门控）请显式分两步调用。
        """
        return self.extract_features(self.prepare_waveforms(waveforms))
