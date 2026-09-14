"""双前端 MLP 声纹门控（ERes2NetV2 + CAM++ 特征 → MLP 裁判）。

为什么要做"MLP 门控"：单模型的余弦相似度只是一个标量，判别"指令音频里
是否含目标说话人"的能力有限；两个模型各自打分后如何加权又很难手工调好。
于是把两个声纹前端各自算出的统计特征拼成 32 维向量，交给一个可训练的
MLP 去融合，让数据自己学出最优组合（"用小网络当裁判"的集成思路）。

特征定义与训练端（sv_backend/extract_features.py）严格一致：
整段余弦 + 滑窗（1.0s/0.2s）余弦统计 + 时长/能量/削波，每个模型 16 维，拼接成 32 维。
裁判为 32->32->16->1 的 MLP，权重与标准化参数来自 backend_dual.pt。

read_wav / SAMPLE_RATE 的 canonical 定义在 midea_qiji.audio（本模块 re-export
以兼容 ``from midea_qiji.gate import read_wav`` 旧导入路径）。

功能：对"唤醒音频（目标说话人注册参考）vs 指令音频（待判定）"这一对波形，
先提取声纹 embedding，再计算 32 维门控特征，最后输出 0~1 的"含目标说话人"
概率，供上层与阈值比较决定是否放行 ASR。

关键导出：
  - ModelScopeEmbedder   封装 ModelScope 声纹模型 + 设备，批量提 L2 归一化 embedding；
  - DualMLPGate          双前端 MLP 门控（本模块主角），score()/accept() 打分；
  - cosine_similarity / sliding_windows  特征计算基础件（也与训练端对拍用）。

依赖：numpy + midea_qiji.audio（模块顶层仅这两个，随时可 import）；
torch / modelscope 在构造函数内延迟导入（见 DualMLPGate.__init__ 的注释）。
本模块是内核层，不依赖上层 szy/；上层入口（python -m midea_qiji.infer 且
config 里 gate_mode=dual_mlp）在运行时才构造 DualMLPGate，因此轻量用法
（--help、numpy-only 单元测试）无需安装 torch。

用法：
    gate = DualMLPGate(sv_model, cam_model, "backend_dual.pt", 0.5334, "cuda")
    accepted, score = gate.accept(wake.wav, command.wav)
"""
from __future__ import annotations  # 延迟注解求值：list[np.ndarray] 这类现代注解在运行时不被真正求值

from collections import defaultdict  # "键不存在时自动建默认值"的字典，用于按帧数分组
from pathlib import Path
from typing import Sequence

import numpy as np

# read_wav / SAMPLE_RATE 的真实定义在 midea_qiji.audio（唯一 canonical 来源，
# 避免两处定义不一致）。这里 re-export 只为兼容旧导入路径：
# ``from midea_qiji.gate import read_wav``。
# ``# noqa: F401`` 告诉 linter："这个 import 确实没在本文件直接用，但别报未使用"。
from midea_qiji.audio import SAMPLE_RATE, read_wav  # noqa: F401  (re-export)

# 每个声纹模型贡献 16 维门控特征（顺序与训练端 sv_backend/extract_features.py
# 严格一致——特征协议不能改，否则 MLP 见到的分布与训练时不同，会整体失效）：
#   第 1 维 cos_whole         整段测试音频 embedding 与注册 embedding 的余弦相似度；
#   第 2~11 维（滑窗统计）     测试音频按 1.0s 窗 / 0.2s 步长滑窗后，各窗口余弦分数
#                             的统计量（个数/极值/均值/标准差/Top3 均值/阈值计数/占比/
#                             跨度），刻画"目标人声音是全程存在还是只出现在局部"；
#   第 12~13 维（时长）        注册/测试音频秒数（音频太短时余弦分数不可信，需显式告诉 MLP）；
#   第 14 维 rms_ratio         测试/注册能量比（音量差异可能暗示不同录音环境/说话人）；
#   第 15~16 维（削波比例）    注册/测试音频中达到满量程的采样点占比（削波=录音爆音失真）。
# 本列表仅供说明与对拍；实际特征在 _model_features() 里按同样顺序拼装，
# 训练端抽特征也以这份顺序为准。
FEATURE_KEYS = [
    "cos_whole", "n_windows", "win_max", "win_mean", "win_std", "win_top3_mean",
    "win_count_ge_25", "win_count_ge_30", "win_count_ge_35", "win_frac_ge_30", "win_span_ge_30",
    "enroll_seconds", "test_seconds", "rms_ratio", "clip_ratio_enroll", "clip_ratio_test",
]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """两个向量的余弦相似度 = 点积 / (模长之积)，值域 [-1, 1]。

    几何含义：两个向量夹角的余弦——方向越一致越接近 1，说明两段声纹
    embedding 越可能来自同一说话人。对已做 L2 归一化的 embedding 而言，
    余弦相似度等价于点积，且不受向量绝对长度（响度）影响。
    任一向量接近零向量（模长 <= 1e-8）时夹角无定义，按"不相似"返回 0。
    """
    # np.linalg.norm 默认算 L2 范数（欧氏长度）。
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def sliding_windows(audio: np.ndarray, window_seconds: float = 1.0, shift_seconds: float = 0.2) -> list[np.ndarray]:
    """把测试音频切成固定长度的滑窗片段（默认窗 1.0s、步长 0.2s）。

    为什么需要滑窗：目标说话人的声音可能只出现在指令音频的一小段里，
    其余是他人说话/背景噪声，整段余弦会被噪声"稀释"；滑窗能捕捉"局部
    最像目标人"的片段，是对整段余弦的重要补充。窗长/步长与训练端
    sv_backend/extract_features.py 严格一致，属于特征协议的一部分，勿动。

    边界处理：
    - 音频短于一个窗：右侧补零到整窗长度，仍返回 1 个窗口（保证任何
      音频都至少参与一次打分，不会因太短而被无声跳过）；
    - 末尾残片不足 0.4s（约半窗）：丢弃——太短的窗口会掺入大量补零或
      过度聚焦单个音素，产生虚假的相似度。
    """
    window = int(window_seconds * SAMPLE_RATE)   # 窗内的采样点数
    shift = int(shift_seconds * SAMPLE_RATE)     # 相邻窗口的滑动步长（采样点）
    if len(audio) <= window:
        # 短音频：右侧补零成整窗，保证输出非空。
        padded = np.zeros(window, dtype=np.float32)
        padded[: len(audio)] = audio
        return [padded]
    segments = []
    # range 的步长=shift，起止控制使每个窗口都完整落在音频内（不越界）。
    for start in range(0, len(audio) - window + 1, shift):
        segments.append(audio[start : start + window])
    # 最后一个完整窗口之后还剩下多少采样点没被覆盖。
    tail = len(audio) - (len(segments) - 1) * shift - window
    if tail >= int(0.4 * SAMPLE_RATE):
        # 残片够长则补上"对齐音频末尾"的最后一个窗口（与训练端对齐逻辑一致）。
        segments.append(audio[len(audio) - window :])
    return segments


def _window_stats(win_scores: list[float]) -> list[float]:
    """把滑窗余弦分数序列压缩成 10 个统计量（门控特征的第 2~11 维）。

    设计意图：原始滑窗分数序列的长度随音频时长变化，不能直接进 MLP
    （MLP 输入维数必须固定），所以用一组统计量把变长序列"压缩"成固定
    长度的描述。这些统计量与阈值（0.25/0.30/0.35）都是训练端定死的
    特征协议，勿动。返回顺序：窗口数 / 最大值 / 均值 / 标准差 / Top3
    均值 / >=0.25 窗口数 / >=0.30 窗口数 / >=0.35 窗口数 / >=0.30
    占比 / >=0.30 连续跨度占比。
    """
    scores = np.asarray(win_scores, dtype=np.float64)
    n = len(scores)
    above = scores >= 0.30               # 布尔掩码：哪些窗口的余弦 >= 0.30
    if above.any():
        indices = np.flatnonzero(above)  # 命中窗口在时间轴上的位置下标
        # "连续跨度"占比：首尾命中之间（含中间未命中的窗口）的距离 /
        # 总窗口数。值大说明目标人声音贯穿全程；零散命中则值小。
        span = (indices[-1] - indices[0] + 1) / n
    else:
        span = 0.0
    return [
        float(n), float(scores.max()), float(scores.mean()), float(scores.std()),
        float(np.sort(scores)[-3:].mean()),          # Top3 均值：最强的几段相似度
        float((scores >= 0.25).sum()), float(above.sum()), float((scores >= 0.35).sum()),
        float(above.mean()), float(span),
    ]


def _rms(audio: np.ndarray) -> float:
    """均方根（Root Mean Square）：音频"响度/能量"的常用度量。

    +1e-12 是数值保护：静音段平方和接近 0，开方前加一个极小常数，避免
    下游做除法（rms_ratio）时出现 0/0 或除零。
    """
    return float(np.sqrt(np.mean(audio**2) + 1e-12))


def _clip_ratio(audio: np.ndarray) -> float:
    """削波比例：采样值达到满量程（|x| >= 0.999）的采样点占比。

    削波（clipping）指录音电平过高被"削顶"，波形失真。削波比例高说明
    录音设备过载、特征失真，也是门控判断音频质量的一个线索。
    """
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= 0.999))


class ModelScopeEmbedder:
    """对任意段波形提 L2 归一化 embedding（自动适配 ERes2NetV2 / CAM++）。

    设计要点：
    - 封装"模型 + 设备 + 批处理"：调用方只需传入波形序列，拿回归一化
      embedding；换模型（ERes2NetV2 / CAM++）不改变调用代码——这正是
      面向对象封装的价值：把"哪个模型、在哪台设备"这些变化关进一个类；
    - 重型依赖延迟导入：modelscope / torch 只在构造函数与提取函数内部
      import（见下方注释），保证 ``import midea_qiji.gate`` 本身不需要
      安装这些包（--help、numpy-only 单元测试都能跑）；
    - 批量按"帧数"分组：embedding 模型要求同一批 feature 在时间维等长
      才能 cat 拼接，而不同时长音频的帧数不同，故先按帧数分组再分小批；
    - torch.inference_mode()：推理专用上下文，不建计算图、不保留中间
      激活，比 no_grad 更快更省显存。
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

    def extract_waveforms(self, waveforms: Sequence[np.ndarray]) -> list[np.ndarray]:
        """批量提取 L2 归一化声纹 embedding，输出顺序与输入一一对应。

        归一化的意义：把每个 embedding 缩放到单位长度，之后余弦相似度
        退化为点积，且消除绝对尺度（响度/模型输出量级）差异。
        """
        if not waveforms:
            return []          # 空输入直接返回空列表，避免下游空批次出错
        import torch          # torch 同样延迟导入（原因见 __init__ 注释）

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
        features = [extract_feature(w.unsqueeze(0) if w.ndim == 1 else w) for w in prepared]
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


class DualMLPGate:
    """双前端 MLP 门控：score >= threshold 判为含目标说话人。

    推理流程：两个声纹模型（ERes2NetV2 主 + CAM++ 辅）各自对
    "注册音频 vs 测试音频"算出 16 维门控特征，拼接成 32 维 -> 标准化
    （训练集 mean/std）-> MLP（32->32->16->1）输出 logit -> sigmoid
    变成 0~1 的"含目标说话人"概率；score >= threshold 即 accept。

    与 infer.SpeakerVerifier（单模型余弦 + 人工阈值）相比，这里由训练
    好的 MLP 自动融合双模型证据；threshold 是历史调参值（默认 0.5334，
    见 infer.Config.dual_threshold）。本类把"两个模型 + 裁判网络 +
    标准化参数"封装成一个对象，外部只需 accept(wake, cmd) 一个入口。
    """

    def __init__(self, sv_model: str, cam_model: str, backend_path: str | Path,
                 threshold: float, device: str, model_root: str | Path | None = None) -> None:
        # 延迟导入（模式说明见 ModelScopeEmbedder.__init__ 的注释）。
        import torch
        from torch import nn

        # resolve_model_path 把 ModelScope 模型名解析到本地缓存路径
        # （model_root 指向缓存根目录，支持离线/固定版本）。它本身是轻量
        # 模块，与 torch 一起延迟导入，让本模块顶层只依赖 numpy 与
        # midea_qiji.audio。
        from midea_qiji.paths import resolve_model_path

        self.threshold = threshold
        # 两个前端：ERes2NetV2（sv，主模型）与 CAM++（cam，辅助模型），
        # 各自由 ModelScopeEmbedder 封装"模型+设备"，互不干扰。
        self.eres_embedder = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
        self.cam_embedder = ModelScopeEmbedder(resolve_model_path(cam_model, model_root), device)

        # backend_dual.pt 是训练端打包的产物，内含 mean/std（标准化参数）
        # 与 MLP 权重。weights_only=False：checkpoint 里除了张量还有
        # dict/list 等普通对象，torch 新版默认的 weights_only=True 会拒绝
        # 加载；这里关闭校验仅因文件来自自家训练端（可信来源），外部
        # checkpoint 不要这么做（反序列化任意对象有代码执行风险）。
        # map_location="cpu" 先加载到 CPU，再按需搬设备。
        bundle = torch.load(str(backend_path), map_location="cpu", weights_only=False)["dual"]
        # 训练集统计的均值/标准差，用于对 32 维特征做 z-score 标准化：
        # 各维量纲差异巨大（秒 vs 比例 vs 计数），不标准化 MLP 会被大数值
        # 维支配；必须用训练端同一组参数，保证推理分布与训练分布一致。
        self.mean = np.asarray(bundle["mean"], dtype=np.float32)
        self.std = np.asarray(bundle["std"], dtype=np.float32)
        # 裁判网络：与训练端 sv_backend 的定义严格一致（层数/宽度/激活/
        # dropout 都不能动，否则权重对不上）。nn.Sequential 把各层串成
        # 一条流水线，forward 自动逐层传递。
        self.mlp = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        # checkpoint 里的权重键带 "net." 前缀（训练端序列化时的命名），
        # 剥掉前缀后与 nn.Sequential 自动编号的键名对上，才能 load_state_dict。
        renamed = {k.replace("net.", ""): v for k, v in bundle["mlp"].items()}
        self.mlp.load_state_dict(renamed)
        self.mlp.eval()   # 切到评估模式：Dropout 关闭（推理时不该随机丢神经元）
        self.torch = torch  # 保存模块引用，score() 里直接 self.torch.no_grad()

    def _model_features(self, embedder: ModelScopeEmbedder, enroll_wav, test_wav, windows) -> list[float]:
        """用单个声纹前端算出 16 维门控特征（DualMLPGate 对两个前端各调一次）。

        与训练端 sv_backend/extract_features.py 的抽法严格一致（各维含义
        见模块级 FEATURE_KEYS 上方的注释）。
        """
        # 一次性把"注册音频 + 测试音频 + 所有滑窗"送去提 embedding：
        # 比逐条调用省开销，也保证所有片段经过完全相同的预处理。
        embeddings = embedder.extract_waveforms([enroll_wav, test_wav, *windows])
        emb_enroll, emb_test = embeddings[0], embeddings[1]
        # 每个滑窗分别与注册 embedding 求余弦，得到滑窗相似度序列。
        win_scores = [cosine_similarity(emb_enroll, w) for w in embeddings[2:]]
        return [
            cosine_similarity(emb_enroll, emb_test),      # 第 1 维：整段余弦
            *_window_stats(win_scores),                   # 第 2~11 维：滑窗统计（10 个）
            len(enroll_wav) / SAMPLE_RATE,                # 第 12 维：注册音频秒数
            len(test_wav) / SAMPLE_RATE,                  # 第 13 维：测试音频秒数
            _rms(test_wav) / max(_rms(enroll_wav), 1e-6),  # 第 14 维：能量比（防除零）
            _clip_ratio(enroll_wav),                      # 第 15 维：注册削波比例
            _clip_ratio(test_wav),                        # 第 16 维：测试削波比例
        ]

    def score(self, wake_audio: Path, command_audio: Path) -> float:
        """对 (唤醒音频, 指令音频) 输出 0~1 的"指令音频含目标说话人"概率。

        wake_audio 充当目标说话人的"注册参考"；对测试音频滑窗，是因为
        目标人的声音可能只出现在指令音频的局部（特征设计见模块 docstring）。
        """
        enroll_wav = read_wav(wake_audio)      # 注册参考音频
        test_wav = read_wav(command_audio)     # 待判定指令音频
        windows = sliding_windows(test_wav)    # 只对测试音频滑窗（注册参考整段都是目标人）
        # 两个前端各算 16 维，拼接成 32 维门控特征。
        features = self._model_features(self.eres_embedder, enroll_wav, test_wav, windows)
        features += self._model_features(self.cam_embedder, enroll_wav, test_wav, windows)
        # z-score 标准化：用训练端同组 mean/std，把各维拉到同一尺度。
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        # no_grad：MLP 纯前向，无需梯度（省显存）；unsqueeze(0) 补 batch 维。
        with self.torch.no_grad():
            logit = self.mlp(self.torch.from_numpy(x).unsqueeze(0)).item()
        # sigmoid 把 logit 压到 (0,1)：1/(1+e^{-x})。
        return float(1.0 / (1.0 + np.exp(-logit)))

    def accept(self, wake_audio: Path, command_audio: Path) -> tuple[bool, float]:
        """门控判定：返回 (是否含目标说话人, 分数)。阈值在构造时给定。"""
        score = self.score(wake_audio, command_audio)
        return score >= self.threshold, score
