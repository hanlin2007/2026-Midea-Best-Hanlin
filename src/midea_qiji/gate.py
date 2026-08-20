"""双前端 MLP 声纹门控（ERes2NetV2 + CAM++ 特征 → MLP 裁判）。

================================================================================
【这个文件在整条流水线里做什么？】

竞赛任务：给定「唤醒音频 kws.wav」（里面有人说了唤醒词，如"你好科慕"）和
「识别音频 cmd.wav」（可能含多个人同时说话），要求识别出 cmd.wav 里
"和说唤醒词的人是同一个人"的那句话。

在花大量算力跑语音识别（ASR）之前，系统要先回答一个问题：
    "cmd.wav 里到底有没有目标说话人在说话？"
这就是「门控（gate）」的职责——一个二分类裁判：有 → 继续识别；没有 → 直接输出空。

为什么要用两个声纹模型（ERes2NetV2 + CAM++）？
    两个模型结构不同、互补。两个模型的"意见"拼成 32 维特征，
    再交给一个小型 MLP（多层感知机）神经网络做最终裁判。
    这个 MLP 是队长用训练数据学出来的（权重在 backend_dual.pt 里），
    而不是人工定规则——这就是"可学习门控"。

特征定义与训练端（sv_backend/extract_features.py）严格一致：
整段余弦 + 滑窗（1.0s/0.2s）余弦统计 + 时长/能量/削波，每个模型 16 维，拼接成 32 维。
裁判为 32->32->16->1 的 MLP，权重与标准化参数来自 backend_dual.pt。

【对外接口】
    DualMLPGate.score(wake_wav, cmd_wav)  -> 浮点概率 p（目标人在说话的概率）
    DualMLPGate.accept(wake_wav, cmd_wav) -> (是否接受, 分数) 元组
================================================================================
"""
from __future__ import annotations  # 让类型注解可以"延迟求值"，可写 forward 引用（详见课程第 1 课）

from collections import defaultdict  # 默认值字典：dict 的高级变体，取值时键不存在自动给默认值
from pathlib import Path  # 面向对象的路径工具，跨平台（Windows 的 \\ 和 Linux 的 / 都兼容）
from typing import Sequence  # 类型标注工具：Sequence 表示"任何可按下标遍历的序列"（list/tuple 等）

import numpy as np  # 数值计算库，本项目所有波形/特征/向量运算的基础

# 全项目统一的音频采样率：16 kHz（一秒钟的音频有 16000 个采样点）
# 为什么是 16000？——语音识别/声纹领域的事实标准，两个声纹模型和 ASR 都是按 16k 训练的
SAMPLE_RATE = 16000

# 每个声纹模型贡献 16 维特征，两个模型拼起来 32 维。这个 list 就是 16 维特征的名字清单，
# 顺序必须与训练端（sv_backend/extract_features.py）以及 _model_features() 的返回顺序严格一致，
# 否则加载的 MLP 权重会对错位（张冠李戴，模型直接失效）。
FEATURE_KEYS = [
    "cos_whole",           # 整段音频的余弦相似度（唤醒整段 vs 识别整段）
    "n_windows",           # 滑窗切出的窗口数量
    "win_max",             # 所有窗口余弦的最大值
    "win_mean",            # 所有窗口余弦的均值
    "win_std",             # 所有窗口余弦的标准差（波动程度）
    "win_top3_mean",       # 余弦最高的 3 个窗口的均值（最像的局部区域有多像）
    "win_count_ge_25",     # 余弦 >= 0.25 的窗口个数
    "win_count_ge_30",     # 余弦 >= 0.30 的窗口个数
    "win_count_ge_35",     # 余弦 >= 0.35 的窗口个数
    "win_frac_ge_30",      # 余弦 >= 0.30 的窗口占比（比例，0~1）
    "win_span_ge_30",      # 余弦 >= 0.30 的窗口在时间轴上的覆盖跨度（最前到最后/总数）
    "enroll_seconds",      # 唤醒音频时长（秒）
    "test_seconds",        # 识别音频时长（秒）
    "rms_ratio",           # 识别/唤醒 的能量（RMS）比值
    "clip_ratio_enroll",   # 唤醒音频的削波比例（有多少采样点顶到最大音量）
    "clip_ratio_test",     # 识别音频的削波比例
]


def read_wav(path: str | Path) -> np.ndarray:
    """用标准库 wave 读 16k 单声道 PCM16 wav，转成 float32 波形数组。

    参数:
        path: wav 文件路径
    返回:
        numpy 一维数组，dtype=float32，数值范围约 [-1.0, 1.0]

    为什么不用 soundfile 而是 wave？
        wave 是 Python 内置标准库，零依赖；同时它对格式做了最严格的检查
        （必须是 16k、单声道、PCM16），恰好符合竞赛数据的格式约定，
        一旦数据不符合预期会立刻报错，而不是静默产出错误结果（fail-fast 风格）。
    """
    import wave  # 标准库模块，函数内导入（lazy import）避免无关路径也加载它

    # with 语句：打开文件，无论中间是否出错，离开缩进块时自动关闭文件（上下文管理器）
    with wave.open(str(path), "rb") as handle:  # "rb" = 只读二进制模式
        # 格式守卫：不是 16k 或不是单声道 → 立即抛异常（数据集契约：所有 wav 都是 16k 单声道 PCM16）
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ValueError(f"unsupported wav format: {path}")
        # 一次性读出所有采样帧，raw 是 bytes（原始字节流），不是数字
        raw = handle.readframes(handle.getnframes())
    # np.frombuffer：把 bytes 按 dtype=int16 解释为整数数组（PCM16 = 每个采样点 2 字节有符号整数）
    # astype(np.float32)：转成 32 位浮点
    # / 32768.0：归一化。int16 取值范围 [-32768, 32767]，除以 32768 后大致落在 [-1, 1)
    #   （声学处理惯例：把振幅缩放到 [-1, 1] 区间，避免后续计算数值过大）
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """两个向量的余弦相似度：衡量"方向一致程度"，与向量长度无关。

    公式: cos(θ) = (a · b) / (‖a‖ × ‖b‖)
      - 结果为 1.0 表示方向完全相同；0 表示正交；-1 表示完全相反。
      - 声纹场景：两个说话人 embedding 的余弦 > 阈值，就认为"是同一个人的声音"。

    为什么用余弦而不是欧氏距离？
        说话人嵌入（embedding）的绝对长度往往受音量、时长等因素影响，
        但"方向"主要由说话人身份决定。余弦相似度天然忽略长度、只看方向，
        对音量不敏感——这正是声纹比较想要的性质。
    """
    # np.linalg.norm 默认算 L2 范数（欧氏长度）：sqrt(各分量平方和)
    # float() 把 numpy 标量转成 Python 原生 float（方便 json 序列化与打印）
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    # 保护分支：任一向量长度接近 0（例如静音段，全 0），分母会爆掉 → 直接返回 0.0
    # 1e-8 是一个"几乎为 0"的经验阈值，避免除以 0 产生 NaN/Inf
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    # np.dot 向量点积；结果 = 余弦值
    return float(np.dot(left, right) / (left_norm * right_norm))


def sliding_windows(audio: np.ndarray, window_seconds: float = 1.0, shift_seconds: float = 0.2) -> list[np.ndarray]:
    """把一段音频切成若干重叠的滑窗片段（window=1.0s，每 0.2s 前进一步）。

    为什么要滑窗？
        一条命令音频可能有 5~10 秒，而目标说话人可能只出现在其中一小段
        （其余是别人说话或噪声）。对"整段"算一次相似度会把目标人的信号
        稀释掉。滑窗相当于把音频切成很多小段分别打分，
        再用统计量（最大值/均值/Top3……）捕捉"局部很像"的证据。

    返回:
        窗口片段的 list；每个元素是 float32 数组（长度 16000）。
        若音频比一个窗口还短 → 后面补 0 凑满一个窗口（保证至少有一个窗口，
        且长度统一，便于后续批量矩阵运算）。
    """
    # 把秒换算成采样点数：window=16000 个点（1 秒），shift=3200 个点（0.2 秒）
    window = int(window_seconds * SAMPLE_RATE)
    shift = int(shift_seconds * SAMPLE_RATE)
    if len(audio) <= window:
        # 音频太短：用 np.zeros 造一个全 0 的 1 秒窗口，把音频塞进开头
        # （padded[:len(audio)] = audio 是 numpy 的切片赋值，只改前 len(audio) 个元素）
        padded = np.zeros(window, dtype=np.float32)
        padded[: len(audio)] = audio
        return [padded]
    segments = []  # 结果列表：装下所有切出的窗口
    # range(起点, 终点, 步长)：起点从 0 开始，每次前进 shift 个点，
    # 终点 len(audio) - window + 1 保证最后一个窗口不会越过数组末尾
    for start in range(0, len(audio) - window + 1, shift):
        segments.append(audio[start : start + window])  # 切片：[start, start+window)，正好 1 秒
    # 尾部处理：range 循环不保证最后恰好停满一个窗口，会剩一段"尾巴"
    # tail = 最后一段窗口起点到音频末尾还差多少点
    tail = len(audio) - (len(segments) - 1) * shift - window
    # 尾巴 >= 0.4 秒（6400 点）才值得再补一个窗口，太短的信息量太小，直接丢弃
    # 补窗口的方式：取音频最后整整 1 秒（audio[len-window:]），保证窗口等长
    if tail >= int(0.4 * SAMPLE_RATE):
        segments.append(audio[len(audio) - window :])
    return segments


def _window_stats(win_scores: list[float]) -> list[float]:
    """把一组窗口余弦分数汇总成 10 个统计特征。

    输入: win_scores = [每个滑窗与唤醒嵌入的余弦相似度, ...]
    返回: [n, max, mean, std, top3_mean, count>=0.25, count>=0.30, count>=0.35, frac>=0.30, span>=0.30]
    """
    # np.asarray：list → numpy 数组；dtype=float64 提高统计精度
    scores = np.asarray(win_scores, dtype=np.float64)
    n = len(scores)  # 窗口总数（也是特征之一：说话时长越长窗口越多，MLP 可以利用这个信息）
    # 布尔掩码：scores >= 0.30 是一个元素为 True/False 的数组（True=该窗口比较像目标人）
    above = scores >= 0.30
    if above.any():  # .any()：数组里只要有一个 True 就返回 True
        # np.flatnonzero(above)：返回所有 True 的位置下标（整数数组）
        # span = (最后一个 True 的下标 - 第一个 True 的下标 + 1) / 总数
        # 含义：目标人"疑似出现"的区域在整个音频时间轴上占多大跨度
        #   （连续说话 → span 大；只在某一瞬间像了一下 → span 小）
        indices = np.flatnonzero(above)
        span = (indices[-1] - indices[0] + 1) / n
    else:
        span = 0.0  # 没有一个窗口达到 0.30 → 跨度记 0
    return [
        float(n),                              # 窗口数
        float(scores.max()),                   # 最大值：最像的局部窗口
        float(scores.mean()),                  # 均值：整体像不像
        float(scores.std()),                   # 标准差：像的程度是否忽高忽低（波动大 = 可疑）
        float(np.sort(scores)[-3:].mean()),    # Top3 均值：排序后取最后 3 个（最大的 3 个）再平均
        float((scores >= 0.25).sum()),         # 达到 0.25 的窗口个数（.sum() 数 True 的个数）
        float(above.sum()),                    # 达到 0.30 的窗口个数
        float((scores >= 0.35).sum()),         # 达到 0.35 的窗口个数
        float(above.mean()),                   # 达到 0.30 的窗口占比（True 的均值 = 比例）
        float(span),                           # 0.30 窗口的时间跨度比例
    ]
    # 注意：return 的顺序必须与 FEATURE_KEYS 的 [1..10] 号位置一一对应（cos_whole 由调用方放在最前）


def _rms(audio: np.ndarray) -> float:
    """RMS（均方根）：衡量一段音频的"响度/能量"。

    公式: sqrt(mean(audio^2))
      - 对每个采样点平方 → 求平均 → 开方。静音 ≈ 0，正常语音 > 0.01。
      - +1e-12 防止全 0 静音时 sqrt(0)=0 带来的除零风险（虽然这里只求值不除，
        但与训练端保持一致，避免数值漂移）。
    """
    return float(np.sqrt(np.mean(audio**2) + 1e-12))


def _clip_ratio(audio: np.ndarray) -> float:
    """削波比例：有多少比例的采样点振幅撞到了 ±1 的极限。

    削波（clipping）现象：录音设备增益过大时，本来应该 >1.0 的振幅被
    硬切到 1.0，波形顶部被"削平"。削波的音频信息已丢失，会影响声纹
    模型的表现。这个特征让 MLP 有机会学会"削波严重 → 分数可疑"。
    """
    if len(audio) == 0:  # 空数组防护（正常数据不会出现，但防御式编程是好习惯）
        return 0.0
    # np.abs(audio) >= 0.999：接近满幅的采样点标记为 True；.mean() 得到比例
    return float(np.mean(np.abs(audio) >= 0.999))


class ModelScopeEmbedder:
    """对任意段波形提取 L2 归一化 embedding（自动适配 ERes2NetV2 / CAM++）。

    什么是 embedding（嵌入）？
        声纹模型把一段"任意长度的音频"压成"固定长度的向量"（ERes2NetV2 是 192 维，
        CAM++ 是 192 维），这个向量就是这段声音的"声纹指纹"。
        同一个人的不同句子 → 向量方向接近（余弦大）；不同人 → 方向差别大。

    L2 归一化：
        把向量除以自己的长度，让所有向量长度都变成 1（变成单位球面上的点）。
        归一化之后，余弦相似度 = 两个向量的点积（因为分母 ‖a‖×‖b‖ = 1×1 = 1），
        计算更简单，且所有比较在同一尺度下进行。

    为什么通过 ModelScope pipeline 加载？
        ERes2NetV2 / CAM++ 是 ModelScope 官方发布的预训练模型
        （iic/speech_eres2netv2_sv_zh-cn_16k-common 等）。
        ModelScope 的 pipeline 封装了模型下载、加载、预处理，本类复用了它的
        加载与预处理逻辑，但"提取 embedding"用的是自定义的批处理路径（更快）。
    """

    def __init__(self, model: str | Path, device: str, batch_size: int = 64) -> None:
        """构造嵌入器。

        参数:
            model: 模型 ID 或本地目录（resolve_model_path 会把 ID 解析成磁盘上的真实目录）
            device: 计算设备，如 "cuda:0"（GPU）或 "cpu"
            batch_size: 一次送进 GPU 的片段数。批处理让 GPU 的矩阵运算保持"吃饱"，
                        远快于一条一条算（GPU 擅长并行大矩阵）。
        """
        # 函数内导入（lazy import）：modelscope 很重，只有真要用声纹模型时才导入
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.batch_size = batch_size
        # 创建 ModelScope 的"说话人确认"pipeline：
        #   task=Tasks.speaker_verification 告诉框架这是一个声纹任务；
        #   model=本地目录 → 直接用本地权重，不联网；
        #   device=cuda:0 → 权重加载到 GPU。
        self.pipeline = pipeline(task=Tasks.speaker_verification, model=str(model), device=device)

    def extract_waveforms(self, waveforms: Sequence[np.ndarray]) -> list[np.ndarray]:
        """批量提取波形序列的声纹 embedding（保持输入顺序）。

        参数:
            waveforms: 波形数组的序列，每个元素是 float32 一维数组
        返回:
            与输入等长的 list，每个元素是该波形的 L2 归一化 embedding（float32 数组）
        """
        if not waveforms:  # 空输入直接返回空列表（不调用模型，省时且避免越界）
            return []
        import torch  # lazy import：本函数是唯一直接触碰 torch 的地方

        # pipeline.preprocess：ModelScope 的预处理（归一化音量、可能的重采样等），
        # 把波形变成模型可直接吃的形式（torch tensor）
        prepared = self.pipeline.preprocess(list(waveforms))
        model = self.pipeline.model  # 底层 torch 模型对象（ERes2NetV2 或 CAM++）
        # 声纹模型由两部分组成：前端特征网络（__extract_feature）→ 后端嵌入网络（embedding_model）。
        # 这里用 dir() 列出对象所有属性名，找到那个以 __extract_feature 结尾的方法。
        # 为什么名字带双下划线？——ModelScope 把内部方法故意起这种怪名字，防止普通用户误调用。
        extract_feature = None
        for attr in dir(model):
            if attr.endswith("__extract_feature"):
                extract_feature = getattr(model, attr)  # getattr：按字符串名字取方法
                break
        if extract_feature is None:  # 找不到就报错：说明模型版本/结构不是预期的
            raise AttributeError(f"no __extract_feature on {type(model).__name__}")
        # 对每个预处理后的波形提前端特征；unsqueeze(0) 把一维波形加一个"批"维度（1, T）
        features = [extract_feature(w.unsqueeze(0) if w.ndim == 1 else w) for w in prepared]
        # 声纹模型按"帧数（时长）"分批：不同长度的音频前端特征维度不同，无法直接
        # cat 成一个大矩阵，所以先按帧数分组，同帧数的凑一批喂 GPU。
        by_frame_count: dict[int, list[int]] = defaultdict(list)  # {帧数: [下标, ...]}
        for index, feature in enumerate(features):
            by_frame_count[int(feature.shape[1])].append(index)
        embeddings: list[np.ndarray | None] = [None] * len(features)  # 预分配结果槽位，按 index 回填
        with torch.inference_mode():  # 推理模式：关闭梯度跟踪，省显存、算得快
            for indices in by_frame_count.values():  # 每个帧数组一批
                for start in range(0, len(indices), self.batch_size):  # 组内再按 batch_size 切
                    batch_indices = indices[start : start + self.batch_size]
                    # torch.cat(dim=0)：把这些片段沿"批"维度拼成一个大张量（B, 帧数, 特征维）
                    # .to(model.device)：搬到模型所在的 GPU
                    feature_batch = torch.cat([features[i] for i in batch_indices], dim=0).to(model.device)
                    # embedding_model：前端特征 → 说话人嵌入（B, 192）
                    # .detach()：与计算图脱钩；.cpu()：搬回内存；.numpy()：转 numpy
                    batch = model.embedding_model(feature_batch).detach().cpu().numpy()
                    for index, embedding in zip(batch_indices, batch):  # zip 打包两个序列一起遍历
                        embeddings[index] = np.asarray(embedding, dtype=np.float32)
        normalized: list[np.ndarray] = []
        for embedding in embeddings:
            if embedding is None:  # 防御：如果有槽位没被回填说明批处理逻辑有 bug
                raise ValueError("embedder did not return every embedding")
            norm = float(np.linalg.norm(embedding))  # L2 范数（向量长度）
            # L2 归一化：向量 / 自身长度 → 长度变成 1。max(norm, 1e-8) 防止除以 0
            normalized.append((embedding / max(norm, 1e-8)).astype(np.float32))
        return normalized


class DualMLPGate:
    """双前端 MLP 门控：score >= threshold 判为含目标说话人。

    它是整条流水线的"第一道关卡"：
        accept(wake, cmd) → True ：cmd.wav 里很可能有目标人 → 放行给 ASR
        accept(wake, cmd) → False：直接拒绝 → 输出空文本，不浪费 ASR 算力
    """

    def __init__(self, sv_model: str, cam_model: str, backend_path: str | Path,
                 threshold: float, device: str, model_root: str | Path | None = None) -> None:
        """构造门控：加载两个声纹前端 + 训练好的 MLP 裁判。

        参数:
            sv_model:  ERes2NetV2 的模型 ID 或路径
            cam_model: CAM++ 的模型 ID 或路径
            backend_path: backend_dual.pt —— 训练端产出的裁判权重包
                          （里面装：32 维特征的均值 mean、标准差 std、MLP 权重 mlp）
            threshold: 冻结的判定阈值（SZY v1 = 0.4460311856655159）
            device: 计算设备
            model_root: 模型根目录（用于把 ID 解析为本地路径）
        """
        import torch
        from torch import nn  # torch 的神经网络模块

        from midea_qiji.infer import resolve_model_path  # 函数内导入，避免模块循环导入

        self.threshold = threshold
        # 两个声纹前端：ERes2NetV2（eres）+ CAM++（cam）
        # resolve_model_path：把 "iic/..." 这样的模型 ID 解析成磁盘真实路径（见 infer.py）
        self.eres_embedder = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
        self.cam_embedder = ModelScopeEmbedder(resolve_model_path(cam_model, model_root), device)

        # 加载训练端保存的裁判权重包。
        # torch.load 反序列化 .pt 文件；map_location="cpu" 表示先加载到内存（GPU 权重也能读）；
        # weights_only=False 是兼容性开关（旧格式 checkpoint 里可能带非张量对象）；
        # ["dual"] 取出字典里 "dual" 这个键对应的子字典。
        bundle = torch.load(str(backend_path), map_location="cpu", weights_only=False)["dual"]
        # mean/std：训练端统计好的 32 维特征均值与标准差（做 z-score 标准化用，见 score()）
        self.mean = np.asarray(bundle["mean"], dtype=np.float32)
        self.std = np.asarray(bundle["std"], dtype=np.float32)
        # 裁判 MLP 结构：32 -> 32 -> 16 -> 1
        #   nn.Linear(32,32)：全连接层，y = xW^T + b，把 32 维输入线性变换到 32 维
        #   nn.ReLU()：非线性激活，max(0, x)。没有它，多层线性叠加还是线性，白搭
        #   nn.Dropout(0.1)：训练时随机丢 10% 神经元防过拟合（eval 模式下自动失效）
        #   最后一层 Linear(16,1)：输出一个标量 logit（未归一化的"分"）
        self.mlp = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        # 训练端保存的键名带 "net." 前缀（当时模型外面包了一层），这里剥掉前缀再装载
        renamed = {k.replace("net.", ""): v for k, v in bundle["mlp"].items()}
        self.mlp.load_state_dict(renamed)  # 把权重逐层灌进刚搭好的结构（结构必须一模一样）
        self.mlp.eval()  # 切换评估模式：固定 Dropout/BN 行为，推理专用
        self.torch = torch  # 存一份引用，方便 score() 里用

    def _model_features(self, embedder: ModelScopeEmbedder, enroll_wav, test_wav, windows) -> list[float]:
        """对单个声纹前端计算 16 维特征。

        参数:
            embedder: ERes2NetV2 或 CAM++ 的嵌入器
            enroll_wav: 唤醒音频波形（作为"注册声纹"）
            test_wav:   识别音频波形（要判断里面有没有目标人）
            windows:    识别音频的滑窗片段列表（由 sliding_windows 切出）
        返回:
            16 个 float 组成的 list（顺序与 FEATURE_KEYS 一致）
        """
        # 一次性批量提取所有波形（唤醒 + 识别整段 + 所有窗口）的 embedding：
        #   同一批内模型只跑一次 GPU 前向，比一个个调快得多
        embeddings = embedder.extract_waveforms([enroll_wav, test_wav, *windows])
        # *windows 是"列表解包"：把 list 里的元素一个个展开成函数参数
        # 拆出前两个：唤醒嵌入（目标人参考）、识别整段嵌入
        emb_enroll, emb_test = embeddings[0], embeddings[1]
        # 每个窗口与唤醒嵌入的余弦相似度（窗口的 embedding 已被 L2 归一化，
        # 所以这里 cosine_similarity 内部实际上是在做归一化后的点积）
        win_scores = [cosine_similarity(emb_enroll, w) for w in embeddings[2:]]
        # 返回 16 维特征：
        #   [0] 整段余弦
        #   [1..10] _window_stats 展开的 10 个统计量（列表解包 * 拼进结果）
        #   [11..12] 唤醒/识别时长（秒）：len(wav)/采样率 = 秒数
        #   [13] 响度比：识别音频响度 / 唤醒响度（max(...,1e-6) 防除零）
        #   [14..15] 两端音频的削波比例
        return [
            cosine_similarity(emb_enroll, emb_test),
            *_window_stats(win_scores),
            len(enroll_wav) / SAMPLE_RATE,
            len(test_wav) / SAMPLE_RATE,
            _rms(test_wav) / max(_rms(enroll_wav), 1e-6),
            _clip_ratio(enroll_wav),
            _clip_ratio(test_wav),
        ]

    def score(self, wake_audio: Path, command_audio: Path) -> float:
        """主入口：给一对（唤醒音频, 识别音频）打一个 0~1 的目标存在概率。"""
        enroll_wav = read_wav(wake_audio)     # 唤醒音频 → 波形
        test_wav = read_wav(command_audio)    # 识别音频 → 波形
        windows = sliding_windows(test_wav)   # 识别音频切滑窗（唤醒音频不切，用整段）
        # 两个前端各算 16 维特征，list 相加拼成 32 维
        features = self._model_features(self.eres_embedder, enroll_wav, test_wav, windows)
        features += self._model_features(self.cam_embedder, enroll_wav, test_wav, windows)
        # z-score 标准化：(x - mean) / std —— 把每维特征缩放到"均值 0、标准差 1"的尺度。
        # 这套 mean/std 是训练时统计的，推理必须用同一套，否则 MLP 输入的分布对不上
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        # 前向推理：numpy 数组 → torch 张量 → unsqueeze(0) 加批维度（1, 32）→ MLP → 标量 logit
        with self.torch.no_grad():  # 不跟踪梯度（推理用，省显存）
            logit = self.mlp(self.torch.from_numpy(x).unsqueeze(0)).item()
        # sigmoid：把 (-inf, +inf) 的 logit 压到 (0, 1) 的概率
        #   公式 1/(1+e^(-z))；z=0 → 0.5；z 越大越接近 1
        return float(1.0 / (1.0 + np.exp(-logit)))

    def accept(self, wake_audio: Path, command_audio: Path) -> tuple[bool, float]:
        """判决：分数是否达到阈值。返回 (是否接受, 分数) 二元组。"""
        score = self.score(wake_audio, command_audio)
        return score >= self.threshold, score  # 元组打包返回（Python 一个 return 两个值）
