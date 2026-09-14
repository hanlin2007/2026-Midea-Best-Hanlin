"""目标说话人语音指令识别管线引擎（SZY 系统核心模块，无 CLI）。

本模块实现整条生产链路的全部决策逻辑（配置、融合门控、逐样本处理），
入口命令行放在 :mod:`szy.cli`，与引擎分离，方便脚本/测试直接 import 引擎。
（"无 CLI"是刻意分工：引擎只管"怎么做"，命令行只管"接用户输入"；
这样测试可以绕过 argparse 直接构造 SzyRunner，脚本也能复用同一引擎。）

模块内部三件套，自上而下组装：
  - :class:`SzyConfig`   frozen dataclass 配置：全部可调参数都在这里；
  - :class:`FusionGate`  融合门控打分器：一次调用产出"接受判定 + 融合分 +
                         特征明细"三样东西；
  - :class:`SzyRunner`   逐样本执行器：把"门控 → 转写 → 音色触发 → 候选
                         选择"整段决策串起来，并负责异常回退。

生产链路一句话：
    唤醒音频定位目标人 → 融合门控判定指令音频是否含目标人 →
    （接受时）dasynth 微调 MossFormer2 盲分离两路 → FireRed ASR 转写 →
    音色触发规则决定用原始转写还是分离通道转写，任何异常回退原始转写。

术语速览（初学者）：
  - 声纹 embedding：把一段语音压缩成固定维度的向量，"说话人越像、
    余弦相似度越接近 1"；ERes2NetV2 是声纹提取模型；
  - 门控（gate）：链路的第一道闸，判定"指令音频里有没有目标说话人"；
    拒绝的样本直接输出空转写（负样本的正确行为），也省下后续所有计算；
  - 盲分离：不给任何先验地把混合语音拆成两路 ms0/ms1，哪路是目标人
    并不知道，要由余弦打分判定；
  - 唤醒锚点（anchor）：从唤醒音频中定位出的"目标人嗓音的代表"，
    之后所有余弦比较都以它为参照。

三个关键设计（相对历史路线的演进，细节见 docs/CURRENT_BEST_PIPELINE_SZY_V4.md）：
1. 主门控 = 两个 ERes2NetV2 前端的线性融合门控（官方预训练 + 分离域微调，
   dataset_dasynth ep6），6 维余弦特征，阈值在 dasynth valid 上标定。
2. 指令候选用微调模型直接余弦（cos(anchor, ms0) vs cos(anchor, ms1)）
   选 ms0/ms1，与门控同一打分原理，且复用门控时刻已算的逐通道余弦，
   不增加额外 embedding 计算。
3. 切换触发 = 音色规则（confidence × 指令关键词 × sv_diff），其中
   sv_diff = max cos(anchor, cmd_ms) − cos(anchor, raw_cmd) 衡量"分离是否
   让目标嗓音比原始混合更突出"——这正是切换本身的物理机制。

依赖方向：本模块只允许向下依赖 szy/（共享工具）与 midea_qiji/（内核）；
torch 分离器在运行时按需延迟导入（保证无 torch 环境也能 --help / 单测）。
"""
# 未来式注解（PEP 604）：让 str | None、tuple[...] 这类新式类型标注在
# Python 3.10 下也能写；注解只作为字符串存在、运行时不求值，零性能开销。
# 必须放在除 docstring/注释外的第一条语句（Python 语法要求）
from __future__ import annotations

import json                                     # 配置 / 融合权重 JSON
from dataclasses import dataclass               # frozen dataclass（不可变配置）
from pathlib import Path                        # 面向对象路径
import time                                     # perf_counter 计时
from typing import Any                           # 动态类型字段的占位标注

import numpy as np                              # 向量化数值计算（点积 / 矩阵乘）

# ---- midea_qiji 内核（下层依赖，声纹/识别/数据模型/决策规则） ----
from midea_qiji.gate import ModelScopeEmbedder, read_wav
from midea_qiji.infer import (
    Sample,
    read_many_jsonl,
    submission_id,
)
from midea_qiji.moss_candidate import (
    select_wake_anchor,
    should_switch_command_timbre,
)
from midea_qiji.paths import resolve_model_path
from midea_qiji.recognizers import FireRedLLMRecognizer, FireRedRecognizer
# ---- szy 共享工具（同层 helpers） ----
from szy.helpers import (
    command_features,
    duration_seconds,
    separation_paths_for,
    transcribe_firered_rows,
)

# 融合门控使用的 6 维特征名（顺序固定，与 model/artifacts/gate_fusion.json 一致，
# 顺序不符会直接报错，防止门控权重与特征错位）。
# 各特征含义（"pre"=官方预训练前端，"ft"=分离域微调前端，同一前端各出 3 个）：
#   {tag}_raw      唤醒与指令两段"原始混合"之间的余弦（分离前的基线相似度）；
#   {tag}_sep      唤醒锚点与"更好的那条分离通道"之间的余弦（分离后目标人浓度）；
#   ch_sim_{tag}   两条分离通道彼此之间的余弦（分离得越开，两路越不相似）。
# 顺序即权重向量的维度含义：JSON 里权重按此顺序排布，代码取特征也必须
# 按此顺序——这就是"隐形契约"，由 FusionGate 加载时校验。
FUSION_FEATURE_KEYS = [
    "pre_raw", "ft_raw", "pre_sep", "ft_sep", "ch_sim_pre", "ch_sim_ft",
]


@dataclass(frozen=True)
class SzyConfig:
    """生产管线配置（frozen dataclass：一次加载后不可变，防止运行中被误改）。

    frozen 的动机：配置是一次性"契约"——整个评测批次期间，阈值、模型
    路径必须保持同一份，否则前半批与后半批口径不一致、结果无法复现。
    frozen 后对字段赋值会当场抛 FrozenInstanceError，把误改挡在运行时，
    而不是让 bug 悄悄流进结果。想"改一个字段得到新配置"，用
    ``SzyConfig(**{**old.__dict__, "字段": 新值})`` 的"复制+覆盖"惯用法
    （见 szy.cli 对 --model-root 的处理）：``__dict__`` 取出全部字段值
    组成字典，展开后覆盖个别键，再重新构造——旧实例保持原样，新实例
    只差一个字段。

    字段全部有默认值，JSON 配置只需覆盖需要的项；未知键会报错（见
    :func:`load_config`），避免拼写错误的配置项被静默忽略。
    """

    device: str = "cuda:0"                 # 主计算设备（分离器 / ASR）；无 GPU 时分离器会自动落到 CPU（见 separator）
    sv_device: str | None = None           # 声纹前端设备；None = 跟随 device（显存紧张时可单独指定）
    model_root: str | None = None          # ModelScope 模型根目录；None = 仓库内默认
    sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"   # 声纹基座模型 id
    # ---- 融合门控三件套（权重文件由微调/标定流程产出） ----
    fusion_backend: str = "model/artifacts/gate_fusion.json"      # 线性融合权重（mean/std/weight/bias）
    finetuned_sv_ckpt: str = "model/artifacts/ep6_datasetA_best.pt"  # 分离域微调 ERes2NetV2 权重
    fusion_threshold: float = -0.6469      # 融合分接受阈值（dasynth valid 标定）；score >= 阈值才接受
    # ---- FireRed ASR 资产 ----
    firered_model: str = "model/external/FireRedASR2-AED"         # AED 模型权重目录
    firered_repo: str = "model/external/FireRedASR2S"             # 推理源码仓库
    firered_vendor: str = "model/runtime/vendor"                  # 运行时依赖（kaldi_native_fbank）
    # ASR 后端："aed" = FireRedASR2-AED（生产基线）；"llm" = FireRedASR2-LLM（Qwen2-7B 解码）
    firered_asr_type: str = "aed"
    firered_llm_batch_size: int = 4       # LLM 后端批量大小
    # ---- 分离器 ----
    moss_model: str = "model/artifacts/mossformer2_dasynth_sweep_ep6.pt"  # dasynth 微调分离器权重
    # ---- 输出口径 ----
    id_mode: str = "relative_stem"        # 样本 id 生成方式（竞赛提交格式要求）
    separation_cache: str = "outputs/szy_cache/separation"  # 分离结果 wav 缓存目录
    batch_writes: int = 1                 # 每批样本后强制写出（checkpoint 粒度）


def load_config(path: str | Path) -> SzyConfig:
    """从 JSON 加载配置；出现配置类里没有的键时立即报错（拼写检查器）。

    "拼写检查器"的意义：JSON 里把 fusion_threshold 误写成
    fusion_threshlod 时，若静默忽略，用户会以为阈值改成功了、实际跑的
    还是默认值——这类静默错误在调参时最致命。这里用 dataclass 的字段表
    逐一核对，任何多余键立刻 ValueError，把错误暴露在启动时刻（fail fast）。
    """
    # Path.read_text(encoding="utf-8")：以明确编码读文件（显式指定 UTF-8，
    # 避免 Windows 默认编码（如 GBK）读中文配置出错）；json.loads 解析成 dict
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # __dataclass_fields__：dataclass 自动维护的"字段名 → Field 元信息"表，
    # 取其键就是全部合法配置项
    fields = SzyConfig.__dataclass_fields__
    # 集合差集：payload 有、字段表没有的键 = 拼写错误或过时配置项；
    # sorted 让报错信息稳定可读（集合本身无序）
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    # **payload 把 JSON 键值对展开为关键字参数构造配置对象；
    # 未提供的字段自动使用类里的默认值——所以 JSON 只需写要改的项
    return SzyConfig(**payload)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """L2 归一化向量间的余弦相似度（embedding 已归一化时即点积）。

    余弦相似度定义 = 点积 / (两向量模长之积)。ERes2NetV2 前端输出的
    embedding 已经 L2 归一化（模长 = 1，见 midea_qiji.gate），分母恒为 1，
    因此这里只剩点积。np.dot 对一维数组 = 逐元素相乘再求和；用 numpy
    一行表达而不是 Python for 循环，是"向量化"：循环在 C 层批量执行，
    快几个数量级。
    """
    return float(np.dot(left, right))


class FusionGate:
    """双前端线性融合门控：预训练 + 分离域微调 ERes2NetV2 的 6 维余弦特征。

    为什么要"双前端"？官方预训练模型见识过海量干净数据（通用性好），
    但没见过竞赛的混合/分离音频；在 dasynth 分离域上微调过的副本更懂
    "分离通道里的目标嗓音"。把两个前端的 6 维余弦特征拼起来做线性融合，
    等于让一个简单的 logistic 打分器自动学会"什么时候信哪个前端"。
    线性融合参数（mean/std/weight/bias）由标定流程在 dasynth valid 上拟合。

    一次 ``score`` 调用同时产出三样东西（复用同批 embedding，不重复推理）：
      1. 接受/拒绝判定（融合分 >= 阈值）；
      2. 融合分 score；
      3. 全部逐通道余弦（供候选选择与音色触发复用）。

    声纹推理（两个大模型各提 5 段波形的 embedding）是全链路最贵的一步，
    所以设计原则是"一次推理的产物榨干用尽"：门控、候选选择、音色触发
    共享同一批 embedding，绝不为同一段音频重复提特征。
    """

    def __init__(
        self,
        sv_model: str,
        finetuned_ckpt: str | Path,
        fusion_backend: str | Path,
        threshold: float,
        device: str,
        model_root: str | Path | None = None,
    ) -> None:
        # 延迟导入 torch：本模块顶部不 import torch，只有真正构造门控
        # （= 真正要跑推理）时才需要。好处：无 torch 的环境也能
        # import szy.pipeline、跑 --help 和单测。import 语句写在函数体内，
        # 执行时机就从"模块导入时"推迟到"函数被调用时"
        import torch

        self.threshold = float(threshold)
        # 两个前端：官方预训练权重 与 分离域微调权重（同一基座结构，只覆盖 encoder）。
        # resolve_model_path 在 model_root（或默认位置）下定位 ModelScope
        # 模型目录，返回可直接加载的本地路径
        self.pretrained = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
        self.finetuned = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
        # torch.load 把 .pt 反序列化：map_location="cpu" 先在 CPU 内存加载
        # （避免加载瞬间占用 GPU 显存峰值，之后再整体搬设备）；
        # weights_only=False 信任本地自产存档（官方新版默认 True 会拒绝
        # 历史格式里的自定义对象，这里显式放开）
        checkpoint = torch.load(str(finetuned_ckpt), map_location="cpu", weights_only=False)
        # 只取存档里的 encoder 权重装进微调前端（存档可能还有别的键）。
        # load_state_dict 按参数名逐一搬运；名字对不上会报错（strict 默认
        # True）——同 separator 里的考量：宁可当场报错，不要静默漏加载
        self.finetuned.pipeline.model.embedding_model.load_state_dict(checkpoint["encoder"])

        # 线性融合参数（在 dasynth 上标定的 logistic：score = (x-mean)/std @ w + b）
        bundle = json.loads(Path(fusion_backend).read_text(encoding="utf-8"))
        if list(bundle["features"]) != FUSION_FEATURE_KEYS:
            # 特征顺序是"隐形契约"：权重向量的每一维对应固定含义的特征。
            # 顺序错位 = 权重装错维度，分数全错且极难排查，宁可当场报错
            raise ValueError(f"fusion backend 特征顺序不符：{bundle['features']}")
        # np.asarray 把 JSON 列表转 numpy 数组；dtype 统一 float32，
        # 与后面特征向量 dtype 对齐，避免隐式类型转换（float64 与 float32
        # 混算会拖慢 GPU/CPU 的向量化路径）
        self.mean = np.asarray(bundle["mean"], dtype=np.float32)
        self.std = np.asarray(bundle["std"], dtype=np.float32)
        self.weight = np.asarray(bundle["weight"], dtype=np.float32)
        self.bias = float(bundle["bias"])

    def score(
        self,
        anchor_wave: np.ndarray,
        raw_wake_wave: np.ndarray,
        raw_command_wave: np.ndarray,
        command0_wave: np.ndarray,
        command1_wave: np.ndarray,
        command_source_empty: bool,
    ) -> tuple[bool, float, dict[str, float]]:
        """对五段波形打分，返回 (是否接受, 融合分, 特征明细)。

        command_source_empty=True（分离通道转写为空）时，把原始指令也作为
        候选一并参与"分离收益"评估，避免空通道污染门控输入。

        类型标注的价值：``tuple[bool, float, dict[str, float]]`` 精确描述
        三个槽位各自是什么，读代码的人不用猜，IDE/静态检查工具也能据此
        提前发现误用（比如把返回值第一个元素当分数算）。
        """
        # 五段波形固定顺序送入两个前端：唤醒锚点、原始唤醒、原始指令、
        # 分离通道 0、分离通道 1
        waves = [anchor_wave, raw_wake_wave, raw_command_wave, command0_wave, command1_wave]
        values: dict[str, float] = {}
        # 两个前端各跑一遍，产出同名的特征集合，用 tag 前缀区分（pre_/ft_）。
        # 循环代替复制粘贴：前端只换实例、其余逻辑完全一致，
        # 以后加第三个前端也只需改元组
        for tag, embedder in (("pre", self.pretrained), ("ft", self.finetuned)):
            # extract_waveforms 一次性提 5 段波形的 embedding（内部按长度
            # 分组批量推理），比逐段调用少 4 次模型进出开销
            e_anchor, e_raw_wake, e_raw_cmd, e_cmd0, e_cmd1 = embedder.extract_waveforms(waves)
            # 分离收益候选：两条分离通道各自与唤醒锚点的余弦
            candidates = [_cosine(e_anchor, e_cmd0), _cosine(e_anchor, e_cmd1)]
            if command_source_empty:
                # 分离通道转写为空说明分离没产出可用内容；把原始指令也拉
                # 进候选竞争，防止"空通道假赢"把门控输入带偏
                candidates.append(_cosine(e_anchor, e_raw_cmd))
            values[f"{tag}_raw"] = _cosine(e_raw_wake, e_raw_cmd)
            values[f"{tag}_sep"] = max(candidates)
            values[f"ch_sim_{tag}"] = _cosine(e_cmd0, e_cmd1)
            # 候选选择（ft 直接余弦）复用，避免二次提 embedding——
            # 这 4 个额外余弦也一并存下，process() 里取用，不再重算
            values[f"{tag}_cos_cmd0"] = _cosine(e_anchor, e_cmd0)
            values[f"{tag}_cos_cmd1"] = _cosine(e_anchor, e_cmd1)
            # 音色触发器的分离收益量：cos(anchor, raw_cmd)
            values[f"{tag}_rawcmd"] = _cosine(e_anchor, e_raw_cmd)
        # ---- 融合打分：z-score 标准化 + 线性加权 ----
        # 按 FUSION_FEATURE_KEYS 的固定顺序取 6 个特征组成向量 x
        #（顺序与权重 JSON 的契约，已在 __init__ 校验）。
        # (x - mean) / std 是 z-score 标准化（减均值、除标准差）：把不同
        # 量纲/尺度的特征拉到同一尺度，否则数值大的特征天然主导权重。
        # 这是 numpy 广播：6 维向量与 6 维数组逐元素相减/相除
        x = (np.asarray([values[key] for key in FUSION_FEATURE_KEYS], dtype=np.float32) - self.mean) / self.std
        # x @ self.weight：numpy 矩阵乘运算符，一维向量 @ 一维向量 = 点积，
        # 再加 bias 得到融合分（z-score 空间里的线性判别值，非概率）
        score = float(x @ self.weight + self.bias)
        return score >= self.threshold, score, values


class SzyRunner:
    """逐样本管线执行器：门控 → 音色触发 → 候选选择，异常一律回退原始转写。

    一个 runner 实例绑定一份配置与一个输出目录，对每个 :class:`Sample`
    调用一次 :meth:`process`，返回可序列化的结果字典（detail.json 一行）。

    面向对象的设计动机：重量级资源（两个声纹模型、分离器、ASR）在
    __init__ 里加载一次，之后 process 只做纯计算。若每处理一个样本都
    重新加载模型（每个模型几百 MB、加载要几十秒），评测几千条样本的
    时间会几乎全耗在加载上——"加载一次、复用多次"是批处理系统的通则。
    """

    def __init__(self, config: SzyConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        # 分离缓存目录：配置里常写相对路径（outputs/szy_cache/separation），
        # 这里统一转成"当前工作目录下的绝对路径"。为什么必须统一：
        # 缓存命中的前提是路径每次都能对上；相对路径随启动目录漂移，
        # 会造成"明明缓存过却找不到、反复重算"
        self.cache_dir = Path(config.separation_cache)
        if not self.cache_dir.is_absolute():
            self.cache_dir = (Path.cwd() / self.cache_dir).resolve()
        # 门控最先构造：它决定大部分样本的去留（拒绝即止步，不需要分离），
        # 是链路的第一道也是最重要的闸。声纹前端设备默认跟随 device，
        # sv_device 单独指定时可分卡部署
        self.gate = FusionGate(
            config.sv_model,
            config.finetuned_sv_ckpt,
            config.fusion_backend,
            config.fusion_threshold,
            config.sv_device or config.device,
            config.model_root,
        )
        # 分离器延迟导入：szy.separator 依赖 torch + vendor 模型定义，
        # 延迟到构造时再 import，保证 --help / 单测在无 torch 环境可用。
        # 注意 import 位置的语义：顶层 import 在"模块被 import"时执行，
        # 函数体内 import 在"函数被调用"时执行——这里推迟的是 torch 依赖
        from szy.separator import MossFormer2Separator

        self.separator = MossFormer2Separator(config.moss_model, config.device)
        # ASR 后端二选一："aed" 生产基线（快、稳）；"llm" 用 Qwen2-7B 解码
        # （上限更高但慢得多，且需要更多显存）。self.asr 标注为 Any：
        # 两种识别器不是同一类型，但都暴露 transcribe_many 与 _model.transcribe
        # 接口（鸭子类型），管线按统一接口使用，类型标注上放宽容许两者
        if config.firered_asr_type == "llm":
            self.asr: Any = FireRedLLMRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
                batch_size=config.firered_llm_batch_size,
            )
        else:
            self.asr = FireRedRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
            )

    def _paths_for(self, sample: Sample) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
        """转发给 helpers.separation_paths_for 计算 (唤醒分离路径对, 指令分离路径对)。

        单独封装成方法的理由：把"路径怎么算"的知识留在 helpers 一处，
        这里只负责把本 runner 的配置（id_mode、缓存目录、分离器实例）
        喂进去；以后改命名规则只动 helpers，不碰 pipeline。
        """
        return separation_paths_for(sample, self.config.id_mode, self.cache_dir, self.separator)

    def process(self, sample: Sample) -> dict[str, Any]:
        """处理一个样本，返回 detail.json 的一行（所有字段都可 JSON 序列化）。

        执行顺序（每个阶段都可能提前返回）：
          1. 算分离缓存路径并批量转写唤醒/指令共 4 段音频；
          2. 选唤醒锚点（哪段唤醒音频最像目标人）；
          3. 融合门控打分；拒绝则带着 route=fusion_reject 直接返回；
          4. 转写原始指令，算字速/关键词/分离收益，判定是否切分离通道；
          5. 不切换则用原始转写返回；切换则用 ft 直接余弦选 ms0/ms1。

        贯穿全程的两个兜底原则：
          - 结果字典在 try 之前就预填好全部字段的默认值——任何环节抛异常，
            外部拿到的永远是"完整的一行"，下游汇总不会因缺键而崩；
          - 异常时先回退原始转写、再回退置空（两层），样本永不"消失"。
        """
        # perf_counter 是单调时钟：只保证"差值"有意义，不受系统校时/时区
        # 影响，专用于测耗时。不要用 time.time() 计时——它可能被 NTP 对时
        # 拨动，出现负耗时
        started = time.perf_counter()
        # submission_id：按竞赛口径由样本路径生成唯一 id（id_mode 决定口径，
        # 如 relative_stem = 相对路径去掉扩展名），提交物按此 id 对齐
        item_id = submission_id(sample, self.config.id_mode)
        # 结果行：全部字段先给默认值。detail.json 每一行的形状由此固定，
        # 下游（reporting/离线分析）可以放心取键；未走到的分支保持默认——
        # 例如被门控拒绝的样本 route 保持 "fusion_reject"、转写保持空串
        result: dict[str, Any] = {
            "id": item_id,
            "content": "",
            "label": sample.label,
            # sample_type：有标注=正样本（必须转写对），无标注=负样本（应拒绝）
            "sample_type": "positive" if sample.label is not None else "negative",
            "sv_score": "",
            "accepted": False,
            "sv_accept_threshold": f"{self.config.fusion_threshold:.6f}",
            "route": "fusion_reject",
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
        try:
            # _paths_for 内部会真实触发分离（或命中缓存直接返回旧路径）
            (wake_paths, command_paths) = self._paths_for(sample)
            # 一次批量转写 4 段音频：唤醒两通道 + 指令两通道。批量调用
            # 摊薄每次调用的固定开销（模型状态准备、结果组装等）
            wake0, wake1, command0, command1 = transcribe_firered_rows(
                self.asr,
                [wake_paths[0], wake_paths[1], command_paths[0], command_paths[1]],
                "szy",
            )
            # 门控唤醒锚点：生产同一规则（一路为空回退原始 kws）。
            # 锚点的作用：唤醒音频里可能同时录进了目标人和家人/噪声，
            # 分离成两路后用文本编辑距离找出"念了唤醒词的那一路"作为
            # 目标人嗓音的代表，后续余弦比较都对着它做
            # 注：曾验证"非空通道直选"（PR/CER 更优），但在计分公式 (1−CER+RR)/2 下
            # RR 权重高，raw 回退版公式分更高（0.7977 vs 0.7939），故保持 raw 回退。
            gate_anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
            # 回退时锚点用原始唤醒音频；否则用选中的分离通道音频。
            # gate_anchor 是 WakeAnchorDecision（frozen dataclass，
            # 字段见 midea_qiji.moss_candidate），不可变、只读决策结果
            anchor_path = sample.wake_audio if gate_anchor.raw_wake_fallback else wake_paths[gate_anchor.selected_source]
            # 任一路分离转写为空 → 分离没产出可用内容，门控会把原始指令
            # 纳入候选（见 FusionGate.score 的 command_source_empty 分支）
            command_source_empty = not command0["text"] or not command1["text"]

            # 融合门控打分：五段波形全部读进内存。read_wav 统一读到
            # 16k 单声道 float32（每秒约 64KB，几秒语音完全放得下）
            accepted, score, gate_values = self.gate.score(
                read_wav(anchor_path),
                read_wav(sample.wake_audio),
                read_wav(sample.command_audio),
                read_wav(command_paths[0]),
                read_wav(command_paths[1]),
                command_source_empty,
            )
            # 把门控产物写进结果行。数值统一 f"{:.6f}" 格式化成字符串：
            # JSON 里浮点会带长尾巴且各平台打印不一致，字符串格式由我们
            # 掌控，离线重扫阈值的脚本也好解析
            result.update({
                "sv_score": f"{score:.6f}",
                "accepted": bool(accepted),
                "gate_features": {k: f"{gate_values[k]:.6f}" for k in FUSION_FEATURE_KEYS},
                "wake_selected_source": int(gate_anchor.selected_source),
                "wake_raw_fallback": bool(gate_anchor.raw_wake_fallback),
                "command_source_empty": bool(command_source_empty),
            })
            if not accepted:
                # 门控拒绝：不跑指令 ASR、不做切换，直接带默认空转写返回。
                # 这是全链路最大的计算量节省点——负样本与明显不含目标的
                # 样本在此止步，后面的分离/转写/打分全部跳过
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # ---- 接受分支：raw ASR + 音色触发 ----
            # 对"原始指令音频"（未分离的混合）转写：这是默认输出口径，
            # 只有音色触发成功才会被分离通道转写替换
            raw_command = transcribe_firered_rows(self.asr, [sample.command_audio], "szy")[0]
            # 字速 = 归一化转写字数 / 音频秒数；关键词分 = 命中 CMD_KW 次数。
            # 二者刻画"转写像不像一条真实指令"（噪声转写往往又短又怪），
            # 与置信度一起进入音色触发规则
            chars_per_sec, command_keyword_score = command_features(
                raw_command["text"], duration_seconds(sample.command_audio)
            )
            # sv_diff：分离后目标嗓音浓度 − 原始混合中的浓度（复用门控已算余弦）。
            # 若分离真的让目标人更突出，sv_diff 显著为正；为负说明分离
            # 反而把目标人抹掉了，此时绝不该切换到分离通道
            sv_diff = max(gate_values["ft_cos_cmd0"], gate_values["ft_cos_cmd1"]) \
                - gate_values["ft_rawcmd"]
            # 音色触发三要素（规则本体在 midea_qiji.moss_candidate）：
            # 转写置信度、指令像度、分离收益，三者组合决定是否把输出
            # 从原始转写换成分离通道转写
            switch = should_switch_command_timbre(
                confidence=raw_command["confidence"],
                command_keyword_score=command_keyword_score,
                sv_diff=sv_diff,
            )
            result.update({
                "content": raw_command["text"],
                # route 记录"这条样本实际走了哪条路"：切换成功记
                # moss_candidate，未触发记 raw——汇总里的路由分布全靠它
                "route": "moss_candidate" if switch else "raw",
                "selected_candidate": "orig",
                "selection_reason": "candidate_pending" if switch else "timbre_switch_rule_not_triggered",
                "switch_triggered": bool(switch),
                "raw_confidence": f"{raw_command['confidence']:.6f}",
                "chars_per_sec": f"{chars_per_sec:.6f}",
                "command_keyword_score": f"{command_keyword_score:.6f}",
                "sv_diff": f"{sv_diff:.6f}",
                # 供触发阈值离线重扫（与 switch 分支重复写入，值一致）。
                # 这些"过程量"不直接进提交物，但 detail.json 里有了它们，
                # 事后可以用同一份结果重跑不同阈值、不必重新推理——
                # 推理很贵，过程数据便宜，这是实验系统的通用做法
                "raw_text": raw_command["text"],
                "command_source0_text": command0["text"],
                "command_source1_text": command1["text"],
                "ft_cos_ms0": f"{gate_values['ft_cos_cmd0']:.6f}",
                "ft_cos_ms1": f"{gate_values['ft_cos_cmd1']:.6f}",
            })
            if not switch:
                # 音色规则未触发：用原始转写收尾（最常见路径之一）
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # 候选选择：ft 直接余弦（与门控同一打分原理，复用门控已算的逐通道余弦）
            # 实测与历史 eres+cam 差分打平，但逻辑更一致，且避开对比通道污染。
            anchor = gate_anchor
            selected = "orig"
            reason = "candidate_pending"
            if command_source_empty:
                # 分离通道没转写出任何内容：即使触发了切换也无可换，
                # 保持原始转写（不切换内容，但记录原因供事后分析）
                reason = "command_source_empty"
            else:
                # 两条分离通道里，与唤醒锚点余弦更大的那路判为目标人通道。
                # >= 保证平局时选 ms0（与历史行为一致）
                selected = "ms0" if gate_values["ft_cos_cmd0"] >= gate_values["ft_cos_cmd1"] else "ms1"
                reason = "ft_direct_cosine"
            # 三选一映射表：字典比 if/elif 链更直观，候选名 → 转写文本
            text_by_candidate = {
                "orig": raw_command["text"],
                "ms0": command0["text"],
                "ms1": command1["text"],
            }
            result.update({
                # selected 为 "orig" 时直接用原始转写；否则取对应分离通道文本
                "content": text_by_candidate[selected] if selected != "orig" else raw_command["text"],
                "selected_candidate": selected,
                "selection_reason": reason,
                # 唤醒锚点的过程量一并落盘（对比通道、文本平局标志、
                # 各通道转写文本），供离线分析锚点选择质量
                "wake_contrast_source": int(anchor.contrast_source),
                "wake_text_tie": bool(anchor.text_tie),
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                "command_source0_text": command0["text"],
                "command_source1_text": command1["text"],
                "ft_cos_ms0": f"{gate_values['ft_cos_cmd0']:.6f}",
                "ft_cos_ms1": f"{gate_values['ft_cos_cmd1']:.6f}",
            })
        except Exception as error:  # preserve raw ASR as the operational fallback
            # 生产兜底：任何环节异常都不让样本"消失"，先试原始转写，再试置空。
            # 为什么兜底是生产系统的关键：一条样本崩溃若直接让整批评测挂掉，
            # 深夜跑 2000 条到第 1900 条时前功尽弃。兜底后最坏情况只是这一
            # 条样本的空转写（扣一点分），批次继续、checkpoint 照常落盘。
            # 捕获最宽泛的 Exception（连底层库的意外错误也收），并把异常
            # 类型和消息记进 error 字段，事后可按类型统计排查
            result["error"] = f"{type(error).__name__}: {error}"
            try:
                # 第一层回退：绕开管线细节，用 ASR 的简易接口直接转写
                # 原始指令音频——"原始转写"在多数异常下仍然可用
                result["content"] = self.asr.transcribe_many([sample.command_audio])[0]
                result["route"] = "raw_fallback_error"
                result["selected_candidate"] = "orig"
            except Exception as raw_error:
                # 第二层回退：连原始转写都失败（ASR 进程死亡、音频损坏等），
                # 置空转写保底并把两段异常信息拼接进 error——两层回退
                # 保证"总有结果可写"，且排查时能看到完整的异常链
                result["error"] += f"; raw_fallback={type(raw_error).__name__}: {raw_error}"
                result["route"] = "empty_fallback_error"
        # 无论正常路径还是异常路径，耗时字段都在最后统一写入。
        # 注意 try 里的正常分支各自提前 return 前也会写 item_duration，
        # 所以每行结果都保证有这个字段（统计总耗时/单条耗时都靠它）
        result["item_duration"] = f"{time.perf_counter() - started:.6f}"
        return result
