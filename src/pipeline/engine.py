"""目标说话人语音指令识别管线引擎（系统核心模块，无 CLI）。

本模块实现整条生产链路的全部决策逻辑（配置、声学路由、融合门控、
逐样本处理），入口命令行放在 :mod:`pipeline.cli`，与引擎分离，方便脚本/测试
直接 import 引擎。（"无 CLI"是刻意分工：引擎只管"怎么做"，命令行只管
"接用户输入"；这样测试可以绕过 argparse 直接构造 PipelineRunner，脚本也能
复用同一引擎。）

模块内部四件套，自上而下组装：
  - :class:`PipelineConfig`   frozen dataclass 配置：全部可调参数都在这里；
  - :class:`AsrCache`    ASR 转写缓存：同一音频在同一解码配置下只解码一次；
  - :class:`FusionGate`  融合门控打分器：一次调用产出"接受判定 + 融合分 +
                         特征明细"三样东西；
  - :class:`PipelineRunner`   逐样本执行器：把"声学路由 → 门控 → 转写 → 选择 →
                         后处理"整段决策串起来，并负责异常回退。

生产链路一句话：
    MossFormer2 盲分离唤醒/指令音频 → 声学单双人路由（双单人走独立短路：
    微调声纹 raw 余弦判同人）→ 拼音锚点定位目标人 → 融合门控接受/拒绝 →
    单人指令只解码原始音频、双人指令只解码 ft 余弦选中的分离通道 →
    选中通道上做保守 MBR N-best 重排 + ELM 重解码 → 任何异常回退原始转写。

术语速览（初学者）：
  - 声纹 embedding：把一段语音压缩成固定维度的向量，"说话人越像、
    余弦相似度越接近 1"；ERes2NetV2 是声纹提取模型；
  - 门控（gate）：链路的第一道闸，判定"指令音频里有没有目标说话人"；
    拒绝的样本直接输出空转写（负样本的正确行为），也省下后续所有计算；
  - 盲分离：不给任何先验地把混合语音拆成两路 ms0/ms1，哪路是目标人
    并不知道，要由余弦打分判定；
  - 唤醒锚点（anchor）：从唤醒音频中定位出的"目标人嗓音的代表"，
    之后所有余弦比较都以它为参照；
  - ELM（外部语言模型）：纠正 ASR 的领域词汇偏向；只作选择后的重解码，
    不参与任何决策，以免污染置信度量程。

三个关键设计（演进细节见 docs/OPTIMIZATION.md）：
1. 主门控 = 两个 ERes2NetV2 前端的线性融合门控（官方预训练 + 分离域微调，
   dataset_dasynth ep6），6 维余弦特征，阈值在 dasynth valid 上标定；
   两个前端共享同一次 Fbank 声学特征（prepare_waveforms 只跑一遍）。
2. 指令路由 = 声学单双人分类（pipeline.single_speaker，纯波形特征、不花 ASR）：
   双单人走 raw/余弦短路；单人指令解码原始音频；双人指令恒切分离通道，
   ft 直接余弦选源、且只解码选中的那一路（延迟解码，省一半命令 ASR）。
3. 所有承载决策的转写一律 ELM 关闭（temporary_elm_weight(0.0)），
   保证 confidence 量程干净；N-best 重排与 ELM 重解码只在路由选定后
   对最终通道各跑一次。唤醒与命令共享同一份 AED 权重（峰值显存
   4818 MiB，约为双实例时代的 2/3）。

依赖方向：本模块只允许向下依赖 pipeline/（同层工具）与 core/（内核）；
torch 分离器在运行时按需延迟导入（保证无 torch 环境也能 --help / 单测）。
"""
# 未来式注解（PEP 604）：让 str | None、tuple[...] 这类新式类型标注在
# Python 3.10 下也能写；注解只作为字符串存在、运行时不求值，零性能开销。
# 必须放在除 docstring/注释外的第一条语句（Python 语法要求）
from __future__ import annotations

import json                                     # 配置 / 融合权重 JSON / ASR 缓存
from dataclasses import dataclass               # frozen dataclass（不可变配置）
from pathlib import Path                        # 面向对象路径
import time                                     # perf_counter 计时
from typing import Any                           # 动态类型字段的占位标注

import numpy as np                              # 向量化数值计算（点积 / 矩阵乘）

# ---- core 内核（下层依赖，声纹/识别/数据模型/决策规则） ----
from core.nbest import conservative_mbr_rerank
from core.audio import read_wav
from core.embeddings import ModelScopeEmbedder
from core.samples import (
    Sample,
    submission_id,
    suppress_model_output,
)
from core.wake_anchor import select_wake_anchor
from core.model_paths import resolve_model_path
from core.firered import FireRedLLMRecognizer, FireRedRecognizer
from core.text import normalize_transcript
# ---- pipeline 同层工具（helpers + 声学单双人路由） ----
from pipeline.helpers import separation_paths_for
from pipeline.single_speaker import BothSingleRouter

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
class PipelineConfig:
    """生产管线配置（frozen dataclass：一次加载后不可变，防止运行中被误改）。

    frozen 的动机：配置是一次性"契约"——整个评测批次期间，阈值、模型
    路径必须保持同一份，否则前半批与后半批口径不一致、结果无法复现。
    frozen 后对字段赋值会当场抛 FrozenInstanceError，把误改挡在运行时，
    而不是让 bug 悄悄流进结果。想"改一个字段得到新配置"，用
    ``PipelineConfig(**{**old.__dict__, "字段": 新值})`` 的"复制+覆盖"惯用法
    （见 pipeline.cli 对 --model-root 的处理）：``__dict__`` 取出全部字段值
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
    # ---- 主解码 beam 配置（唤醒通道用 temporary_beam_size(1) 临时压回 1） ----
    firered_beam_size: int = 3            # 命令通道主解码束宽（beam=3 顺带产出 N-best 候选）
    # ---- N-best 保守 MBR 重排（路由选定后对选中通道执行一次） ----
    firered_nbest_mode: str = "off"       # "off" or "conservative_mbr"
    firered_nbest_min_confidence_gain: float = 0.03   # 换人闸 1：最低置信度增益
    firered_nbest_min_mbr_risk_reduction: float = 0.05  # 换人闸 2：最低风险下降
    # ---- ELM 外部语言模型（只作选择后重解码，不参与主解码/决策） ----
    firered_elm_dir: str | None = None    # ELM 模型目录；None = 不启用
    firered_elm_weight: float = 0.0       # ELM 融合权重（生产 0.15）
    # ELM 重解码使用的 beam。ELM 不再参与主解码，只在路由选定后对选中
    # 通道重解码一次，因此它的 beam 与主解码的 firered_beam_size 解耦。
    firered_elm_beam_size: int = 3
    # ---- 声学单双人路由（双单人独立短路；生产显式开启） ----
    both_single_mode: str = "off"         # "off"（关闭短路，全走门控）or "sft_raw_cosine"
    both_single_backend: str = "configs/single_speaker_router.json"  # 20 维 logistic 分类器 bundle
    both_single_sft_threshold: float = 0.29  # 双单人分支的同人余弦阈值
    # ---- 分离器 ----
    moss_model: str = "model/artifacts/mossformer2_dasynth_sweep_ep6.pt"  # dasynth 微调分离器权重
    # ---- 输出口径 ----
    id_mode: str = "relative_stem"        # 样本 id 生成方式（竞赛提交格式要求）
    separation_cache: str = "outputs/cache/separation"  # 分离结果 wav 缓存目录
    batch_writes: int = 100               # 每累计 N 条新结果写一次盘（checkpoint/ASR 缓存粒度）
    # False 时只保留进程内 ASR 缓存（供同一条音频的 N-best 复用），
    # 不读写 asr_cache.json（交付/评测口径：结果只依赖当次运行）
    asr_cache_persistent: bool = True


def load_config(path: str | Path) -> PipelineConfig:
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
    fields = PipelineConfig.__dataclass_fields__
    # 集合差集：payload 有、字段表没有的键 = 拼写错误或过时配置项；
    # sorted 让报错信息稳定可读（集合本身无序）
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    # **payload 把 JSON 键值对展开为关键字参数构造配置对象；
    # 未提供的字段自动使用类里的默认值——所以 JSON 只需写要改的项
    return PipelineConfig(**payload)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """L2 归一化向量间的余弦相似度（embedding 已归一化时即点积）。

    余弦相似度定义 = 点积 / (两向量模长之积)。ERes2NetV2 前端输出的
    embedding 已经 L2 归一化（模长 = 1，见 core.gate），分母恒为 1，
    因此这里只剩点积。np.dot 对一维数组 = 逐元素相乘再求和；用 numpy
    一行表达而不是 Python for 循环，是"向量化"：循环在 C 层批量执行，
    快几个数量级。
    """
    return float(np.dot(left, right))


class AsrCache:
    """ASR 转写缓存：键 = ``<音频路径>|<解码配置标签>``，值 = 转写结果。

    同一段音频会在多种解码配置下被转写——主解码（ELM 关、beam=3）、
    唤醒解码（beam=1）、ELM 重解码（ELM 开）——所以键必须带配置标签
    （如 ``b3|elm0``），只按路径做键会把"别的配置解出来的文本"错当成
    本配置的结果（静默污染，最难排查的那类 bug）。

    两层用法：
      - 进程内缓存始终开启：同一次运行里，主解码存下的 N-best 候选可被
        后处理直接复用，不必对同一音频二次解码；
      - 落盘持久化由配置控制（asr_cache_persistent）：开启时跨运行复用，
        调参重跑 ~10x 提速；评测/交付口径下关闭，保证结果只依赖当次运行。

    写盘是批量的（每 batch_writes 条新结果写一次）：缓存 JSON 动辄几千条，
    每条解码都全量重写一次会把 IO 放大成瓶颈。
    """

    def __init__(self, cache_path: Path, batch_writes: int = 100, persistent: bool = True):
        self.cache_path = cache_path
        self.batch_writes = max(int(batch_writes), 1)
        self.persistent = bool(persistent)
        self.dirty_entries = 0
        self.cache: dict[str, dict] = {}
        if self.persistent and cache_path.exists():
            self.load()

    def load(self):
        """Load cache from disk."""
        with open(self.cache_path, "r", encoding="utf-8") as f:
            self.cache = json.load(f)

    def save(self):
        """Save cache to disk."""
        if not self.persistent:
            self.dirty_entries = 0
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        self.dirty_entries = 0

    def save_if_needed(self) -> None:
        """每累计指定数量的新转写结果后写盘，避免每次解码都重写整个 JSON。"""
        if self.dirty_entries >= self.batch_writes:
            self.save()

    def flush(self) -> None:
        """程序正常结束时强制保存最后不足一个批次的缓存。"""
        if self.dirty_entries:
            self.save()

    def get(self, path_str: str) -> dict | None:
        """Get cached result for a path."""
        return self.cache.get(path_str)

    def put(self, path_str: str, result: dict):
        """Store result for a path."""
        self.cache[path_str] = result
        self.dirty_entries += 1


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
      3. 全部逐通道余弦（供候选选择与双单人短路复用）。

    声纹推理（两个大模型各提 5 段波形的 embedding）是全链路最贵的一步，
    所以设计原则是"一次推理的产物榨干用尽"：两个前端共享同一次 Fbank
    （prepare_waveforms 只跑一遍，各自前向 embedding_model），门控、
    候选选择、双单人短路共享同一批 embedding，绝不为同一段音频重复
    提特征。
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
        # import pipeline.pipeline、跑 --help 和单测。import 语句写在函数体内，
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

        command_source_empty=True（分离通道不可用/指令为单人）时，把原始
        指令也作为候选一并参与"分离收益"评估，避免空通道污染门控输入。

        类型标注的价值：``tuple[bool, float, dict[str, float]]`` 精确描述
        三个槽位各自是什么，读代码的人不用猜，IDE/静态检查工具也能据此
        提前发现误用（比如把返回值第一个元素当分数算）。
        """
        # 五段波形固定顺序送入两个前端：唤醒锚点、原始唤醒、原始指令、
        # 分离通道 0、分离通道 1
        waves = [anchor_wave, raw_wake_wave, raw_command_wave, command0_wave, command1_wave]
        # 两套 ERes2NetV2 的声学前端完全相同，只是 embedding_model 权重
        # 不同。Fbank 只计算一次（prepare_waveforms），两套模型仍串行
        # 前向，不增加并发激活显存。
        shared_features = self.pretrained.prepare_waveforms(waves)
        values: dict[str, float] = {}
        # 两个前端各跑一遍 embedding，产出同名的特征集合，用 tag 前缀
        # 区分（pre_/ft_）。循环代替复制粘贴：前端只换实例、其余逻辑
        # 完全一致，以后加第三个前端也只需改元组
        for tag, embedder in (("pre", self.pretrained), ("ft", self.finetuned)):
            e_anchor, e_raw_wake, e_raw_cmd, e_cmd0, e_cmd1 = embedder.extract_features(shared_features)
            # 分离收益候选：两条分离通道各自与唤醒锚点的余弦
            candidates = [_cosine(e_anchor, e_cmd0), _cosine(e_anchor, e_cmd1)]
            if command_source_empty:
                # 分离通道不可用说明分离没产出可用内容；把原始指令也拉
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

    def finetuned_raw_cosine(self, wake_wave: np.ndarray, command_wave: np.ndarray) -> float:
        """只用微调 ERes2NetV2 计算原始唤醒与原始命令的余弦。

        双单人短路专用：两段音频都是单人时，一次 raw/raw 余弦足以判
        "是不是同一个人"，不必走 6 维门控（那套特征为混合/分离场景设计）。
        """
        wake_embedding, command_embedding = self.finetuned.extract_waveforms(
            [wake_wave, command_wave]
        )
        return _cosine(wake_embedding, command_embedding)


class PipelineRunner:
    """逐样本管线执行器：声学路由 → 门控 → 选择 → 后处理，异常一律回退原始转写。

    一个 runner 实例绑定一份配置与一个输出目录，对每个 :class:`Sample`
    调用一次 :meth:`process`，返回可序列化的结果字典（detail.json 一行）。

    面向对象的设计动机：重量级资源（两个声纹模型、分离器、ASR）在
    __init__ 里加载一次，之后 process 只做纯计算。若每处理一个样本都
    重新加载模型（每个模型几百 MB、加载要几十秒），评测几千条样本的
    时间会几乎全耗在加载上——"加载一次、复用多次"是批处理系统的通则。
    """

    def __init__(self, config: PipelineConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        # 分离缓存目录：配置里常写相对路径（outputs/cache/separation），
        # 这里统一转成"当前工作目录下的绝对路径"。为什么必须统一：
        # 缓存命中的前提是路径每次都能对上；相对路径随启动目录漂移，
        # 会造成"明明缓存过却找不到、反复重算"
        self.cache_dir = Path(config.separation_cache)
        if not self.cache_dir.is_absolute():
            self.cache_dir = (Path.cwd() / self.cache_dir).resolve()

        # ASR 转写缓存与分离缓存同目录的父级（asr_cache.json）；进程内
        # 缓存始终开启，是否跨进程落盘由配置控制
        asr_cache_path = self.cache_dir.parent / "asr_cache.json"
        self.asr_cache = AsrCache(
            asr_cache_path, config.batch_writes, config.asr_cache_persistent,
        )

        # 门控最先构造：它决定大部分样本的去留（拒绝即止步），是链路的
        # 第一道也是最重要的闸。声纹前端设备默认跟随 device，sv_device
        # 单独指定时可分卡部署
        print("[初始化] 加载 FusionGate (预训练 + 微调 ERes2NetV2)...", flush=True)
        self.gate = FusionGate(
            config.sv_model,
            config.finetuned_sv_ckpt,
            config.fusion_backend,
            config.fusion_threshold,
            config.sv_device or config.device,
            config.model_root,
        )
        print("[初始化] ✓ FusionGate 加载完成", flush=True)

        # 声学单双人路由器（纯波形特征，不花 ASR）：生产配置显式开启；
        # "off" 时不启用短路，全部样本走融合门控流程
        if config.both_single_mode not in {"off", "sft_raw_cosine"}:
            raise ValueError("both_single_mode must be off or sft_raw_cosine")
        self.both_single_router = (
            BothSingleRouter(config.both_single_backend, config.both_single_sft_threshold)
            if config.both_single_mode == "sft_raw_cosine" else None
        )
        if self.both_single_router is not None:
            print(
                "[初始化] ✓ 双单人独立路由已启用 "
                f"(single_p>={self.both_single_router.single_threshold:.6f}, "
                f"SFT cosine>={self.both_single_router.same_speaker_threshold:.3f})",
                flush=True,
            )

        # 分离器延迟导入：pipeline.separator 依赖 torch + vendor 模型定义，
        # 延迟到构造时再 import，保证 --help / 单测在无 torch 环境可用。
        # 注意 import 位置的语义：顶层 import 在"模块被 import"时执行，
        # 函数体内 import 在"函数被调用"时执行——这里推迟的是 torch 依赖
        print("[初始化] 加载 MossFormer2 分离器...", flush=True)
        from pipeline.separator import MossFormer2Separator

        self.separator = MossFormer2Separator(config.moss_model, config.device)
        print("[初始化] ✓ MossFormer2 分离器加载完成", flush=True)

        # ASR 后端二选一："aed" 生产基线（快、稳）；"llm" 用 Qwen2-7B 解码
        # （上限更高但慢得多，且需要更多显存）。self.asr 标注为 Any：
        # 两种识别器不是同一类型，但都暴露 transcribe_many 与 _model.transcribe
        # 接口（鸭子类型），管线按统一接口使用，类型标注上放宽容许两者
        print("[初始化] 加载 FireRed ASR...", flush=True)
        if config.firered_asr_type == "llm":
            self.asr: Any = FireRedLLMRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
                batch_size=config.firered_llm_batch_size,
            )
            self.elm_asr = None
        else:
            # 唤醒与命令通道共用一份 AED 权重。唤醒侧原本是独立的
            # FireRedRecognizer(beam=1, 无 ELM)，等价于在共享实例上临时把
            # beam 压到 1 且把 elm_weight 置 0（见 _transcribe_wake）。
            # 省掉整份 AED 权重的重复驻留（峰值 4818 MiB vs ~7.4 GB），
            # 也消除两份模型同时在显存里带来的数值/内核选择抖动。
            self.asr = FireRedRecognizer(
                config.firered_model,
                config.device,
                config.firered_repo,
                config.firered_vendor,
                beam_size=config.firered_beam_size,
                nbest=config.firered_beam_size,
                elm_dir=config.firered_elm_dir,
                elm_weight=config.firered_elm_weight,
            )
            # ELM 与主解码共用同一份 AED 权重。主解码一律用
            # temporary_elm_weight(0.0) 关闭 ELM，ELM 只作为选择后的
            # 重解码通道，因此不会污染 confidence，也就不会干扰路由。
            self.elm_asr = self.asr if config.firered_elm_dir else None
            print("[初始化] ✓ FireRed ASR 加载完成 (唤醒/命令共享同一份 AED 权重)", flush=True)
            print("[初始化]   - Wake ASR: beam=1, ELM off (临时切换)", flush=True)
            print(f"[初始化]   - Command ASR: beam={config.firered_beam_size} + ELM", flush=True)
            print(f"[初始化]   - N-best mode: {config.firered_nbest_mode}", flush=True)
            if config.firered_elm_dir:
                print(f"[初始化]   - ELM: enabled (weight={config.firered_elm_weight})", flush=True)
            else:
                print("[初始化]   - ELM: disabled", flush=True)

    # ------------------------------------------------------------------
    # 转写层：统一走 ASR 缓存，按解码配置打标签
    # ------------------------------------------------------------------

    def _transcribe_with_asr(
        self, paths: list[Path], asr_instance, *, cache_tag: str = "default",
    ) -> list[dict[str, Any]]:
        """Transcribe paths using a specific ASR instance.

        ``cache_tag`` identifies the decode configuration (beam + ELM weight).
        The same wav is now decoded under more than one configuration — the
        primary pass with ELM off and the post-selection pass with ELM on — so
        keying the cache on the path alone would silently return a transcript
        produced by the wrong configuration.

        （中文）统一转写入口：先查缓存（命中直接返回），未命中才真实
        解码并把结果（含 N-best 候选）写进缓存。返回与输入顺序一一对应
        的 [{"text", "confidence", "nbest"?}] 列表。
        """
        if not paths:
            return []

        # 先按缓存键把输入分成"命中"与"未命中"两组，记录下标以便最后
        # 按原顺序回填
        results = []
        uncached_paths = []
        uncached_indices = []

        for idx, path in enumerate(paths):
            path_str = str(path)
            cached = self.asr_cache.get(f"{path_str}|{cache_tag}")
            if cached is not None:
                results.append((idx, cached))
            else:
                uncached_paths.append(path)
                uncached_indices.append(idx)

        # 全部命中：直接按原顺序返回（不再碰模型）
        if not uncached_paths:
            return [r for _, r in sorted(results, key=lambda x: x[0])]

        # 未命中的批量送解码；uttid 用下标生成，行序按 uttid 重排回输入序
        utterance_ids = [f"utt_{index}" for index in range(len(uncached_paths))]
        with suppress_model_output():
            rows = asr_instance._model.transcribe(utterance_ids, [str(path) for path in uncached_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(uncached_paths):
            # 数量对不上说明底层漏返回/重复返回——宁可报错也不要静默
            # 输出错位结果（错位会把 A 的文本安到 B 头上，CER 全错）
            raise RuntimeError("FireRed returned an unexpected result count")

        for idx, identifier in enumerate(utterance_ids):
            row = mapped[identifier]
            confidence = row.get("confidence")
            try:
                confidence = float(confidence)
                if not np.isfinite(confidence):
                    confidence = 1.0
            except (TypeError, ValueError):
                confidence = 1.0

            result = {
                "text": normalize_transcript(str(row.get("text", ""))),
                "confidence": confidence,
            }
            # FireRed 在主 beam=3 解码时已经生成了 N-best。把它与 top-1
            # 一起保存，最终通道选定后即可直接做 MBR，无需对同一音频
            # 再次解码。
            normalized_nbest: list[dict[str, Any]] = []
            for hypothesis in row.get("nbest", []) or []:
                if not isinstance(hypothesis, dict):
                    continue
                hypothesis_confidence = hypothesis.get("confidence", 0.0)
                try:
                    hypothesis_confidence = float(hypothesis_confidence)
                    if not np.isfinite(hypothesis_confidence):
                        hypothesis_confidence = 0.0
                except (TypeError, ValueError):
                    hypothesis_confidence = 0.0
                normalized_nbest.append({
                    "text": normalize_transcript(str(hypothesis.get("text", ""))),
                    "confidence": hypothesis_confidence,
                })
            if normalized_nbest:
                result["nbest"] = normalized_nbest

            path_str = str(uncached_paths[idx])
            self.asr_cache.put(f"{path_str}|{cache_tag}", result)
            results.append((uncached_indices[idx], result))

        # 只在累计达到 batch_writes 后写盘；最后一个不足批次由
        # flush_caches 强制刷新
        self.asr_cache.save_if_needed()

        return [r for _, r in sorted(results, key=lambda x: x[0])]

    def _primary_transcribe(
        self, paths: list[Path], *, beam_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Primary AED decode with ELM disabled.

        Every decision-carrying transcript (wake anchors, separated command
        channels, the raw command) comes from here.  ELM is held at weight 0
        so ``confidence`` keeps the scale the routing was calibrated on;
        ELM is applied later, once, to the selected channel only.

        （中文）主解码：所有"承载决策"的转写都从这里出。LLM 后端没有
        temporary_elm_weight（它本就不带 ELM），hasattr 探测后直接解码。
        """
        beam = self.config.firered_beam_size if beam_size is None else beam_size
        tag = f"b{beam}|elm0"
        if not hasattr(self.asr, "temporary_elm_weight"):
            return self._transcribe_with_asr(paths, self.asr, cache_tag=tag)
        with self.asr.temporary_elm_weight(0.0):
            if beam_size is None:
                return self._transcribe_with_asr(paths, self.asr, cache_tag=tag)
            with self.asr.temporary_beam_size(beam_size):
                return self._transcribe_with_asr(paths, self.asr, cache_tag=tag)

    def _transcribe(self, paths: list[Path]) -> list[dict[str, Any]]:
        """Transcribe the separated command channels (primary beam, no ELM)."""
        return self._primary_transcribe(paths)

    def _transcribe_wake(self, paths: list[Path]) -> list[dict[str, Any]]:
        """Transcribe wake channels at beam=1 with no ELM (stable anchors).

        （中文）唤醒通道只求"稳"：锚点选择需要的是确定性而非探索性，
        beam=1 贪心解码结果最稳定；与命令通道共享同一份 AED 权重，
        靠 temporary_beam_size(1) 临时压束宽。
        """
        return self._primary_transcribe(paths, beam_size=1)

    def _elm_redecode(self, audio: Path, baseline_text: str) -> tuple[str, bool, str | None]:
        """Re-decode the selected channel with ELM enabled.

        Runs after routing, so it can no longer alter the gate or the switch
        decision.  An empty baseline is left untouched: a language model must
        not invent a transcript where the acoustic model produced none.

        （中文）ELM 重解码：路由已定之后对最终通道跑一次。空基线不重解
        ——语言模型不许在声学模型"什么都没听到"的地方凭空造出文本。
        """
        if self.elm_asr is None or not baseline_text:
            return baseline_text, False, None
        beam = self.config.firered_elm_beam_size
        weight = self.config.firered_elm_weight
        tag = f"b{beam}|elm{weight:g}"
        with self.elm_asr.temporary_elm_weight(weight):
            with self.elm_asr.temporary_beam_size(beam):
                rows = self._transcribe_with_asr([audio], self.elm_asr, cache_tag=tag)
        elm_text = rows[0]["text"]
        if not elm_text:
            return baseline_text, False, elm_text
        return elm_text, True, elm_text

    def _nbest_redecode(
        self, audio: Path, baseline_text: str, baseline_confidence: float,
    ) -> dict[str, Any]:
        """Conservative MBR rerank of the selected channel.

        （中文）优先复用主解码已经产生并缓存的 N-best 候选（零额外
        推理）；兼容旧缓存——其中没有 nbest 字段时才执行历史二次解码。
        """
        disabled = {"text": baseline_text, "nbest_applied": False, "nbest_reason": "nbest_disabled"}
        if self.config.firered_nbest_mode == "off" or not baseline_text:
            return disabled
        primary_tag = f"b{self.config.firered_beam_size}|elm0"
        cached = self.asr_cache.get(f"{str(audio)}|{primary_tag}")
        hypotheses = list(cached.get("nbest", [])) if cached is not None else []
        if not hypotheses:
            try:
                with self.asr.temporary_elm_weight(0.0):
                    hypotheses = self.asr.transcribe_many_nbest([audio])[0]
            except Exception as error:
                return {**disabled, "nbest_reason": f"nbest_error_keep_baseline:{type(error).__name__}"}
        if not hypotheses:
            return {**disabled, "nbest_reason": "no_hypotheses"}
        baseline = {"text": baseline_text, "confidence": baseline_confidence}
        if len(hypotheses) == 1:
            return {**disabled, "nbest_reason": "single_hypothesis"}
        return conservative_mbr_rerank(
            baseline=baseline,
            nbest=[h for h in hypotheses if h["text"] != baseline_text],
            min_confidence_gain=self.config.firered_nbest_min_confidence_gain,
            min_mbr_risk_reduction=self.config.firered_nbest_min_mbr_risk_reduction,
        )

    def _postprocess_text(
        self, audio: Path, baseline_text: str, baseline_confidence: float,
    ) -> dict[str, Any]:
        """N-best rerank then ELM, applied to whichever channel routing chose.

        Previously N-best only ever ran on the raw command, so it was
        discarded whenever the switch rule picked a separated channel.

        （中文）选择后处理两件套：先 MBR 重排、再 ELM 重解码，只作用在
        路由选中的那一路。它们不再可能反向影响门控/路由——这是 Phase 2
        的核心不变量（详见 docs/OPTIMIZATION.md）。
        """
        nbest = self._nbest_redecode(audio, baseline_text, baseline_confidence)
        final_text, elm_applied, elm_text = self._elm_redecode(audio, nbest["text"])
        return {
            "content": final_text,
            "baseline_content": baseline_text,
            "elm_applied": elm_applied,
            "elm_content": elm_text,
            "nbest_applied": nbest["text"] != baseline_text,
            "nbest_reason": nbest.get("nbest_reason", "unknown"),
        }

    def flush_caches(self) -> None:
        """保存最后不足 batch_writes 条的 ASR 缓存。"""
        self.asr_cache.flush()

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
          1. 分离唤醒/指令音频（或命中缓存）并读入 6 段波形；
          2. 声学单双人判定（纯波形，不花 ASR）；双单人进入独立短路：
             微调声纹 raw 余弦判同人，接受则只解码原始指令；
          3. 唤醒两通道 beam=1 转写 → 拼音相似度选锚点；
          4. 融合门控打分；拒绝则带 route=fusion_reject 直接返回；
          5. 单人指令：只解码原始指令；双人指令：ft 余弦选路后只解码
             选中的分离通道（首选为空才补解码另一路）；
          6. 选中通道统一做 N-best 重排 + ELM 重解码后处理。

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
            "wake_text": sample.wake_text,
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
            "both_single": False,
            "both_single_wake_probability": None,
            "both_single_command_probability": None,
            "both_single_sft_raw_cosine": None,
            "command_is_single": None,
            "separated_command_source_empty": None,
        }
        try:
            # _paths_for 内部会真实触发分离（或命中缓存直接返回旧路径）
            (wake_paths, command_paths) = self._paths_for(sample)
            # 先从波形判断唤醒/指令是否均为单人。该判断不依赖 ASR 文本——
            # "便宜特征先行"：20 维声学统计量的代价远低于一次 ASR 解码
            raw_wake_wave = read_wav(sample.wake_audio)
            raw_command_wave = read_wav(sample.command_audio)
            wake0_wave, wake1_wave = read_wav(wake_paths[0]), read_wav(wake_paths[1])
            command0_wave, command1_wave = read_wav(command_paths[0]), read_wav(command_paths[1])
            both_single = (
                self.both_single_router.decide_both(
                    raw_wake_wave, wake0_wave, wake1_wave,
                    raw_command_wave, command0_wave, command1_wave,
                )
                if self.both_single_router is not None else None
            )
            if both_single is not None:
                result.update({
                    "both_single": bool(both_single.is_both_single),
                    "both_single_wake_probability": f"{both_single.wake.probability:.6f}",
                    "both_single_command_probability": f"{both_single.command.probability:.6f}",
                    "command_is_single": bool(both_single.command.is_single),
                })

            # 双单人是独立的短路分支：不再使用六维门控或分离通道文本。
            # 单人录音里分离通道没有额外信息，一次 raw/raw 余弦判同人即可
            if both_single is not None and both_single.is_both_single:
                score = self.gate.finetuned_raw_cosine(raw_wake_wave, raw_command_wave)
                accepted = self.both_single_router.accept_same_speaker(score)
                result.update({
                    "sv_score": f"{score:.6f}",
                    "sv_accept_threshold": f"{self.both_single_router.same_speaker_threshold:.6f}",
                    "accepted": bool(accepted),
                    "both_single_sft_raw_cosine": f"{score:.6f}",
                    "route": "both_single_raw" if accepted else "both_single_reject",
                    "selected_candidate": "orig" if accepted else "",
                    "selection_reason": "both_single_sft_raw_cosine",
                    "switch_triggered": False,
                })
                if not accepted:
                    result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                    return result
                raw_command = self._primary_transcribe([sample.command_audio])[0]
                result.update({
                    **self._postprocess_text(
                        sample.command_audio, raw_command["text"], raw_command["confidence"],
                    ),
                    "raw_confidence": f"{raw_command['confidence']:.6f}",
                    "raw_text": raw_command["text"],
                })
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # ---- 常规分支：拼音锚点 + 融合门控 ----
            # Wake channels: beam=1 (stable anchor selection)
            wake0, wake1 = self._transcribe_wake([wake_paths[0], wake_paths[1]])
            # 指令路由完全由声学单双人分类器决定（单人指令的分离通道里
            # 没有额外信息，直接解码原始音频省一半命令 ASR）。
            command_is_single = bool(
                both_single is not None and both_single.command.is_single
            )
            # 命令 ASR 延迟到门控接受、余弦选源完成之后。此处不解码
            # Raw、cmd0 或 cmd1，因此被拒识的样本不会浪费任何命令 ASR。
            command0: dict[str, Any] | None = None
            command1: dict[str, Any] | None = None
            # 门控唤醒锚点：生产同一规则（一路为空回退原始 kws；拼音
            # 相似度打平也回退原始 kws——规则见 core.moss_candidate）
            # 注：曾验证"非空通道直选"（PR/CER 更优），但在计分公式
            # (1−CER+RR)/2 下 RR 权重高，raw 回退版公式分更高
            #（0.7977 vs 0.7939），故保持 raw 回退。
            gate_anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
            # 只有单人指令才允许门控把 Raw CMD 纳入候选；双人指令保持恒切
            gate_use_raw_command = command_is_single

            anchor_wave = (
                raw_wake_wave if gate_anchor.raw_wake_fallback
                else (wake0_wave, wake1_wave)[gate_anchor.selected_source]
            )
            # 融合门控打分：五段波形一次打分，产出接受判定 + 融合分 +
            # 全部逐通道余弦（候选选择直接复用，不重复提 embedding）
            accepted, score, gate_values = self.gate.score(
                anchor_wave,
                raw_wake_wave,
                raw_command_wave,
                command0_wave,
                command1_wave,
                gate_use_raw_command,
            )
            # 把门控产物写进结果行。数值统一 f"{:.6f}" 格式化成字符串：
            # JSON 里浮点会带长尾巴且各平台打印不一致，字符串格式由我们
            # 掌控，离线重扫阈值的脚本也好解析
            result.update({
                "sv_score": f"{score:.6f}",
                "accepted": bool(accepted),
                "gate_features": {k: f"{gate_values[k]:.6f}" for k in FUSION_FEATURE_KEYS},
                "wake_selected_source": int(gate_anchor.selected_source),
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                "wake_raw_fallback": bool(gate_anchor.raw_wake_fallback),
                "wake_text_tie": bool(gate_anchor.text_tie),
                "wake_contrast_source": int(gate_anchor.contrast_source),
                "wake_reason": gate_anchor.reason,
                "wake_source0_sim": f"{gate_anchor.source0_sim:.6f}" if gate_anchor.source0_sim is not None else None,
                "wake_source1_sim": f"{gate_anchor.source1_sim:.6f}" if gate_anchor.source1_sim is not None else None,
                # ASR 尚未执行，分离源是否为空要在双人指令分支解码后才能确定。
                "command_source_empty": None,
                "command_is_single": bool(command_is_single),
                "separated_command_source_empty": None,
            })
            if not accepted:
                # 门控拒绝：不跑指令 ASR、不做切换，直接带默认空转写返回。
                # 这是全链路最大的计算量节省点——负样本与明显不含目标的
                # 样本在此止步，后面的转写/后处理全部跳过
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            if command_is_single:
                # 单人指令：门控接受后只解码一次 Raw CMD。
                raw_command = self._primary_transcribe([sample.command_audio])[0]
                result.update({
                    "content": raw_command["text"],
                    "route": "raw_command_single",
                    "selected_candidate": "orig",
                    "selection_reason": "acoustic_command_single",
                    "switch_triggered": False,
                    "switch_mode": "acoustic_single_double",
                    "raw_confidence": f"{raw_command['confidence']:.6f}",
                    "raw_text": raw_command["text"],
                    "command_source0_text": "",
                    "command_source1_text": "",
                    "ft_cos_ms0": f"{gate_values['ft_cos_cmd0']:.6f}",
                    "ft_cos_ms1": f"{gate_values['ft_cos_cmd1']:.6f}",
                })
                result.update(self._postprocess_text(
                    sample.command_audio, raw_command["text"], raw_command["confidence"],
                ))
                result["item_duration"] = f"{time.perf_counter() - started:.6f}"
                return result

            # 双人指令恒切：先按 FT 余弦选源，只对选中通道执行一次 ASR。
            # 仅当首选通道文本为空时，才付出第二次 ASR 的开销并改选另一
            # 通道；绝不解码 Raw CMD（它的内容对双人指令没有参考价值）。
            anchor = gate_anchor
            preferred_index = (
                0 if gate_values["ft_cos_cmd0"] >= gate_values["ft_cos_cmd1"] else 1
            )
            selected_index = preferred_index
            selected_result = self._transcribe([command_paths[selected_index]])[0]
            if selected_index == 0:
                command0 = selected_result
            else:
                command1 = selected_result
            reason = "ft_direct_cosine_lazy_asr"

            if not selected_result["text"]:
                # 首选分离源识别为空时，才补充解码另一通道。
                selected_index = 1 - preferred_index
                selected_result = self._transcribe([command_paths[selected_index]])[0]
                if selected_index == 0:
                    command0 = selected_result
                else:
                    command1 = selected_result
                reason = "preferred_empty_fallback_other_separated"

            selected = f"ms{selected_index}"
            selected_audio = command_paths[selected_index]
            selected_confidence = selected_result["confidence"]
            selected_text = selected_result["text"]
            final_source_empty = not bool(selected_text)
            result.update({
                # 路由已定：N-best 重排 + ELM 只作用在选中的那一路上
                **self._postprocess_text(
                    selected_audio, selected_text, selected_confidence,
                ),
                "selected_candidate": selected,
                "selection_reason": reason,
                "route": "moss_candidate",
                "switch_triggered": True,
                "switch_mode": "acoustic_single_double",
                "wake_contrast_source": int(anchor.contrast_source),
                "wake_text_tie": bool(anchor.text_tie),
                "wake_source0_text": wake0["text"],
                "wake_source1_text": wake1["text"],
                # 未执行 ASR 的通道保留 None，便于确认延迟解码确实生效。
                "command_source0_text": command0["text"] if command0 is not None else None,
                "command_source1_text": command1["text"] if command1 is not None else None,
                "command_source_empty": final_source_empty,
                "separated_command_source_empty": final_source_empty,
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
