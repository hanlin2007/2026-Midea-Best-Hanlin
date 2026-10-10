"""dasynth 域适配微调的 MossFormer2_SS_16K torch 分离器（SZY 音频前端）。

把一段混合语音盲分离成两路（ms0/ms1），整波推理、peak→0.5 归一化、
``_ms0``/``_ms1`` 落盘缓存。模型定义 vendor 在 ``third_party/mossformer2/``
（外部代码，未改动），权重可以是：

  - ClearerVoice 官方预训练（ModelScope iic/ClearerVoice-Studio）；
  - 经 ``pipeline.train`` 在 dataset_dasynth 上域适配微调的 .pt
    （生产用法：修复官方分离器在混响/掩盖样本上的崩塌）。

"盲分离"指模型不知道谁在说话，只是把混合语音拆成两路（ms0/ms1），哪路
是目标说话人由下游（融合门控的余弦打分）决定——本模块不做任何判定，
只做"拆"这一件事，职责单一。MossFormer2 是端到端神经网络（约 40M 参数），
整段波形一次前向就出两路估计，不需要传统方法的迭代过程。

输入音频读取共用 ``pipeline.helpers.read_mono_16k``（严格 16k 单声道 float32）。

注意：本模块在 import 时会把 ``third_party/`` 加入 sys.path 并引入
mossformer2 模型定义（依赖 torch），因此管线里采用延迟导入。
（"延迟导入"= 调用方把 ``import pipeline.separator`` 写进函数体内，而不是模块
顶部——只要不构造 PipelineRunner，就不会触发本模块的 import，也就不会碰到
torch 依赖。）
"""
from __future__ import annotations

import sys                                     # 修改模块搜索路径（vendor 注入）
from pathlib import Path                       # 路径计算
from types import SimpleNamespace              # 把超参字典当"轻量配置对象"用（点号访问）

import numpy as np                             # 波形数组与峰值归一化
import soundfile as sf                         # 分离结果 wav 写出

from core.audio import SAMPLE_RATE       # 16k 常量（全系统统一口径）
from pipeline.helpers import read_mono_16k          # 严格输入校验（采样率/单声道/finite）

# vendor 模型定义：third_party/mossformer2 是独立第三方代码，未打进 pip 包，
# 这里手动把仓库内的 third_party 目录放进模块搜索路径再 import。
# 为什么必须改 sys.path：Python import 时只搜索 sys.path 里的目录（site-packages、
# 当前目录等）；third_party 里的 mossformer2 既不在 pip 里、也不在 src 下，
# 不注入路径 import 会直接 ModuleNotFoundError。insert(0, ...) 插到最前面，
# 保证优先于任何同名包（防止与已安装的第三方包撞名）。insert 前先判重，
# 避免重复 import 本模块时路径列表越积越长。
_REPO_ROOT = Path(__file__).resolve().parents[2]
# 逐级说明：__file__ 是本文件路径；resolve() 消除符号链接/相对段；
# parents[2] 是"上溯两级"：parents[0]=src/szy，parents[1]=src，
# parents[2]=仓库根目录——从文件自身位置推导仓库根，不依赖"从哪里启动"
_THIRD_PARTY = _REPO_ROOT / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

# noqa: E402 是给 linter 的豁免："模块级 import 不在文件顶部"——因为上面
# 必须先改 sys.path，这个 import 才能成功；位置是刻意为之，不是疏忽
from mossformer2.mossformer2 import MossFormer2_SS  # noqa: E402


def load_mossformer2(checkpoint: str | Path, device: str):
    """构建 MossFormer2_SS_16K 模型并加载 .pt 权重，返回 (模型, 实际设备)。

    兼容两种 checkpoint 格式：裸 state_dict 或带 ``"model"`` 顶层键的
    训练存档（train.py 的产物）。权重加载用 strict=True：键不匹配立即报错，
    避免静默漏加载。

    注意返回值是 (模型, 实际设备) 二元组而不是模型：请求 cuda 但机器
    没有 GPU 时会静默落到 cpu，调用方必须知道真实落在哪，否则后续把
    数据 .to(device) 时会出现"张量在 cpu、模型在 gpu"的错位。
    """
    # 延迟导入 torch：本函数被调用才需要 torch（模型加载都在 torch 上）。
    # train.py / sanity_check.py 调用本函数时本来就有 torch，而 pipeline
    # 通过延迟导入 separator 间接延迟了这里——链条上是同一套逻辑
    import torch

    # expanduser() 展开 ~（用户主目录）；resolve() 转成绝对路径并消解符号
    # 链接——权重路径常以 ~/... 形式写在配置里，不展开后面 is_file 会找不到
    ckpt = Path(checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        # 启动即检查、报错带完整路径：权重缺失是最常见的环境问题，
        # 与其让 torch.load 抛晦涩的底层错误，不如这里给出明确信息
        raise FileNotFoundError(f"MossFormer2 torch checkpoint not found: {ckpt}")
    # 模型超参与官方预训练一致（2 说话人 / 16k 采样率配置）。
    # SimpleNamespace 把 kwargs 变成"可用点号访问属性的轻量对象"——
    # vendor 代码期望 cfg.num_spks 这样的属性访问，一个 dict 满足不了，
    # 为此定义一个完整类又太重，SimpleNamespace 正好
    cfg = SimpleNamespace(
        num_spks=2, encoder_kernel_size=16, encoder_embedding_dim=512,
        mossformer_sequence_dim=512, num_mossformer_layer=24,
    )
    model = MossFormer2_SS(cfg).model        # .model 取包装器内的 nn.Module 本体
    # torch.load 反序列化 .pt；map_location="cpu" 先把张量读进 CPU 内存，
    # 之后整体 .to(device) 一次搬过去——避免加载瞬间在 GPU 上产生内存峰值
    payload = torch.load(str(ckpt), map_location="cpu")
    # 两种存档格式自适应：训练存档带 "model" 顶层键（train.py 的 torch.save
    # 写的是 {"model": state_dict, ...}），官方/裸存档本身就是 state_dict
    # strict=True：加载键与模型键必须完全一一对应，多/少/错名都报错。
    # 看似严格，实则是最省心的选择——静默漏加载几层权重，模型跑起来
    # 分数莫名其妙变差，排查成本远大于当场报错
    model.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)
    # 设备协商：配置写 cuda 且机器真有 CUDA 才用 GPU，否则回退 CPU。
    # startswith("cuda") 覆盖 "cuda" / "cuda:0" / "cuda:1" 等写法
    use = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    # .to(use) 把参数搬到目标设备；.eval() 切评估模式——关掉 dropout 等
    # 只在训练时生效的随机行为，推理结果才可复现
    return model.to(use).eval(), use


class MossFormer2Separator:
    """产线唯一的分离器实现（torch 版）。

    接口约定（与 helpers.separation_paths_for 配合）：``separate(source,
    output_prefix)`` 返回两条输出 wav 的 Path 元组；两文件都已存在时视为
    缓存命中，直接返回不再推理。

    为什么缓存检查放在这里而不是调用方？因为"文件是否有效"的知识属于
    分离器（它知道自己的输出长什么样）；调用方只关心"给我两条路径"，
    不必知道缓存规则。此外缓存键 = 输出文件名，而文件名由调用方给出，
    这也保证"同样的输入前缀 → 同样的路径 → 同样的缓存"。
    """

    def __init__(self, model_path: str | Path, device: str) -> None:
        # 构造时完成模型加载（一次，几百 MB）；之后每次 separate 都是纯推理。
        # "贵的东西只初始化一次"是管线各环节共用的原则
        self.model, self.device = load_mossformer2(model_path, device)
        # 记录加载的真实权重路径（绝对路径）：manifest/日志里可以复现
        self.model_path = Path(model_path).expanduser().resolve()

    def separate(self, source: Path, output_prefix: Path) -> tuple[Path, Path]:
        import torch                             # 延迟导入：见模块 docstring

        # 先建目录再推理：推理很贵，若因目录不存在导致写出失败，几分钟
        # 白算；目录类错误应该在最便宜的时刻暴露
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        # with_name(新名)：替换文件名、保留父目录；输出名 = 前缀 + "_ms0/_ms1"
        outputs = (output_prefix.with_name(output_prefix.name + "_ms0.wav"),
                   output_prefix.with_name(output_prefix.name + "_ms1.wav"))
        # 缓存命中判定：两个文件都在 → 直接返回路径，不推理。
        # "两个都在"是完整条件：只有一半（上次写到一半被中断）不算命中，
        # 会重新推理并把两个文件都重写——缓存要么完整可用，要么重算
        if all(path.is_file() for path in outputs):
            return outputs
        # 输入严格校验（16k/单声道/finite），失败立刻 ValueError，
        # 不让坏音频进入昂贵的推理
        waveform = read_mono_16k(source)
        # no_grad：纯推理，不建计算图，省显存；[None, :] 加 batch 维。
        # 展开讲：
        #   torch.no_grad() 上下文里所有张量运算都不记录梯度图——训练要
        #   靠梯度图反传，推理不需要；关掉后显存占用大幅下降，速度也更快；
        #   torch.from_numpy(waveform) 把 numpy 数组"零拷贝"包装成张量
        #   （共享同一块内存，不复制数据，大波形也几乎零开销）；
        #   .float() 显式 float32（from_numpy 继承 numpy 的 dtype，保险起见）；
        #   .to(self.device) 搬到 GPU；[None, :] 等价于 unsqueeze(0)——
        #   模型按 (batch, time) 批量处理，单条输入也要有 batch 维
        with torch.no_grad():
            out = self.model(torch.from_numpy(waveform).float().to(self.device)[None, :])
        # out 形状 (1, 2, T)：batch=1、两路输出、时间轴。逐路落盘
        for index, output_path in enumerate(outputs):
            # 该路输出：squeeze(0) 去掉 batch 维 → .float() 转 float32 →
            # .cpu() 搬回主机内存 → .numpy() 转 numpy；np.asarray 兜底类型
            signal = np.asarray(out[index].squeeze(0).float().cpu().numpy(), dtype=np.float32)
            # 输出归一化到 ±0.5（与历史 ONNX 口径一致，防止爆音）。
            # 为什么归一化：模型输出幅度无保证，可能远超 ±1（16bit PCM 的
            # 满刻度），直接写文件会削波爆音；压到 ±0.5 与历史 ONNX 分离器
            # 的幅度口径一致，下游（声纹/ASR）在同样的响度分布上工作
            peak = float(np.max(np.abs(signal))) if signal.size else 0.0
            if peak > 0.0:
                signal = signal / peak * 0.5
            # subtype="PCM_16"：写成 16bit 整数 PCM。人类对话存档一般就用
            # 16bit（CD 标准）；float32 wav 体积大一倍且部分工具不识别。
            # 参数顺序：路径、波形、采样率、子类型
            sf.write(str(output_path), signal, SAMPLE_RATE, subtype="PCM_16")
        return outputs
