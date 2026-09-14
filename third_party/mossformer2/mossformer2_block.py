"""
Implementation for MossFormer2 block
This source code is rewritten by Shengkui Zhao based on https://github.com/lucidrains/FLASH-pytorch

（以下为新增中文说明，原 docstring 内容保留）

本文件在模型里的角色
====================
本文件是 MossFormer2 分离网络的"积木库"，提供分离网络用到的全部构件：
- 小工具函数：identity / exists / default / padding_to_multiple_of 等
- ScaleNorm：按 L2 范数归一化
- ScaledSinuEmbedding：正弦位置编码（给特征注入"时间先后"信息）
- OffsetScale：从同一份 QK 投影派生出 4 路注意力输入
- FFConvM / FFConvM_Small / FFM / GroupLinear：几种前馈 + 卷积模块
- FLASH_ShareA_FFConvM：本文件的核心——FLASH 风格的分组注意力层
  （平方注意力 + 线性注意力共享同一套 Q/K，"ShareA"即共享注意力）
- Gated_FSMN(_dilated)：门控 FSMN 记忆块（u 支路走记忆卷积、v 支路做门）
- Gated_FSMN_Block(_Dilated)：带瓶颈卷积的门控 FSMN 完整块
- MossformerBlock_GFSMN / MossformerBlock：把上述积木按 depth 层堆叠的
  最终块，前者 = Flash 注意力 + 门控 FSMN 交替，后者只有 Flash 注意力

被谁调用
========
    mossformer2.py:
        MossFormerM  -> MossformerBlock_GFSMN
        MossFormerM2 -> MossformerBlock
        MossFormer_MaskNet -> ScaledSinuEmbedding

说明：本文件是 vendor 进来的第三方代码，功能零改动，只新增中文注释。
其中 math / torchinfo.summary / identity / GLU / GLayerNorm / GlobLayerNorm /
ILayerNorm 等在本文件内未被实际使用（vendor 遗留导入），按规则保留不动。
"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torchinfo import summary
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from .conv_module import ConvModule, GLU, FFConvM_Dilated
from .fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from .layer_norm import CLayerNorm, GLayerNorm, GlobLayerNorm, ILayerNorm
# functions

# 恒等函数：输入什么就返回什么。*args/**kwargs 让它能"吞掉"任何
# 多余的参数——常用于需要一个可调用对象占位、但实际不做任何事的场合。
# （本文件中该函数未实际使用，vendor 遗留，保留。）
def identity(t, *args, **kwargs):
    return t

# 在张量尾部追加 num_dims 个大小为 1 的维度。view 不复制数据、只改
# "视角"，追加的 1 维可以让后续广播对齐形状。
def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

# 判断值是否"存在"（不为 None）。写成函数是为了让主代码更贴近
# 自然语言：if exists(mask): 读起来比 if mask is not None: 更顺。
def exists(val):
    return val is not None

# "取默认值"：val 存在就用 val，否则用 d。常用于可选参数的处理。
def default(val, d):
    return val if exists(val) else d

# 计算"把 n 补到 mult 的整数倍还需要补几个 0"。分组注意力要求
# 序列长度是组大小 group_size 的整数倍，末尾不够一组时补零凑整。
def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

class ScaleNorm(nn.Module):
    """按向量 L2 范数做归一化（scalenorm）。

    与 LayerNorm 的区别：LayerNorm 除以的是"标准差"（先减均值），
    ScaleNorm 除以的是"整个向量的 L2 范数"（不减均值），再乘一个
    可学习的标量 g。实现极简、没有逐维参数（g 只是一个标量），
    是 Transformer 类模型的轻量替代方案。
    """
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        # scale = 1/sqrt(dim)：先按维度开方缩放范数，使不同维度的输入
        # 归一化后的尺度可比（维度越大范数天然越大）。
        # g：可学习的全局缩放标量（形状 [1]），初始化 1（恒等起步）。
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

        # torch.norm(x, dim=-1, keepdim=True)：沿最后一维求 L2 范数，
        # keepdim 保留维度便于广播相除。clamp(min=eps) 防止范数为 0
        # 时除零（把最小值钳制在 eps）。
    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    """正弦位置编码（Transformer 论文式的绝对位置编码）。

    为什么需要位置编码：注意力机制本身对"顺序"不敏感（打乱输入顺序，
    注意力分数不变），必须额外把"每个 token 处于第几个时间位置"的信息
    注入特征，模型才知道先后关系。正弦编码用一组不同频率的 sin/cos
    函数生成每个位置的编码，无需训练（是确定性函数），且外推到更长
    序列时依然有定义。
    """
    def __init__(self, dim):
        super().__init__()
        # inv_freq：频率倒数序列 [1, 1/10000^(0/dim), 1/10000^(2/dim), ...]
        # 共 dim/2 个频率。10000 是 Transformer 论文的约定底数；
        # 频率从低到高指数分布——低频捕捉"粗粒度"位置、高频捕捉"细粒度"。
        self.scale = nn.Parameter(torch.ones(1,))
        # register_buffer：inv_freq 不是可学习参数（不需要梯度），但要随
        # 模型一起 .to(device) 搬移、随 state_dict 保存——buffer 正是
        # 为这种"模型附属常量张量"设计的。
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 取序列长度 n（输入 x 的第 2 维是时间）与所在设备。
    def forward(self, x):
        n, device = x.shape[1], x.device
        # einsum('i , j -> i j')：外积——位置 i 与频率 j 相乘，得到
        # [n, dim/2] 的"角度表"。einsum 是爱因斯坦求和约定的简写，
        # 这里等价于 t[:, None] * inv_freq[None, :]。
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        # sin 与 cos 沿最后一维拼接成 [n, dim]，使相邻两维是同一频率的
        # sin/cos 对（一个位置编码的经典设计，保证任意两个位置编码的
        # 点积只与它们的"相对距离"有关）。
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

class OffsetScale(nn.Module):
    """从同一份 QK 特征派生出 4 路注意力输入（quad_q/lin_q/quad_k/lin_k）。

    "Offset"指 beta 偏移、"Scale"指 gamma 缩放：对每个 head（heads=4）
    用一对可学习的 gamma/beta 做逐维缩放平移，同一个输入 x 在不同 head
    下被投影成不同的东西。FLASH 的"共享注意力"正是靠它：一份 QK 投影
    同时喂给平方注意力和线性注意力，靠 per-head 的缩放偏移区分用途。
    """
    def __init__(self, dim, heads = 1):
        # gamma/beta 形状 [heads, dim]：每个 head 一套缩放/偏移。
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        # gamma 初始化为均值 1、标准差 0.02 的高斯噪声：起步时接近恒等
        # 但略有扰动，帮助 4 个 head 分化出不同的投影。
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

        # einsum('... d, h d -> ... h d')：x 的最后一维 d 与 gamma 的
        # 最后一维 d 逐元素相乘，同时在倒数第二维"广播出" head 维 h，
        # 得到 [..., h, d]，再加 beta。`...` 是省略号：匹配任意多的
        # 前置维度（batch、时间等），即这条规则与前置维度无关。
    def forward(self, x):
        # unbind(dim=-2)：沿 head 维拆开，返回 4 个 [..., d] 张量的元组，
        # 调用方用 quad_q, lin_q, quad_k, lin_k = ... 一次性接住。
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class FFConvM(nn.Module):
    """前馈 + 卷积模块（FFN 的卷积增强版）。

    流程：LayerNorm（pre-norm）-> Linear 升维 -> SiLU -> ConvModule
    （深度卷积，见 conv_module.py）-> Dropout。卷积让前馈层带上"局部
    时序上下文"的感知能力，弥补纯线性层只看当前 token 的不足。
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        # norm_klass 是"类"而非实例（默认 nn.LayerNorm），调用方可以
        # 换成 ScaleNorm 等其他归一化实现——把类当参数传的设计。
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class GroupLinear(nn.Module):
    """分组线性层：用分组 1x1 卷积代替全连接做通道混合，大幅省参数。

    普通 Linear(dim_in, dim_out) 的参数量是 dim_in*dim_out；这里先用
    groups=dim_in//K 的分组卷积（每 K 个输入通道共享一组权重，参数
    只有原来的 1/K），输出 hidden=dim_in//2 个通道，再用 LayerNorm +
    一个较小的 Linear 补足。是 dim 很大时控制参数量的常用手段。
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        K = 4
    ):
        super().__init__()
        hidden = dim_in // 2
        # kernel_size=1 的 Conv1d（逐点卷积）+ groups=dim_in//K：
        # 沿通道维做"分组线性混合"，不涉及时序邻域。
        self.group_conv = nn.Conv1d(dim_in, hidden, groups=dim_in//K, kernel_size=1)
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, dim_out)

    def forward(
        self,
        x,
    ):
        # Conv1d 要求 [B, C, T]（通道在前），而本模块输入是 [B, T, C]，
        # 所以先 transpose(2,1) 换过去，做完再换回来——与 ConvModule
        # 内部的 Transpose 层是同一个套路。
        x1 = x.transpose(2,1)
        conv_out = self.group_conv(x1)
        x2 = self.norm(conv_out.transpose(2,1))
        x3 = self.linear(x2)
        return x3

class FFConvM_Small(nn.Module):
    """FFConvM 的轻量版：中间的 Linear 换成 GroupLinear 以减少参数。

    结构其余部分与 FFConvM 完全一致：pre-norm -> 升维（分组线性）->
    SiLU -> ConvModule -> Dropout。reduction 参数留作扩展位，当前实现
    未使用（vendor 保留）。
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1,
        reduction = 4
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            GroupLinear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class FFM(nn.Module):
    """最朴素的前馈模块：pre-norm -> Linear -> SiLU -> Dropout。

    没有卷积、没有门控，就是标准 Transformer FFN 去掉第二层 Linear 的
    变体。与 FFConvM / FFConvM_Small 保持同样的输入输出约定，
    方便按配置互相替换。
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class FLASH_ShareA_FFConvM(nn.Module):
    """本文件的核心：FLASH 风格的分组注意力层（MossFormer2 的主注意力）。

    "FLASH" 指一种高效注意力（源自 lucidrains/FLASH-pytorch）：
    - 分组（group）：把长序列切成 group_size 长的组，注意力只在组内做，
      复杂度从 O(L^2) 降到 O(L * group_size)；
    - 双路注意力：一路"平方注意力"（relu(QK^T/sqrt(g))^2 做权重，替代
      softmax）、一路"线性注意力"（先算 K^T V 再乘 Q，避免构造完整的
      注意力矩阵），两路结果相加——"ShareA"表示两路共享同一套 Q/K 投影。
    - 门控输出：两个值分支 u/v，一个做门控，控制另一个的信息流量。

    输入输出形状都是 [B, T, dim]（时间在前、特征在后），
    这是"块"的基本约定，保证可以按 depth 层层堆叠。
    """
    def __init__(
        self,
        *,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 1.,
        causal = False,
        dropout = 0.1,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = True
    ):
        super().__init__()
        # `*` 之后的参数都是"关键字专用参数"（keyword-only）：调用时必须
        # 写参数名（dim=..., group_size=...），防止一大串位置参数传错位。
        # hidden_dim：值分支的维度 = dim * expansion_factor。
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings
        # 旋转位置编码对象由外部（MossformerBlock）传入并共享。
        self.rotary_pos_emb = rotary_pos_emb
        # norm
        self.dropout = nn.Dropout(dropout)
        # projections
        
        # to_hidden：把输入投影成"值"特征（维度 hidden_dim，后面会切成 u/v
        #           两半，所以 hidden_dim 必须是偶数）。
        self.to_hidden = FFConvM(
            dim_in = dim,
            dim_out = hidden_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        # to_qk：把输入投影成"查询/键"特征（维度 query_key_dim），
        #       两路注意力共享这一份投影（ShareA 的含义）。
        self.to_qk = FFConvM(
            dim_in = dim,
            dim_out = query_key_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )

        # 用 4 个 head 的缩放偏移把同一份 qk 拆成 quad_q/lin_q/quad_k/lin_k。
        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)

        # to_out：把注意力输出投影回 dim。输入维度是 dim*2，因为 u 与 v
        # 两路的输出沿通道拼接后一起送进来。
        self.to_out = FFConvM(
            dim_in = dim*2,
            dim_out = dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        
        # 门控激活：Sigmoid 把"门"压到 (0,1) 区间，决定信息通过的比例。
        self.gateActivate=nn.Sigmoid() 

    def forward(
        self,
        x,
        *,
        mask = None
    ):

        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        # prenorm
        #x = self.fsmn(x)
        # pre-norm 在这里被注释掉了（vendor 简化版）：normed_x 直接等于 x，
        # 归一化由外层 Gated_FSMN_Block 的 CLayerNorm 承担。
        normed_x = x #self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen
        # 原注释：token shift——来自深圳一位独立 AI 研究者的"零成本"技巧。
        # 先保存原始输入，最后做残差连接（residual = x）。
        residual = x

        # token shift（token 平移）：把特征沿最后一维（通道）切成两半，
        # 一半在时间轴上整体"右移一帧"（左侧补 0、右侧裁掉一帧，
        # F.pad 的 (0,0,1,-1) 表示：最后两维=时间维，左补 1 右裁 1），
        # 再与未平移的另一半拼回。效果：每个 token 的特征里混入了
        # 相邻 token 的信息——给注意力提供"顺序感"的最廉价手段，
        # 没有增加任何参数和计算量。
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections
        # 值特征切成 u、v 两半（各 [B, T, hidden_dim/2]）：
        # v 与注意力输出相乘得到"主体信息"，u 用于构造"门"。
        v, u = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # offset and scale
        # 一份 qk 经 4 个 head 的缩放偏移，拆成平方/线性两路的 Q 和 K。
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        # 计算注意力：返回 (att_v, att_u)，分别是 v、u 两路加权求和的结果。
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)

        # 门控融合：att_u*v 是"主体"，att_v*u 经 Sigmoid 变成"门"，
        # 两者逐元素相乘——门控决定主体信息通过多少（GLU 思想的注意力版）。
        out = (att_u*v ) * self.gateActivate(att_v*u)
        # 残差连接 + 输出投影：x + to_out(out)。residual 变量保存的原始
        # 输入在此用上（写成了 x，语义相同）。
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask = None):
        # 本方法是注意力计算的核心，流程：mask 处理 -> 旋转位置编码 ->
        # 分组 -> 平方注意力 -> 线性注意力 -> 两路相加并还原序列。
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

            # rearrange('... -> ... 1')：在末尾追加一个 1 维，使 mask 形状
            # 对齐 lin_k 以便广播。masked_fill(~lin_mask, 0.)：把"被屏蔽
            # 位置"的 key 置零——key 全零时该位置的注意力分数恒为 0，
            # 等效于"看不见"。
        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            # 旋转位置编码（RoPE）：对 Q/K 施加与位置相关的旋转，使注意力
            # 分数天然携带"相对位置"信息（两个位置编码的点积只取决于
            # 它们的相对距离），无需学习参数。map(函数, 元组) 对 4 个
            # 张量逐个应用旋转。
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups
        # 分组要求序列长度是组大小的整数倍：算出差多少帧。
        padding = padding_to_multiple_of(n, g)

            # 时间轴末尾补 padding 个零帧：(0, 0, 0, padding) 对应最后两维
            # 的 (左, 右) 填充量——倒数第二维不补、最后一维（时间）右补。
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # group along sequence
        # 分组的关键一步：rearrange('b (g n) d -> b g n d', n=group_size)
        # 把 [B, T, d] 重排成 [B, g, n, d]——长度为 T 的序列被切成 g 段、
        # 每段 n 个 token。之后 einsum 里的 `...` 会作用在 g/n 维上，
        # 即"每个组内部"独立做注意力（组与组之间互不关注）。
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v, u))

            # mask 同步分组：[B, g*1*j] -> [B, g, 1, j]，
            # 中间补一个 1 维以便与注意力矩阵 [B, g, n, n] 广播。
        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # calculate quadratic attention output
        # ===== 平方注意力（quadratic attention）=====
        # sim：Q 与 K 的"缩放点积"分数。einsum('... i d, ... j d -> ... i j')
        # 即 sim[b,g,i,j] = sum_d quad_q[b,g,i,d] * quad_k[b,g,j,d]。
        # 除以 g（组大小）做缩放，防止点积随维度增大而数值爆炸
        # （与经典注意力除以 sqrt(d) 同理，这里是 FLASH 的约定）。
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        # relu(sim)**2：先 ReLU 把负相关全部置 0，再平方放大强相关、
        # 压缩弱相关——FLASH 论文的核心改动，用"平方 ReLU"替代 softmax：
        # 不再需要指数运算，数值更稳、计算更快，且天然产生"稀疏"的
        # 非负注意力（大多数位置是 0）。
        attn = F.relu(sim) ** 2
        # 对注意力权重做 dropout：随机把一些权重置零再整体缩放，
        # 防止注意力过度依赖少数位置（正则化手段）。
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        # 因果掩码：triu(1) 生成"对角线以上为 True"的上三角矩阵。
        # 把"未来位置"（j > i）的注意力置零——每个 token 只能看
        # 自己和过去的 token。语音分离离线处理时 causal=False，
        # 双向看；流式（实时）处理时才需要 causal=True。
        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        # 注意力加权求和：attn[b,g,i,j] 与 v[b,g,j,d] 相乘对 j 求和，
        # 得到每个位置 i 的"上下文向量"（对 v、u 两路各做一次）。
        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # calculate linear attention output
        # ===== 线性注意力（linear attention）=====
        # 思想：把 softmax(QK^T)V 的计算顺序换成 Q(K^T V)。先对 K^T V
        # 求和得到"摘要矩阵"，再乘 Q——复杂度从 O(n^2) 降到 O(n)。
        if self.causal:
        # 因果版本：摘要矩阵沿组内序列做"前缀和"（cumsum），
        # 即第 i 个位置只累计 0..i-1 的 key（exclusive：先 cumsum
        # 再整体右移一位、最前面补零），保证不"偷看未来"。
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            # exclusive cumulative sum along group dimension
            lin_ku = lin_ku.cumsum(dim = 1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
        # 非因果版本：对整个组一次性求和得到摘要矩阵 [B, d, e]，
        # 除以 n（组内 token 数）做平均归一化。
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # fold back groups into full sequence, and excise out padding
        # 两路注意力输出相加（平方 + 线性），再把分组拼回完整序列
        # [B, g, n, d] -> [B, g*n, d]，并裁掉末尾为凑组而补的 padding
        # 帧（[:, :n] 只保留前 n 帧）。map 返回迭代器，由调用方
        # att_v, att_u = ... 解包成两个张量。
        return map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v+lin_out_v, quad_out_u+lin_out_u))

class Gated_FSMN(nn.Module):
    """门控 FSMN：FSMN 记忆块 + 门控机制。

    u 支路：先过 FFConvM 升维，再走 FSMN 记忆卷积（捕捉长程时序上下文）；
    v 支路：只过 FFConvM，作为"门"；
    最后 x_v * x_u：门控逐元素控制记忆信息的通过量，再加回原始输入
    （残差）。门控让网络学会"哪些位置、哪些通道的记忆信息值得保留"。
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()
        # u 支路：升维到 hidden_size（FSMN 的输入维）。
        self.to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        # v 支路：同样的升维，但结果只做门控用。
        self.to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        # 核心记忆块：见 fsmn.py（线性投影 + 记忆卷积 + 残差）。
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def forward(
        self,
        x,
    ):
        # input 保存原始输入，最后做模块级残差。
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x) 
        # u 支路走记忆块：输出"带长程上下文的特征"。
        x_u = self.fsmn(x_u)
        # 门控：v 支路乘 u 支路（逐元素），再加残差。
        x = x_v * x_u + input               
        return x

class Gated_FSMN_dilated(nn.Module):
    """门控 FSMN 的膨胀版：记忆块换成 UniDeepFsmn_dilated（感受野更大）。

    结构与 Gated_FSMN 完全一致（u 支路记忆、v 支路门控、残差输出），
    唯一区别是 u 支路里的 FSMN 使用膨胀稠密卷积（DilatedDenseNet），
    能用更少的层看到更远的上下文。
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()
        # u / v 支路与 Gated_FSMN 相同（注释见上）。
        self.to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        # 膨胀版记忆块（内部是 DilatedDenseNet，感受野指数增长）。
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(
        self,
        x,
    ):
        # forward 与 Gated_FSMN 完全相同的门控 + 残差流程（注释见上）。
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x) 
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input               
        return x

class Gated_FSMN_Block(nn.Module):
    """Gated-FSMN block."""

    # （补充说明）带瓶颈（bottleneck）的门控 FSMN 完整块，输入输出同形状
    # [B, T, dim]，可堆叠。流程：
    # conv1（1x1 卷积把 dim 降到 inner_channels，瓶颈压缩计算量）-> PReLU
    # -> CLayerNorm -> Gated_FSMN（门控记忆）-> CLayerNorm
    # -> conv2（1x1 卷积升回 dim）-> 与输入相加（残差）。
    # 注意 __init__ 里选出的 norm_klass 变量在本类中未使用（vendor 遗留，
    # 归一化实际用的是 CLayerNorm），按规则保留不动。
    def __init__(self,
                 dim,
                 inner_channels = 256,
                 group_size = 256, 
                 norm_type = 'scalenorm',
                 ):
        super(Gated_FSMN_Block, self).__init__()
        if norm_type == 'scalenorm':
        # 按字符串选择归一化类（结果未使用，见类头注释的说明）。
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # 下面这行是被注释掉的旋转位置编码（vendor 保留的备选方案，未启用）。
        # rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # conv1：1x1 卷积（逐点卷积）降维 + PReLU——"瓶颈"结构：
        # 先在低维（inner_channels）里做记忆建模，计算量小，
        # 最后再升回 dim。
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        # 门控 FSMN 记忆块（lorder=20：单侧上下文 19 帧）。
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):        
        # 输入 [B, T, dim]。Conv1d 要求通道在前，所以 transpose(2,1)
        # 换成 [B, dim, T]；而 Gated_FSMN 的输入约定又是 [B, T, C]
        # （时间在前），喂给它之前再换回去。本文件里 Conv1d 层与
        # 序列模块交替使用时，反复出现这个"来回转置"套路。
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        # 残差：模块级恒等路径，堆叠多层不退化。
        return conv2.transpose(2,1) + input

class Gated_FSMN_Block_Dilated(nn.Module):
    """Gated-FSMN block with dilitations."""

    # （补充说明）与 Gated_FSMN_Block 结构完全相同，只是中间的 Gated_FSMN
    # 换成膨胀版（Gated_FSMN_dilated），记忆块感受野更大。MossFormer2 的
    # 24 层分离网络里，与 Flash 注意力层交替堆叠的就是这个块。
    def __init__(self,
                 dim,
                 inner_channels = 256,
                 group_size = 256, 
                 norm_type = 'scalenorm',
                 ):
        super(Gated_FSMN_Block_Dilated, self).__init__()
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # 瓶颈：1x1 卷积降维 + PReLU（同 Gated_FSMN_Block）。
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        # 下一行原注释说明：带门控的膨胀记忆块。
        #block dilated with gating
        # 膨胀版门控记忆块。
        self.gated_fsmn = Gated_FSMN_dilated(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        # forward 流程与 Gated_FSMN_Block 一致（注释见上）。
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2,1) + input

class MossformerBlock_GFSMN(nn.Module):
    """MossFormer2 的最终块：depth 层"Flash 注意力 + 门控 FSMN"交替堆叠。

    forward 里每轮迭代：先过一层 FLASH_ShareA_FFConvM（分组注意力，
    负责"谁和谁相关"的全局交互），再过一层 Gated_FSMN_Block_Dilated
    （局部时序上下文与记忆），重复 depth 次。两种机制互补，
    构成本仓库分离网络的核心。
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size = 256, #384, #128, #256,
        query_key_dim = 128, #256, #128,
        expansion_factor = 4.,
        causal = False,
        attn_dropout = 0.1,
        norm_type = 'scalenorm',
        shift_tokens = True
    ):
        super().__init__()
        # 参数合法性检查：norm_type 只允许两种取值，传错立刻报错，
        # 避免"静默地用了意料之外的归一化"。
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # 按字符串选择归一化类，传给注意力层的 FFConvM 使用。
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # 旋转位置编码：只对前 min(32, query_key_dim) 维做旋转——
        # "部分旋转"（partial RoPE），源自 GPT-J（Wang et al.）的做法：
        # 保留部分维度不旋转，兼顾位置信息与内容信息。
        # 该对象在 depth 个注意力层之间共享（参数只存一份）。
        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        # nn.ModuleList：存放子模块的专用列表。必须用 ModuleList 而不是
        # 普通 list——只有 ModuleList 里的模块会被自动"登记"进模型，
        # 其参数才会出现在 model.parameters() 里（被优化器更新）、
        # 随 model.to(device) 搬移、随 state_dict 保存。
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])
        # 列表推导式批量创建 depth 个注意力层，共享同一个 rotary_pos_emb。
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens) for _ in range(depth)])
  
        # 辅助方法：把 repeats 个 UniDeepFsmn 串成 nn.Sequential。
        # （本类中该方法未被调用，vendor 从工具链带来的遗留，保留不动。）
    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        # 交替堆叠：Flash 注意力层 -> 门控 FSMN 块，共 depth 轮。
        # （ii 手动计数是 vendor 的写法，等价于 enumerate 的索引，保留。）
        ii = 0
        for flash in self.layers:
            x = flash(x, mask = mask)
            # 与第 ii 个门控 FSMN 块配对。
            x = self.fsmn[ii](x)
            ii = ii + 1
        return x

class MossformerBlock(nn.Module):
    """只含 Flash 注意力层的块（原版 MossFormer 的 masknet 内核）。

    与 MossformerBlock_GFSMN 的区别：没有 fsmn 模块列表，forward 里只
    逐层过注意力。由 mossformer2.py 的 MossFormerM2 使用（当前主模型
    走的是 MossFormerM -> MossformerBlock_GFSMN，本类是保留的原版路径）。
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size = 256, #384, #128, #256,
        query_key_dim = 128, #256, #128,
        expansion_factor = 4.,
        causal = False,
        attn_dropout = 0.1,
        norm_type = 'scalenorm',
        shift_tokens = True
    ):
        super().__init__()
        # 参数检查与归一化选择：同 MossformerBlock_GFSMN。
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # 部分旋转位置编码（同 MossformerBlock_GFSMN 的说明）。
        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        # depth 个 Flash 注意力层（nn.ModuleList 的意义同上）。
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens) for _ in range(depth)])

        # 同 MossformerBlock_GFSMN：本类中未使用的辅助方法（vendor 遗留）。
    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        # 逐层过 Flash 注意力。（ii 计数在本类中更显冗余——没有 fsmn
        # 需要配对索引，vendor 原样保留。）
        ii = 0
        for flash in self.layers:
            x = flash(x, mask = mask)
            ii = ii + 1
        return x

