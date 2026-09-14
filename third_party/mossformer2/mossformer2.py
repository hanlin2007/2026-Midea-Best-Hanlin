"""
modified from https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py
#Author: Shengkui Zhao


（以下为新增中文说明，原 docstring 内容保留）

本文件在模型里的角色
====================
本文件是 MossFormer2 语音分离模型的主装配文件，也是本包对外的唯一入口。
整体结构（端到端）：
    波形 [B, T] -> Encoder（一维卷积把波形变成特征序列，相当于"可学习的
    STFT"）-> MaskNet（24 层 MossFormer 分离网络，估计每个说话人的掩码）
    -> mask 乘特征 -> Decoder（转置卷积把掩蔽后的特征还原成波形）
    -> 两路说话人波形（list，长度 = num_spks）。

被谁调用
========
    src/szy/separator.py -> from mossformer2.mossformer2 import MossFormer2_SS
        （load_mossformer2：用 SimpleNamespace 构造 args 后实例化并加载权重）
    src/szy/train.py / sanity_check.py / diag.py / pipeline.py
        （经 separator.py 间接使用；train.py 是域适配微调入口）

关于 __main__：本文件没有 `if __name__ == "__main__"` 训练入口——
vendor 原版的训练脚本未随包进来；本仓库的训练入口在 src/szy/train.py。

说明：本文件是 vendor 进来的第三方代码，功能零改动，只新增中文注释。
其中 math / copy 在本文件内未实际使用（vendor 遗留导入），保留不动；
GlobalLayerNorm / CumulativeLayerNorm / select_norm 与 conv_module.py
中的同名实现几乎相同（vendor 冗余，两处均保留，注释见 conv_module.py）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .mossformer2_block import ScaledSinuEmbedding, MossformerBlock_GFSMN, MossformerBlock


EPS = 1e-8


# （vendor 冗余）GlobalLayerNorm / CumulativeLayerNorm / select_norm 三个
# 定义与 conv_module.py 中的同名实现几乎相同，原理与详细注释见
# conv_module.py，这里不再重复展开。
class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


# 同 GlobalLayerNorm 的说明：与 conv_module.py 的同名实现重复（vendor 冗余）。
class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


# 同 GlobalLayerNorm 的说明：与 conv_module.py 的同名工厂函数重复。
def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


# ======================== 编码器 ========================
# Encoder：把一维波形变成二维特征序列——用"可学习的卷积"替代固定
# 的 STFT 基函数，思想与 conv_stft.py 的 ConvSTFT 一脉相承，
# 但基函数、窗函数全部由训练决定。
class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

# stride=kernel_size//2：相邻两帧重叠 50%（帧移=半窗长），
# 与经典 STFT 的 hop 设置一致；输出帧数约 T//stride。
    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
# bias=False：卷积无偏置，防止学习到的"直流偏移"干扰后续归一化。
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

# 前向：波形 -> 特征序列。
    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
# 输入约定是 [B, L]（单通道波形没有通道维），在中间补出大小为 1
# 的通道维变成 [B, 1, L]，满足 Conv1d 的输入形状要求。
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
# 一次卷积完成"分帧 + 投影"（卷积核即一组可学习的基函数）。
# ReLU 保证特征非负——与后面 MaskNet 输出的非负掩码配套：
# 特征与掩码都是非负的，乘积仍是非负的，解码稳定。
        x = self.conv1d(x)
        x = F.relu(x)

        return x


# ======================== 解码器 ========================
# Decoder：继承 nn.ConvTranspose1d（转置卷积）。把分离后的特征
# "上采样"回波形：每个特征帧展开成 kernel_size 个采样点，相邻帧
# 按 stride 错开叠加——相当于 iSTFT 的重叠相加，只不过基函数是
# 可学习的。注意：Decoder 接收 [B, N, L]，其中 N 是特征维，
# 在 MossFormer 里由 mask_net 输出的掩码特征提供。
class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

# 前向：特征 -> 波形，并整理输出维度。
    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

# 防御性检查：只接受 2 维或 3 维输入（缺少通道维时自动补上）。
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
# 转置卷积本体：输出长度约 (L-1)*stride + kernel_size。
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

# 整理维度：去掉多余的 1 维，返回 [B, T] 波形。
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


# IdentityBlock：恒等占位块。dual-path 原设计允许 intra/inter 之一
# 为恒等变换，此类保留该能力（当前主模型未使用）。注意它连
# nn.Module 都不是（没有可学习参数），只用 __call__ 模拟层接口。
class IdentityBlock:
    """This block is used when we want to have identity transformation within the Dual_path block.

    Example
    -------
    >>> x = torch.randn(10, 100)
    >>> IB = IdentityBlock()
    >>> xhat = IB(x)
    """

    def _init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x


# MossFormerM：num_blocks 个"Flash 注意力 + 门控 FSMN"块（GFSMN）
# 之后接一个最终 LayerNorm。它是 Computation_Block 里的分离内核。
class MossFormerM(nn.Module):
    """This class implements the MossFormer2 block.

    Arguments
    ---------
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = MossFormerM(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        attn_dropout = 0.1
    ):
        super().__init__()

# 核心：depth=num_blocks 的 GFSMN 块（见 mossformer2_block.py）。
        self.mossformerM = MossformerBlock_GFSMN(
                           dim=d_model,
                           depth=num_blocks,
                           group_size=group_size,
                           query_key_dim=query_key_dim,
                           expansion_factor=expansion_factor,
                           causal=causal,
                           attn_dropout=attn_dropout
                              )
# 输出前再过一层 LayerNorm（post-norm）：稳定块堆叠后的输出分布。
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    def forward(
        self,
        src,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
# 前向：直接过块 + 归一化，输入输出同形状 [B, L, N]。
        output = self.mossformerM(src)
        output = self.norm(output)

        return output

# MossFormerM2：原版 MossFormer 的 masknet 内核——只有 Flash
# 注意力层、没有 FSMN 块。当前主模型走 MossFormerM（MossFormer+），
# 本类是保留的原版路径（Computation_Block 里被注释掉的那行用的它）。
class MossFormerM2(nn.Module):
    """This class implements the MossFormer block.

    Arguments
    ---------
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = MossFormerM2(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        attn_dropout = 0.1
    ):
        super().__init__()

# 只有 depth 个 Flash 注意力层（MossformerBlock，无 FSMN）。
        self.mossformerM = MossformerBlock(
                           dim=d_model,
                           depth=num_blocks,
                           group_size=group_size,
                           query_key_dim=query_key_dim,
                           expansion_factor=expansion_factor,
                           causal=causal,
                           attn_dropout=attn_dropout
                              )
# 同样的最终 LayerNorm。
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
# 前向与 MossFormerM 相同。
        output = self.mossformerM(src)
        output = self.norm(output)

        return output

# Computation_Block："dual-path"（双路径）命名遗产——原 DPRNN 分
# intra（块内）与 inter（块间）两条处理路径，MossFormer2 只保留
# intra 路径（inter 的跨块信息由 FSMN 记忆块覆盖）。
class Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> comp_block = Computation_Block(64)
        >>> x = torch.randn(10, 64, 100)
        >>> x = comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100])
    """

    def __init__(
        self,
        num_blocks,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Computation_Block, self).__init__()

# 两条可选内核（原注释保留）：当前启用 MossFormer+（带循环/FSMN），
# 下一行注释掉的是原版 MossFormerM2。
        ##MossFormer+: MossFormer with recurrence
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        ##MossFormerM2: the orignal MossFormer
        #self.intra_mdl = MossFormerM2(num_blocks=num_blocks, d_model=out_channels)
# 是否在 intra 路径外加残差（默认开启）。
        self.skip_around_intra = skip_around_intra

        # Norm
# 可选归一化（默认 "ln"），作用在 intra 输出之后。
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

# 前向：[B, N, S] -> 换轴 -> intra 分离内核 -> 换回 -> 归一化 -> 残差。
    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, S].
            where, B = Batchsize,
               N = number of filters
               S = sequence time index 
        """
        B, N, S = x.shape
# [B, N, S] -> [B, S, N]：序列模型（注意力/FSMN）习惯"时间在前、
# 特征在后"，所以先把特征维 N 换到最后一维。.contiguous() 整理
# 内存布局（permute 只改视图，底层内存未变，部分算子要求连续张量）。
        # [B, S, N]
        intra = x.permute(0, 2, 1).contiguous() #.view(B, S, N)

# 24 层分离内核在此执行（大部分计算量都在这里）。
        intra = self.intra_mdl(intra)

        # [B, N, S]
# [B, S, N] -> [B, N, S]：换回卷积习惯的"通道在前"。
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
# intra 之后的归一化（如 norm 为 None 则跳过）。
            intra = self.intra_norm(intra)

        # [B, N, S]
# 残差连接：intra + 原始输入，恒等路径保证深层堆叠不退化。
        if self.skip_around_intra:
            intra = intra + x

        out = intra
        return out


# ======================== 分离网络（MaskNet）========================
# MossFormer_MaskNet：估计每个说话人掩码的网络本体。流程：
# 归一化 -> 1x1 卷积调通道 -> 位置编码 -> 24 层分离块 -> PReLU
# -> 输出通道扩展成 N*num_spks（每个说话人一套掩码）-> 门控输出
# -> 1x1 卷积还原 -> 整形为 [spks, B, N, L] 的掩码堆。
class MossFormer_MaskNet(nn.Module):
    """The MossFormer MaskNet for predicting mask for encoder output features.
       The MossFormer2 model uses an upgraded MaskNet structure

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> mossformer_masknet = MossFormer_MaskNet(64, 64, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = mossformer_masknet(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=24,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer_MaskNet, self).__init__()
# 说话人数（本仓库恒为 2）。
        self.num_spks = num_spks
        self.num_blocks = num_blocks
# 输入归一化（默认 "ln"）。
        self.norm = select_norm(norm, in_channels, 3)
# 1x1 卷积：调整通道数 in_channels -> out_channels（不带偏置）。
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
# 是否加全局位置编码（默认开启）。
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
# 正弦位置编码（见 mossformer2_block.py 的 ScaledSinuEmbedding）。
            self.pos_enc = ScaledSinuEmbedding(out_channels)

# 24 层分离内核（num_blocks=24），skip_around_intra 开启残差。
        self.mdl = Computation_Block(
                    num_blocks,
                    out_channels,
                    norm,
                    skip_around_intra=skip_around_intra,
                )

# 1x1 卷积把通道数扩展 num_spks 倍：每个说话人分到一套独立的
# 掩码通道。
        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
# 掩码输出前再卷积回 in_channels（与编码器输出通道数对齐）。
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
# PReLU：负半轴斜率可学习，比 ReLU 更柔和。
        self.prelu = nn.PReLU()
# 掩码的最后激活用 ReLU——保证掩码非负（理由见 Encoder 注释）。
        self.activation = nn.ReLU()
        # gated output layer
# gated output layer：门控输出层 = Tanh 的"值" x Sigmoid 的"门"，
# 让网络自己决定每个时频点的掩码值保留多少（GLU 思想）。
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

# 前向：特征 -> 掩码堆。行注释里标明了每一步之后的形状变化。
    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, S]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               S = the number of time frames
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
# [B, N, L]：先归一化（gln/cln/ln 之一）。
        x = self.norm(x)

        # [B, N, L]
# [B, N, L]：1x1 卷积调通道。
        x = self.conv1d_encoder(x)
# 位置编码：把 ScaledSinuEmbedding 生成的 [T, C] 编码加到时间轴上。
# 加之前先把特征换轴成 [B, T, C]（时间在前）以对齐位置编码的形状，
# 加完再加回原特征（base）上——位置编码是"加性"的。
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1) 
            x = base + emb
            

        # [B, N, S]
# [B, N, S]：24 层分离内核（最重的计算部分）。
        x = self.mdl(x)
# PReLU 非线性。
        x = self.prelu(x)

        # [B, N*spks, S]
# [B, N*spks, S]：通道扩展 num_spks 倍，每个说话人一套掩码通道。
        x = self.conv1d_out(x)
        B, _, S = x.shape

        # [B*spks, N, S]
# [B*spks, N, S]：把"说话人"维折进 batch 维——这样后面的卷积
# 可以并行处理所有说话人，不用写循环；最后再展开回来。
        x = x.view(B * self.num_spks, -1, S)

        # [B*spks, N, S]
# [B*spks, N, S]：门控输出（Tanh 值 x Sigmoid 门）。
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, S]
# [B*spks, N, S]：卷积回 in_channels，得到最终的掩码。
        x = self.conv1_decoder(x)

        # [B, spks, N, S]
        _, N, L = x.shape
# [B, spks, N, L]：把 batch 维重新展开成 (batch, 说话人)。
        x = x.view(B, self.num_spks, N, L)
# ReLU 保证掩码非负：掩码的语义是"每个时频点归该说话人的比例"，
# 必须 >= 0；配合编码器的 ReLU（特征非负），掩蔽结果才稳定。
        x = self.activation(x)

        # [spks, B, N, S]
# [spks, B, N, L]：说话人维移到最前——与 MossFormer.forward 里
# torch.stack([x]*num_spks) 的形状对齐，便于广播相乘。
        x = x.transpose(0, 1)

        return x

# ======================== 端到端模型 ========================
# MossFormer：把 Encoder + MaskNet + Decoder 装配成完整分离模型：
# 输入混合波形，输出一个 list（长度 = num_spks），第 i 个元素是
# 第 i 个说话人的估计波形 [B, T]。
class MossFormer(nn.Module):
    """ The E2E Encoder-MaskNet-Decoder MossFormer model for speech separation
        The MossFormer2 model uses an upgraded MaskNet
    ---------
    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the MossFormer2 blocks.
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> mossformer = MossFormer(num_spks=2)
    >>> x = torch.randn(1, 10000)
    >>> x = mossformer(x)
    >>> x
    x[0]: torch.Size([1, 10000])
    x[1]: torch.Size([1, 10000])
    """
    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        num_blocks=24,
        kernel_size=16,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer, self).__init__()
# 保存说话人数，forward 里组装输出时用。
        self.num_spks = num_spks
# 编码器：波形 -> 特征（kernel_size=16, stride=8，50% 重叠）。
        self.enc = Encoder(kernel_size=kernel_size, out_channels=in_channels, in_channels=1)
# 分离网络：特征 -> 掩码（24 层，num_spks 套掩码）。
        self.mask_net = MossFormer_MaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
# 解码器：特征 -> 波形。stride 与编码器一致（kernel//2），
# 保证特征帧与波形采样点的时间对齐关系可逆。
        self.dec = Decoder(
           in_channels=out_channels,
           out_channels=1,
           kernel_size=kernel_size,
           stride = kernel_size//2,
           bias=False
        )
# 前向：完整分离流程（encoder -> masknet -> mask 乘特征 -> decoder）。
    def forward(self, input):
# 第 1 步：编码。
        x = self.enc(input)
# 第 2 步：估计掩码 [spks, B, N, L]。
        mask = self.mask_net(x)
# 第 3 步：把编码特征沿新维度复制 num_spks 份（[spks, B, N, L]）。
        x = torch.stack([x] * self.num_spks)
# 第 4 步：逐说话人掩蔽——每个说话人的掩码乘在特征上，
# 得到"这个说话人占有的特征部分"。
        sep_x = x * mask

        # Decoding
# 第 5 步：逐说话人解码。对每个说话人的掩蔽特征做转置卷积得到
# 波形，unsqueeze(-1) 后在最后一维拼接成 [B, T, num_spks]。
        est_source = torch.cat(
            [
                self.dec(sep_x[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )
# 第 6 步：长度对齐。编码-解码过程有卷积取整，重建波形可能比
# 原始波形差几个采样点：长了裁剪、短了补零，保证输出与输入等长。
        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

# 第 7 步：拆成 list 返回，out[i] 是第 i 个说话人的波形 [B, T]。
# （separator.py 拿到的就是这两路：ms0 / ms1。）
        out = []
        for spk in range(self.num_spks):
            out.append(est_source[:,:,spk])
        return out


# ======================== 对外包装 ========================
# MossFormer2_SS：给外部调用方用的薄包装。接收一个 args 命名空间
# 对象（本仓库在 separator.py 里用 types.SimpleNamespace 构造），
# 从 args 里读出 6 个配置项组装 MossFormer；forward 直接转发。
# 这样外部不需要了解内部类的构造细节，换配置只改 args。
class MossFormer2_SS(nn.Module):
    """MossFormer2 model wrapper for outside calling"""

    def __init__(self, args):
        super(MossFormer2_SS, self).__init__()
# 用 args 的字段组装内部模型（各字段含义见 MossFormer 的注释）。
        self.model = MossFormer(
            in_channels=args.encoder_embedding_dim,
            out_channels=args.mossformer_sequence_dim,
            num_blocks=args.num_mossformer_layer,
            kernel_size=args.encoder_kernel_size,
            norm="ln",
            num_spks=args.num_spks,
            skip_around_intra=True,
            use_global_pos_enc=True,
            max_length=20000)

# 转发：输入混合波形，输出说话人波形 list。
    def forward(self, x):
        outputs = self.model(x)
        return outputs
