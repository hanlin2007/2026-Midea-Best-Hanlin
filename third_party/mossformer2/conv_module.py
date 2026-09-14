"""卷积模块库：归一化层、激活函数、深度可分离卷积与 Conformer 式卷积模块。

在本模型里的角色
================
本文件是 MossFormer 工具链的"基础建材仓库"：
- 归一化：GlobalLayerNorm / CumulativeLayerNorm / select_norm（工厂函数）
- 激活与门控：Swish、GLU（门控线性单元）
- 卷积积木：DepthwiseConv1d（深度卷积）、PointwiseConv1d（逐点卷积）、
  ConvModule（Conformer 风格的卷积模块，带残差）
- 组合件：DilatedDenseNet（膨胀稠密卷积）、FFConvM_Dilated（前馈 + 膨胀卷积）

被谁调用
========
    mossformer2_block.py:
        FLASH_ShareA_FFConvM 内的 FFConvM -> ConvModule
        FFConvM_Dilated -> DilatedDenseNet / nn.SiLU
        （GlobalLayerNorm/CumulativeLayerNorm/select_norm/Swish/GLU/Linear/
          DepthwiseConv1d/PointwiseConv1d 在本仓库未被直接调用，是 vendor
          工具链自带的完整实现；mossformer2.py 里另有一份几乎相同的
          GlobalLayerNorm/CumulativeLayerNorm/select_norm，属 vendor 冗余。）

说明：本文件是 vendor 进来的第三方代码，功能零改动，只新增中文注释。
其中 torch.nn.functional（F）在本文件内未实际使用（vendor 遗留导入）。
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F

# 数值稳定小常数：加在分母/方差里防止除零（1e-8 足够小，不影响精度）。
EPS = 1e-8

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

    # （补充说明）"Global"指统计量是全局的：在"除 batch 维与通道维之外的
    # 所有维"上求均值/方差，每个样本只得到一对全局统计量（形状 [N,1,1]
    # 或 [N,1,1,1]），只对齐整体尺度、不破坏帧间与通道间的相对关系。
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # weight/bias 的形状做成 [dim, 1] 或 [dim, 1, 1]：第 1 维是通道
        # 维（与输入第 2 维对齐），其余维为 1，广播时正好"逐通道"缩放。
        # shape 参数告诉本层输入是 3 维还是 4 维。
        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
        # elementwise_affine=False 时不需要可学习参数：用
        # register_parameter(name, None) 显式登记"没有这个参数"，
        # 这样 state_dict/parameters 的行为才与参数存在时一致。
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
        # 3 维输入 [N, C, L]：在 (1, 2) 两维（通道 + 时间）上求全局统计量。
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

        # 4 维输入 [N, C, K, S]：同理在 (1, 2, 3) 上求全局统计量。
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

    # （补充说明）名字源自 DPRNN 论文的 cLN（cumulative layer normalization，
    # 推理时用随序列累计的统计量做流式归一化）。这里的实现做了简化：继承
    # nn.LayerNorm，把通道维挪到最后一维后做"逐时间帧、跨通道"归一化
    # （与 layer_norm.py 的 CLayerNorm 机制相同），统计量不累计。
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
        # 4 维输入：permute 把通道维 C 挪到最后一维，让 LayerNorm 对每个
        # (N, K, S) 位置的通道向量归一化。
        if x.dim() == 4:
        # .contiguous() 保证转置后的张量在内存中连续（转置只是改了"视图"，
        # 内存布局没变；很多算子要求连续张量，这里主动整理好）。
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        # 3 维输入 [N, C, L]：同样的"转置 -> 归一化 -> 转回"套路。
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


# （补充说明）工厂函数：根据字符串名字返回对应的归一化层实例。
# 好处是调用方可以用配置项（字符串）选择归一化类型，而不是硬编码类名。
# 'ln' 返回 nn.GroupNorm(1, dim)：GroupNorm 的组数设为 1 时，
# 统计量在"整个通道 + 空间"上求——行为等价于 LayerNorm。
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

# （补充说明）Swish 公式：f(x) = x * sigmoid(x)。它是"带门控的恒等"：
# x 大且正时 sigmoid→1，输出≈x（保留信息）；x 很负时 sigmoid→0，
# 输出≈0（抑制）。全程光滑可导，梯度不会像 ReLU 那样在 0 点断裂。
# PyTorch 内置别名是 nn.SiLU。
class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    # forward 的参数/返回值类型标注（Tensor）是 PyTorch 官方推荐的
    # 写法，方便 IDE 提示与静态检查，运行时不强制。
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


# （补充说明）GLU（门控线性单元）：把输入沿指定维切成两半，
# 一半当"值"、一半当"门"（sigmoid 压到 (0,1) 区间），逐元素相乘。
# "门"决定每个位置让多少信息通过——0 表示完全关断、1 表示完全放行。
# 语音分离里门控是控制信息流的核心手段（哪些时频点属于哪个说话人，
# 本质就是一组 0/1 附近的掩码）。
class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    # chunk(2, dim) 沿 dim 维把张量等分成两块（要求该维长度是偶数）。
    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

# （补充说明）为什么要把转置包成 nn.Module？因为 nn.Sequential 只接受
# Module 对象，不接受 torch.transpose 这样的裸函数——包装成类之后，
# 转置就可以作为"一层"放进 Sequential，随模型一起管理。
class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    # *self.shape 解包元组：transpose(0, 1) 表示交换第 0、1 维。
    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

# （补充说明）这个包装类存在的意义是"初始化"：默认 nn.Linear 的初始化
# 未必理想，这里改用 Xavier 初始化（见下），bias 置零。
class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Xavier（Glorot）初始化：把权重采样自均匀分布 U(-a, a)，其中
        # a 与输入/输出神经元数相关（sqrt(6/(fan_in+fan_out))）。
        # 目的：让前向传播的激活值方差、反向传播的梯度方差在网络各层之间
        # 大致保持稳定，避免"方差爆炸/消失"导致训练困难。
        # 名字带下划线结尾（xavier_uniform_）表示 in-place 操作，直接
        # 改写传入张量的值。
        init.xavier_uniform_(self.linear.weight)
        # 偏置置零：从"无偏移"的对称起点开始训练。
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

# （补充说明）深度卷积：groups=in_channels 意味着"每个输入通道配一组
# 独立的卷积核"，通道之间完全不相通——因此只做"时间邻域"上的滤波，
# 不混合通道。参数量是普通卷积的 1/out_channels，是轻量化的关键手段。
# 每个输入通道可以输出 K 个通道（out_channels = K * in_channels），
# 这也是构造里 assert 的要求。
class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        # assert 是"约定检查"：分组卷积要求输出通道数是输入通道数的整数倍，
        # 不满足说明调用方传错了参数，宁可当场报错也不让模型带病运行。
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        # groups=in_channels 即深度卷积的写法（见类注释）。
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


# （补充说明）逐点卷积：kernel_size=1，即"每个时间点独立地做一次
# 跨通道的线性组合"，完全不做时间邻域滤波。它常与 DepthwiseConv1d
# 配对使用，合称"深度可分离卷积"——先深度卷积处理时间、再逐点卷积
# 混合通道，参数比等价的大卷积核少一个数量级。也常用于改变通道数。
class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        # kernel_size=1：逐点卷积（见类注释）。
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


# （补充说明）Conformer 是语音识别里的经典结构（Transformer + 卷积），
# 它的卷积模块 = 逐点升维 -> GLU -> 深度卷积 -> BN -> Swish -> 逐点降维。
# 这里做了大幅简化：只保留"深度卷积 + 残差"两步（升维/门控/逐点卷积
# 都被省略），输入输出形状不变 [B, T, dim]，输入时间在前、特征在后，
# 所以内部先用 Transpose 转成卷积习惯的 [B, dim, T]。
class ConvModule(nn.Module):
    """
    Modified from Conformer convolution module to perform depth-wise convolution
    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 17, 
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        # 两个约定检查：'SAME' 填充（输出与输入等长）要求卷积核长度为奇数，
        # 才能左右对称地各补 (kernel_size-1)//2 个零；本实现只支持
        # expansion_factor=2（原版 Conformer 升维 2 倍的结构）。
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        # nn.Sequential：按顺序串起若干层，前一层输出直接喂给下一层，
        # 不需要自己写 forward 里的逐层调用。这里只放两层：
        #   Transpose(1,2)：[B, T, C] -> [B, C, T]（卷积要求通道在前）
        #   DepthwiseConv1d：沿时间的深度卷积（padding=(k-1)//2 保持等长）
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # 卷积后的结果转回 [B, T, C]，再与原始输入相加——残差连接：
        # 模块只学习"修正量"，梯度直通，深层堆叠不退化。
        return inputs + self.sequential(inputs).transpose(1, 2)

# （补充说明）与 fsmn.py 里同名类几乎相同（vendor 重复实现），
# 唯一的区别：forward 开头多了一步 unsqueeze，直接接受 3 维输入
# [B, T, C] 而不是 [B, C, T, 1]。膨胀卷积 + 稠密连接的原理见
# fsmn.py 中同名类的注释。
class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad / self.twidth 在 forward 中未使用（vendor 遗留属性）。
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = lorder*2-1
        self.kernel_size = (self.twidth, 1)
        for i in range(self.depth):
        # 膨胀系数按 2 的幂增长。
            dil = 2 ** i
        # 因果式填充量公式：保证输出等长且覆盖膨胀后的感受野。
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
        # setattr 动态注册子模块（nn.Module 会自动登记，原理见 fsmn.py）。
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels*(i+1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1), groups=self.in_channels, bias=False))
        # InstanceNorm2d：逐样本逐通道归一化，适合小 batch（原理见 fsmn.py）。
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
        # PReLU：负半轴斜率可学习。
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

        # 输入 [B, T, C]：先补通道维再摆成 [B, C, T, 1]（套路同 fsmn.py）。
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x_per = x.permute(0, 3, 2, 1)
        # skip 保存当前全部特征（稠密连接的起点）。
        skip = x_per
        for i in range(self.depth):
        # 每一步：padding -> 膨胀卷积 -> 归一化 -> 激活（同 fsmn.py）。
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
        # 稠密连接：新特征与旧特征沿通道维拼接，喂给下一层。
            skip = torch.cat([out, skip], dim=1)
        out1 = out.permute(0, 3, 2, 1)
        return out1.squeeze(1)

# （补充说明）前馈层 + 膨胀稠密卷积的组合模块（FFN 的卷积增强版）。
# 流程：LayerNorm（pre-norm，先归一化再变换）-> Linear 升维 -> SiLU
# 非线性 -> DilatedDenseNet 捕捉长程上下文 -> Dropout 防过拟合。
# pre-norm 是现在的主流：把归一化放在残差分支里（而不是输出之后），
# 训练更稳定。与 mossformer2_block.py 的 FFConvM 的区别是中间用
# 膨胀稠密卷积代替普通 ConvModule。
class FFConvM_Dilated(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        # norm_klass 是"类"而不是实例：默认 nn.LayerNorm，调用时用
        # norm_klass(dim_in) 实例化——允许调用方换成别的归一化类（如
        # ScaleNorm），这是"把类当参数传"的常见设计。
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            DilatedDenseNet(depth=2, lorder=17, in_channels=dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

