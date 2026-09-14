"""FSMN（Feedforward Sequential Memory Network，前馈序列记忆网络）模块。

在本模型里的角色
================
FSMN 的核心思想：用"可学习的 FIR 卷积滤波器"（称为记忆块，memory block）
代替 RNN 的循环连接来捕捉时序上下文。每个时刻的输出可以依赖前后若干帧
（lorder 阶的上下文），但所有帧是并行计算的——没有 RNN 那种"必须一帧
一帧串行喂"的依赖，训练和推理都快得多。MossFormer2 里，本文件的类被
mossformer2_block.py 的 Gated_FSMN / Gated_FSMN_dilated 包装成"门控记忆块"，
与 Flash 分组注意力交替堆叠，组成分离网络的主体。

被谁调用
========
    mossformer2_block.py:
        Gated_FSMN             -> UniDeepFsmn
        Gated_FSMN_dilated     -> UniDeepFsmn_dilated（内部用 DilatedDenseNet）
        MossformerBlock_GFSMN / MossformerBlock 的 _build_repeats -> UniDeepFsmn

说明：本文件是 vendor 进来的第三方代码，功能零改动，只新增中文注释。
其中 np / os / Parameter 在本文件内未实际使用（vendor 遗留导入），保留不动。
"""
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.nn.parameter import Parameter
import numpy as np
import os

class UniDeepFsmn(nn.Module):
    """基础 FSMN 记忆块：线性投影 + 沿时间的逐通道卷积（记忆滤波器）+ 残差。

    输入输出都是 [B, T, C]（batch, 时间帧, 特征维），输出与输入同形状——
    这是"块"的基本要求，保证它可以像积木一样随意堆叠。
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # 注意这个 early return：lorder 为 None 时直接返回、模块处于
        # "半初始化"状态（后续属性都不存在）。这是 vendor 的防御性写法，
        # 实际调用方（Gated_FSMN 等）总是显式传 lorder，正常不会触发。
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        # 线性层把特征升维到 hidden_size：先在更大的空间里做非线性变换，
        # 记忆块才有足够的容量建模。
        self.linear = nn.Linear(input_dim, hidden_size)
        # 投影回 output_dim。bias=False：偏移项由记忆卷积的"直流分量"
        # （卷积核中心附近权重）承担，避免参数冗余。
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        # 记忆卷积：Conv2d 的卷积核形状 [2*lorder-1, 1] —— 沿时间维看
        # 2*lorder-1 帧（前后各 lorder-1 帧 + 当前帧），高度维尺寸 1 不滑动。
        # groups=output_dim 是分组卷积：每个通道一个独立滤波器（等价于
        # 逐通道一维卷积 / depthwise），参数极少、通道间互不干扰。
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):
        # 输入形状约定为 [B, T, C]。
        # 第 1 步：升维 + ReLU 非线性。
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        # 第 3~6 步是把 [B, T, C] 摆成 Conv2d 认识的 [B, C, T, 1]：
        #   unsqueeze 在 C 前插入高度维 -> [B, 1, T, C]
        #   permute(0, 3, 2, 1) 把 C 换到第 2 维（通道维）-> [B, C, T, 1]
        # 这样卷积核 [2*lorder-1, 1] 就沿 T（宽维）滑动、在高度维（尺寸 1）
        # 上不滑动——等效于"对每个通道沿时间做一维卷积"。
        x = th.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        # 时间维两侧各补 lorder-1 个零（'SAME' 填充）：卷积后输出与输入
        # 等长，且当前帧能同时看到"过去 lorder-1 帧"和"未来 lorder-1 帧"
        # （非因果、双向上下文）。语音分离一般离线处理整段录音，
        # 允许看未来帧，所以这里用两侧对称填充而非只补左侧。
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        # 记忆卷积 + 残差：x_per + 卷积结果。残差结构让记忆块只学习
        # "相对原特征的修正量"，梯度可以沿恒等路径直通，深层网络更好训练。
        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        # 模块级残差连接：与内部残差呼应，保证堆叠多层时信息不衰减。
        return input + out1.squeeze()

class UniDeepFsmn_dual(nn.Module):
    """两级记忆卷积的 FSMN：第二级在"每 4 个通道一组"的组内混合通道。

    与 UniDeepFsmn 的区别：级联了第二个记忆卷积 conv2，且它的分组数只有
    第一级的 1/4（groups=output_dim//4）——第一级每个通道完全独立，
    第二级则让每 4 个通道共享一组滤波器，引入少量通道间交互，
    同时扩大有效感受野（两级卷积的感受野叠加）。
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn_dual, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # 与 UniDeepFsmn 相同的防御性 early return，正常调用不触发。
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        # 第一级记忆卷积：逐通道独立（groups=output_dim）。
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim, bias=False)
        # 第二级记忆卷积：每 4 个通道一组（groups=output_dim//4），组内共享
        # 滤波器，因此参数更少且能实现组内通道信息交换。
        self.conv2 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim//4, bias=False)

    def forward(self, input):

        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        # [B, 1, T, C] -> [B, C, T, 1]：让时间维 T 成为卷积滑动的宽维
        # （permute/unsqueeze 的意义与 UniDeepFsmn 相同，后同）。
        x = th.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        # 第一级：记忆卷积 + 残差。
        conv1_out = x_per + self.conv1(y)
        # 第二级输入同样先做两侧 'SAME' 填充。
        z = F.pad(conv1_out, [0, 0, self.lorder - 1, self.lorder - 1])
        # 第二级：直接作用在第一级输出上（仍是线性滤波 + 残差，非线性只
        # 来自最前面的 ReLU），两层记忆块叠加，感受野翻倍。
        out = conv1_out + self.conv2(z)
        out1 = out.permute(0, 3, 2, 1)
        # 模块级残差（同 UniDeepFsmn）。
        return input + out1.squeeze()

class DilatedDenseNet(nn.Module):
    """膨胀稠密卷积网络：用 4 层（depth=4）膨胀卷积 + 稠密连接捕捉长程上下文。

    两个关键技巧：
    - 膨胀（dilation）：第 i 层的卷积核采样点之间隔 2**i 个点，感受野随层数
      指数增长（第 4 层能"看"到很远的帧），但参数量不随感受野增加——
      用很小的代价换取长程上下文。
    - 稠密（dense）：第 i 层的输入是之前所有层输出的拼接
      （通道数 in_channels*(i+1)），信息流最大化、梯度传播更顺畅。
    每层之后跟 InstanceNorm + PReLU 提供归一化与非线性。
    """
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        # 注意：self.pad / self.twidth 在 forward 里没有被使用（vendor
        # 遗留属性，每层实际用的是下面循环里注册的 pad1..padN），保留不动。
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = lorder*2-1
        self.kernel_size = (self.twidth, 1)
        for i in range(self.depth):
        # 膨胀系数按 2 的幂增长：第 0 层普通卷积，之后逐层翻倍。
            dil = 2 ** i
        # pad_length 公式同时保证两点：卷积输出与输入等长；只往"过去"
        # 一侧多补零（因果式填充），且补的量正好覆盖膨胀后感受野
        # 需要的范围，使每一层都能看到与第 1 层同样多的历史帧。
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
        # setattr 动态注册子模块：nn.Module 重载了 __setattr__，凡是
        # nn.Module 类型的属性都会被自动"登记"进模型，其参数才会出现在
        # model.parameters() 里、被优化器更新、随 .to(device) 搬移。
        # 用循环 + setattr 代替手写 pad1/conv1/... 更省代码，效果等价。
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels*(i+1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1), groups=self.in_channels, bias=False))
        # InstanceNorm2d：逐样本、逐通道算均值方差。语音是变长序列、
        # batch 常常很小（甚至 1），InstanceNorm 不依赖 batch 统计量，
        # 比 BatchNorm 更适合这种场景。affine=True 额外学一个缩放/平移。
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
        # PReLU：负半轴斜率可学习的 ReLU，比 ReLU 更灵活（负数区域
        # 不是一刀切为 0，而是保留一个可学习的比例）。
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        # 输入 x 形状 [B, C, T, 1]（由调用方 UniDeepFsmn_dilated 摆好）。
        skip = x
        for i in range(self.depth):
        # 每一步：padding -> 膨胀卷积 -> 归一化 -> 激活，
        # 然后把新特征拼回 skip（稠密连接），作为下一层的输入。
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)            
        # 稠密连接：新特征与旧特征沿通道维拼接。
            skip = th.cat([out, skip], dim=1)
        return out

class UniDeepFsmn_dilated(nn.Module):
    """膨胀版 FSMN 记忆块：把 UniDeepFsmn 的记忆卷积换成 DilatedDenseNet。

    前缀步骤（升维 -> ReLU -> 投影 -> 摆成 [B, C, T, 1]）与 UniDeepFsmn
    完全一致，只是记忆部分换成膨胀稠密卷积网络，感受野更大、层间信息
    流动更丰富；结尾同样是模块级残差 input + out。
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, depth=2):
        super(UniDeepFsmn_dilated, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth=depth
        # 同样保留 vendor 的防御性 early return，正常调用不触发。
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        # 记忆部分：膨胀稠密卷积网络（depth 层），见 DilatedDenseNet。
        self.conv = DilatedDenseNet(depth=self.depth, lorder=lorder, in_channels=output_dim)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = th.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        # DilatedDenseNet 内部自带因果 padding，这里无需再手动 pad。
        out = self.conv(x_per)
        out1 = out.permute(0, 3, 2, 1)

        # 模块级残差（同 UniDeepFsmn）。
        return input + out1.squeeze()
