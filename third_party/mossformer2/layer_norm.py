#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

"""各种"通道维"层归一化的变体（配合 [B, C, T] 数据布局使用）。

在本模型里的角色
================
归一化（normalization）是深度网络训练的基石：把每层输出的分布拉回
"均值 0、方差 1"附近，可以缓解梯度消失/爆炸、加快收敛、允许更大的
学习率。语音模型习惯的数据布局是 [B, C, T]（batch, 通道, 时间帧），
而 PyTorch 自带的 nn.LayerNorm 归一化的是"最后几个维度"、nn.InstanceNorm1d
归一化的是"第 2 维（通道维）"。本文件用"先转置、再调标准层、再转置回来"
的方式，把标准归一化层改造成符合语音布局习惯的版本。

本文件最重要的知识点：为什么每个类都要 transpose？
------------------------------------------------
以 [B, C, T] 为例，transpose(1, 2) 之后变成 [B, T, C]，通道维 C 被挪到
最后一维。此时 nn.LayerNorm(dim) 归一化的对象就是"每个 (batch, 时间帧)
位置上的整条通道向量"——即跨通道归一化。做完再转置回来，对外仍是
[B, C, T]。这就是"用转置把任意维度对齐到归一化层要求的位置"的通用手法。

被谁调用
========
    mossformer2_block.py:
        Gated_FSMN_Block / Gated_FSMN_Block_Dilated -> CLayerNorm
        （ILayerNorm / GLayerNorm / GlobLayerNorm 在本仓库未被直接调用，
          是 vendor 从 MossFormer 工具链带过来的配套实现）

说明：本文件是 vendor 进来的第三方代码，功能零改动，只新增中文注释。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization."""

    # （补充说明）名字叫 "channel-wise"，但实际机制是"逐时间帧、跨通道"
    # 归一化：转置成 [N, T, C] 后，LayerNorm 对最后一维（通道维 C）求
    # 均值方差，即每个 (batch, 帧) 位置独立归一化——帧与帧之间不共享统计量，
    # 统计量也不依赖 batch（这点区别于 BatchNorm）。
    def __init__(self, *args, **kwargs):
        super(CLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        # 防御性检查：本层只接受 3 维输入 [N, C, T]，形状不对直接报错，
        # 让调用方尽早发现维度约定错误（而不是在奇怪的形状上算出错误结果）。
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # （补充说明）把通道维挪到最后一维，让父类 LayerNorm 对"每个时间
        # 帧的通道向量"求统计量（原理见文件头 docstring）。
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # LayerNorm
        sample = super().forward(sample)
        # （补充说明）还原布局再返回。
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample

class ILayerNorm(nn.InstanceNorm1d):
    """Channel-wise layer normalization."""

    # （补充说明）ILayerNorm 实际继承的是 nn.InstanceNorm1d，语义与
    # CLayerNorm 不同（vendor 沿用了同一句 docstring）：转置成 [N, T, C]
    # 后，InstanceNorm1d 对每个样本、每个通道沿长度维 T 求统计量——
    # 即"逐通道"归一化（把每个通道的能量拉到同一尺度）。instance norm
    # 完全不依赖 batch 统计量，训练/推理行为一致，适合小 batch 场景。
    def __init__(self, *args, **kwargs):
        super(ILayerNorm, self).__init__(*args, **kwargs)

    def forward(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        # 防御性检查与转置套路同 CLayerNorm，不重复展开。
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # LayerNorm
        sample = super().forward(sample)
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample

class GLayerNorm(nn.Module):
    """Global Layer Normalization for TasNet."""

    # （补充说明）"Global"指统计量是全局的：整个样本（所有帧、所有通道）
    # 只算一对均值/方差（形状 [N, 1, 1]）。这是 TasNet 系列分离网络的
    # 标配归一化——只对齐整体尺度、不破坏帧间与通道间的相对大小关系，
    # 特别适合"幅度敏感"的语音分离任务。
    def __init__(self, channels, eps=1e-5):
        super(GLayerNorm, self).__init__()
        self.eps = eps
        self.norm_dim = channels
        # gamma/beta 是可学习的逐通道缩放/平移参数（形状 [channels]）：
        # 归一化后的分布被允许"再整形"，由训练决定最合适的尺度。
        self.gamma = nn.Parameter(torch.Tensor(channels))
        self.beta = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    # 显式调用参数初始化（见下）。
    def reset_parameters(self):
        # gamma 初始化为 1、beta 初始化为 0：归一化层起步时等价于
        # "恒等变换"（乘 1 加 0），让训练从"不改变分布"开始，收敛更稳。
        # nn.init.ones_/zeros_ 是就地（in-place）初始化函数。
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # 在 (1, 2) 两维（时间 + 通道）上同时求均值/方差，keepdim=True 保留
        # 维度便于广播：每个样本（batch 中每一条）只得到一对全局统计量。
        # Mean and variance [N, 1, 1]
        mean = torch.mean(sample, (1, 2), keepdim=True)
        var = torch.mean((sample - mean)**2, (1, 2), keepdim=True)
        # 标准归一化公式 (x - mean)/sqrt(var + eps)，eps 防止方差为 0 时
        # 除零；再乘 gamma 加 beta（形状 [channels]，与最后一维对齐广播）。
        sample = (sample - mean) / torch.sqrt(var + self.eps) * \
                self.gamma + self.beta
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample

class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    # （补充说明）下划线开头的类名是 Python 惯例：表示"模块内部使用的基类"，
    # 不建议被外部直接实例化。它把 gamma/beta 参数的管理和"乘回来"的操作
    # 抽出来，供子类复用。
    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        # 可学习的逐通道缩放/平移参数，初始化为 1 和 0（恒等变换起步）。
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    # 把通道维挪到最后一维再乘 gamma 加 beta，再挪回来。
    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        # 为什么要 transpose？gamma 的形状是 [channel]，numpy/torch 的广播
        # 规则是"从最后一维往前对齐"——只有通道维在最后，[channel] 才能
        # 与 [batch, ..., channel] 正确逐通道对齐相乘。
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLayerNorm(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    # （补充说明）与上面的 GLayerNorm 等价，但实现更通用：支持任意 >2 维的
    # 输入 [batch, chan, *]，统计量在"batch 维之外的所有维"上求——
    # 同样是每个样本一对全局均值/方差。是 MossFormer 工具链中常用的写法。
    def forward(self, x):
        """ Applies forward pass.
        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        # dims = [1, 2, ..., 最后一维]：即"除 batch 维外的所有维"。
        # 在这些维上统一求统计量（每个样本一对全局统计量）。
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        # 归一化后再乘 gamma 加 beta（见基类 apply_gain_and_bias），
        # 1e-8 防除零。
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())
