"""MossFormer2 语音分离模型定义包（vendor 第三方代码，功能零改动，仅加注释）。

本目录是从开源项目（ClearerVoice-Studio / MossFormer2，作者 Shengkui Zhao 等）
vendor 进来的 MossFormer2 语音分离模型本体。在本系统中它扮演"分离前端"：
把一段混合语音盲分离成两路说话人信号（本仓库竞赛任务的前置处理步骤）。

本包不包含训练/测试入口（包内没有 __main__ 脚本），外部调用方都在本包之外：

    src/szy/separator.py    ->  from mossformer2.mossformer2 import MossFormer2_SS
                                （load_mossformer2 构建模型并加载权重）
    src/szy/pipeline.py     ->  整条处理管线中调用 separator 做盲分离
    src/szy/train.py        ->  dasynth 域适配微调的训练入口
    src/szy/sanity_check.py ->  加载模型做正确性验证
    src/szy/diag.py         ->  加载模型做诊断

包内文件分工：

    mossformer2.py       主模型装配（Encoder -> 24 层 MossFormer 分离网络
                         -> 估计 mask -> Decoder），以及对外包装类 MossFormer2_SS
    mossformer2_block.py 分离网络的核心积木（Flash 分组注意力、门控 FSMN 块、
                         位置编码、ScaleNorm 等）
    fsmn.py              FSMN 前馈序列记忆网络（用卷积滤波器捕捉时序上下文）
    conv_module.py       卷积模块库（深度/逐点卷积、各类归一化、激活函数、
                         Conformer 式卷积模块）
    layer_norm.py        各种"通道维"层归一化变体（配合 [B, C, T] 数据布局）
    conv_stft.py         用 Conv1d 实现的 STFT / iSTFT 频域前端（主模型未直接
                         使用，是配套工具与自带的自检脚本）

注意：本包是 vendor 外部代码，除注释外不允许任何功能改动。
"""
