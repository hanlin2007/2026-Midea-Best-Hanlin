"""用卷积实现短时傅里叶变换（STFT）及其逆变换（iSTFT）的频域前端模块。

在本模型里的角色
================
语音分离的经典路线是：波形 -> STFT -> 时频谱 -> 分离网络估计每个说话人的
mask -> mask 乘回时频谱 -> iSTFT -> 恢复每个说话人的波形。本文件提供的就是
"STFT / iSTFT"这两个环节的卷积实现（ConvSTFT / ConviSTFT），相当于
librosa.stft / librosa.istft 的 GPU 友好版本。

被谁调用
========
本仓库的 MossFormer2 主模型（mossformer2.py）实际使用自己内置的
Encoder/Decoder（可学习的卷积编解码器，思想与本文件一脉相承），因此
本文件在本系统中主要作为配套工具与学习材料存在；文件底部的 test_*
函数是其自带的开发期自检脚本（与 librosa 对比、STFT->iSTFT 往返重建测试），
由 __main__ 入口触发。

核心思想：为什么用卷积做 FFT？
==============================
STFT 的数学本质是：把信号按 hop（帧移）滑动分帧，每帧乘窗函数后与一组
傅里叶基函数做内积（投影）。"滑动 + 内积"恰好就是一维卷积的定义——
把每个频率的基函数做成一个卷积核（kernel），F.conv1d 一次调用就能并行
完成所有帧、所有频率的投影，完全利用 GPU 上高度优化的卷积算子，比显式
写循环分帧快得多；而且卷积核可以注册成 nn.Parameter（可学习参数），
让"STFT 基底"也参与训练——这就是"可学习的 STFT"。

注意：本文件是 vendor 进来的第三方代码（源自 torch-stft 风格实现），
功能零改动，只新增中文注释。
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    """构造 STFT / iSTFT 的卷积核与窗函数，供 ConvSTFT / ConviSTFT 使用。

    Args:
        win_len: 窗长（一帧包含的采样点数，相当于 librosa 的 win_length）
        win_inc: 帧移（相邻两帧起点之间的距离，即卷积的 stride、STFT 的
                 hop_length；小于 win_len 时帧与帧之间有重叠）
        fft_len: FFT 点数（决定频域分辨率与输出频率通道数）
        win_type: 窗函数类型（'hamming'/'hanning'/None 等），传给
                  scipy.signal.get_window
        invers: False 构造正向分析核；True 构造 iSTFT 用的合成核（伪逆）

    Returns:
        kernel: 形状 [fft_len+2, 1, win_len] 的卷积核。前 fft_len//2+1 行是
                各频率的实部基，后 fft_len//2+1 行是虚部基。形状之所以是
                [out_channels, 1, kernel_size]，是因为 nn.Conv1d 的权重
                约定就是 (输出通道, 输入通道/组数, 卷积核长度)。
        window: 形状 [1, win_len, 1] 的窗向量，留给 iSTFT 做重叠相加
                归一化用。

    实现思路见函数体内的行注释。
    """
    # 1) 生成窗函数。fftbins=True 表示要"周期性"窗（periodic window）：
    #    适合做频谱分析（端点不重复加权，能量统计更一致）。
    #    win_type 为 None 或字符串 'None' 时退化为全 1 矩形窗（等价于不加窗）。
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        # **0.5：对窗开根号。正向（分析）和逆向（合成）各乘一次 sqrt(窗)，
        # 合起来等于乘一次完整窗——这是保证重叠相加（overlap-add）重建后
        # 信号幅度不失真的标准做法（满足 COLA 类条件）。
        window = get_window(win_type, win_len, fftbins=True)**0.5
    
    # 2) 构造傅里叶基矩阵。np.fft.rfft 是"实数输入的 FFT"：因为输入是
    #    实数，频谱具有共轭对称性，只保留前 N//2+1 个频率通道即可（省一半）。
    #    np.eye(N) 的每一行是一个单位脉冲；对它逐行做 rfft，结果第 i 行
    #    第 k 列就是第 k 个复指数基 e^{-j*2*pi*k*i/N} 在采样点 i 的取值，
    #    即"每个频率通道对应的基向量"。取 [:win_len] 只保留窗长内的部分。
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    # 实部、虚部沿列拼接再转置：[win_len, dim] 与 [win_len, dim] 拼成
    # [win_len, fft_len+2]，转置后每"行"是一个频率通道的基向量：
    # 前 dim 行是各频率的实部，后 dim 行是各频率的虚部。
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    
    # 3) 逆变换时改用伪逆（Moore-Penrose pseudo-inverse，最小二乘广义逆）。
    #    为什么需要伪逆：帧与帧之间有重叠（hop < win_len），正向分析矩阵
    #    不是方阵、不可逆；伪逆给出"最小二乘意义下最优"的合成核，配合
    #    后面的窗归一化，让 iSTFT 尽可能精确地重建原波形。
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    # 4) 每个频率的基向量逐点乘窗（numpy 广播：kernel 的最后一维与 window
    #    对齐相乘）。分析端与合成端各乘一次 sqrt(窗)，理由见第 1 步。
    kernel = kernel*window
    # 5) 增加"通道维"：[fft_len+2, win_len] -> [fft_len+2, 1, win_len]，
    #    对齐 nn.Conv1d 权重布局 (out_channels, in_channels/groups, kernel_size)。
    kernel = kernel[:, None, :]
    # 6) numpy -> torch 张量。先 astype(np.float32) 复制一份再 from_numpy：
    #    from_numpy 与原数组共享内存，先复制可以避免 torch 侧操作意外
    #    改动 numpy 数组（也统一成 PyTorch 默认的单精度浮点）。
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

    """正向短时傅里叶变换：把时域波形变成时频谱（幅值谱 + 相位谱）。

    对外等价于 librosa.stft，但内部是一层 F.conv1d，支持 GPU 与端到端训练。
    输入 [B, T]（或 [B, 1, T]），输出 (mags, phase) 两个 [B, F, T_out] 张量，
    其中 F = fft_len//2+1（实数谱的频率通道数），
    T_out = (T - win_len)//win_inc + 1（帧数）。
    """
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__() 
        
        # fft_len 未指定时取"不小于 win_len 的最小 2 的幂"（如 win_len=400 -> 512）。
        # 2 的幂长度是 FFT 的最快情形；这里虽是卷积实现，也沿用 STFT 的惯例。
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # 把 numpy 算好的卷积核注册成 nn.Parameter：
        #   - 注册为 Parameter（而非普通张量属性），优化器才能在
        #     model.parameters() 里找到并更新它；
        #   - requires_grad=(not fix)：fix=True 时冻结梯度（当作固定的
        #     数学变换，纯粹复刻标准 STFT）；fix=False 时卷积核参与训练，
        #     这就是"可学习的 STFT"。
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        # 输入若只有 [B, T]（缺少通道维），在第 1 维前插入大小为 1 的
        # 通道维，变成 Conv1d 要求的 [B, 1, T]。
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)

        # 一次卷积同时完成"分帧 + 加窗 + 基投影"：输出形状
        # [B, fft_len+2, T_out]，前一半通道是各频率的实部、后一半是
        # 虚部（布局见 init_kernels）。
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)
         
        # 'complex' 模式：直接返回实部/虚部拼接的结果，即"复数谱"的
        # 实虚分离表示（PyTorch 对复数支持有限，语音模型常把复数谱
        # 拆成两个实张量处理；ConviSTFT 正好按此布局接收）。
        if self.feature_type == 'complex':
            return outputs
        else:
            # 拆出实部、虚部两个 [B, F, T_out] 张量。
            dim = self.dim//2+1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            # 极坐标表示：幅值谱 = sqrt(实²+虚²)，相位 = atan2(虚, 实)。
            # 分离任务里网络一般只处理幅值谱（mask 作用在幅值上），
            # 重建波形时再用"估计幅值 + 原始相位"合成复数谱。
            mags = torch.sqrt(real**2+imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase

class ConviSTFT(nn.Module):

    """逆向短时傅里叶变换：把时频谱（幅值 + 相位）重建回时域波形。

    内部用转置卷积（ConvTranspose1d）实现"重叠相加"（overlap-add）：
    每一帧的谱先被投影回时域（一帧 -> win_len 个采样点），相邻帧的输出
    按帧移错开叠放、重叠部分相加；最后除以每个采样点被多少个窗平方
    覆盖过，消除加窗引入的幅度起伏。
    输入 [B, N+2, T]（复数谱，实虚沿通道拼接）或 [B, N//2+1, T]（幅值谱
    + phase 参数），输出重建波形 [B, 1, T_out]，T_out 约等于 (T-1)*hop+win_len。
    """
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        # invers=True：拿到伪逆合成核（为什么用伪逆见 init_kernels 第 3 步）。
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.win_inc = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        # register_buffer 注册"不参与训练、但要随模型一起管理的张量"：
        #   - window：合成窗，形状 [1, win_len, 1]，用于重叠相加归一化；
        #   - enframe：单位矩阵做成的卷积核 [win_len, 1, win_len]，作用是把
        #     每帧的窗平方按帧移"撒回"时间轴（实现重叠计数）。
        # 它们不是 Parameter，优化器不会更新；但会被 model.to(device)、
        # model.state_dict() 一并管理——这一点与普通 Python 属性完全不同。
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """ 

        # 传入了 phase 说明 inputs 是幅值谱：用 幅值*cos(相位)、幅值*sin(相位)
        # 重建实部与虚部，再沿通道维拼接，恢复成 ConvSTFT 'complex' 模式
        # 输出的那种"实虚分离复数谱"布局（前 dim 通道实部、后 dim 通道虚部）。
        if phase is not None:
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        # 转置卷积（transposed convolution，也常被叫"反卷积"）：它是普通
        # 卷积的"反向"算子——每个输出通道的卷积核把"频域的一帧"展开成
        # "时域的 win_len 个点"，stride=帧移让相邻帧的展开结果错开 hop 个
        # 点再相加。这就是 overlap-add（重叠相加）的卷积实现。
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride) 

        # ===== 重叠相加归一化（以下两行源自 torch-stft）=====
        # this is from torch-stft: https://github.com/pseeth/torch-stft
        # 1) t：把合成窗逐元素平方，并按"帧数"重复 -> [1, win_len, n_frames]。
        t = self.window.repeat(1,1,inputs.size(-1))**2
        # 2) coff：用单位矩阵核（enframe）做一次同样的转置卷积。enframe 的
        #    每个通道相当于一个"把单帧窗平方放回时间轴"的卷积核，转置卷积
        #    把所有帧的结果重叠相加，得到每个输出采样点上的
        #    "参与合成该点的所有帧的窗平方之和"（重叠计数）。
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        # 3) 除以重叠计数做归一化：消除加窗与重叠造成的幅度起伏，让
        #    iSTFT 整体成为能量守恒的重建。+1e-8 防止信号两端（覆盖不全、
        #    coff 接近 0 的位置）出现除零。下面那行被注释掉的 torch.where
        #    写法是原作者留下的另一种等价的防除零写法。
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs/(coff+1e-8) 
        return outputs

def test_fft():
    # 自检 1：把本实现的幅值谱与 librosa.stft 的结果对比，打印两者差的
    # 均方值（应接近 0）。librosa 是音频社区公认的 STFT 参考实现，用
    # 它做"标准答案"来验证卷积版 STFT 的正确性。
    # torch.manual_seed(20) 固定随机种子，保证测试可复现。
    torch.manual_seed(20)
    win_len = 320
    win_inc = 160 
    fft_len = 512
    inputs = torch.randn([1, 1, 16000*4])
    fft = ConvSTFT(win_len, win_inc, fft_len, win_type='hanning', feature_type='real')
    # import 放在函数内是"延迟导入"：只有真的运行本测试时才需要 librosa，
    # 平时 import 本模块不需要它。
    import librosa

    outputs1 = fft(inputs)[0]
    outputs1 = outputs1.numpy()[0]
    np_inputs = inputs.numpy().reshape([-1])
    # center=False：librosa 默认会在两端补零"居中"，这里关掉才能与本实现
    # （不做居中）逐点对齐比较。
    librosa_stft = librosa.stft(np_inputs, win_length=win_len, n_fft=fft_len, hop_length=win_inc, center=False)
    print(np.mean((outputs1 - np.abs(librosa_stft))**2))


def test_ifft1():
    # 自检 2：读入一段真实语音（../../wavs/ori.wav，相对本文件的位置），
    # 走一遍 STFT -> iSTFT 完整往返，把重建波形写成 conv_stft.wav 供人工
    # 试听，并打印重建误差（标签写着 MSE，实际算式是平均绝对误差，
    # vendor 原文如此，保留原样）。
    import soundfile as sf
    N = 100
    inc = 75
    fft_len=512
    torch.manual_seed(N)
#    inputs = torch.randn([1, 1, N*3000])
    data = sf.read('../../wavs/ori.wav')[0]
    inputs = data.reshape([1,1,-1])
    # 用 'complex' 模式：STFT 输出实虚拼接的复数谱，iSTFT 直接接收，
    # 中间不做幅值/相位拆分（无损往返，重建误差只来自窗归一化的数值精度）。
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    
    inputs = torch.from_numpy(inputs.astype(np.float32))
    outputs1 = fft(inputs)
    outputs2 = ifft(outputs1)
    sf.write('conv_stft.wav', outputs2.numpy()[0,0,:],16000)
    # 重建长度可能与原长差几个采样点（卷积取整导致），比较时先截齐。
    print('wav MSE', torch.mean(torch.abs(inputs[...,:outputs2.size(2)]-outputs2)))


def test_ifft2():
    # 自检 3：与 test_ifft1 相同的往返测试，但输入换成人造带限噪声
    # （随机数缩放后裁到 [-1,1]），不依赖外部音频文件，适合自动化验证。
    N = 400
    inc = 100
    fft_len=512
    np.random.seed(20)
    torch.manual_seed(20)
    t = np.random.randn(16000*4)*0.005
    t = np.clip(t, -1, 1)
    #input = torch.randn([1,16000*4]) 
    input = torch.from_numpy(t[None,None,:].astype(np.float32))
    
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    
    out1 = fft(input)
    output = ifft(out1)
    print('random MSE', torch.mean(torch.abs(input-output)**2))
    import soundfile as sf
    sf.write('zero.wav', output[0,0].numpy(),16000)


if __name__ == '__main__':
    # ===================== 测试入口说明 =====================
    # `if __name__ == "__main__":` 是 Python 的固定套路：
    #   - 直接运行本文件（python conv_stft.py）时，解释器把 __name__ 置为
    #     字符串 "__main__"，条件成立，执行下面这些开发期自检脚本；
    #   - 本文件被其他模块 import 时，__name__ 是模块名 "conv_stft"，
    #     条件不成立，测试脚本不会执行。
    # 这样同一个文件既能当库用（被 import），又能当脚本用（直接跑测试），
    # 互不干扰。想跑哪个自检就取消哪一行的注释。
    #test_fft()
    test_ifft1()
    #test_ifft2()
