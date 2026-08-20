"""SZY v4：szy_v2 全管线 + dataset_dasynth 域适配微调的 MossFormer2 分离前端。

动机：离线 MossFormer2 只在干净语料上预训练，在 DatasetA 风格的混响/掩盖
输入上分离质量崩塌（dasynth valid 实测 reverb 子集 SI-SDR 比混合还差
6~14dB），而分离输出同时是唤醒锚点选择、融合门控特征、候选 ASR 三个环节
的共同输入。szy_v4 不动任何后端逻辑，只把分离器换成在 dasynth（18k 条
DatasetA 风格合成对，腐蚀前干净源为训练目标）上微调后的 torch 版本。

组成：
  - ``metrics.py``           分离质量指标（si_sdr / oracle_sisdr / si_snr /
                             pit_sisnr_loss 的唯一实现，供其余文件共享）；
  - ``separator.py``        torch 分离器（与 ONNX 版同接口，产线接入点）；
  - ``train_mossformer2.py`` dasynth 域适配微调训练（PIT SI-SNR）；
  - ``diag_sisdr.py``        分离质量诊断（按腐蚀类型分组的 SI-SDR 水位）；
  - ``sanity_check.py``      torch 权重 vs 产线 ONNX 的对照验证；
  - ``pipeline.py``          产线入口（复用 szy_v2 主流程，配置指向 .pt 即启用）。

数据集与权重见 docs/ 与 third_party/mossformer2/README.md。
"""
