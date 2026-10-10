# third_party/mossformer2

MossFormer2 单通道语音分离模型的 PyTorch 实现，vendor 自
[modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
（`train/speech_separation/models/mossformer2/`，2026-08-17 clone，经 gh-proxy）。

改动仅限一处：`from models.mossformer2.xxx` 绝对导入改为包内相对导入 `from .xxx`，
使其可作为独立 package 使用。模型结构、权重兼容性均未动。

配套预训练权重（不在本仓库内，需自行下载）：
ModelScope `iic/ClearerVoice-Studio` → `MossFormer2_SS_16K/last_best_checkpoint.pt`（约 670MB）。

运行时依赖（yuyin 环境已装）：`torch`、`einops`、`rotary_embedding_torch`、`torchinfo`。

用途：`src/szy_v4/` 的分离器推理与 dasynth 域适配微调训练。
