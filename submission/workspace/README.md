# 抗干扰语音指令识别交付目录

本目录是 `lsz_v4_5`（V100 基线及双单人独立路由）的 Docker 构建上下文。
当前阶段已整理推理代码、配置和模型资源，但尚未构建或导出 Docker 镜像。

## 1. 目录结构

```text
/workspace/
├── README.md
├── demo_run_test.sh
├── run_test.py
├── requirements.txt
├── config/                 # 运行配置、门控配置
├── model/                  # 全部权重，运行时不下载
├── src/                    # 最新推理流程
├── third_party/            # MossFormer2 与 FireRedASR2 源码
├── vendor/                 # 离线 Python 二进制依赖
├── doc/                    # 参赛方案说明材料
├── cache/                  # 运行时分离/ASR 缓存
└── outputs/                # result.json
```

## 2. 输入格式

`--test-json` 支持 JSON 数组或 JSONL。每条样本至少包含：

```json
{
  "id": 1,
  "唤醒音频": "pos/kws_1.wav",
  "唤醒文本": "hi colmo",
  "识别音频": "pos/cmd_1.wav"
}
```

音频相对路径以 `--data-root` 为根目录。输入应为可读取的 WAV；当前模型工作在
16 kHz 单声道域，双声道输入会在前端转为单声道，但不负责采样率转换。

## 3. 运行方式

```bash
DATA_ROOT=/data/dataset \
TEST_JSON=/data/dataset/test.jsonl \
OUTPUT=/workspace/outputs/result.json \
DEVICE=cuda:0 \
bash /workspace/demo_run_test.sh
```

性能统计：

```bash
PROFILE=1 DATA_ROOT=/data/dataset TEST_JSON=/data/dataset/test.jsonl \
bash /workspace/demo_run_test.sh
```

输出格式：

```json
{
  "result": {
    "results": [
      {"id": 1, "content": "灯光亮度调到百分之三十"}
    ]
  }
}
```

开启 `PROFILE=1` 后还会输出 `avg_dur`、`cpu_mem` 和 `gpu_mem`。

## 4. 模型与流程

- MossFormer2：对唤醒和命令音频分别进行双路分离；
- 拼音归一化唤醒选源：用唤醒文本选择目标说话人锚点；
- ERes2NetV2：预训练与分离域微调模型构成六维融合拒识门控；
- 双单人短路：分类为唤醒单人且命令单人时，使用微调 ERes2NetV2 的
  raw/raw 余弦和固定阈值 `0.29`；
- FireRedASR2-AED：beam=3，复用 N-best，最终选中通道再进行 ELM 重解码；
- 每次只处理一条样本（batch=1），模型仅在 `load_model` 中初始化一次。

## 5. 可复现与离线约束

- 默认随机种子为 `2026`；
- `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；
- ModelScope 模型使用交付目录内的本地权重；
- 推理期间不允许安装依赖或下载模型；
- `cache/` 和 `outputs/` 必须可写。

## 6. Docker 阶段待办

后续使用 CUDA 不高于 12.4 的 Ubuntu 20.04 基础镜像，离线安装
`requirements.txt` 对应的 wheel。当前随目录保存的 `kaldi_native_fbank` 二进制
扩展使用 CPython 3.12 ABI，因此 Docker 阶段应固定 Python 3.12，或在镜像内
重新安装与所选 Python 版本匹配的离线 wheel。完成容器内冒烟测试后，再执行：

```bash
docker save -o 申请人.tar 镜像名:标签
```

最终文件必须按实际申请人姓名重命名，不能直接使用“申请人.tar”。
