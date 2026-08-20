# SZY v1 本地复现完整指南

本指南带你从零把仓库跑起来，在自己的电脑上复现队长冻结的 DatasetA 结果
（CER `0.3174937` / PR `0.8790323` / RR `0.8396624`）。

---

## 0. 复现需要哪些东西（先看清全局）

| 资产 | 状态 | 获取方式 |
|---|---|---|
| 代码（本仓库） | ✅ 已就位 | 本仓库 |
| DatasetA 数据集 | ✅ 已就位 | 你已上传 zip，已解压到 `dataset/` |
| ERes2NetV2 / CAM++ / MossFormer2 | ⬜ 需下载 | `scripts/download_models.py` 自动下载 |
| `backend_dual.pt`（双 MLP 门控权重） | ⬜ 需复制 | 队长工作区（私有产物，无法自动下载） |
| FireRedASR2-AED 模型 + FireRedASR2S 源码 | ⬜ 需复制/获取 | 队长工作区（见第 5 节） |
| Python 环境（torch/modelscope/onnxruntime 等） | ⬜ 需安装 | 第 2 节 |

---

## 1. 硬件与系统要求

- **强烈建议有 NVIDIA GPU**（CUDA ≥ 11.8，显存 ≥ 8GB）。全量 1838 条在单卡上约 18 分钟。
- **纯 CPU 也能跑通，但很慢**：FireRed AED 在 CPU 上单条可能数秒~数十秒，全量会非常久。
  建议 CPU 环境只用 `--limit 5` 做冒烟验证链路是否通。
- **操作系统**：Windows / Linux 均可。但注意——FireRed 的依赖 `kaldi_native_fbank`
  在 Windows 上**可能没有预编译 wheel**（见第 5、8 节），若安装失败建议改用
  WSL2 或一台 Linux 机器（如 AutoDL）跑全量。

---

## 2. 安装 Python 环境

建议用 conda 建独立环境（避免污染你已有的 PyTorch 环境）：

```bash
conda create -n szy_v1 python=3.10 -y
conda activate szy_v1
```

装依赖（**先装 torch，再装 requirements**，避免 torch 从错误源重复下载）：

```bash
# 有 GPU（CUDA 12.8 为例；你的 CUDA 版本不同请去 pytorch.org 查对应命令）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
pip install -r requirements.txt
```

> `requirements.txt` 里第一行 `--extra-index-url` 是为 torch 准备的备用源；
> 若你已手动装好 torch，可忽略。`onnxruntime` 若要用 GPU 加速 Moss 分离，
> 另装 `pip install onnxruntime-gpu`（版本需与 CUDA 匹配）。

---

## 3. 下载公开模型（ERes2NetV2 / CAM++ / MossFormer2）

在仓库根目录执行：

```bash
python scripts/download_models.py
```

它会用 ModelScope 把 3 个公开模型下载到 `model/workspace/midea/models/modelscope/`，
并自动校验 MossFormer2 `simple_model.onnx` 的 sha256。

> 前提：`pip install "modelscope>=1.17.0"`，且系统装有 `git` + `git-lfs`
> （ModelScope 大文件走 LFS）。Windows 可 `winget install Git.LFS` 后
> `git lfs install`。

---

## 4. 数据集（你已就位）

你上传的 `datasetA/dataset.zip` 已解压并整理到：

```text
dataset/dataset_test/datasetA/datasetA/
├── pos.jsonl          # 1364 条正样本
├── neg.jsonl          # 474 条负样本
├── pos/               # 1364×2 个 wav（kws + cmd）
├── neg/               # 474×2 个 wav
├── data_reader.py     # 官方读数据示例（保留作参考）
└── readme.txt(.bak)
```

JSONL 字段与代码预期完全一致：`id / 唤醒音频 / 唤醒文本 / 识别音频 / 识别文本`。
无需任何改动。原始的 `datasetA/dataset.zip` 仍保留在你放置的位置，可自行删除以省空间。

---

## 5. 从队长工作区复制的私有资产（关键）

以下资产无法自动下载，需要你从队长的工作区（`/home/hustdun/szy_workspace`）
或队长提供的网盘里复制，放到对应位置：

### 5.1 双 MLP 门控权重（必选，缺了 SZY v1 跑不了）

```
源（队长工作区）: szy_workspace/sv_backend/backend_dual_v2_out/backend_dual.pt
目标（本仓库）  : model/workspace/sv_backend/backend_dual_v2_out/backend_dual.pt
```

复制后校验 sha256（应与 `58eefbda...72d2ec6b` 一致，`check_models.py` 会自动验）：

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('model/workspace/sv_backend/backend_dual_v2_out/backend_dual.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

### 5.2 FireRedASR2-AED 模型 + FireRedASR2S 源码（必选）

```
目标目录: model/external/FireRedASR2-AED   （模型权重目录）
          model/external/FireRedASR2S      （推理源码，含 examples_infer/asr/fireredasr2）
          model/runtime/vendor             （依赖 kaldi_native_fbank）
```

**方式 A（推荐）：直接从队长工作区复制。** 队长工作区里这三处分别对应：
`/home/hustdun/models/FireRedASR2-AED`、`/home/hustdun/zhy_workspace/FireRedASR2S`、
以及 vendor 目录。

**方式 B（无法拿到队长文件时的备选）：** FireRedASR 是小红书开源的 ASR 系统，
GitHub 仓库 `FireRedTeam/FireRedASR`、HuggingFace 上有 `FireRedTeam/FireRedASR-AED`
系列权重。注意：本仓库的封装 `FireRedRecognizer` 按队长 fork 的接口写
（`from fireredasr2.asr import FireRedAsr2`、`FireRedAsr2.from_pretrained("aed", ...)`），
官方原版仓库的包名/接口可能不同，需要你核对 `examples_infer/asr` 下的实际导入路径后微调。
**若你拿不到队长的 FireRedASR2S，这一步是你最需要向队长确认的地方。**

`kaldi_native_fbank` 可尝试 `pip install kaldi_native_fbank`（或装进 vendor：
`pip install --target model/runtime/vendor kaldi_native_fbank`）。Windows 若无对应
wheel，请改用 WSL2/Linux。

---

## 6. 运行

### 6.1 自检（先跑这个）

```bash
python scripts/check_models.py
```

全部 `[ok]` 后继续；有问题它会列出缺失项和 sha256 差异。

### 6.2 冒烟测试（5 条，验证链路）

**推荐方式：用仓库根目录的启动器 `run_szy_v1.py`**（自动把 `src/` 加入模块路径并切到仓库根，
无需 `PYTHONPATH`，Windows cmd / PowerShell / WSL / Git Bash 通用）：

```bash
python run_szy_v1.py --data-root dataset/dataset_test/datasetA/datasetA \
  --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl \
  --jsonl dataset/dataset_test/datasetA/datasetA/neg.jsonl \
  --limit 5 --output outputs/smoke_submission.json \
  --final-output outputs/smoke_detail.json \
  --checkpoint-output outputs/smoke_checkpoint.json \
  --manifest-output outputs/smoke_manifest.json
```

**等价方式一（一次性安装后可用 `python -m`）：**

```bash
pip install -e .          # 开发模式安装本包，之后任何目录都能 import midea_qiji / szy_v1
python -m szy_v1.pipeline --data-root ... --jsonl ...   # 其余参数同上
```

**等价方式二（Git Bash / Linux 的 shell 脚本）：**

```bash
bash scripts/run_szy_v1.sh --limit 5 \
  --output outputs/smoke_submission.json \
  --final-output outputs/smoke_detail.json \
  --checkpoint-output outputs/smoke_checkpoint.json \
  --manifest-output outputs/smoke_manifest.json
```

> 注意：用 `python -m szy_v1.pipeline` 或 shell 脚本时，**必须在仓库根目录执行**（配置里的
> `model/...`、`outputs/...` 都是相对仓库根解析的）；`run_szy_v1.py` 会自动切目录，不挑位置。

### 6.3 全量运行

```bash
python run_szy_v1.py --data-root dataset/dataset_test/datasetA/datasetA \
  --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl \
  --jsonl dataset/dataset_test/datasetA/datasetA/neg.jsonl
# 中断续跑：加 --resume（用已写好的 checkpoint 跳过已完成的样本）
```

GPU 编号：`CUDA_VISIBLE_DEVICES=2 python run_szy_v1.py ...`；
CPU-only：改配置里 `"device": "cpu"`（或临时复制一份配置改设备），只建议 `--limit` 冒烟。

---

## 7. 产物与预期结果

运行结束会产出（`outputs/` 下）：

| 文件 | 内容 |
|---|---|
| `szy_v1_submission.json` | 提交格式结果（id/content/label/cer） |
| `szy_v1_detail.json` | 逐条明细（路由、置信度、差分分数、错误） |
| `szy_v1_manifest.json` | 本次运行用的配置快照 + 模型 sha256 |
| `szy_v1_checkpoint.json` | 断点（可 --resume） |
| `szy_v1_cache/separation/` | Moss 分离的中间 wav 缓存 |

全量预期（与队长 baseline 一致）：

```text
CER = 0.3174937   PR = 0.8790323   RR = 0.8396624
路由分布：raw 870 / moss_candidate 408 / primary_reject 560（合计 1838）
```

若指标有细微差异（第 6 位小数级），通常是 GPU 浮点累加顺序导致，属正常；若差异明显，
请用 `check_models.py` 核对 sha256，并确认 `FireRedASR2S` 源码版本一致。

---

## 8. 常见问题（Windows 重点）

| 现象 | 原因与解决 |
|---|---|
| `No module named 'midea_qiji'` | 没设 `PYTHONPATH=src`，或在错误目录运行。用 `run_szy_v1.sh` 或按 6.2 设环境变量 |
| `No module named 'fireredasr2'` | `model/external/FireRedASR2S` 没就位或路径不对，见第 5.2 节 |
| `No module named 'kaldi_native_fbank'` | FireRed 依赖未装，见第 5.2 节；Windows 无 wheel 时用 WSL2 |
| `No module named '_kaldi_native_fbank'` | vendor 里的 .so 是给别的 Python 版本编译的（如 cpython-312，而你的环境是 3.10）。修复：`pip install --target model/runtime/vendor kaldi_native_fbank==1.22.3` 装成你当前 Python 版本的 .so |
| `CUDA out of memory` | 显存不足。试 `SZY_DEVICE=cuda:0` 换卡、减小 batch，或先跑 `--limit` |
| `unknown szy_v1 config keys` | 配置文件里有多余键。JSON 不支持注释，别往 `szy_v1_firered.json` 里加 `//` 或说明字段 |
| `FileNotFoundError: .../simple_model.onnx` | Moss 模型没下载或目录名不对，重跑 `download_models.py` |
| 符号链接创建失败（relocate 工具） | Windows 需要开发者模式/管理员；普通复现不需要该工具 |
| 中文输出乱码 | 确保终端编码 UTF-8（PowerShell 可 `chcp 65001`） |

---

## 9. 本仓库相对队长原版的改动（可审计）

1. **模型目录从"仓库同级 `../model`"改为"仓库内 `model/`"**，使整个文件夹可整体搬迁；
   相应更新了 `configs/szy_v1_firered.json`、`run_szy_v1.sh`、`infer.py` 的 `PACKAGE_MODEL_ROOT`
   与 `pipeline.py` 的 `dual_backend` 默认值（默认值仅兜底，正式运行由 JSON 覆盖）。
2. **补充 `requirements.txt`** 缺失的 `onnxruntime`、`soundfile`。
3. **`check_models.py` 重写**为"按本仓库路径自检 + sha256 校验"（原版依赖队长工作区的 `model/MANIFEST.json`）。
4. 删除了 `.tar.gz` 内嵌的空文件与 `__pycache__`；数据集解压到 `dataset/`。
5. **所有算法逻辑、阈值、决策规则、输出格式均未改动**——这是复现的前提。
