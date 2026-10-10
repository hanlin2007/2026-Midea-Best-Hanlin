# Midea Best 目录与文件结构

## 1. 顶层关系

`midea-best` 和 `model` 是 `/home/hustdun/szy_workspace` 下的两个同级目录：

```text
/home/hustdun/szy_workspace/
├── midea-best/                         # 当前最优 SZY v1 代码包
└── model/                              # 整个工作区的模型总目录
```

`midea-best` 负责流程代码和运行入口，`model` 负责模型实体。二者分离后，代码包不会重复占用大型权重空间，也不会把历史模型混入最优代码目录。

## 2. midea-best 结构

```text
midea-best/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── szy_v1_firered.json
├── scripts/
│   ├── run_szy_v1.sh
│   ├── check_models.py
│   └── relocate_workspace_models.py
├── src/
│   ├── midea_qiji/
│   │   ├── __init__.py
│   │   ├── gate.py
│   │   ├── infer.py
│   │   └── moss_candidate.py
│   └── szy_v1/
│       ├── __init__.py
│       └── pipeline.py
├── tests/
│   ├── test_infer.py
│   └── test_moss_candidate.py
├── docs/
│   ├── CURRENT_BEST_PIPELINE_SZY_V1.md
│   ├── SZY_V1.md
│   ├── MOSS_CANDIDATE_REPRODUCTION.md
│   └── MIDEA_BEST_STRUCTURE.md
└── outputs/
    └── baseline/
        ├── szy_v1_submission.json
        ├── szy_v1_detail.json
        └── szy_v1_manifest.json
```

### 2.1 根目录文件

| 文件 | 作用 |
|---|---|
| `README.md` | 快速介绍、运行命令、指标和文档入口。 |
| `pyproject.toml` | Python 包名、版本和 `src/` 包发现配置。 |
| `requirements.txt` | 当前项目使用的 Python 依赖声明。 |

### 2.2 配置

| 文件 | 作用 |
|---|---|
| `configs/szy_v1_firered.json` | 冻结 SZY v1 的设备、声纹模型名、双 MLP 权重、FireRed、MossFormer2、阈值和输出策略。模型路径统一指向同级 `../model/`。 |

关键固定参数包括：

- 双 MLP 主门控阈值：`0.4460311856655159`；
- 唤醒与命令采样率：16 kHz、单声道；
- MossFormer2 执行提供者：`auto`；
- 样本 ID 模式：`relative_stem`；
- 分离缓存：`outputs/szy_v1_cache/separation`。

### 2.3 运行脚本

| 文件 | 作用 |
|---|---|
| `scripts/run_szy_v1.sh` | 从包根目录启动完整 SZY v1，自动设置 `PYTHONPATH`、ModelScope 缓存、DatasetA 路径和输出路径。 |
| `scripts/check_models.py` | 检查 `model/MANIFEST.json`、模型实体、原路径兼容链接和外部 FireRed 链接。 |
| `scripts/relocate_workspace_models.py` | 将工作区模型资产迁入同级 `model/workspace/`，并在原路径建立相对符号链接；通常只在模型整理时使用。 |

### 2.4 源码

| 文件 | 作用 |
|---|---|
| `src/midea_qiji/gate.py` | ERes2NetV2 与 CAM++ 双前端特征提取、滑窗统计、标准化和双 MLP 目标说话人主门控。 |
| `src/midea_qiji/infer.py` | 音频读取、JSONL 样本解析、文本归一化、CER/PR/RR 报告和 FireRed ASR 包装。 |
| `src/midea_qiji/moss_candidate.py` | 唤醒文本与分离源的编辑距离选择、命令候选仲裁、空源回退和置信度切换规则。 |
| `src/szy_v1/pipeline.py` | SZY v1 端到端编排：主门控、raw FireRed、MossFormer2 候选、唤醒 anchor、差分声纹选源和 raw fallback。 |
| `src/*/__init__.py` | Python 包初始化文件。 |

### 2.5 测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_infer.py` | 文本归一化、稳定样本 ID、唤醒文本保留和报告指标/提交结构。 |
| `tests/test_moss_candidate.py` | 唤醒源选择、空唤醒源回退、置信度规则、空命令源回退和差分分数平局。 |

### 2.6 文档与基线结果

| 路径 | 作用 |
|---|---|
| `docs/CURRENT_BEST_PIPELINE_SZY_V1.md` | 当前批准路线的完整流程、阈值、模型、回退策略和 DatasetA 指标。 |
| `docs/SZY_V1.md` | SZY v1 的补充说明和运行边界。 |
| `docs/MOSS_CANDIDATE_REPRODUCTION.md` | MossFormer2 候选路线的复现实验说明。 |
| `docs/MIDEA_BEST_STRUCTURE.md` | 本文件，说明代码包和模型总目录结构。 |
| `outputs/baseline/szy_v1_submission.json` | 基线提交格式和每条输出文本。 |
| `outputs/baseline/szy_v1_detail.json` | 基线逐条结果、路由信息和指标明细。 |
| `outputs/baseline/szy_v1_manifest.json` | 基线运行时的输入、配置和模型摘要。 |

## 3. model 结构

```text
model/
├── MANIFEST.json
├── workspace/
│   ├── midea/models/                 # ModelScope/Hugging Face 模型缓存
│   ├── midea/outputs3/...            # Qwen LoRA adapter bundle
│   ├── midea/outputs/...             # 关键词/文本条件实验权重
│   ├── midea/tmp/...                 # 文本条件实验权重
│   ├── exp_sep/...                   # Moss anchor selector 权重
│   ├── sv_asr/...                    # ASR adapter bundle
│   ├── sv_backend/...                # SV 双前端 MLP 和 KWS 权重
│   ├── sv_finetune/...               # ERes2NetV2 微调权重
│   ├── sv_probe/...                  # probe head 权重
│   ├── sv_selector/...               # selector 权重
│   ├── sv_tse/...                    # ULUNAS-TSE checkpoint
│   └── sv_firered/...                # FireRed 微调模型
├── external/
│   ├── FireRedASR2-AED -> 工作区外的 FireRed AED 模型
│   └── FireRedASR2S -> 工作区外的 FireRed 推理源码
└── runtime/
    └── vendor -> 工作区内的 kaldi_native_fbank 依赖
```

### 3.1 `model/workspace`

该目录保留模型原来在 `szy_workspace` 中的相对层级，便于从清单反查来源。迁移实体包括：

- `midea/models`：ModelScope、Hugging Face、MossFormer2、ERes2NetV2、CAM++、Qwen、SepFormer 等缓存；
- `sv_backend`：`backend_dual*.pt`、`backend_kws.pt` 等门控权重；
- `sv_asr` 与 `midea/outputs3`：完整 LoRA adapter 目录及其 tokenizer 配置；
- `sv_firered`：FireRed 微调的 `best`/`last` 模型目录；
- `sv_tse`、`sv_finetune`、`sv_probe`、`sv_selector`：TSE、声纹微调和选择器 checkpoint；
- `midea/outputs`、`midea/tmp`、`exp_sep` 中实际保存的训练权重。

数据集音频、Dataset Probe 的 embedding、日志、虚拟环境和 Python 源码中的 `models` 模块不属于模型资产，不迁入此目录。

### 3.2 `model/MANIFEST.json`

模型清单记录：

- 原始工作区路径 `original_path`；
- 集中后的路径 `canonical_path`；
- 文件或目录类型 `kind`；
- 字节数和目录文件数；
- 单文件 SHA-256；
- 工作区外依赖的链接目标。

原路径的符号链接只负责兼容旧脚本，`model/workspace/` 才是模型实体的唯一存放位置。

## 4. 运行时数据流

```text
唤醒音频 + 唤醒文本 + 识别音频
              |
              v
ERes2NetV2/CAM++ 双前端 MLP 主门控
       |                       |
    拒绝                     接受
  输出空文本            raw FireRed ASR
                              |
                    低可靠时触发候选
                              |
                    MossFormer2 分离
                              |
              唤醒文本选 anchor，差分声纹选命令源
                              |
                    任意失败回退 raw
```

详细阈值、分支条件和指标定义以 `docs/CURRENT_BEST_PIPELINE_SZY_V1.md` 为准。

## 5. 常用命令

```bash
cd /home/hustdun/szy_workspace

# 检查模型清单和兼容链接
/home/hustdun/.conda/envs/yuyin/bin/python midea-best/scripts/check_models.py

# 运行单元测试
cd midea-best
PYTHONPATH=src /home/hustdun/.conda/envs/yuyin/bin/python -m unittest discover -s tests -p 'test_*.py'

# 运行完整流程
cd /home/hustdun/szy_workspace
PYTHON_BIN=/home/hustdun/.conda/envs/yuyin/bin/python midea-best/scripts/run_szy_v1.sh
```
