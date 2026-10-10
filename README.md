# 目标说话人语音指令识别（Target-Speaker Command Recognition）

给定一段**唤醒音频**（目标说话人念唤醒词）和一段**指令音频**（可能混有
目标说话人、干扰说话人与噪声），系统判断指令音频中是否含有目标说话人：
有则输出该说话人的指令文本，没有则输出空（拒识）。

本仓库是该系统的**参考实现**：结构清晰、注释完备、测试自足，供团队复盘
与学习。评测集（DatasetA，1838 样本）公式分 **(1−CER+RR)/2 = 0.8937**
（PR 0.9150 / RR 0.9304 / CER 0.1430）。

## 系统一图流

```text
wake + command audio
        │
        ▼
  MossFormer2 盲分离（dasynth 微调，整波推理，落盘缓存）
        │── wake ms0/ms1  └── command ms0/ms1
        ▼
  声学单双人判定（20 维波形特征 logistic，零 ASR）
        │
   双单人 ──→ ft 微调声纹 cos(raw_wake, raw_cmd) ≥ 0.29 ?
        │        接受 → 只解码 raw command ─┐
        │        拒绝 → 输出空              │
        ▼ （否则）                           │
  唤醒通道 beam=1 转写 → 拼音相似度选锚点    │
        ▼                                   │
  融合门控（双 ERes2NetV2 前端，6 维余弦    │
            线性融合）≥ −0.6469 ?           │
        │                                   │
   拒绝 → 输出空（不跑指令 ASR）            │
   接受 → 指令单人？ 是 → 只解码 raw ───────┤
        │          否 → ft 余弦选路，       │
        │          只解码选中通道 ──────────┤
        ▼                                   ▼
  后处理（仅选定通道）：MBR N-best 重排 → ELM 重解码
        ▼
  任何异常 → 回退原始转写（再失败置空，样本永不消失）
```

各环节"为什么这么做"见 `docs/ALGORITHM.md`；每一项优化来自哪里、代价是
什么见 `docs/OPTIMIZATION.md`。

## 快速开始

```bash
# 环境：conda（torch 需 CUDA 版，见 requirements.txt）
pip install -r requirements.txt

# 单元测试（19 个用例，numpy-only，无需模型/GPU）
PYTHONPATH=src python -m unittest discover -s tests

# 冒烟：分层抽样 5 条（需要模型资产与数据集，见"资产准备"）
bash scripts/run.sh smoke --limit 5

# 全量评测：产物收进 outputs/baseline/，自动生成人类可读报告
bash scripts/run.sh baseline

# 断点续跑 / 指定 GPU / 跑 datasetX
bash scripts/run.sh baseline --resume
bash scripts/run.sh baseline --gpu 0
bash scripts/run.sh x_run --dataset X
```

每次运行产出五件：`submission.json`（提交物）、`detail.json`（逐样本
全字段明细）、`detail_report.txt`（人类可读报告）、`checkpoint.json`
（断点存档）、`manifest.json`（配置 + 权重 SHA-256 指纹）。

## 资产准备（不进 git）

`model/`、`dataset/`、`outputs/` 由 `.gitignore` 排除。`model/` 目录
结构与填充方式见其内的 README（本地文件）；关键资产：

| 资产 | 路径 |
|---|---|
| MossFormer2 分离器（dasynth 微调） | `model/artifacts/mossformer2_dasynth_sweep_ep6.pt` |
| 融合门控权重 | `model/artifacts/gate_fusion.json` |
| 微调 SV encoder | `model/artifacts/ep6_datasetA_best.pt` |
| ELM 外部语言模型 | `model/artifacts/firered_elm/` |
| FireRedASR2-AED 权重与源码 | `model/external/` |
| ERes2NetV2 预训练基座 | `model/workspace/.../iic/speech_eres2netv2_sv_zh-cn_16k-common` |
| 单双人路由器 bundle | `configs/single_speaker_router.json`（随 git 分发） |

数据集：官方 DatasetA 置于 `dataset/dataset_test/datasetA/datasetA/`
（`pos.jsonl` / `neg.jsonl` + 音频）。

## 仓库结构

```text
├── configs/                 # pipeline.json（生产配置）+ single_speaker_router.json（路由器 bundle）
├── scripts/run.sh           # 统一入口：命名运行、分层冒烟、断点续跑、自动报告
├── src/
│   ├── core/                # 内核层：全仓库公共部件（只依赖标准库/numpy，重依赖延迟导入）
│   │   ├── text.py          #   文本归一化 / 编辑距离（冻结语义）
│   │   ├── phoneme.py       #   拼音音素相似度与唤醒词别名表
│   │   ├── audio.py         #   采样率常量与严格 WAV 读取
│   │   ├── model_paths.py   #   模型路径解析（ModelScope 缓存定位）
│   │   ├── samples.py       #   样本数据模型与 JSONL 清单读取
│   │   ├── nbest.py         #   N-best 候选的保守 MBR 重排序
│   │   ├── embeddings.py    #   声纹 embedding 提取（ModelScope 前端封装）
│   │   ├── firered.py       #   FireRedASR2 识别后端（AED / LLM）
│   │   └── wake_anchor.py   #   唤醒锚点选择的冻结决策规则
│   └── pipeline/            # 系统层：生产管线
│       ├── cli.py           #   命令行入口（python -m pipeline.cli）
│       ├── engine.py        #   管线引擎（配置 / ASR 缓存 / 融合门控 / 逐样本执行）
│       ├── single_speaker.py#   单人判别与双单人余弦路由（纯波形特征）
│       ├── separator.py     #   MossFormer2 torch 分离器（dasynth 微调权重）
│       ├── metrics.py       #   分离质量指标（SI-SDR / PIT 损失）
│       ├── train.py         #   dasynth 域适配微调训练
│       ├── helpers.py       #   共享工具（哈希 / 音频读取 / 缓存路径）
│       └── reporting.py     #   指标汇总、提交物写出、人类可读报告
├── tests/                   # 19 个 numpy-only 单测（无需模型/GPU）
├── third_party/mossformer2/ # vendor 的 MossFormer2 模型定义（未改动结构）
├── docs/                    # 三篇文档（见下）
└── model/ dataset/ outputs/ # 资产 / 数据 / 产物（不进 git）
```

分层规则：**依赖只能向下**（scripts/tests → pipeline → core → third_party）。
核心代码配有面向初学者的逐段中文注释——不仅说明"做什么"，更说明
"为什么这样做"。

## 文档地图（按推荐阅读顺序）

1. **本 README** — 系统总览与运行方式
2. `docs/ARCHITECTURE.md` — 架构与数据流：分层设计、逐文件职责、
   一条样本从 jsonl 到提交物的完整旅程
3. `docs/ALGORITHM.md` — 算法详解：管线每一环节的算法、阈值、
   缓存与性能数字
4. `docs/OPTIMIZATION.md` — 优化历程：从基线 0.8679 到 0.8937 的逐项
   实验（什么有效、为什么、什么被否决）
