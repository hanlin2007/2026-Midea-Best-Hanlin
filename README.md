# Midea Best：SZY v1（美的「复杂交互场景抗干扰语音指令识别」最佳路线）

本仓库是队长工作区里**当前批准的最优版本代码**的本地组织版，目标是：**在你的电脑上
一步不落地复现出冻结结果**（DatasetA 全量：CER `0.3174937`、PR `0.8790323`、RR `0.8396624`）。

任务一句话说明：给定「唤醒音频 kws.wav（有人说了唤醒词）」+「识别音频 cmd.wav（可能多人
重叠说话）」，识别出 cmd.wav 里**与唤醒词说话人是同一人**的那句话；负样本（目标人不在）要拒绝。

## 最优流程

```text
ERes2NetV2/CAM++ 双前端 MLP 主门控
  -> 原始音频 FireRed ASR
  -> 低可靠结果才生成 MossFormer2 候选
  -> 唤醒文本选择唤醒 anchor
  -> 差分声纹选择命令源
  -> 任意候选失败均回退原始音频
```

## 快速开始（3 步）

```bash
# 0) 安装依赖（建议用 conda 独立环境，见 SETUP 指南第 2 节）
pip install -r requirements.txt

# 0.5) 可选但推荐：以开发模式安装本包，之后 python -m szy_v1.pipeline 也能直接跑
pip install -e .

# 1) 下载 3 个公开模型（ERes2NetV2 / CAM++ / MossFormer2）
python scripts/download_models.py

# 2) 自检（数据集你本地已有，已解压到 dataset/；backend_dual.pt 和 FireRed 需从队长工作区复制）
python scripts/check_models.py

# 冒烟测试（先跑 5 条验证链路；run_szy_v1.py 会自动设好模块路径，无需 PYTHONPATH）
python run_szy_v1.py --data-root dataset/dataset_test/datasetA/datasetA \
  --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl \
  --jsonl dataset/dataset_test/datasetA/datasetA/neg.jsonl \
  --limit 5 --output outputs/smoke_submission.json \
  --final-output outputs/smoke_detail.json --checkpoint-output outputs/smoke_checkpoint.json \
  --manifest-output outputs/smoke_manifest.json

# 全量运行（断点续跑加 --resume）
python run_szy_v1.py --data-root dataset/dataset_test/datasetA/datasetA \
  --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl \
  --jsonl dataset/dataset_test/datasetA/datasetA/neg.jsonl
```

> 完整的环境搭建、模型/数据获取、Windows 注意事项，见 **[docs/SETUP_本地复现指南.md](docs/SETUP_本地复现指南.md)**。

## 目录结构

```text
midea_qiji_best_hanlin/
├── README.md                     # 本文件（入口）
├── run_szy_v1.py                 # 一键启动器（免 PYTHONPATH，python run_szy_v1.py ...）
├── requirements.txt              # 依赖清单
├── configs/szy_v1_firered.json   # 冻结配置（阈值/模型路径）
├── scripts/
│   ├── run_szy_v1.sh             # 一键运行（Git Bash）
│   ├── download_models.py        # 下载 3 个公开模型
│   ├── check_models.py           # 环境自检（模型/数据/sha256）
│   └── relocate_workspace_models.py  # 队长工作区迁移工具（一般不用）
├── src/
│   ├── midea_qiji/               # 工具包（全部带详细中文注释）
│   │   ├── gate.py               # 双前端 MLP 声纹门控
│   │   ├── infer.py              # 样本读取/文本归一化/ASR/指标
│   │   └── moss_candidate.py     # Moss 候选冻结决策规则
│   └── szy_v1/
│       └── pipeline.py           # SZY v1 端到端主流程（全部带详细注释）
├── tests/                        # 单元测试（numpy-only，可直接跑）
├── docs/
│   ├── CURRENT_BEST_PIPELINE_SZY_V1.md   # 队长原始文档（权威）
│   ├── MIDEA_BEST_STRUCTURE.md           # 目录与文件说明
│   ├── MOSS_CANDIDATE_REPRODUCTION.md    # 候选路线复现说明
│   ├── SZY_V1.md                         # SZY v1 补充说明
│   ├── SETUP_本地复现指南.md              # ★ 本地复现完整指南
│   └── lessons/                          # ★ 学习课程（Python 初学者 → 工程实战）
├── dataset/                      # DatasetA（已解压，.gitignore 排除）
├── model/                        # 模型实体（下载/复制到此，.gitignore 排除）
└── outputs/
    └── baseline/                 # 队长产出的基线结果（复现对照）
```

## 学习课程（从 Python 初学者到工程实战）

代码已全部加详细中文注释。另配 5 课教程，从 list/dict/class 基础讲到工程实战，
再以一条样本贯穿全链路讲解原理：

| 课程 | 内容 | 适合 |
|---|---|---|
| [第 1 课](docs/lessons/lesson1_gate.py_逐行笔记.md) | gate.py 逐行精讲：list/dict/class/numpy 基础 | Python 初学者 |
| [第 2 课](docs/lessons/lesson2_pipeline.py_逐行笔记.md) | pipeline.py 逐行精讲：工程实战（配置/断点/异常兜底） | 进阶工程化 |
| [第 3 课](docs/lessons/lesson3_样本之旅_上_入口与主门控.md) | 一条样本之旅·上：JSONL → 主门控判决 | 串联理解 |
| [第 4 课](docs/lessons/lesson4_样本之旅_中_ASR与切换.md) | 一条样本之旅·中：FireRed ASR、文本归一化、切换规则 | 串联理解 |
| [第 5 课](docs/lessons/lesson5_样本之旅_下_分离与差分声纹.md) | 一条样本之旅·下：Moss 分离、差分声纹、回退与指标 | 串联理解 |

## 指标口径

```text
CER = 正样本总 Levenshtein 编辑距离 / 正样本参考字符总数
PR  = 正样本最终非空输出数 / 正样本总数
RR  = 负样本最终空输出数 / 负样本总数
J   = (1 - CER) + RR
```

## 从队长工作区还需要复制什么

| 资产 | 位置 | 是否公开 |
|---|---|---|
| `backend_dual.pt`（双 MLP 门控权重） | `model/workspace/sv_backend/backend_dual_v2_out/` | 否，队长训练产物 |
| `FireRedASR2-AED` + `FireRedASR2S` | `model/external/` | 见 SETUP 第 5 节 |
| ERes2NetV2 / CAM++ / MossFormer2 | `model/workspace/midea/models/modelscope/` | 是，`download_models.py` 可下 |

详见 [docs/SETUP_本地复现指南.md](docs/SETUP_本地复现指南.md) 的「资产清单」一节。
