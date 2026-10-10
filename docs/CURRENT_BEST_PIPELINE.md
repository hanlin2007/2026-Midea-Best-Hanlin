# 现役最佳管线（LSZ v4.5，Score 0.8937）

本文档描述**当前生产管线**的算法构成、关键阈值、运行方式与复现口径。
历史阶段（v1→v2→v4，Score 0.8679 及之前）见 `docs/history/`；
每一项优化"为什么有效、代价是什么"的完整实验记录见
`docs/OPTIMIZATION_SUMMARY.md`。

## 1. 系统构成

- **分离前端**：dasynth 域适配微调的 MossFormer2_SS_16K torch 分离器
  （联合去混响+去噪+分离，修复官方分离器在混响/掩盖样本上的崩塌），
  唤醒与指令音频各分成两路 ms0/ms1。
- **声学单双人路由**：20 维波形统计特征 + 线性 logistic
  （`configs/moss_single_speaker_router_v2.json`，训练与定阈零 DatasetA）。
  双单人走独立短路：微调 ERes2NetV2 的 `cos(raw_wake, raw_cmd)` ≥ 0.29 判同人。
- **唤醒锚点**：拼音音素相似度（同音字不罚分）+ 唤醒词别名表
  （hicolmo ↔ haikemu）；打平或单路缺失回退原始唤醒音频。
- **融合门控**：预训练 + 分离域微调双 ERes2NetV2 前端（共享一次 Fbank），
  6 维余弦特征 z-score 线性融合，阈值 **−0.6469**（dasynth valid 标定，
  零 DatasetA 调参）。
- **指令路由与延迟解码**：单人指令只解码原始指令；双人指令恒切分离通道，
  ft 直接余弦选路后**只解码选中的那一路**（首选为空才补解另一路），
  原始指令不解码。被拒样本不跑任何指令 ASR。
- **后处理（仅选定通道，各跑一次）**：保守 MBR N-best 重排
  （置信度增益 ≥ 0.03 且 MBR 风险下降 ≥ 0.05 才允许换人）→ ELM 重解码
  （领域语言模型，weight 0.15，beam 3；空文本不重解）。
- **ASR 共享实例**：唤醒与命令共用同一份 FireRedASR2-AED 权重
  （峰值显存 4818 MiB，约为双实例时代的 2/3）；所有承载决策的转写一律
  `temporary_elm_weight(0.0)`，唤醒侧 `temporary_beam_size(1)`。

## 2. Pipeline 流程

```text
wake + command audio
        │
        ▼
  MossFormer2 separation（dasynth 微调 .pt，整波推理，peak→0.5 落盘缓存）
        │
        ├── wake ms0/ms1  └── command ms0/ms1
        ▼
  声学单双人判定（纯波形，20 维特征 logistic）
        │
   双单人 ──→ ft 微调前端 cos(raw_wake, raw_cmd) ≥ 0.29 ?
        │        接受 → 只解码 raw command → 后处理 → 输出
        │        拒绝 → 输出空
        ▼ （否则）
  唤醒通道 beam=1 转写 → 拼音相似度选锚点（打平/缺失回退原始唤醒）
        ▼
  FusionGate（6 维余弦 → 线性融合）≥ −0.6469 ?
        │
   拒绝 → 输出空（不跑指令 ASR）
   接受 → 指令单人？ 是 → 只解码 raw command
        │          否 → ft 余弦选路，只解码选中通道（空则补另一路）
        ▼
  后处理：MBR N-best 重排 → ELM 重解码（仅选定通道）
        ▼
  任何异常 → 回退原始转写（再失败置空）
```

## 3. 关键数字（DatasetA，1838 样本）

| 阶段 | PR | RR | CER | Score |
|---|---|---|---|---|
| v4 基线（融合门控 + 音色规则，beam=1） | 0.9157 | 0.9304 | 0.1945 | 0.8679 |
| + 拼音音素锚点与别名表 | 0.9164 | 0.9283 | 0.1831 | 0.8726 |
| + N-best 重排序 | 0.9142 | 0.9304 | 0.1822 | 0.8741 |
| + ELM 集成（最大单项收益） | 0.9135 | 0.9177 | 0.1524 | 0.8867 |
| + 唤醒/命令通道分离解码配置 | 0.9150 | 0.9304 | 0.1438 | 0.8933 |
| + ELM 后处理化 + 阈值重扫（Phase 2） | 0.9150 | 0.9304 | 0.1430 | **0.8937** |

现役系统在此之上加入声学单双人路由与延迟解码（结构性收益：显存与时延
大幅下降，分数持平；病理性的"规则退化成恒切"不再可能发生）。
完整消融与统计显著性讨论见 `docs/OPTIMIZATION_SUMMARY.md`。

## 4. 阈值与配置速查

| 项 | 值 | 位置 |
|---|---|---|
| 融合门控阈值 | **−0.6469**（dasynth valid 标定） | `configs/pipeline_firered.json` |
| 6 维特征顺序 | pre_raw, ft_raw, pre_sep, ft_sep, ch_sim_pre, ch_sim_ft | `pipeline.py FUSION_FEATURE_KEYS` |
| 双单人同人余弦阈值 | **0.29** | `both_single_sft_threshold` |
| 单人概率阈值 | 0.8436（router bundle 内置） | `configs/moss_single_speaker_router_v2.json` |
| 主解码 beam / N-best | 3 / conservative_mbr（闸 0.03 + 0.05） | `firered_beam_size` / `firered_nbest_*` |
| ELM | weight 0.15，beam 3，仅选择后重解码 | `firered_elm_*` |
| 音色规则切点（历史重放用） | 0.88 / 0.76 / −0.03（新 confidence 量程） | `firered_switch_*` |
| 分离器 cfg | num_spks=2, kernel=16, dim=512, 24 层 | `szy/separator.py` |
| ASR 缓存键 | `<路径>\|b<beam>\|elm<weight>` | `pipeline.py AsrCache` |
| checkpoint/缓存落盘粒度 | 每 100 条 | `batch_writes` |

**切勿混用两套音色规则切点**：默认值 0.70/0.57/0.05 标定于 ELM 融入主解码
时代（confidence ~0.60 量程）；0.88/0.76/−0.03 标定于 ELM 后处理化之后
（~0.88 量程）。跨量程使用实测掉 0.0027。

## 5. 运行

```bash
# 冒烟（分层抽样 5 条）；去掉 --limit 跑全量
bash scripts/run.sh --limit 5

# 命名实验：全部产物收进 outputs/<NAME>/，自动生成人类可读报告
bash scripts/run_exp.sh baseline --gpu 0
bash scripts/run_exp.sh baseline --resume      # 断点续跑
bash scripts/run_exp.sh test --limit 3         # 快速验证

# 旧 detail.json 补生成报告
python scripts/generate_report.py outputs/<NAME>/<NAME>_detail.json
```

每次运行产出五件：`submission.json`（交卷）、`detail.json`（全字段病历）、
`detail_report.txt`（人类可读报告，自动生成）、`checkpoint.json`（断点）、
`manifest.json`（配置 + 三份权重 sha256 指纹）。

## 6. 缓存（同名重跑约 10× 提速）

| 缓存 | 位置 | 内容 | 命中效果 |
|---|---|---|---|
| 分离缓存 | `outputs/<NAME>/cache/separation/*.wav` | MossFormer2 两路分离音频 | 跳过分离推理 |
| ASR 缓存 | `outputs/<NAME>/cache/asr_cache.json` | 按 `路径\|配置` 键的转写（含 N-best） | 跳过 FireRed 解码 |

迭代模式：首次全量 ~2.3 s/utt；同名重跑（仅改决策代码）~0.2–0.7 s/utt。
生产配置 `asr_cache_persistent: false` 只保留进程内缓存（保证评测结果
只依赖当次运行）；本地迭代想跨运行复用时改为 `true`。
强制全新运行：删掉对应 `outputs/<NAME>/` 目录。

## 7. 模型资产

| 资产 | 路径（仓库内） |
|---|---|
| MossFormer2（dasynth 微调分离器） | `model/artifacts/mossformer2_dasynth_sweep_ep6.pt` |
| 融合门控权重 | `model/artifacts/gate_fusion.json` |
| 微调 SV encoder | `model/artifacts/ep6_datasetA_best.pt` |
| ELM 外部语言模型 | `model/artifacts/firered_elm/` |
| FireRedASR2-AED | `model/external/FireRedASR2-AED` |
| ERes2NetV2（预训练基座） | `model/workspace/.../iic/speech_eres2netv2_sv_zh-cn_16k-common` |
| 双单人路由器 | `configs/moss_single_speaker_router_v2.json`（随 git 分发） |

`model/` 不进 git；资产填充与自检见 `model/README.md`、`scripts/check_models.py`。

## 8. 与交付工作区的关系

`submission/workspace/` 是同一套系统的**平铺单文件交付形态**（Docker 构建
上下文，`run_test.py` 实现主办方 `load_model`/`infer_model` 接口）。它的
`src/szy_v2.py` 与本仓库 `src/szy/pipeline.py` 决策逻辑一致，仅 import
路径与默认路径不同。日常开发在本仓库进行，交付内容随版本冻结。
