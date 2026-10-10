# 目标说话人识别 Pipeline：SZY v2

## 1. 文档定位

本文档描述 SZY v2 目标说话人命令识别管线（`midea-best/src/szy_v2/pipeline.py`），
是当前的最佳路线。v2 与 v1（`docs/CURRENT_BEST_PIPELINE_SZY_V1.md`）共享任务定义、
样本读取、输出 schema、指标口径和大部分回退逻辑；本文档重点写**架构差异和每处
改进的证据**，与 v1 相同的细节不再展开，请对照 v1 文档阅读。

当前结果（DatasetA，1838 trials：pos 1364 / neg 474）：

| 版本 | PR | RR | CER | 口径 |
|---|---|---|---|---|
| szy_v1 生产基线 | 0.879 | 0.840 | 0.3175 | 官方 |
| **szy_v2 定稿（融合门控+音色触发+raw回退锚点）** | **0.8519** | **0.9008** | **0.3054** | 官方 |

计分公式 (1−CER+RR)/2 下 v2 定稿 = **0.7977**，经全变体扫描为公式最优区
（v1 基线 0.7665）。已验证未采纳的改进见 §9（新 SV 模型、SeACo 锚点、
非空锚点——均在 PR/CER 侧更优但公式分不更高）。

## 2. 与 v1 的差异总表

v2 在 v1 的"门控 → raw ASR → 切换触发 → 分离候选选择"骨架上改了四处：

| # | 环节 | v1 | v2 | 收益 |
|---|---|---|---|---|
| 1 | 主门控 | 双前端 MLP（eres+cam，滑窗统计特征） | **双 ERes2NetV2 融合 logistic**（预训练+分离域微调，6 维余弦） | RR +6.1pt |
| 2 | 分离时机 | 触发后才分离 | **分离提前到门控之前**（门控直接消费分离通道） | 门控可见分离证据 |
| 3 | 切换触发 | conf × 关键词 × **chars_per_sec** | conf × 关键词 × **sv_diff（音色差）** | CER −1.2pt vs v1 规则 |
| 4 | 候选选择 | eres+cam **同次分离差分** | **微调模型直接余弦** | 打平（1021 vs 1013 字错），但去掉 cam++ 依赖、与门控同原理 |

注：另有两处已验证未采纳——锚点"非空通道直选"（PR/CER 优但 RR 损，公式分更低）
与新 SV 模型（cal2 域 r1s1ep2，EER 0.1285 vs 0.1367 但公式分不升），均存档备查。

另有一处**已验证未默认启用**的改进：SeACo 热词高置信锚点修正
（CER 再 −0.3pt、RR +0.4pt，重放口径，需额外挂载 SeACo 模型），见 §9。

## 3. 总体数据流

```
(kws.wav, cmd.wav)
  │
  ├─ MossFormer2 分离（两路都分）── wake_ms0/1, cmd_ms0/1        【差异 2】
  ├─ FireRed 转写 4 路分离通道（wake0/1, cmd0/1）
  ├─ 唤醒文本选 anchor（与 v1 相同：编辑距离 + 一路为空回退原始 kws）
  │
  ├─ 融合门控（双 eresv2，6 维余弦 → logistic 分数 ≥ -0.6469 ?）  【差异 1】
  │     └─ 拒绝 → content 为空（route=fusion_reject）
  │
  ├─ 接受 → raw cmd FireRed（文本 + confidence）
  ├─ 音色切换触发：conf × 关键词 × sv_diff                        【差异 3】
  │     └─ 不触发 → 输出 raw 文本（route=raw）
  │
  └─ 触发 → ft 直接余弦选 cmd_ms0/ms1（一路为空回退 orig）        【差异 4】
        └─ 输出所选通道文本（route=moss_candidate）
```

与 v1 的关键结构差别：v1 是"门控用原始音频、触发后才分离"；v2 把分离提前，
门控的 6 维特征里有 4 维直接来自分离通道——门控从"猜原始音频里有没有目标人"
变成"看分离之后目标人在不在"。

## 4. 阶段一：融合门控（核心改进）

实现：`szy_v2/pipeline.py` 的 `FusionGate`。

### 4.1 双前端

- **预训练前端**：ModelScope `speech_eres2netv2_sv_zh-cn_16k-common`，原样冻结。
- **微调前端**：同一基座 + `dataset_dasynth/out/ep6_datasetA_best.pt` 的 encoder 权重。
  该 checkpoint 在 DatasetA 风格合成数据（dataset_dasynth，CN-Celeb 说话人、
  模拟比赛损毁配方）上做分离域适配；**epoch 6 是迁移甜点**（更晚的 epoch 在合成
  valid 上单调变好、在 DatasetA 上反而域过拟合，ep17/20 EER 回升）。
  教训：第一版用 AISHELL 朗读重叠数据适配，迁移失败（EER 0.171 vs 预训练 0.156），
  合成域匹配是前提。

### 4.2 六维余弦特征

对每条 trial 提取 5 个 embedding（anchor、raw_wake、raw_cmd、cmd_ms0、cmd_ms1），
每个前端算 3 个特征：

| 特征 | 定义 | 含义 |
|---|---|---|
| `{tag}_raw` | cos(raw_wake, raw_cmd) | 不分离时两原始音频的相似度 |
| `{tag}_sep` | max cos(anchor, cmd_ms*)（任一空通道则 raw_cmd 入候选） | 分离协议得分 |
| `ch_sim_{tag}` | cos(cmd_ms0, cmd_ms1) | 两通道音色是否同人（重叠指示） |

### 4.3 训练与阈值口径（干净）

- 训练集：dataset_dasynth valid 构造的 trial（**负样本跨 mix 配对**：enroll 来自
  记录 A、test 来自说话人互斥的记录 B；同 mix 对方通道做负样本在 max 取分下与正样本
  同分，第一版融合器因此没学起来）。
- 标准化参数 + 权重 + 阈值 **-0.6469**（valid 上扫 balanced 点）全部来自外部合成
  数据，DatasetA 未参与。
- 产物：`dataset_dasynth/out/gate_fusion.json`。

### 4.4 效果与上限

- 相对 v1 双 MLP 门控：RR 0.840 → 0.901（+6.1pt），PR 0.879 → 0.851（−2.8pt）。
- DatasetA 上分数 AUC = 0.940；oracle 阈值扫描 balanced 仅 0.8778 vs 当前 0.8765
  ——**该特征集合的阈值已没有调优空间**，再提升只能改善分数本身（见 §10 方向 1）。

## 5. 阶段二：唤醒锚点选择（与 v1 相同：raw 回退）

`moss_candidate.select_wake_anchor`：FireRed 转写两路 wake 分离通道，与已知唤醒
文本比编辑距离选近者；一路为空 → 回退原始 kws 音频。锚点是整条 trial 的 SV 基准，
喂给门控的 4 个余弦和候选选择。

曾验证"一路为空时直接用非空通道"变体：PR +1.7pt、CER −1pt，但 RR −3.4pt；
在计分公式 (1−CER+RR)/2 下公式分反而更低（0.7939 vs 0.7977），**已回退为 raw
回退并保持不变**。

## 6. 阶段三：音色切换触发（替代语速规则）

实现：`moss_candidate.should_switch_command_timbre`。

判决"raw 转写够不够用、要不要动用分离候选"。触发特征从 v1 的
`chars_per_sec`（绝对语速假设，DatasetA 手调常数）换成**音色差**：

```
sv_diff = max[cos(anchor, cmd_ms0), cos(anchor, cmd_ms1)] − cos(anchor, raw_cmd)
```

物理含义：分离前后目标嗓音浓度的增量——切换有收益的机制本身。
干净 cmd：raw 里全是目标人，diff≈0 或为负 → 不切；重叠 cmd：混合拖低 raw 余弦、
分离捞回目标人 → diff 显著为正 → 切。两个 embedding 门控时已算好，零额外成本。

规则（conf 为 FireRed 对 raw cmd 的自估置信度，kw 为指令关键词命中数）：

```
conf ≤ 0.57         → 切（必坏线：ASR 自认崩溃）
0.57 < conf ≤ 0.70  → kw ≥ 1 或 sv_diff > 0.05 则切（灰区旁证）
conf > 0.70         → sv_diff > 0.05 且 kw = 0 则切
                      （自信转写但不像指令 + 音色说分离更好 ⇒ 自信地转错了人）
```

### 6.1 为什么换掉 chars_per_sec

cps 是对"ASR 把两路混着转写 ⇒ 文本异常快"的间接假设，且假设了绝对语速，
常数来自 DatasetA 手调。sv_diff 是对同一现象的直接声学测量。消融证据：

| 触发器（同一门控/选择器） | 切换数 | CER |
|---|---|---|
| 永不切（全 raw） | 0 | 0.3820 |
| cps 规则（v1） | 294 | 0.3181 |
| 单置信度最优阈值 | 582 | 0.3313 |
| d_sel 文本比对（中间迭代版） | 290 | 0.3092 |
| **sv_diff 音色规则（当前）** | 359 | **0.3058** |
| oracle 触发（知道答案才切，天花板） | 232 | 0.2662 |

### 6.2 稳健性与常数来源（诚实声明）

- sv_diff 判别力 AUC = 0.795（单特征）；τ 在 [0.02, 0.08] 盆地平（0.3044~0.3059），
  取中值 0.05，未取 DatasetA argmax。
- conf 切点 0.57/0.70 是 v1 遗产（DatasetA 手调），但敏感度验证：c1∈[0.50,0.70]、
  c2∈[0.65,0.80] 矩形内 CER 起伏 ≤0.4pt（如 0.60/0.80 → 0.3079）——结构重要、数值不敏感。
- kw 的价值集中在高置信否决角色（去掉两处 kw → 0.3194；只留否决 → 0.3052）。
- 曾尝试在外部数据（AISHELL overlap）上训练学习版触发器替代整棵决策树：
  迁移失败（最好 0.3243），决策边界不随分布迁移——该负结果存档于
  `AISHELL1_OVERLAP_HANDOFF/out/trigger/`。

## 7. 阶段四：候选选择（ft 直接余弦）

触发后，用微调前端的 `cos(anchor, cmd_ms0)` vs `cos(anchor, cmd_ms1)` 选通道
（门控时已算，直接复用）；一路分离为空 → 回退 orig。

替换 v1 的 eres+cam 同次分离差分（`cos(anchor,ms) − cos(contrast,ms)`）的理由：

- 打平：DatasetA switch 集上 1021 vs 1013 字错（统计噪声内）；
- 差分形式在"对比通道有真语音"时被污染（准确率 67% vs 直接余弦 84%）；
- 与门控同一打分原理，**cam++ 前端就此移除**，v2 只依赖 eresv2 双前端。

## 8. 异常和回退矩阵

与 v1 一致，核心几条：门控拒绝 → 空输出；任一路分离为空 → 原始音频/原始文本兜底；
任何模型异常 → 回退 raw FireRed 输出（`route=raw_fallback_error`）。
SeACo 锚点修正若启用（§9），其失败回退为 FireRed 文本选源。

## 9. 已验证待启用：SeACo 热词高置信锚点

思路（重放验证通过，未默认启用）：用 SeACo-Paraformer（热词=唤醒文本）转写两路
wake 分离通道，**仅当两路编辑距离差 ≥ 2 时才翻转锚点选择**，否则维持 FireRed 选源；
空回退机制不变。

重放口径（v2 音色触发版之上）：PR 0.8519 / RR 0.9051 / CER 0.3030
（vs 不启用 0.8512 / 0.9008 / 0.3058）。无差别采纳 SeACo 会出现假象（坏锚点压低
门控分，PR/RR 互搏），高置信限制（58 条翻转）后正负样本进出平衡。
成本：管线需挂载 SeACo 模型，每条 trial +2 路 kws 转写（~0.2s）。
证据：`AISHELL1_OVERLAP_HANDOFF/replay_hotword_anchor.py`。

**反向验证（已证伪，勿再试）**：同一热词机制搬到 cmd 侧不成立。将 raw cmd 与两路
分离通道的内容文本换成 SeACo 热词转写（热词=唤醒文本 / 唤醒文本+36 指令词表，
触发、判空、门控仍用 FireRed），pos raw 编辑距离 2988(FireRed)→3809/3776，
端到端 CER 0.3058→0.3621/0.3592（PR/RR 基本不变）。原因：指令词是常见词，
FireRed 本来就识别得好，热词偏置反而扭曲识别、并会在分离垃圾通道脑补热词文本
（neg ms1 非空 246→380）；热词的价值仅限于唤醒词这类稀有词。关键词数量不是瓶颈，
SeACo 基座在该数据上的识别率才是。证据：`transcribe_dataseta_cmd_seaco.py`、
`replay_hotword_anchor.py --seaco-cmd/--seaco-raw`、`exp_sep/*_seaco_{wake,wlex}.json`。

**TSE 替换分离候选（已证伪，第二次）**：SoloSpeech（当前 TSE 可懂度 SOTA，级联
生成式+GeCo 校正）零样本，enrollment=生产同款唤醒锚点，mixture=raw cmd，全量 1838 条。
触发时内容换 TSE 文本（两档：分离判空仍回退 raw / 触发即用）端到端 CER
0.3058→0.331/0.3355，PR/RR 不变——**明确变差**。机理：低置信子集（conf≤0.70，n=232）
上 TSE 条件 CER 0.695，虽优于 raw 的 0.883，但输给 MossFormer2 分离 oracle 的 0.505；
英文训练的零样本 TSE 打不过"分离+声纹选源"的内容质量。TSE 空文本极少（pos 21、
neg 18），无拒识信号红利。此为继 ULUNAS-TSE（v1 档案，CER 0.447→0.696）之后第二次
TSE 证伪；剩余唯一未试路径是域内微调 TSE（WeSep pDPCCN + 自造三元组），预期收益
已显著下修。证据：`run_tse_solospeech.py`、`build_tse_manifest.py`、
`transcribe_dataseta_tse.py`、`exp_sep/moss_cmd_text_tse_solospeech.json`、
`replay_hotword_anchor.py --tse-text`。

## 10. 指标口径与已知上限

指标公式与 v1 逐字一致（PR=非空输出 pos 比例、RR=空输出 neg 比例、CER=被拒全错计）。

按环节分解的剩余空间（DatasetA 实测）：

| 环节 | 当前贡献 | oracle | 可压缩 |
|---|---|---|---|
| 门控拒识的 200 条 pos | 1320 字（0.139 CER） | 362 字 | ~10pt（受限于分数 AUC 0.94） |
| raw 直出（触发未命中） | 711 字 | 364 字 | ~3.6pt |
| 候选选择选错 | 989 字 | 692 字 | ~3.1pt |

结论：规则层（触发/选择/阈值）已接近其特征能力上限；**唯一有大块收益预期的剩余
方向是门控分数质量本身**——方向 1：用 CN-Celeb dev/eval 新说话人造 8-10k 条门控
专用数据 + 滑窗统计特征训 MLP 门控（约束：不能用 dasynth train 的说话人，encoder
见过其分数有偏）。

## 11. 运行方式

```bash
cd midea-best && CUDA_VISIBLE_DEVICES=3 bash scripts/run_szy_v2.sh
# --limit N 冒烟；--resume 断点续跑
```

产物：`outputs/szy_v2_detail.json`（逐条明细 + summary）、`szy_v2_submission.json`、
`szy_v2_manifest.json`（含各模型 sha256）。分离缓存 `outputs/szy_v2_cache/`。

## 12. 主要证据索引

- 门控：`dataset_dasynth/train_gate_fusion.py`、`out/gate_fusion.json`
- 微调：`AISHELL1_OVERLAP_HANDOFF/train_eres2netv2_da.py`、`dataset_dasynth/out/ep6_datasetA_best.pt`
- 触发器扫描/消融：`eval_confidence_sweep.py`、`eval_switch_rule_ablation.py`、
  `eval_trigger_features.py`、`eval_timbre_trigger.py`
- 候选选择扫描：`eval_candidate_rules.py`、`eval_always_separate.py`、`eval_candidate_diff.py`
- 锚点实验：`replay_hotword_anchor.py`、`exp_sep/moss_kws_text_seaco.json`
- 学习版触发器（负结果）：`build_trigger_data_v2.py`、`train_trigger.py`、`out/trigger/`
