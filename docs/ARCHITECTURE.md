# 架构与数据流（ARCHITECTURE）

本文档回答两个问题：**仓库是怎么组织的**（第 1–3 章）、**一条样本在系统
里经历了什么**（第 4 章）。读完你应该能不看代码就讲出每个文件的职责，
以及任意一个 detail.json 字段是在哪一步被写入的。

面向读者：Python 新手也能跟上——每个设计决策都附"为什么"。
算法细节（阈值、模型结构、数字）在 `docs/ALGORITHM.md`，本文不重复。

---

## 1. 分层架构：依赖只能向下

```text
入口层     scripts/run.sh、tests/
                 │  调用 python -m pipeline.cli 或直接 import
                 ▼
系统层     src/pipeline/    生产管线：配置、引擎、声学路由、融合门控、
                            分离器、训练、汇总写出
                 │
                 ▼
内核层     src/core/        全仓库公共部件：文本口径、拼音、音频读取、
                            模型路径、样本模型、N-best、声纹 embedding、
                            FireRed 后端、锚点规则
                 │
                 ▼
vendor     third_party/mossformer2/   外部模型定义（ClearerVoice-Studio）
资产层     model/  dataset/  outputs/（不进 git）
```

规则只有一条：**只允许向下依赖**。`pipeline` 可以 import `core`，反向
不行；`core` 里出现 `import pipeline` 就是分层错误。单向依赖让每一层都
能被单独理解、测试和替换——想换识别后端只动 `core/firered.py`，想换
门控只动 `pipeline/engine.py`，不会牵一发动全身。

配套约定：

- **重型依赖延迟导入**：torch / modelscope / FireRed 包一律在函数体内
  import，于是没有 GPU/模型权重的机器也能 `--help`、跑全部单测
  （`pipeline/separator.py` 是例外——它 import 时引入 vendor 模型定义，
  所以引擎把它延迟到构造 Runner 时才 import）；
- **逻辑进包，脚本做壳**：所有决策/计算逻辑都在 `src/` 的两个包里，
  `scripts/` 只做入口编排——逻辑因此可被单测直接覆盖；
- **冻结语义上测试锁**：文本归一化口径、锚点决策边界、默认配置这些
  "改一个字就改变历史数字"的地方，都有 numpy-only 单测逐点锁死。

---

## 2. 仓库布局

```text
├── README.md                # 项目门面：简介、运行命令、文档索引
├── requirements.txt         # 依赖清单（torch 需 CUDA 版）
├── pyproject.toml           # 打包配置（src 布局，包自动发现）
├── .gitignore               # 排除 model/ dataset/ outputs/ 等资产与产物
│
├── configs/
│   ├── pipeline.json                # 生产配置（全部可调参数的唯一来源）
│   └── single_speaker_router.json   # 单双人路由器 bundle（20 维 logistic 权重）
│
├── scripts/
│   └── run.sh               # 唯一入口：命名运行 / 分层冒烟 / 断点续跑 / 自动报告
│
├── src/
│   ├── core/                # 内核层（9 个模块，见 §3.1）
│   └── pipeline/            # 系统层（9 个模块，见 §3.2）
│
├── tests/                   # 19 个 numpy-only 单测（4 个文件，见 §3.3）
│
├── third_party/mossformer2/ # vendor 的 MossFormer2 模型定义（仅改包内相对导入）
│
├── docs/                    # ARCHITECTURE（本文）/ ALGORITHM / OPTIMIZATION
│
├── model/                   # 【不进 git】模型资产（artifacts + 基座 + FireRed + vendor）
├── dataset/                 # 【不进 git】官方数据 DatasetA
└── outputs/                 # 【不进 git】每次运行一个目录：分离/ASR 缓存 + 五件产物
```

---

## 3. 逐文件职责

### 3.1 内核层 `src/core/`（9 个模块）

内核层的自我修养：**轻量**（顶层只依赖标准库/numpy）、**不反向依赖**、
**口径唯一**（全仓库的文本/音频/路径口径都收敛在这里）。

| 文件 | 干什么 | 关键导出 |
|---|---|---|
| `text.py` | **全仓库文本口径唯一实现**（冻结语义）：数字/百分号转中文、去标点、小写归一化；字符级编辑距离（滚动两行 O(n) 空间） | `normalize_transcript`, `levenshtein` |
| `phoneme.py` | 拼音音素相似度：文本 → pypinyin（TONE3 去声调）紧凑拼音串 → 编辑距离取负；唤醒词别名表（hicolmo ↔ haikemu）双向展开取最优；pypinyin 缺失退化为小写字符比较 | `phoneme_similarity`, `WAKE_WORD_ALIASES` |
| `audio.py` | 采样率常量（16k，全仓库唯一）与严格 WAV 读取（不校验通过就报错，绝不悄悄重采样） | `SAMPLE_RATE`, `read_wav` |
| `model_paths.py` | 模型 id → 本地路径的**唯一解析入口**（直接路径 → --model-root → MODELSCOPE_CACHE → 默认缓存，兼容 `. → ___` 转义） | `resolve_model_path`, `PACKAGE_MODEL_ROOT`, `DEFAULT_FIRED_VENDOR` |
| `samples.py` | 数据入口层：样本数据模型 + JSONL 清单读取 + 提交 id 口径 + 模型噪音静默器 | `Sample`, `read_many_jsonl`, `submission_id`, `suppress_model_output` |
| `nbest.py` | N-best 候选的保守 MBR 重排序：选"与其他候选置信度加权差异最小"的假设，置信度增益与风险下降**双闸齐过**才允许换掉 top-1 | `conservative_mbr_rerank`, `normalized_edit_distance` |
| `embeddings.py` | 声纹 embedding 提取：prepare（Fbank，可双前端共享）+ extract（前向 embedding_model）两步接口，按帧数分组批处理，L2 归一化输出 | `ModelScopeEmbedder` |
| `firered.py` | FireRedASR2 识别后端：AED（生产：beam/nbest/ELM 可配，`temporary_beam_size`/`temporary_elm_weight` 支持单实例临时切换）+ LLM（Qwen2-7B 解码，备选）；鸭子类型统一接口 | `FireRedRecognizer`, `FireRedLLMRecognizer` |
| `wake_anchor.py` | 唤醒锚点选择的冻结决策规则（纯函数）：拼音相似度像者胜，打平/单路缺失回退原始唤醒音频 | `select_wake_anchor`, `WakeAnchorDecision` |

### 3.2 系统层 `src/pipeline/`（9 个模块）

| 文件 | 干什么 | 关键导出 |
|---|---|---|
| `cli.py` | 命令行入口（薄）：解析参数 → 加载配置 → 读样本 → 断点续跑 → 逐样本交引擎 → 汇总写出五件产物 | `build_parser`, `main` |
| `engine.py` | **管线引擎**（核心）：`PipelineConfig`（frozen 配置 + 未知键报错）、`AsrCache`（按"路径+解码配置"做键的转写缓存）、`FusionGate`（双前端共享 Fbank 的 6 维余弦线性融合门控）、`PipelineRunner`（逐样本执行器：声学路由、拼音锚点、延迟解码、N-best+ELM 后处理、两级兜底） | `PipelineConfig`, `load_config`, `AsrCache`, `FusionGate`, `PipelineRunner`, `FUSION_FEATURE_KEYS` |
| `single_speaker.py` | 单人判别器：20 维**通道交换不变**波形统计特征 + 外部训练的线性 logistic；`BothSingleRouter` 做"双单人"联合判定与同人余弦裁决，全程纯 numpy、零 ASR | `BothSingleRouter`, `feature_mapping` |
| `separator.py` | MossFormer2_SS_16K torch 分离器：整波推理、peak→0.5 归一化、`_ms0/_ms1` 落盘缓存；import 时注入 third_party 并引入 vendor 模型定义（故引擎延迟导入它） | `load_mossformer2`, `MossFormer2Separator` |
| `metrics.py` | 分离质量指标：numpy 诊断口径 SI-SDR、torch 训练口径 SI-SNR / PIT 损失 | `si_sdr`, `oracle_sisdr`, `si_snr`, `pit_sisnr_loss` |
| `train.py` | dasynth 域适配微调训练：PIT SI-SNR、2s 随机裁剪、bf16、按腐蚀类型分组评估选模 | `DasynthPairs`, `evaluate`, `main` |
| `helpers.py` | 共享工具（收敛为唯一实现）：文件分块 SHA-256、严格 16k 音频读取、分离缓存路径计算 | `sha256_file`, `read_mono_16k`, `separation_paths_for` |
| `reporting.py` | 评测汇总与产出：按 label 分正负算 CER/PR/RR + 路由分布；写竞赛格式 submission.json 与全字段 detail.json；生成人类可读的逐样本"病历"报告 | `summarize`, `write_outputs`, `generate_report` |

### 3.3 `tests/`（4 文件，19 用例，numpy-only）

| 文件 | 锁定什么 |
|---|---|
| `test_text.py` | 文本归一化口径、id 口径、JSONL 中文字段、提交物 schema |
| `test_wake_anchor.py` | 锚点规则的四个分支（胜负/打平/单路/同音字） |
| `test_phoneme.py` | 拼音相似度与别名表、MBR 重排双闸 |
| `test_helpers.py` | 哈希对照 oracle、严格 16k 读取契约 |

运行：`PYTHONPATH=src python -m unittest discover -s tests`。
这些测试的另一重身份是**规格说明书**——读测试比读实现更快懂规则。

### 3.4 `configs/`（2 个）

- `pipeline.json`：生产配置。`PipelineConfig` 的 37 个字段全部有默认值，
  JSON 只覆盖要改的项；出现未知键直接报错（fail fast，防拼写错误）。
  关键项：门控阈值 −0.6469、beam=3、N-best conservative_mbr、ELM 0.15、
  双单人 sft_raw_cosine（同人阈值 0.29）、batch_writes 100。
- `single_speaker_router.json`：20 维 logistic 的 mean/std/weight/bias +
  单人概率阈值 0.8436。训练与定阈零评测集参与；加载时逐项校验特征顺序。

---

## 4. 一条数据的完整旅程

以下跟着一条样本走完三个阶段的全部旅程。每步给出**源码位置**与
**为什么这样做**；字段口径与阈值细节见 `docs/ALGORITHM.md`。

### 阶段一：样本加载与分离（cli → separator）

**① 配置加载**（`engine.load_config`）。JSON 读成 `PipelineConfig`——
frozen dataclass，加载后不可变，防止整批评测期间口径漂移；未知键当场
报错（拼写检查器），不让"以为改了阈值实际没改"的静默事故流入结果。

**② 样本读取**（`core/samples.read_many_jsonl`）。jsonl 每行一个 JSON
（官方中文字段：唤醒音频/识别音频/识别文本/唤醒文本），相对路径按
`--data-root` 拼成绝对路径，装进 frozen `Sample`。`label=None` 即负样本
（期望拒识）。`--limit N` 按全集正负比例**分层抽样**（seed 固定）——
只取前 N 条会 skew 到几乎全是正样本，冒烟验不出拒识路径。

**③ 断点续跑**（`cli`）。`--resume` 读回 checkpoint 里已完成样本的 id
集合，按同一 id 口径过滤后只补跑剩余样本。checkpoint 按
`batch_writes`（100）批量落盘——中断最多损失最近一个批次；逐条写会把
几 MB 的 JSON 重写几千次，IO 反成瓶颈。

**④ 分离与缓存**（`helpers.separation_paths_for` → `separator.separate`）。
唤醒与指令各被 MossFormer2 盲分离成两路：
`{id}_wake_ms{0,1}.wav` / `{id}_command_ms{0,1}.wav`。整波一次前向，
峰值归一化到 ±0.5 后 PCM_16 落盘。两个文件都在 = 缓存命中，跳过推理
——分离是全链路最贵的步骤之一，缓存让重跑几乎免费。盲分离没有固定
通道序（PIT 训练），"哪路是目标人"由下游余弦判定。

### 阶段二：识别与门控决策（engine.PipelineRunner.process）

结果字典在 try 之前**预填全部字段默认值**——无论从哪个分支离开，
detail.json 的行结构永远完整。默认 `route=fusion_reject`、`content=""`：
没被明确接受之前，先按拒绝处理（安全一侧的默认）。

**⑤ 声学单双人判定**（`single_speaker.BothSingleRouter`）。只吃 6 段
波形，提取 20 维通道交换不变的统计特征（分帧 RMS、强弱通道活跃比、
弱通道能量占比、重构误差、包络相关性等），过线性 logistic。**零 ASR**
——便宜特征先行短路贵流程。全部特征按"强/弱通道"排序后计算，因为
分离器通道序随机，同一音频无论目标在哪一路特征都必须一样。

**⑥ 双单人短路**。唤醒和指令都判为单人时，跳过拼音锚点与 6 维门控，
微调声纹前端对两段原始音频算一次余弦，≥ 0.29 判同人。两段都是单人时
"像不像同一个人"一次余弦就是最直接证据。接受 → 只解码原始指令；
拒绝 → 置空返回，不跑任何指令 ASR。

**⑦ 唤醒通道转写与拼音锚点**（`core/firered` + `core/wake_anchor`）。
常规分支先转写唤醒两通道：与指令**共享同一份 AED 权重**，靠
`temporary_beam_size(1)` 临时压束宽求稳；所有承载决策的转写一律
`temporary_elm_weight(0.0)`，保持 confidence 量程干净。随后用
**拼音相似度**在两路里选"念了唤醒词的那一路"：同音字不罚分、别名表
自动展开；两路打平或一路为空都回退原始唤醒音频当锚点（混合音频里
至少有完整的目标人嗓音——保守方向）。

**⑧ 融合门控**（`engine.FusionGate.score`）。两个 ERes2NetV2 前端
（官方预训练 + 分离域微调）**共享一次 Fbank**，各提 5 段 embedding，
算 6 维余弦特征，z-score 后线性融合出分。一次调用产出三样：接受判定、
融合分、全部逐通道余弦（选路直接复用，零额外 embedding）。指令判为
单人时原始指令加入 `*_sep` 候选，避免近静默分离通道污染门控。
融合分 < −0.6469 拒绝：置空返回，**不跑指令 ASR**——全链路最大的
计算量节省点。

**⑨ 单/双人路由的延迟解码**。接受后按 ⑤ 的判定分两路：**单人指令**
只解码原始指令（分离通道没有额外信息，省两次 ASR）；**双人指令**恒切
分离通道，ft 余弦（门控时已算好，零成本）选路后**只解码选中的那一路**，
首选为空才补解另一路，原始指令始终不解码。"先选路再解码"比"全解码
再选路"省一半以上指令 ASR；所有解码经 `AsrCache`（键 = 路径+解码配置，
如 `b3|elm0`），重跑全部命中。

**⑩ 后处理两件套**（仅选定通道，各一次）。先**保守 MBR N-best 重排**：
beam=3 主解码顺手产出的候选已缓存，选"与其他候选加权差异最小"者，
且置信度增益 ≥ 0.03 与风险下降 ≥ 0.05 双闸齐过才允许换人——宁可不换，
不可换错。再 **ELM 重解码**：领域语言模型纠正同音错误（全系统最大
单项提分），空文本不重解（语言模型不许凭空造文本）。两件套不可能
反向影响门控/路由——这是置信度量程不被污染的结构保证。

**⑪ 两级异常兜底**。任何环节异常：先回退原始指令转写
（`raw_fallback_error`），再失败置空（`empty_fallback_error`），异常
类型写进 error 字段。一条坏样本不能拖垮整批；样本永远不消失；
坏样本事后可按 error 类型统计排查。

### 阶段三：汇总与产出（reporting → cli）

**⑫ 汇总**（`reporting.summarize`）。按 label 分正负：正样本算归一化
编辑距离口径的 CER（分子分母各自求和再除，不是逐条平均）与 PR
（转写非空比例），负样本算 RR（输出为空比例），另统计路由分布、
切换次数、异常数。

**⑬ 五件套写出**（`reporting.write_outputs` + `generate_report` + `cli`）：

| 产物 | 角色 |
|---|---|
| `submission.json` | 提交物：result.results + avg_cer/avg_rr/duration(ms) |
| `detail.json` | 病历：summary + 每样本全字段（门控特征、路由、各通道文本、后处理痕迹） |
| `detail_report.txt` | 人读报告：逐样本"标注 vs 输出 vs 决策依据"，自动生成 |
| `checkpoint.json` | 断点存档：version/duration/results |
| `manifest.json` | 指纹：完整配置 + 三份权重 SHA-256（哪份配置+哪份权重跑出的，可追溯） |

---

## 5. 贯穿全局的设计约定

1. **冻结语义不得改动**：`normalize_transcript` 归一化口径、
   `wake_anchor` 决策边界、`PipelineConfig` 默认值、`gate_fusion.json`
   特征顺序（与 `FUSION_FEATURE_KEYS` 不一致当场报错）。改动前先跑
   `tests/`，再对照 `docs/ALGORITHM.md` 的基线数字。
2. **fail fast 优于静默降级**：配置未知键报错、特征顺序错位报错、
   音频采样率不符报错、ASR 返回行数不符报错——错误在离根源最近的
   地方爆炸，排查成本最低。少数例外都有意为之（置信度非法回退 1.0 =
   保守不切换；异常样本两级兜底 = 生产容错）。
3. **一次推理的产物榨干用尽**：门控、选路、双单人短路共享同一批
   embedding；双前端共享同一次 Fbank；主解码的 N-best 候选缓存给
   后处理复用。声纹推理与 ASR 是全链路最贵的两步，绝不为同一段音频
   重复计算。
4. **缓存即键，键含配置**：分离缓存以输出文件名做键；ASR 缓存以
   `路径|解码配置` 做键——同一音频在不同配置下的结果绝不混淆。
5. **生产容错三件套**：结果行预填默认值、两级异常兜底、批量
   checkpoint——任何单点失败最坏只损失一条样本或一个批次。
