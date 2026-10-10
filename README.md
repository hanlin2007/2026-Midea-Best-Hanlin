# Midea Best：美的目标说话人语音指令识别（LSZ v4.5 最终系统）

给定唤醒音频与指令音频，判断指令音频是否含目标说话人，含则输出其指令文本
（美的语音竞赛：目标说话人语音指令识别）。仓库为重构后的最终形态：单一
系统包 `src/szy/` + 基线内核 `src/midea_qiji/` + 交付工作区
`submission/workspace/`，历史阶段包已全部删除。

**当前最好成绩：DatasetA 公式分 (1−CER+RR)/2 = 0.8937**（1838 样本）。

## 当前系统构成

- **分离前端**：dasynth 域适配微调的 MossFormer2_SS_16K torch 分离器
  （联合去混响+去噪+分离，修复官方分离器在混响/掩盖样本上的崩塌）。
- **声学单双人路由**：20 维波形特征 logistic（纯 numpy、零 ASR）判
  唤醒/指令是否各为单人；双单人走独立短路（微调声纹 raw 余弦 ≥ 0.29）。
- **唤醒锚点**：拼音音素相似度（同音字不罚分）+ 唤醒词别名表；打平或
  单路缺失回退原始唤醒音频。
- **融合门控**：预训练 + 分离域微调双 ERes2NetV2 前端（共享一次 Fbank），
  6 维余弦特征线性融合，阈值 −0.6469（dasynth valid 标定，零 DatasetA 调参）。
- **延迟解码**：单人指令只解码原始指令；双人指令 ft 余弦选路后只解码
  选中通道；被拒样本不跑任何指令 ASR。
- **后处理**：保守 MBR N-best 重排 + ELM 重解码，仅作用在选定通道。
- **ASR**：FireRedASR2-AED 单实例共享（唤醒 beam=1 / 命令 beam=3 临时切换，
  峰值显存 4818 MiB）。

## Pipeline 全景（plain text）

下图为生产管线全链路（`scripts/run.sh` → `python -m szy.cli`）：
框内编号 1.~13. 与下一节「架构细节解读」的 [1]~[13] 逐项对应
（训练侧 [14] 不在生产管线内）。

```text
    输入：pos.jsonl + neg.jsonl（官方中文字段）
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 1. 配置加载（load_config）                       │
│ · JSON 出现未知键 → fail-fast 立即报错           │
│ · SzyConfig：frozen dataclass，加载后不可变      │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 2. 样本读取（read_many_jsonl）                   │
│ · 官方中文字段：唤醒音频 / 识别音频 / 识别文本   │
│ · --limit 分层抽样冒烟；id 口径 relative_stem    │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 3. 断点续跑（--resume 可选）                     │
│ · checkpoint 按 batch_writes(100) 批量落盘       │
│ · ASR 缓存同粒度；正常结束统一 flush             │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 4. 资源装配（SzyRunner.__init__）                │
│ · FusionGate：预训练 + 分离域微调双 ERes2NetV2   │
│ · BothSingleRouter：20 维波形 logistic           │
│ · MossFormer2Separator：dasynth 微调分离器       │
│ · FireRedRecognizer：单实例 AED（beam=3 + ELM）  │
└────────────────────────┬─────────────────────────┘
                         ▼
       逐样本处理：SzyRunner.process(sample)
                         ▼
┌──────────────────────────────────────────────────┐
│ 5. 分离与缓存（separation_paths_for）            │
│ · 唤醒 ×2 + 指令 ×2 共 4 段分离 wav（整波推理）  │
│ · 双文件都在 = 缓存命中，跳过推理                │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 6. 声学单双人判定（BothSingleRouter）            │
│ · 20 维通道交换不变波形特征，纯 numpy，零 ASR    │
│ · 判定唤醒/指令各自是否单人，写入 detail         │
└────────────────────────┬─────────────────────────┘
                         ▼
              ┌──────────┴──────────┐
              │ 双单人？            │
              ▼                     ▼
┌───────────────────────────┐   （否 → 常规分支，接 7.）
│ 6a. 双单人短路            │
│ · ft 微调前端 raw 余弦    │
│ · ≥0.29 判同人            │
│ · 接受：只解码 raw 指令   │
│   → 后处理（12.）         │
│ · 拒绝：输出空            │
└───────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 7. 唤醒通道转写（_transcribe_wake）              │
│ · 与命令共享一份 AED；beam 临时压到 1（求稳）    │
│ · ELM 一律关闭（temporary_elm_weight(0.0)）      │
│ · 经 AsrCache（键 = 路径|解码配置）              │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 8. 拼音锚点（select_wake_anchor）                │
│ · 拼音相似度高者胜（同音字不罚分）               │
│ · 别名表展开：hicolmo ↔ haikemu                  │
│ · 打平 / 一路空 → 锚点回退原始唤醒音频           │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 9. 融合门控（FusionGate.score）                  │
│ · 双前端共享一次 Fbank，各提 5 段 embedding      │
│ · 6 维余弦特征 → z-score → 线性融合分            │
│ · 一次打分产出：接受判定 + 融合分 + 逐通道余弦   │
└────────────────────────┬─────────────────────────┘
                         ▼
              ┌──────────┴──────────────┐
              │                         │
              ▼                         ▼
       score ≥ −0.6469           score < −0.6469
       （接受 → 继续）                 │
              │                         ▼
              │                     ┌───────────────────────────┐
              │                     │ Reject（样本结束）        │
              │                     │ · route=fusion_reject     │
              │                     │ · content=""（拒识）      │
              │                     │ · 不跑任何指令 ASR        │
              │                     └───────────────────────────┘
              ▼
┌──────────────────────────────────────────────────┐
│ 10. 单/双人路由的延迟解码                        │
│ · 指令单人：只解码原始指令（raw_command_single） │
│ · 指令双人：ft 余弦选路，只解码选中通道          │
│   （首选为空才补解另一路；绝不解码 raw）         │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 11. 异常兜底（两级，全程包裹）                   │
│ · 任何异常：先回退原始转写（raw_fallback_error） │
│ · 再失败：置空（empty_fallback_error）           │
│ · 结果行预填默认值，样本永不消失                 │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 12. 后处理两件套（仅选定通道，各一次）           │
│ · 保守 MBR N-best 重排（复用主解码缓存的候选）   │
│ · ELM 重解码（weight 0.15；空文本不重解）        │
│ · 不反向影响门控/路由（Phase 2 核心不变量）      │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 13. 汇总与五件套（summarize → write_outputs）    │
│ · CER / PR / RR + 路由分布；公式分 (1−CER+RR)/2  │
│ · submission / detail / detail_report /          │
│   checkpoint / manifest（配置+权重 sha256）      │
└──────────────────────────────────────────────────┘
```

## 架构细节解读（对照源码、算法与伪代码）

以下按 Pipeline 图阶段编号逐项解读：每节给 **源码位置**、**具体实现** 与
**算法/伪代码**。更细的字段口径、缓存命名与阈值含义见
`docs/一条数据的流动全景-01/02/03`（样本加载与分离前端 / 识别与门控决策 /
汇总与提交产出）。

### [1] 配置层：SzyConfig + load_config

源码：`src/szy/pipeline.py:108-204`。

- `SzyConfig` 是 **frozen dataclass**：加载后不可变（运行时赋值当场抛
  `FrozenInstanceError`），保证整批评测期间阈值/模型路径口径一致；
  全部字段有默认值，JSON 只需覆盖要改的项。
- `load_config` 是"拼写检查器"：用 `SzyConfig.__dataclass_fields__` 当
  白名单与 JSON 键比对，**未知键立即 ValueError**（fail fast），避免拼错
  的配置项被静默忽略、按默认值跑出另一条结果。
- `--model-root` 覆盖用 frozen 对象的"复制+覆盖"惯用法：

```python
config = SzyConfig(**{**config.__dict__, "model_root": args.model_root})
```

### [2] 数据入口：Sample / read_many_jsonl / submission_id

源码：`src/midea_qiji/infer.py`（Sample / read_many_jsonl / submission_id）。

- JSONL 每行一个 JSON（官方中文字段：`唤醒音频` / `识别音频` /
  `识别文本` / `唤醒文本`）；相对路径以 `--data-root` 为基准拼成绝对路径，
  装进 frozen `Sample`。`label=None` 即负样本（期望拒识）。
- `submission_id(sample, id_mode)` 按竞赛口径生成提交 id，生产用
  `relative_stem` = 相对路径去掉 `.wav` 扩展名 + 正斜杠统一：

```python
path = Path(sample.command_relative_path)
return (path.parent / path.stem).as_posix()
```

- `--limit N` 按全集正负比例**分层抽样**（seed 固定可复现）：只取前 N 条
  会 skew 到几乎全是正样本，冒烟既验不出拒识路径也算不出有意义的 RR。

### [3] 断点续跑与 ASR 缓存

源码：`src/szy/cli.py`（续跑循环）、`src/szy/pipeline.py`（AsrCache）。

- `--resume` 读回 checkpoint 里已完成样本的 id 集合（`set`，O(1) 判重），
  按同一 id 口径过滤后只补跑剩余样本，并继承历史耗时。
- checkpoint 按 `batch_writes`（现役 100）**批量落盘**：中断最多损失
  最近一个批次；逐条写会把几 MB 的 JSON 重写几千次，IO 反成瓶颈。
- `AsrCache` 缓存键 = `<音频路径>|<解码配置>`（如 `b3|elm0`、`b3|elm0.15`）：
  同一音频会在多种解码配置下被转写，键必须带配置标签，否则会把"别的配置
  解出的文本"错当本配置结果。beam=3 主解码顺手产出的 N-best 候选一并缓存，
  后处理 MBR 重排直接复用，零额外推理。`asr_cache_persistent: false` 时
  只保留进程内缓存（评测口径：结果只依赖当次运行）；本地迭代改 `true`
  可跨运行复用，重跑约 10× 提速。

### [4] 资源装配与延迟导入

源码：`src/szy/pipeline.py:437-560`。

- 门控最先构造（决定大部分样本去留的第一道闸）；`szy.separator` 在
  构造函数内**延迟导入**（它 import 时向 `sys.path` 注入 `third_party/`
  并引入 torch 依赖），保证无 torch 环境也能 `--help` / 跑 numpy-only 单测。
- ASR 后端二选一：`firered_asr_type: "aed"`（生产基线）或 `"llm"`
  （Qwen2-7B 解码，上限更高但慢得多）。
- AED 单实例共享：唤醒与命令通道共用一份权重，唤醒侧用
  `temporary_beam_size(1)` + 决策转写一律 `temporary_elm_weight(0.0)`
  临时覆盖（峰值显存 4818 MiB，比双实例时代省约 2.5 GB，还消除了
  两份模型同时驻留导致的数值/内核选择抖动）。

### [5] 分离前端：MossFormer2Separator（torch 版）

源码：`src/szy/separator.py`；模型定义 vendor 于
`third_party/mossformer2/mossformer2.py`（ClearerVoice-Studio，仅改包内
相对导入，结构零改动）。

**模型结构**（`MossFormer2_SS`，cfg：`num_spks=2, encoder_kernel_size=16,
encoder_embedding_dim=512, mossformer_sequence_dim=512,
num_mossformer_layer=24`，约 55.7M 参数）：

```text
波形 [B, T]
  → Encoder       Conv1d(1→512, k=16, s=8) + ReLU        （可学习 STFT，50% 重叠）
  → MaskNet       归一化 → 1×1 conv → 正弦位置编码
                  → 24 × GFSMN 块（Flash 分组注意力 + 门控 FSMN 记忆块交替）
                  → 门控输出 Tanh(值) × Sigmoid(门) → 掩码 [2, B, 512, T/8]
  → 掩蔽          sep_x = 特征 ⊙ 掩码                    （逐说话人）
  → Decoder       ConvTranspose1d(512→1, k=16, s=8)      （可学习 iSTFT，重叠相加）
                  → 两路波形 out[0]/out[1]（= 下游的 ms0/ms1）
```

- 缓存键 = 输出文件名（`{item_id}_wake/command_ms{0,1}.wav`）：调参重跑
  几十遍时只有首次真正分离。
- 盲分离无固定通道序（PIT 训练），"哪路是目标人"完全由下游余弦打分判定。
- 生产权重经 dasynth 域适配微调（数字见 `docs/history/CURRENT_BEST_PIPELINE_SZY_V4.md`）。

### [6] 声学单双人判定：BothSingleRouter

源码：`src/szy/single_speaker.py`；bundle：`configs/moss_single_speaker_router_v2.json`。

- 20 维**通道交换不变**波形特征：分帧 RMS（20ms）强弱通道活跃比、弱通道
  能量占比（mean/p50/p90）、主导翻转率、包络相关性、重构误差（含最小二乘
  拟合版）、波形余弦等。单人录音里分离器的弱通道几乎没有能量，这套统计量
  足以判单双人。
- 全部特征按"强/弱通道"排序后计算而非直接用 ms0/ms1——分离器通道序
  随机（PIT 训练），同一段音频无论目标在哪一路特征都一样。
- 线性 logistic：z-score → 点积 → sigmoid；阈值 0.8436 与权重同 bundle，
  训练与定阈零 DatasetA。加载时逐项校验特征名顺序（与门控同一套
  "隐形契约"防错位思路）。
- **双单人短路**（`both_single_mode: "sft_raw_cosine"`）：唤醒和指令都
  判为单人时，跳过拼音锚点与 6 维门控，微调前端 `cos(raw_wake, raw_cmd)`
  ≥ 0.29 判同人；接受只解码原始指令，拒绝直接置空。

### [7] ASR 后端：FireRedASR2-AED（共享实例 + 缓存）

源码：`src/midea_qiji/recognizers.py:93-320`；转写入口
`src/szy/pipeline.py:_transcribe_with_asr`。

- AED = Attention Encoder-Decoder（无 LLM 解码器，快且省显存）；构造时向
  `sys.path` 注入 vendor（`kaldi_native_fbank`）与 FireRed 源码仓库目录。
- 解码配置（生产）：主解码 `beam_size=3, nbest=3`（顺手产出 MBR 候选）、
  `softmax_smoothing=1.25`、`aed_length_penalty=0.6`；唤醒侧临时 beam=1。
- 两个上下文管理器让单实例替代双实例：`temporary_beam_size`（nbest 自动
  clamp 到束宽，退出恢复）、`temporary_elm_weight`（ELM 权重常驻，解码时
  按 config.elm_weight 开关）。
- 所有承载决策的转写（唤醒锚点、各指令通道、原始指令）都在 ELM 关闭下
  解码——`raw_confidence` 保持路由标定时的量程（~0.88）。ELM 只在路由
  选定后对最终通道重解码一次。

**文本归一化五步**（`src/midea_qiji/text.py`，全仓库唯一口径、冻结语义）：

```text
normalize_transcript(t):
    1. 删除模型特殊 token（<|...|> 形式）
    2. "(\d+)%"        → "百分之\1"        （30% → 百分之30）
    3. "百分之(\d+)"    → 百分之 + 中文数字  （→ 百分之三十）
    4. 剩余 "\d+"       → 中文读法          （25 → 二十五）
    5. 删除全部标点/空白 + 转小写
```

### [8] 拼音锚点：select_wake_anchor

源码：`src/midea_qiji/moss_candidate.py:111-172`（规则）、
`src/midea_qiji/phoneme.py`（拼音相似度与别名表）。

唤醒音频可能同时录进目标人与家人/噪声，分离成两路后用**拼音相似度**找出
"念了唤醒词的那一路"作为目标人嗓音的注册锚点：

```text
select_wake_anchor(wake_text, s0_text, s1_text):
    ref = normalize_transcript(wake_text)
    一路空 → selected 记有声路，锚点回退原始唤醒音频（raw_wake_fallback=True）
    sim0, sim1 = phoneme_similarity(s0, ref), phoneme_similarity(s1, ref)
    selected = 0 if sim0 >= sim1 else 1      # 像者胜；平手先记 source0
    tie（或 ref 为空）→ 锚点同样回退原始唤醒音频  # 读音一样像，无法可靠裁决
```

- `phoneme_similarity`：文本 → pypinyin（TONE3 去声调）紧凑拼音串 →
  编辑距离取负（0 = 读音完全一致）。同音字不罚分（"科慕"vs"科目"距离 0）
  ——相对旧字符编辑距离版，这是 DatasetA 上最大的单项护城河
  （51 条锚点选择差异带来 81 字优势）。
- 别名表：`hicolmo ↔ haikemu` 双向展开取最高分，覆盖唤醒词的中英变体。

### [9] 融合门控（全链路核心）

源码：`src/szy/pipeline.py:285-431`（FusionGate）、
`src/midea_qiji/gate.py:163-290`（ModelScopeEmbedder）。

**双前端共享 Fbank**：同一 ERes2NetV2 基座的两个实例——官方预训练
（泛化好）+ 分离域微调（懂分离通道里的目标嗓音，只覆盖 encoder 权重）。
两个前端声学前端完全相同，Fbank 只算一次（`prepare_waveforms`），各自
前向 `embedding_model`（`extract_features`）——计算量省一半，不增加
并发激活显存。

**五段波形**（固定顺序）：唤醒锚点、原始唤醒、原始指令、分离通道 0、
分离通道 1 → 每前端各出 5 个 L2 归一化 embedding。

**6 维特征**（顺序即权重向量维度含义，加载时与 `gate_fusion.json` 的
`features` 字段逐一校验，错位立即报错）：

| 特征 | 公式（tag ∈ {pre, ft}） | 含义 |
|---|---|---|
| `{tag}_raw` | `cos(E(raw_wake), E(raw_cmd))` | 原始混合间相似度（分离前基线） |
| `{tag}_sep` | `max(cos(E(anchor),E(cmd0)), cos(E(anchor),E(cmd1))[, cos(E(anchor),E(raw_cmd)) 当指令单人时])` | 分离后目标人浓度 |
| `ch_sim_{tag}` | `cos(E(cmd0), E(cmd1))` | 两通道分离度 |
| `{tag}_cos_cmd0/1` | `cos(E(anchor), E(cmd0/1))` | **过程量**：选路直接复用 |
| `{tag}_rawcmd` | `cos(E(anchor), E(raw_cmd))` | **过程量** |

**融合打分**（z-score + 线性加权）：

```text
x     = (features[6 维，按 FUSION_FEATURE_KEYS 顺序] − mean) / std
score = x · weight + bias
accept = score ≥ −0.6469
```

- 线性融合参数在 **dasynth valid** 上拟合 logistic（零 DatasetA 调参）。
- 一次 `score` 产出三样：接受判定、融合分、全部逐通道余弦——门控与
  选路共享同一批 embedding，绝不为同一段音频重复提特征（声纹推理是
  全链路最贵一步）。
- 拒绝的样本直接返回：不跑指令 ASR——全链路最大的计算量节省点。

### [10] 单/双人路由的延迟解码

源码：`src/szy/pipeline.py:894-980`。

- **指令单人**（`raw_command_single`）：只解码原始指令一次。单人录音的
  分离通道没有额外信息，省两次 ASR。
- **指令双人**（`moss_candidate`）：恒切分离通道；ft 余弦（门控时已算好，
  零成本）选路后**只解码选中的那一路**；首选转写为空才补解另一路；
  原始指令始终不解码（双人混合的 raw 转写没有参考价值）。
- "先选路再解码"相对旧版"全解码再选路"省一半以上指令 ASR；AsrCache
  保证重跑时这些解码全部免费。

历史说明：此前的"音色触发规则"（confidence × 关键词 × sv_diff）已不在
生产路径上，规则本体（`should_switch_command_timbre`，切点已参数化）保留
供重放与消融；它在新 confidence 量程上的重扫值（0.88/0.76/−0.03）也保留
在配置里。演进始末见 `docs/OPTIMIZATION_SUMMARY.md` 附录。

### [11] 两级异常兜底

源码：`src/szy/pipeline.py:981-997`。

结果行在 `try` 之前就**预填全部字段默认值**（完整 schema），任何路径返回
的都是完整一行；异常时先回退原始转写、再回退置空，样本永不消失：

```text
try:
    …正常链路…
except Exception as e:
    error = "Type: msg"
    try:   content = ASR.transcribe_many([原始指令])[0];  route = "raw_fallback_error"
    except Exception as e2:
        error += "; raw_fallback=…";  content = "";  route = "empty_fallback_error"
```

### [12] 后处理两件套：N-best 重排 + ELM 重解码

源码：`src/szy/pipeline.py:640-720`、
`src/midea_qiji/firered_nbest.py`。

- **保守 MBR N-best 重排**：从主解码缓存的 N-best 候选里选"与其他候选
  置信度加权差异最小"的假设（候选集合的"中位数"，期望错误最少）；只有
  置信度增益 ≥ 0.03 且 MBR 风险下降 ≥ 0.05 **同时**成立才允许换掉 top-1
  ——宁可不换，不可换错。
- **ELM 重解码**：外部语言模型熟悉智能家居领域词汇，纠正"调到/掉到"类
  同音错误（全系统最大单项提分，+0.0126）；`temporary_elm_weight(0.15)`
  下对选定通道重解码一次；空文本不重解（语言模型不许凭空造文本）。
- **Phase 2 核心不变量**：两件套只作用在选定通道、只跑一次，不可能
  反向影响门控/路由——旧版 ELM 融在主解码里会把 confidence 从 ~0.88
  压到 ~0.60，让依赖置信度的规则退化。

### [13] 汇总与五件套

源码：`src/szy/reporting.py`、`src/szy/cli.py:122-180`。

- **指标口径**（按 label 分正负；文本一律先归一化再比）：

```text
CER = Σ lev(norm(hyp), norm(ref)) / Σ len(norm(ref))     # 正样本，越低越好
PR  = 正样本转写非空比例                                  # 空转写整条 0 分
RR  = 负样本输出为空比例                                  # 负样本输出任何文字整条 0 分
公式分 = (1 − CER + RR) / 2                              # 竞赛计分，本仓库 = 0.8937
```

- **submission.json**（交卷）：`{"result": {"results": [{id, content, label,
  cer}], "avg_cer", "avg_rr", "duration(ms)"}}`；
- **detail.json**（病历）：summary + 每样本全字段（6 维门控特征、路由、
  耗时、各通道文本、N-best/ELM 痕迹）——事后可离线重扫阈值/重放门控；
- **detail_report.txt**（人读）：逐样本病历报告，自动生成
  （`scripts/generate_report.py` 可为旧产物补生成）；
- **checkpoint.json**：断点存档；
- **manifest.json**（指纹）：完整配置快照 + `moss_model` /
  `fusion_backend` / `finetuned_sv_ckpt` 三份权重的 SHA-256。

### [14] 训练侧：dasynth 域适配微调（不在生产管线内，离线执行）

源码：`src/szy/train.py`（分离器）；声纹微调与融合标定在仓库外工作区。
完整实验见 `docs/history/CURRENT_BEST_PIPELINE_SZY_V4.md`。

- **数据**：dataset_dasynth train 18k 条（AISHELL-1 + CN-Celeb 合成，
  DatasetA 风格腐蚀链：重叠/混响/噪声/削波），每条含腐蚀混合 + **腐蚀前
  对齐干净双源**——干净目标迫使模型联合学会去混响+去噪+分离。
- **训练**：PIT SI-SNR 损失、2s 随机裁剪、bf16 autocast、batch 2 × 梯度
  累积 2、Adam lr 5e-5 / weight_decay 1e-5、梯度裁剪 10.0。
- **选模**：每 epoch 在 valid 前 200 条整波推理 oracle 排列 SI-SDR，按
  ALL 均值存 best。训练/选模零 DatasetA 参与（DatasetA 只做冻结权重的
  一次性应用）。

### 关键数字速查

| 项 | 值 | 源码位置 |
|---|---|---|
| 融合门控阈值 | **−0.6469**（dasynth valid 标定） | `configs/pipeline_firered.json` |
| 6 维特征顺序 | pre_raw, ft_raw, pre_sep, ft_sep, ch_sim_pre, ch_sim_ft | `pipeline.py FUSION_FEATURE_KEYS` |
| 双单人同人阈值 | 0.29（单人概率阈 0.8436 在 bundle 内） | `both_single_sft_threshold` |
| 主解码 beam / N-best | 3 / conservative_mbr（双闸 0.03 + 0.05） | `firered_beam_size` / `firered_nbest_*` |
| ELM | weight 0.15，beam 3，仅选择后重解码 | `firered_elm_*` |
| 音色规则切点（重放用） | 0.88 / 0.76 / −0.03（新量程；旧默认 0.70/0.57/0.05 勿混用） | `firered_switch_*` |
| 分离器 cfg | num_spks=2, kernel=16, dim=512, 24 层 | `separator.py` |
| 分离输出 | peak→0.5 归一化, PCM_16 落盘 | `separator.py` |
| 采样率 | 16000（全仓库唯一常量） | `midea_qiji/audio.py` |
| 置信度回退 | 非法/NaN/Inf → 1.0（保守不切换） | `helpers.py` / `pipeline.py` |
| id 口径 | relative_stem | `infer.py` |
| ASR 缓存键 | `<路径>\|b<beam>\|elm<weight>` | `pipeline.py AsrCache` |

## 质量分

DatasetA 公式分 **(1−CER+RR)/2 = 0.8937**（PR 0.9150 / RR 0.9304 /
CER 0.1430，1838 样本）。从 v4 基线 0.8679 到 0.8937 的逐项优化实验
（拼音锚点 / N-best / ELM / 通道分离 / Phase 2 重扫）见
`docs/OPTIMIZATION_SUMMARY.md`；现役管线完整口径见
`docs/CURRENT_BEST_PIPELINE.md`。

## 目录布局

```text
midea_qiji_best_hanlin/
├── configs/               # pipeline_firered.json（生产）+ 消融配置 + 双单人 router bundle
├── scripts/               # run.sh（固定输出）/ run_exp.sh（命名实验）/ 编排与重放工具
│   └── experiments/       # 离线实验工具（阈值重扫 / N-best 分析 / 测 q / gate_v2）
├── src/szy/               # 最终系统包：cli / pipeline / single_speaker / separator /
│                          #   metrics / train / diag / sanity_check / helpers / reporting
├── src/midea_qiji/        # 基线内核：text / phoneme / firered_nbest / audio / paths /
│                          #   gate / moss_candidate / recognizers / infer
├── tests/                 # 29 个 numpy-only 单测（无需模型/GPU）
├── third_party/mossformer2/  # vendor 模型定义（ClearerVoice-Studio，未改动）
├── submission/workspace/  # 交付工作区（自包含推理代码 + vendor 依赖 + run_test.py）
├── docs/                  # 现役文档 + 流动全景系列 + history/（阶段存档）
├── model/                 # 资产（不进 git）：artifacts + 声纹基座 + FireRed + vendor
├── dataset/               # 官方数据（不进 git）：dataset_test/datasetA/datasetA/
└── outputs/               # 产物（不进 git）：分离/ASR 缓存 / 提交物 / checkpoint
```

`model/`、`dataset/`、`outputs/` 由 `.gitignore` 排除；资产填充方式见
`model/README.md`。

## 运行与验证

```bash
conda activate yuyin

# 冒烟：分层抽样 5 条；去掉 --limit 跑全量
bash scripts/run.sh --limit 5

# 命名实验：产物收进 outputs/<NAME>/，自动生成人类可读报告
bash scripts/run_exp.sh baseline

# 资产自检（依赖 model/MANIFEST.json，由 relocate_workspace_models.py 生成）
python scripts/check_models.py

# 单元测试（29 个用例，numpy-only）
PYTHONPATH=src python -m unittest discover -s tests
```

默认读取仓库内的 `dataset/` 与 `model/`；可用 `PYTHON_BIN`、
`CUDA_VISIBLE_DEVICES`、`MODELSCOPE_CACHE`、`HF_HUB_OFFLINE` 覆盖运行环境。
等价命令行入口见 `python -m szy.cli --help`（或 `docs/代码库结构解读.md`
第 3 节）。

## 文档索引

阅读路径（新手按此顺序）：

1. **本 README** — 系统总览与运行方式
2. `docs/代码库结构解读.md` — 逐文件解读与入口关系（仓库地图）
3. `docs/一条数据的流动全景-01-样本加载与分离前端.md`
4. `docs/一条数据的流动全景-02-识别与门控决策.md`
5. `docs/一条数据的流动全景-03-汇总与提交产出.md`
6. `docs/CURRENT_BEST_PIPELINE.md` — 现役管线算法细节与 0.8937 的口径
7. `docs/OPTIMIZATION_SUMMARY.md` — 逐项优化实验（为什么有效 / 代价 / 未采用项）
8. `docs/history/` — v1→v4 阶段演进存档（含 0.8679 时代的完整实验）
9. `submission/workspace/README.md` — 交付工作区说明（Docker 构建上下文）
