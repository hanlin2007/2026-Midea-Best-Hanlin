# Midea Best：美的目标说话人语音指令识别（SZY 最终系统）

给定唤醒音频与指令音频，判断指令音频是否含目标说话人，含则输出其指令文本
（美的语音竞赛：目标说话人语音指令识别）。仓库现为重构后的最终形态：单一
系统包 `src/szy/` + 基线内核 `src/midea_qiji/`，历史阶段包已全部删除。

## 当前系统构成

- **融合门控**：预训练 + dasynth 分离域微调的双 ERes2NetV2 前端，6 维余弦
  特征线性融合打分，阈值 −0.6469（dasynth valid 标定，零 DatasetA 调参）。
- **分离前端**：dasynth 域适配微调的 MossFormer2_SS_16K torch 分离器
  （联合去混响+去噪+分离，修复官方分离器在混响/掩盖样本上的崩塌）。
- **ASR**：FireRedASR2-AED 主后端（LLM 后端可选）。
- **决策链**：唤醒文本选锚点 → 融合门控接受/拒绝 → 音色触发（sv_diff）
  → 微调前端直接余弦选分离通道转写 → 任何异常回退原始转写。

## Pipeline 全景（plain text）

下图为生产管线全链路（`scripts/run.sh` → `python -m szy.cli`）：
框内编号 1.~12. 与下一节「架构细节解读」的 [1]~[12] 逐项对应
（训练侧 [13] 不在生产管线内）。

```text
    输入：pos.jsonl + neg.jsonl（官方中文字段）
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 1. 配置加载（load_config）                       │
│                                                  │
│ · JSON 出现未知键 → fail-fast 立即报错           │
│ · SzyConfig：frozen dataclass，加载后不可变      │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 2. 样本读取（read_many_jsonl）                   │
│                                                  │
│ · 官方中文字段：唤醒音频 / 识别音频 / 识别文本   │
│ · 相对路径按 data_root 拼绝对路径 → Sample       │
│ · --limit 截断冒烟；id 口径 relative_stem        │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 3. 断点续跑（--resume 可选）                     │
│                                                  │
│ · 读 checkpoint 已完成 id 集合，同口径过滤       │
│ · 每样本落盘一次，中断最多损失一条               │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 4. 资源装配（SzyRunner.__init__）                │
│                                                  │
│ · 重型资源加载一次，复用全批                     │
│ · FusionGate：预训练 + 分离域微调双 ERes2NetV2   │
│ · MossFormer2Separator：dasynth 微调分离器       │
│ · FireRedRecognizer：FireRedASR2-AED（beam=1）   │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
       逐样本处理：SzyRunner.process(sample)
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 5. 分离与缓存（separation_paths_for）            │
│                                                  │
│ · 唤醒 ×2 + 指令 ×2 共 4 段分离 wav（整波推理）  │
│ · 命名 {id}_wake/command_ms{0,1}.wav             │
│ · 双文件都在 = 缓存命中，跳过推理                │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 6. 批量转写（transcribe_firered_rows）           │
│                                                  │
│ · 4 路分离通道一次批量送入 FireRed ASR           │
│ · 按 uttid 重排回输入顺序，数量不符即报错        │
│ · 文本归一化；置信度非法回退 1.0                 │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 7. 唤醒锚点（select_wake_anchor）                │
│                                                  │
│ · 两路都有文本 → 与 wake_text 编辑距离小者胜     │
│ · 一路空 → 锚点回退原始唤醒音频                  │
│ · 平手取 source0（复刻历史口径）                 │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 8. 融合门控（FusionGate.score）                  │
│                                                  │
│ · 双前端各提 5 段 embedding（L2 归一化）         │
│ · 6 维余弦特征 → z-score → 线性融合分            │
│ · 一次打分产出：接受判定 + 融合分 + 逐通道余弦   │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
              ┌──────────┴──────────────┐
              │                         │
              ▼                         ▼
       score ≥ −0.6469           score < −0.6469
       （接受 → 继续）
              │                         │
              │                         ▼
              │                     ┌───────────────────────────┐
              │                     │ Reject（样本结束）        │
              │                     │ · route=fusion_reject     │
              │                     │ · content=""（拒识）      │
              │                     └───────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────┐
│ 9. 原始转写 + 音色触发（音色规则）               │
│                                                  │
│ · raw_command = ASR(原始指令音频)                │
│ · 特征：字速 / 关键词命中 / sv_diff 音色差       │
│ · conf≤0.70：kw≥0.5 或 sv_diff>0.05 或 conf≤0.57 │
│ · 触发即切换；高置信区需 sv_diff>0.05 且 kw<0.5  │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
              ┌──────────┴──────────────┐
              │                         │
              ▼                         ▼
            触发                     不触发
       （切分离通道）
              │                         │
              │                         ▼
              │                     ┌───────────────────────────┐
              │                     │ raw（样本结束）           │
              │                     │ · route=raw               │
              │                     │ · content=原始转写        │
              │                     └───────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────┐
│ 10. 候选选择（ft 直接余弦）                      │
│                                                  │
│ · cos(anchor,ms0) ≥ cos(anchor,ms1) ? ms0 : ms1  │
│ · 分离通道转写为空 → 回退 orig                   │
│ · route=moss_candidate                           │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 11. 异常兜底（两级，全程包裹）                   │
│                                                  │
│ · 任何异常：先回退原始转写（raw_fallback_error） │
│ · 再失败：置空（empty_fallback_error）           │
│ · 结果行预填默认值，样本永不消失                 │
│                                                  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 12. 汇总与三件套（summarize → write_outputs）    │
│                                                  │
│ · CER / PR / RR + 路由分布；公式分 (1−CER+RR)/2  │
│ · submission.json（交卷）+ detail.json（明细）   │
│ · manifest.json：配置快照 + 三份权重 sha256      │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 架构细节解读（对照源码、算法与伪代码）

以下按 Pipeline 图阶段编号逐项解读：每节给 **源码位置**、**具体实现** 与
**算法/伪代码**。更细的字段口径、缓存命名与阈值含义见
`docs/一条数据的流动全景-01/02/03`（样本加载与分离前端 / 识别与门控决策 /
汇总与提交产出）。

### [1] 配置层：SzyConfig + load_config

源码：`src/szy/pipeline.py:90-151`。

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

源码：`src/midea_qiji/infer.py:95-219`。

- JSONL 每行一个 JSON（官方中文字段：`唤醒音频` / `识别音频` /
  `识别文本` / `唤醒文本`）；相对路径以 `--data-root` 为基准拼成绝对路径，
  装进 frozen `Sample`。`label=None` 即负样本（期望拒识）。
- `submission_id(sample, id_mode)` 按竞赛口径生成提交 id，生产用
  `relative_stem` = 相对路径去掉 `.wav` 扩展名 + 正斜杠统一：

```python
path = Path(sample.command_relative_path)
return (path.parent / path.stem).as_posix()
```

### [3] 断点续跑

源码：`src/szy/cli.py:87-119`。

- `--resume` 读回 checkpoint 里已完成样本的 id 集合（`set`，O(1) 判重），
  按同一 id 口径过滤后只补跑剩余样本，并继承历史耗时。
- **每处理一条样本就落盘一次 checkpoint**（写盘毫秒级 vs 推理秒级）：
  任何时刻中断最多损失一条。全量 1838 样本约 26~46 分钟，中断即续跑。

### [4] 资源装配与延迟导入

源码：`src/szy/pipeline.py:299-345`。

- 门控最先构造（决定大部分样本去留的第一道闸）；`szy.separator` 在
  构造函数内**延迟导入**（它 import 时向 `sys.path` 注入 `third_party/`
  并引入 torch 依赖），保证无 torch 环境也能 `--help` / 跑 numpy-only 单测。
- ASR 后端二选一：`firered_asr_type: "aed"`（生产基线）或 `"llm"`
  （Qwen2-7B 解码，上限更高但慢得多）。

### [5] 分离前端：MossFormer2Separator（torch 版）

源码：`src/szy/separator.py:56-169`；模型定义 vendor 于
`third_party/mossformer2/mossformer2.py`（ClearerVoice-Studio，仅改包内
相对导入，结构零改动）。

**模型结构**（`MossFormer2_SS`，cfg：`num_spks=2, encoder_kernel_size=16,
encoder_embedding_dim=512, mossformer_sequence_dim=512,
num_mossformer_layer=24`，约 55.7M 参数；历史 ONNX 轻量变体 3.97M）：

```text
波形 [B, T]
  → Encoder       Conv1d(1→512, k=16, s=8) + ReLU        （可学习 STFT，50% 重叠）
                  → 特征 [B, 512, T/8]
  → MaskNet       归一化 → 1×1 conv → 正弦位置编码
                  → 24 × GFSMN 块（Flash 分组注意力 + 门控 FSMN 记忆块交替）
                  → PReLU → 1×1 conv 通道×2（每说话人一套）
                  → 门控输出 Tanh(值) × Sigmoid(门) → 1×1 conv → ReLU
                  → 掩码 [2, B, 512, T/8]（非负）
  → 掩蔽          sep_x = 特征 ⊙ 掩码                    （逐说话人）
  → Decoder       ConvTranspose1d(512→1, k=16, s=8)      （可学习 iSTFT，重叠相加）
                  → 两路波形 out[0]/out[1]（= 下游的 ms0/ms1）
```

**分离调用伪代码**（`separator.py:125-169`）：

```text
separate(source, output_prefix):
    outputs = [prefix_ms0.wav, prefix_ms1.wav]
    if 两个文件都存在: return outputs          # 缓存命中（"两个都在"才算，半成品不算）
    wav = read_mono_16k(source)               # 严格 16k 单声道 float32；不合法直接报错
    with torch.no_grad():
        out = model(wav[None, :])             # [1, 2, T]，整波一次前向
    for i in (0, 1):
        s = out[i].squeeze(0) → numpy
        s = s / max|s| × 0.5                  # peak→0.5 归一化（与历史 ONNX 口径一致，防爆音）
        sf.write(outputs[i], s, 16000, PCM_16)
    return outputs
```

- 缓存键 = 输出文件名（`{item_id}_wake/command_ms{0,1}.wav`，item_id 中
  `/` 换成 `_`，见 `helpers.py:180-201`）：调参重跑几十遍时只有首次真正分离。
- 盲分离无固定通道序（PIT 训练），"哪路是目标人"完全由下游余弦打分判定。
- 生产权重经 dasynth 域适配微调（修复官方分离器在混响/掩盖样本上的崩塌，
  数字见 `docs/CURRENT_BEST_PIPELINE_SZY_V4.md` §2/§3）。

### [6] ASR 后端：FireRedASR2-AED

源码：`src/midea_qiji/recognizers.py:93-156`；批量转写封装
`src/szy/helpers.py:138-177`。

- AED = Attention Encoder-Decoder（无 LLM 解码器，快且省显存）；构造时向
  `sys.path` 注入 vendor（`kaldi_native_fbank`）与 FireRed 源码仓库目录。
- 解码配置（历史调参值）：`beam_size=1`（贪心）、`nbest=1`、
  `softmax_smoothing=1.25`、`aed_length_penalty=0.6`。
- `transcribe_firered_rows` 一次批量转 4 路分离通道：给每条配 uttid →
  `asr._model.transcribe(uttids, paths)` 返回行 → **按 uttid 重排回输入顺序**
  （行序与请求序未必一致），数量对不上立即报错（防文本错位污染 CER）。
- 每个结果统一口径：文本过 `normalize_transcript`、置信度过
  `asr_confidence`（**非法/NaN/Inf 回退 1.0**——置信度只用于"怀疑时才切
  换"，回退最自信 = 保守不切换）。

**文本归一化五步**（`src/midea_qiji/text.py:71-122`，全仓库唯一口径、
冻结语义，改动即破坏 CER/PR/RR 复现）：

```text
normalize_transcript(t):
    1. 删除模型特殊 token（<|...|> 形式）
    2. "(\d+)%"        → "百分之\1"        （30% → 百分之30）
    3. "百分之(\d+)"    → 百分之 + 中文数字  （→ 百分之三十）
    4. 剩余 "\d+"       → 中文读法          （25 → 二十五；<10/20/100/1000 分档）
    5. 删除全部标点/空白 + 转小写
```

CER 与编辑距离（`text.py:125-177`）：`levenshtein` 是字符级动态规划，
滚动两行数组把空间从 O(m·n) 降到 O(n)；`cer = lev(hyp, ref) / len(ref)`
（以标注长度为分母）。

### [7] 唤醒锚点：select_wake_anchor

源码：`src/midea_qiji/moss_candidate.py:109-171`；调用
`src/szy/pipeline.py:418-422`。

唤醒音频可能同时录进目标人与家人/噪声，分离成两路后用文本编辑距离找出
"念了唤醒词的那一路"作为目标人嗓音的注册锚点，后续所有余弦比较都对着它。

```text
select_wake_anchor(wake_text, s0_text, s1_text):
    ref = normalize_transcript(wake_text)
    has0, has1 = bool(s0), bool(s1)          # 通道转写只 strip，不二次归一化（复现优先）
    if has0 and not has1: return (selected=0, contrast=1, raw_wake_fallback=True,
                                  reason="source1_empty_raw_wake")      # 对称分支同理
    e0, e1 = lev(s0, ref), lev(s1, ref)
    selected = 0 if e0 <= e1 else 1          # 编辑距离小者胜；平手取 source0（复刻历史）
    if not has0 and not has1:                # 双空退化分支：保留历史行为保证重放确定性
        return (selected, 0, True, tie, e0, e1, "both_sources_empty_raw_wake")
    return (selected, 1-selected, False, e0==e1, e0, e1, "wake_text_edit_distance")
```

锚点音频：`raw_wake_fallback=True` 时用整段原始唤醒音频，否则用选中分离
通道。注：曾验证"非空通道直选"PR/CER 更优，但在计分公式 (1−CER+RR)/2 下
RR 权重高，raw 回退版公式分更高（0.7977 vs 0.7939），故保持 raw 回退
（见 `pipeline.py:416-417` 注释）。

### [8] 融合门控（全链路核心）

源码：`src/szy/pipeline.py:166-284`（FusionGate）、
`src/midea_qiji/gate.py:163-249`（ModelScopeEmbedder）。

**双前端**：同一 ERes2NetV2 基座的两个实例——官方预训练（泛化好）+ 分离域
微调（懂分离通道里的目标嗓音，只覆盖 encoder 权重：
`torch.load(ep6_datasetA_best.pt, map_location="cpu") →
load_state_dict(checkpoint["encoder"])`）。

**embedding 提取**（`extract_waveforms`，每前端对 5 段波形各出一个 L2
归一化向量）：`pipeline.preprocess` → 反射查找 `__extract_feature`（ERes2NetV2
与 CAM++ 通用）→ **按帧数分组**（同批 cat 要求时间维等长）→ 每组按
batch 64 切小批 → `embedding_model` 前向（`torch.inference_mode`）→
L2 归一化。归一化后余弦 = 点积（`pipeline.py:154-163` 的 `_cosine` 只做
`np.dot`）。

**五段波形**（固定顺序送入两个前端）：唤醒锚点、原始唤醒、原始指令、
分离通道 0、分离通道 1 → 得到 `e_anchor, e_raw_wake, e_raw_cmd, e_cmd0, e_cmd1`。

**6 维特征定义**（`FUSION_FEATURE_KEYS`，顺序即权重向量维度含义，加载时
与 `gate_fusion.json` 的 `features` 字段逐一校验，错位立即报错）：

| 特征 | 公式（tag ∈ {pre 预训练, ft 微调}） | 含义 |
|---|---|---|
| `{tag}_raw` | `cos(E(raw_wake), E(raw_cmd))` | 原始混合间相似度（分离前基线） |
| `{tag}_sep` | `max(cos(E(anchor),E(cmd0)), cos(E(anchor),E(cmd1))[, cos(E(anchor),E(raw_cmd)) 当分离通道转写为空时])` | 分离后目标人浓度 |
| `ch_sim_{tag}` | `cos(E(cmd0), E(cmd1))` | 两通道分离度（分得越开越不相似） |
| `{tag}_cos_cmd0/1` | `cos(E(anchor), E(cmd0/1))` | **过程量**：候选选择直接复用 |
| `{tag}_rawcmd` | `cos(E(anchor), E(raw_cmd))` | **过程量**：sv_diff 复用 |

**融合打分**（z-score 标准化 + 线性加权，`pipeline.py:280-284`）：

```text
x     = (features[6 维，按 FUSION_FEATURE_KEYS 顺序] − mean) / std    # z-score：统一量纲
score = x · weight + bias                                              # 线性判别值（非概率）
accept = score ≥ −0.6469
```

- 线性融合参数（mean/std/weight/bias）由标定流程在 **dasynth valid** 上
  拟合 logistic（零 DatasetA 调参）；门控消融显示该阈值已坐在 DatasetA
  oracle 上限上，调优空间为零（`docs/CURRENT_BEST_PIPELINE_SZY_V4.md` §5.3）。
- 一次 `score` 调用产出三样：接受判定、融合分、**全部逐通道余弦**——
  门控、候选选择、音色触发共享同一批 embedding，绝不为同一段音频重复提
  特征（声纹推理是全链路最贵一步）。
- 拒绝的样本直接返回：不跑指令 ASR、不做切换——全链路最大的计算量节省点。

### [9] 音色触发：should_switch_command_timbre

源码：`src/szy/pipeline.py:456-503`；规则本体
`src/midea_qiji/moss_candidate.py:215-235`。

**三个输入特征**：

- `confidence`：原始指令转写置信度（非法回退 1.0，见 [6]）；
- `command_keyword_score`：指令文本命中 `CMD_KW` 关键词的次数
  （`helpers.py:41-45`：打开/关闭/空调/灯光/窗帘/温度/音量/…，正则
  `r"灯光?"` 的 `?` 让"灯"字可有可无）；
- `sv_diff`：**分离收益量**（`pipeline.py:468-469`，复用门控已算余弦）：

```text
sv_diff = max(ft_cos_cmd0, ft_cos_cmd1) − ft_rawcmd
        = max(cos(anchor, ms0), cos(anchor, ms1)) − cos(anchor, raw_cmd)
```

直接测量"分离是否让目标嗓音比原始混合更突出"——切换本身的物理机制。
为负说明分离反而抹掉了目标人，绝不切换。

**决策树**（与历史文本规则同构的两段式，阈值全部冻结，
`tests/test_moss_candidate.py` 逐边界断言）：

```text
should_switch_command_timbre(conf, kw, sv_diff):
    if conf <= 0.70:                       # 低置信区：三个"或"，宽松放行
        return kw >= 0.5 or sv_diff > 0.05 or conf <= 0.57
    return sv_diff > 0.05 and kw < 0.5     # 高置信区：音色证据强且内容不像指令才切
```

阈值语义：`0.57` 必坏线（极低置信必切）；`0.05` 音色收益门槛（τ 盆地
[0.02, 0.08] 平坦）；`0.70` 置信切点（在 [0.50,0.70]×[0.65,0.80] 内平坦）——
"盆地平坦"说明规则没有过拟合到幸运数字。

另有两个辅助特征（`helpers.command_features`，与上述规则配套）：
字速 `chars_per_sec = len(normalize(text)) / max(duration, 0.1)`、关键词
命中数——真实指令语速正常且必含关键词，噪声转写又短又怪又快。

### [10] 候选选择：ft 直接余弦

源码：`src/szy/pipeline.py:505-540`。

```text
if 分离通道转写为空:                              # 分离失败，无可换
    selected = "orig";  reason = "command_source_empty"
else:
    selected = "ms0" if ft_cos_cmd0 >= ft_cos_cmd1 else "ms1"   # 平手取 ms0（历史口径）
    reason = "ft_direct_cosine"
content = {"orig": raw 文本, "ms0": cmd0 文本, "ms1": cmd1 文本}[selected]
```

- 与门控同一打分原理（微调前端对唤醒锚点的余弦），**复用门控时刻已算的
  逐通道余弦，零额外 embedding 计算**；实测与历史 eres+cam 差分打平，
  且避开了对比通道污染。

### [11] 两级异常兜底

源码：`src/szy/pipeline.py:541-564`。

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

### [12] 汇总与三件套

源码：`src/szy/reporting.py:29-146`、`src/szy/cli.py:122-149`。

- **指标口径**（按 label 分正负；文本一律先归一化再比）：

```text
CER = Σ lev(norm(hyp), norm(ref)) / Σ len(norm(ref))     # 正样本，越低越好
PR  = 正样本转写非空比例                                  # 空转写整条 0 分
RR  = 负样本输出为空比例                                  # 负样本输出任何文字整条 0 分
公式分 = (1 − CER + RR) / 2                              # 竞赛计分，本仓库 = 0.8679
```

- **submission.json**（交卷）：`{"result": {"results": [{id, content, label,
  cer}], "avg_cer", "avg_rr", "duration(ms)"}}`；
- **detail.json**（病历）：summary + 每样本全字段（6 维门控特征、路由、
  耗时、sv_diff、各通道文本）——事后可离线重扫阈值/重放门控，不必重推理；
- **manifest.json**（指纹）：完整配置快照 + `moss_model` /
  `fusion_backend` / `finetuned_sv_ckpt` 三份权重的 SHA-256（1MB 分块流式
  哈希），证明"这次结果由哪份配置+哪份权重跑出"。

### [13] 训练侧：dasynth 域适配微调（不在生产管线内，离线执行）

源码：`src/szy/train.py`（分离器）；声纹微调与融合标定在仓库外工作区
（sv_backend）。完整实验见 `docs/CURRENT_BEST_PIPELINE_SZY_V4.md`。

- **数据**：dataset_dasynth train 18k 条（AISHELL-1 + CN-Celeb 合成，
  DatasetA 风格腐蚀链：重叠/混响/噪声/削波），每条含腐蚀混合 + **腐蚀前
  对齐干净双源**——干净目标迫使模型联合学会去混响+去噪+分离。
- **训练**：PIT SI-SNR 损失（自动最优通道对应，不罚通道顺序）、2s 随机
  裁剪（与官方预训练一致）、bf16 autocast、batch 2 × 梯度累积 2、Adam
  lr 5e-5 / weight_decay 1e-5、梯度裁剪 10.0。
- **选模**：每 epoch 在 valid 前 200 条整波推理 oracle 排列 SI-SDR，
  按 ALL 均值存 best；生产采用**扫描轮 ep6**（公式分 0.8679）。
  训练/选模零 DatasetA 参与（DatasetA 只做冻结权重的一次性应用）。

### 关键数字速查

| 项 | 值 | 源码位置 |
|---|---|---|
| 融合门控阈值 | **−0.6469**（dasynth valid 标定） | `configs/pipeline_firered.json` |
| 6 维特征顺序 | pre_raw, ft_raw, pre_sep, ft_sep, ch_sim_pre, ch_sim_ft | `pipeline.py FUSION_FEATURE_KEYS` |
| 音色触发阈值 | conf≤0.70 分档 / 0.57 必坏线 / sv_diff>0.05 / kw≥0.5 | `moss_candidate.py:233-235` |
| 分离器 cfg | num_spks=2, kernel=16, dim=512, 24 层 | `separator.py:83-86` |
| 分离输出 | peak→0.5 归一化, PCM_16 落盘 | `separator.py:158-168` |
| ASR 解码 | beam=1, nbest=1, smoothing=1.25, len_penalty=0.6 | `recognizers.py:125-129` |
| 采样率 | 16000（全仓库唯一常量） | `midea_qiji/audio.py:35` |
| 置信度回退 | 非法/NaN/Inf → 1.0（保守不切换） | `helpers.py:120-135` |
| id 口径 | relative_stem | `infer.py:213-215` |
| 分离缓存命名 | `{id}_wake/command_ms{0,1}.wav` | `helpers.py:194-195` |

## 质量分

DatasetA 公式分 **(1−CER+RR)/2 = 0.8679**（PR 0.9157 / RR 0.9304 /
CER 0.1945，1838 样本，全量约 26~46 分钟）。完整实验口径、训练与消融
数字见 `docs/CURRENT_BEST_PIPELINE_SZY_V4.md`。

## 目录布局

```text
midea_qiji_best_hanlin/
├── configs/               # pipeline_firered.json（生产）+ pipeline_pretrained_firered.json（消融）
├── scripts/               # run.sh（现役入口）+ sweep/ablate 编排 + 资产自检/重放工具
├── src/szy/               # 最终系统包：cli / pipeline / separator / metrics / train / diag / sanity_check / helpers / reporting
├── src/midea_qiji/        # 基线内核：text / audio / paths / gate / moss_candidate / recognizers / infer
├── tests/                 # 16 个 numpy-only 单测（无需模型/GPU）
├── third_party/mossformer2/  # vendor 模型定义（ClearerVoice-Studio，未改动）
├── model/                 # 资产（不进 git）：artifacts 三件套 + 声纹基座 + FireRed + vendor
├── dataset/               # 官方数据（不进 git）：dataset_test/datasetA/datasetA/
└── outputs/               # 产物（不进 git）：分离缓存 / 提交物 / checkpoint
```

`model/`、`dataset/`、`outputs/` 由 `.gitignore` 排除；资产填充方式见
`model/README.md`。

## 运行与验证

```bash
conda activate yuyin

# 冒烟：前 5 条；去掉 --limit 跑全量
bash scripts/run.sh --limit 5

# 资产自检（依赖 model/MANIFEST.json，由 relocate_workspace_models.py 生成）
python scripts/check_models.py

# 单元测试（16 个用例，numpy-only）
PYTHONPATH=src python -m unittest discover -s tests
```

默认读取仓库内的 `dataset/` 与 `model/`；可用 `PYTHON_BIN`、
`CUDA_VISIBLE_DEVICES`、`MODELSCOPE_CACHE`、`HF_HUB_OFFLINE` 覆盖运行环境。
等价命令行入口见 `python -m szy.cli --help`（或 `docs/代码库结构解读.md`
第 3 节）。

## 文档索引

- `docs/代码库结构解读.md` — 重构后最终系统的逐文件解读与入口关系（仓库地图）
- `docs/一条数据的流动全景-01-样本加载与分离前端.md`
- `docs/一条数据的流动全景-02-识别与门控决策.md`
- `docs/一条数据的流动全景-03-汇总与提交产出.md`
- `docs/CURRENT_BEST_PIPELINE_SZY_V4.md` — 现役管线算法细节与 0.8679 的完整实验
- `docs/CURRENT_BEST_PIPELINE_SZY_V1.md` / `_V2.md` / `_V3_EXPERIMENTAL.md`、
  `docs/SZY_V1.md` — 阶段演进历史（v1→v2→v4 设计决策与实验记录）
- `docs/MOSS_CANDIDATE_REPRODUCTION.md`、`docs/MIDEA_BEST_STRUCTURE.md` — 历史复现与结构记录
