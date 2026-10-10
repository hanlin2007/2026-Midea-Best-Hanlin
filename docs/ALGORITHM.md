# 算法详解（ALGORITHM）

本文档逐项讲清生产管线每个环节的**算法、模型、阈值与数字**。结构与
`docs/ARCHITECTURE.md` 第 4 章的旅程一一对应；各项优化的实验依据见
`docs/OPTIMIZATION.md`。

评测口径：DatasetA 1838 样本，公式分 **(1−CER+RR)/2 = 0.8937**
（PR 0.9150 / RR 0.9304 / CER 0.1430）。

---

## 1. 分离前端：dasynth 微调 MossFormer2

源码：`src/pipeline/separator.py`；模型定义 vendor 于
`third_party/mossformer2/`（ClearerVoice-Studio，仅改包内相对导入）。

**任务**：把混合语音盲分离成两路 ms0/ms1（唤醒与指令各一次）。"盲"指
模型不知道谁在说话，哪路是目标人由下游余弦判定（PIT 训练，通道序随机）。

**模型结构**（`MossFormer2_SS`，约 55.7M 参数）：

```text
波形 [B, T]
  → Encoder   Conv1d(1→512, k=16, s=8) + ReLU      （可学习 STFT，50% 重叠）
  → MaskNet   归一化 → 1×1 conv → 正弦位置编码
              → 24 × GFSMN 块（Flash 分组注意力 + 门控 FSMN 记忆块交替）
              → 门控输出 Tanh(值) × Sigmoid(门) → 掩码 [2, B, 512, T/8]
  → 掩蔽      sep_x = 特征 ⊙ 掩码                  （逐说话人）
  → Decoder   ConvTranspose1d(512→1, k=16, s=8)    （可学习 iSTFT，重叠相加）
              → 两路波形（= 下游的 ms0/ms1）
```

配置：`num_spks=2, encoder_kernel_size=16, encoder_embedding_dim=512,
mossformer_sequence_dim=512, num_mossformer_layer=24`。

**分离调用**（缓存即文件名）：

```text
separate(source, output_prefix):
    outputs = [prefix_ms0.wav, prefix_ms1.wav]
    if 两个文件都存在: return outputs          # 缓存命中（"两个都在"才算）
    wav = read_mono_16k(source)               # 严格 16k 单声道；不合法直接报错
    out = model(wav[None, :])                 # [1, 2, T]，整波一次前向
    for i in (0, 1):
        s = out[i] / max|out[i]| × 0.5        # peak→0.5 归一化（防爆音）
        sf.write(outputs[i], s, 16000, PCM_16)
```

**为什么必须微调**：官方预训练分离器在 DatasetA 风格的混响/掩盖样本上
崩塌（dasynth valid 实测 reverb 子集 SI-SDR 比混合还差 6~14dB）。
dasynth 提供 18k 条"腐蚀混合 + 腐蚀前对齐干净双源"，做联合
去混响+去噪+分离的域适配微调（训练侧见 §8）。生产权重：
`model/artifacts/mossformer2_dasynth_sweep_ep6.pt`。

---

## 2. 声学单双人路由：BothSingleRouter

源码：`src/pipeline/single_speaker.py`；bundle：`configs/single_speaker_router.json`。

**任务**：不花 ASR，纯波形判断"唤醒/指令各是不是单人录音"，驱动两件事：
双单人短路（§3）与单/双人指令路由（§6）。

**20 维特征**（全部**通道交换不变**——按强/弱通道排序后计算，与
ms0/ms1 的随机通道序无关）：

| 组 | 特征 | 直觉 |
|---|---|---|
| 能量 | raw/strong/weak log_rms、能量对数比 | 单人录音弱通道近乎静音 |
| 活跃度 | 20ms 帧 RMS 的活跃占比（共享阈值）、双活跃/单活跃比例、活跃差 | 单人时弱通道几乎不活跃 |
| 弱通道占比 | weak_share 的 mean/p50/p90 | 弱通道能量占两路之和的比例 |
| 主导性 | dominance_flip_ratio、envelope_correlation | 双人时主导通道随时间切换 |
| 重构 | raw−ms0−ms1 残余、最小二乘拟合残余、波形余弦 ×3 | 分离保真度：raw ≈ ms0+ms1 |

**分类器**：z-score → 线性打分 → sigmoid 的 logistic（20 维 mean/std/
weight/bias + 阈值 0.8436 全在 bundle 里；训练与定阈零 DatasetA）。
加载时逐项校验特征名顺序（与门控同一套"防错位"契约）。

---

## 3. 双单人短路：一次余弦判同人

源码：`engine.PipelineRunner.process`（双单人分支）+
`FusionGate.finetuned_raw_cosine`。

唤醒和指令都判为单人时，跳过拼音锚点与 6 维门控：

```text
score = cos(E_ft(raw_wake), E_ft(raw_cmd))     # 微调 ERes2NetV2，一次推理两段
accepted = score ≥ 0.29                        # both_single_sft_threshold
```

接受（`both_single_raw`）：只对原始指令做一次主解码 → 后处理。
拒绝（`both_single_reject`）：置空返回，不跑指令 ASR。

为什么独立成支：两段都是单人时，分离通道没有额外信息，6 维门控里
"分离收益"类特征失去意义；一次 raw/raw 余弦就是最直接、最便宜的证据。

---

## 4. 拼音锚点：select_wake_anchor

源码：`src/core/wake_anchor.py`（规则）+ `src/core/phoneme.py`（相似度）。

**任务**：唤醒音频可能同时录进目标人与家人/噪声；分离成两路后，找出
"念了唤醒词的那一路"作为目标人嗓音的注册锚点，后续所有余弦比较都
对着它。

**拼音相似度**：

```text
phoneme_similarity(a, b):
    py_a, py_b = 拼音串(a), 拼音串(b)      # pypinyin TONE3 去声调，紧凑小写
    if b 命中别名表: b 展开为所有变体       # hicolmo ↔ haikemu
    return max(-levenshtein(py_a, alias) for alias in variants(b))
```

同音字（ASR 最高频错误，"科慕" vs "科目"）在拼音空间距离为 0——字符
编辑距离会冤枉好通道，拼音不会。这是全系统在 DatasetA 上最大的单项
护城河（51 条锚点选择差异带来 81 字优势）。

**裁决四分支**：

| 情形 | 锚点 | reason |
|---|---|---|
| 只有一路有转写 | 原始唤醒音频（回退） | `source{0,1}_empty_raw_wake` |
| 两路分出胜负 | 相似度高的分离通道 | `wake_text_phoneme_similarity` |
| 两路打平（一样像） | 原始唤醒音频（回退） | 同上，`text_tie=True` |
| 两路都空（退化） | 原始唤醒音频，对比项钉 ms0 | `both_sources_empty_raw_wake` |

为什么保留回退：实验验证过"非空通道直选"PR/CER 更好，但在计分公式
(1−CER+RR)/2 下 RR 权重高，raw 回退版总分更高（0.7977 vs 0.7939）。
**指标口径决定设计取舍。**

---

## 5. 融合门控：FusionGate

源码：`src/pipeline/engine.py`（FusionGate）+
`src/core/embeddings.py`（ModelScopeEmbedder）。

**任务**：链路的安全闸——指令音频里有没有目标说话人？拒绝的样本直接
置空返回，不跑指令 ASR（全链路最大的计算量节省点）。

**双前端共享 Fbank**：同一 ERes2NetV2 基座的两个实例——官方预训练
（泛化好）+ 分离域微调（懂分离通道里的目标嗓音，只覆盖 encoder 权重：
`model/artifacts/ep6_datasetA_best.pt`）。两个前端声学前端完全相同，
Fbank 只算一次（`prepare_waveforms`），各自前向 `embedding_model`
——计算量省一半，不增加并发激活显存。

**6 维特征**（顺序是硬契约，加载时与 `gate_fusion.json` 逐一校验）：

| 特征 | 公式（tag ∈ {pre, ft}） | 含义 |
|---|---|---|
| `{tag}_raw` | cos(E(raw_wake), E(raw_cmd)) | 原始混合间相似度（分离前基线） |
| `{tag}_sep` | max(cos(E(anchor),E(ms0)), cos(E(anchor),E(ms1))[, cos(E(anchor),E(raw_cmd)) 指令单人时]) | 分离后目标人浓度 |
| `ch_sim_{tag}` | cos(E(ms0), E(ms1)) | 两通道分离度 |
| 过程量 | `{tag}_cos_cmd0/1`, `{tag}_rawcmd` | 选路与分析复用，零额外计算 |

**融合打分**：

```text
x      = (features[6 维，按 FUSION_FEATURE_KEYS 顺序] − mean) / std   # z-score
score  = x · weight + bias                                             # 线性判别值
accept = score ≥ −0.6469                  # dasynth valid 标定，零 DatasetA 调参
```

线性融合参数在 dasynth valid 上拟合 logistic 得到（`model/artifacts/
gate_fusion.json`）。门控消融显示该阈值已坐在 DatasetA oracle 上限上。

---

## 6. 延迟解码与 ASR 共享实例

源码：`src/core/firered.py`（后端）+ `src/pipeline/engine.py`
（_primary_transcribe / _transcribe_wake / AsrCache）。

**单实例共享**：唤醒与命令通道共用一份 FireRedASR2-AED 权重（峰值显存
4818 MiB，比双实例时代省约 2.5 GB，还消除了两份模型同时驻留导致的
数值/内核选择抖动）。两个上下文管理器让临时配置成为可能：

- `temporary_beam_size(n)`：with 块内改束宽（nbest 自动 clamp），退出恢复；
- `temporary_elm_weight(w)`：ELM 权重常驻，解码时按 config.elm_weight 开关。

**三条解码通道**（全部经过 AsrCache，键 = `路径|配置`）：

| 通道 | 配置 | 用途 |
|---|---|---|
| `_transcribe_wake` | beam=1, ELM off（`b1\|elm0`） | 唤醒锚点：求稳不求探索 |
| `_primary_transcribe` | beam=3, ELM off（`b3\|elm0`） | 所有承载决策的指令转写 |
| `_elm_redecode` | beam=3, ELM 0.15（`b3\|elm0.15`） | 仅选定通道的最终重解码 |

**为什么决策转写一律 ELM 关闭**：ELM 会把 confidence 从 ~0.88 压到
~0.60 量程；路由与后处理依赖干净的置信度，ELM 只允许在最后对选定
通道重解码一次（Phase 2 核心不变量，见 OPTIMIZATION.md）。

**单/双人路由**：

```text
指令单人（raw_command_single）：只解码原始指令一次
指令双人（moss_candidate）：
    preferred = 0 if ft_cos_cmd0 ≥ ft_cos_cmd1 else 1   # 复用门控余弦，零成本
    解码 command_paths[preferred]；转写为空才补解另一路
    原始指令始终不解码（双人混合的 raw 转写没有参考价值）
```

---

## 7. 后处理：MBR N-best + ELM

源码：`src/core/nbest.py` + `engine._postprocess_text`。

**保守 MBR 重排**：beam=3 主解码顺手产出的 N-best 候选已随 AsrCache
存下（零额外推理）。MBR 风险 = 与其他候选的置信度加权编辑距离之和
——选风险最小者（候选集合的"中位数"）。**双闸**：置信度增益 ≥ 0.03
且风险下降 ≥ 0.05 同时成立才允许换掉 top-1。宁可不换，不可换错。

**ELM 重解码**：外部语言模型熟悉智能家居领域词汇（纠正"调到/掉到"
类同音错误，全系统最大单项提分 +0.0126）。`temporary_elm_weight(0.15)`
下对选定通道重解码一次；**空文本不重解**——语言模型不许在声学模型
"什么都没听到"的地方凭空造文本。

---

## 8. 训练侧：dasynth 域适配微调（离线，不在生产管线内）

源码：`src/pipeline/train.py`（分离器）；声纹微调与融合标定在仓库外
工作区完成。

- **数据**：dataset_dasynth train 18k 条（AISHELL-1 + CN-Celeb 合成，
  DatasetA 风格腐蚀链：重叠/混响/噪声/削波），每条含腐蚀混合 +
  腐蚀前对齐干净双源——干净目标迫使模型联合学会去混响+去噪+分离。
- **训练**：PIT SI-SNR 损失（自动最优通道对应）、2s 随机裁剪（与官方
  预训练一致）、bf16 autocast、batch 2 × 梯度累积 2、Adam
  lr 5e-5 / weight_decay 1e-5、梯度裁剪 10.0。
- **选模**：每 epoch 在 valid 前 200 条整波推理 oracle 排列 SI-SDR，
  按 ALL 均值存 best。**训练/选模零 DatasetA 参与**（DatasetA 只做
  冻结权重的一次性应用）——评测集挑模型会让分数虚高，上线必现落差。

---

## 9. 关键数字速查

| 项 | 值 | 位置 |
|---|---|---|
| 融合门控阈值 | **−0.6469**（dasynth valid 标定） | `configs/pipeline.json` |
| 6 维特征顺序 | pre_raw, ft_raw, pre_sep, ft_sep, ch_sim_pre, ch_sim_ft | `engine.FUSION_FEATURE_KEYS` |
| 双单人同人余弦阈值 | **0.29** | `both_single_sft_threshold` |
| 单人概率阈值 | 0.8436（bundle 内置） | `configs/single_speaker_router.json` |
| 主解码 beam / N-best | 3 / conservative_mbr（双闸 0.03 + 0.05） | `firered_beam_size` / `firered_nbest_*` |
| ELM | weight 0.15，beam 3，仅选择后重解码 | `firered_elm_*` |
| 分离器 cfg | num_spks=2, kernel=16, dim=512, 24 层 | `separator.py` |
| 分离输出 | peak→0.5 归一化, PCM_16 落盘 | `separator.py` |
| 采样率 | 16000（全仓库唯一常量） | `core/audio.py` |
| 置信度回退 | 非法/NaN/Inf → 1.0（保守不切换） | `engine._transcribe_with_asr` |
| id 口径 | relative_stem | `core/samples.py` |
| ASR 缓存键 | `<路径>\|b<beam>\|elm<weight>` | `engine.AsrCache` |
| checkpoint/缓存落盘粒度 | 每 100 条 | `batch_writes` |

## 10. 指标口径

```text
CER = Σ lev(norm(hyp), norm(ref)) / Σ len(norm(ref))   # 正样本；分子分母各自求和
PR  = 正样本转写非空比例                                # 空转写整条 0 分
RR  = 负样本输出为空比例                                # 负样本输出任何文字整条 0 分
公式分 = (1 − CER + RR) / 2                            # 竞赛计分，本仓库 = 0.8937
```

文本一律先过 `normalize_transcript` 再比对（与官方评测同口径）；该函数
是冻结语义，改动即破坏全部数字的复现（`tests/test_text.py` 锁死）。

## 11. 缓存与性能

| 缓存 | 位置 | 键 | 命中效果 |
|---|---|---|---|
| 分离缓存 | `outputs/<NAME>/cache/separation/*.wav` | 输出文件名 | 跳过分离推理 |
| ASR 缓存 | `outputs/<NAME>/cache/asr_cache.json` | `路径\|配置` | 跳过 FireRed 解码 |

典型迭代（仅改决策代码）：首次全量 ~2.3 s/utt，同名重跑 ~0.2–0.7 s/utt
（约 10× 提速）。生产配置 `asr_cache_persistent: false` 只保留进程内
缓存（评测结果只依赖当次运行）；本地迭代改 `true` 跨运行复用。
强制全新运行：删掉对应 `outputs/<NAME>/` 目录。
