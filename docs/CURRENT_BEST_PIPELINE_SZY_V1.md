# 当前最佳目标说话人识别 Pipeline：SZY v1

## 1. 文档定位

本文档描述当前已通过完整 DatasetA 验收、仍被批准使用的 SZY v1 路线。它不是 ULUNAS-TSE 的实验方案，也不是 SZY v2 中未通过验收的 TSE、presence 或候选替换分支。

当前批准版本的核心原则是：

1. 唤醒音频用于建立目标说话人的声纹参考。
2. 唤醒文本只用于在唤醒分离源中确认“哪一路说了唤醒词”。
3. 识别音频先经过主声纹门控，拒绝样本不运行 ASR。
4. 原始识别音频和原始 FireRed 文本永远保留。
5. MossFormer2 只生成候选，不被当作目标说话人真值。
6. 任一分离、ASR、声纹或候选选择步骤失败，都回退到原始命令；原始命令也失败时才输出空文本。

当前版本的完整实测结果为：CER `0.3174937`、PR `0.8790323`、RR `0.8396624`。对应的逐条结果和运行清单分别位于 `outputs/szy_v1_detail.json`、`outputs/szy_v1_submission.json` 和 `outputs/szy_v1_manifest.json`。

## 2. 任务定义

每条样本包含：

| 输入 | 含义 | 是否直接做最终 ASR |
|---|---|---|
| `唤醒音频` / `kws.wav` | 唤醒阶段的多人重叠语音，目标人是说出唤醒词的人 | 否，主要用于声纹参考和候选锚定 |
| `唤醒文本` | 已知的唤醒短语 | 否，作为唤醒源选择条件 |
| `识别音频` / `cmd.wav` | 需要识别的命令音频，可能含重叠说话和噪声 | 是，默认使用原始命令音频 |
| `识别文本` | 正样本标签；负样本为空或缺失 | 只用于评估 CER，不作为推理输入 |

目标是识别 `识别音频` 中与唤醒音频里唤醒词说话人相同的人所说的内容，并拒绝目标说话人不存在的负样本。

## 3. 总体数据流

```text
pos.jsonl / neg.jsonl + 原始 WAV
              |
              v
       稳定样本 ID 和音频读取
              |
              v
  ERes2NetV2 + CAM++ 双前端 MLP 主门控
       |                         |
       | score < 0.446031        | score >= 0.446031
       v                         v
    Reject                 FireRed 原始命令 ASR
    输出空文本                    |
                                 v
                置信度 / 命令特征二维切换规则
                         |                 |
                         | 不触发          | 触发
                         v                 v
                    保留 raw 文本       MossFormer2 候选路线
                                             |
                         +-------------------+-------------------+
                         |                                       |
                         v                                       v
                  分离唤醒音频                         分离识别音频
                  得到 wake0/wake1                    得到 cmd0/cmd1
                         |                                       |
                         v                                       |
               FireRed 转写两路唤醒源                       |
                         |                                       |
                         v                                       |
               与唤醒文本编辑距离比较                       |
                         |                                       |
                         v                                       |
               选 target anchor + contrast                 |
                         |                                       |
                         +-------------------+-------------------+
                                             |
                                             v
                         ERes2NetV2/CAM++ 同次分离差分声纹选择
                                             |
                         +-------------------+-------------------+
                         |                                       |
                    cmd0/cmd1 任一路空                    两路均非空
                         |                                       |
                         v                                       v
                   回退 raw 文本                         选择 ms0 或 ms1
                                                               |
                                                               v
                                                        输出候选文本

任何异常 -> raw command FireRed -> raw 也失败时输出空文本
```

这里的“主门控”和“候选选择”是两个不同决策：主门控负责目标是否存在，候选选择只在原始 FireRed 结果被判断为不可靠时尝试替换文本。

## 4. 阶段一：样本读取和稳定 ID

实现入口：`src/szy_v1/pipeline.py`。

运行时直接读取 DatasetA 的 `pos.jsonl`、`neg.jsonl` 和对应 WAV，不依赖历史 `exp_sep` 转写结果。样本 ID 默认使用 `relative_stem`，从相对音频路径派生，避免 JSONL 中重复的原始 `id` 导致结果覆盖。

音频要求：

- 采样率：16 kHz。
- 声道：单声道。
- 唤醒音频完整保留，不做硬裁剪、拼接或通用降噪。
- 识别音频完整保留，raw command 是始终存在的最终候选。

每条样本处理完成后写入 checkpoint，因此长任务可中断并用 `--resume` 继续。checkpoint 保存已完成结果和累计耗时，不会重新处理已经完成的样本。

## 5. 阶段二：双前端 MLP 主门控

实现：`src/midea_qiji/gate.py`。

### 5.1 声纹前端

使用两个独立的 speaker verification 前端：

- ERes2NetV2：`iic/speech_eres2netv2_sv_zh-cn_16k-common`。
- CAM++：`iic/speech_campplus_sv_zh-cn_16k-common`。

每个前端都比较完整唤醒音频和完整识别音频，同时对识别音频生成 1.0 秒窗口、0.2 秒步长的局部片段。

### 5.2 每个前端的 16 维特征

特征包括：

1. 整段余弦相似度。
2. 局部窗口数、最大值、均值、标准差、top-3 均值。
3. 相似度超过 `0.25`、`0.30`、`0.35` 的窗口数。
4. 相似度超过 `0.30` 的窗口比例和连续覆盖跨度。
5. 唤醒时长、识别时长。
6. 识别音频与唤醒音频的 RMS 比值。
7. 两段音频的削波比例。

两个前端拼接后得到 32 维向量。训练得到的均值、标准差和 MLP 权重保存在：

```text
/home/hustdun/szy_workspace/sv_backend/backend_dual_v2_out/backend_dual.pt
```

MLP 结构为 `32 -> 32 -> 16 -> 1`，中间使用 ReLU 和 Dropout。推理时加载标准化参数和权重，输出目标存在概率。

### 5.3 主门控决策

当前冻结阈值：

```text
dual_threshold = 0.4460311856655159
```

决策规则：

```text
score >= 0.4460311856655159  -> 接受，继续 FireRed
score <  0.4460311856655159  -> Reject，最终 content 为空
```

主门控拒绝后不运行 FireRed，不运行 MossFormer2，也不生成候选文本。这样可以控制负样本误接，同时避免把后续 ASR 的随机非空输出误计为接受。

## 6. 阶段三：raw command FireRed 基线

实现：`src/midea_qiji/infer.py:FireRedRecognizer`。

配置：

```json
{
  "firered_model": "/home/hustdun/models/FireRedASR2-AED",
  "firered_repo": "/home/hustdun/zhy_workspace/FireRedASR2S",
  "asr_backend": "firered"
}
```

主门控接受后，先把原始 `cmd.wav` 送入 FireRed。原始转写结果同时承担两个作用：

1. 当结果看起来可靠时，直接作为最终输出。
2. 当结果触发候选切换规则时，作为 raw fallback 和候选比较基准。

FireRed 输出会经过统一的 `normalize_transcript`：去除特殊标记、统一数字表达、去除空格和常见标点。CER、PR、RR 都基于归一化文本计算。

## 7. 阶段四：raw 结果的候选切换规则

实现：`src/midea_qiji/moss_candidate.py:should_switch_command`。

系统从 FireRed 结果中读取：

- FireRed confidence。
- 归一化文本长度除以音频时长得到的 `chars_per_sec`。
- 命令关键词数量，例如“打开、关闭、空调、灯光、音量、播放、暂停、温度”等。

冻结规则为：

```text
如果 confidence <= 0.70：
    command_keyword_score >= 0.5
    或 chars_per_sec >= 3.0
    或 confidence <= 0.57
    -> 进入候选路线

如果 confidence > 0.70：
    confidence <= 1.01 且 chars_per_sec >= 3.0
    且 command_keyword_score < 0.5
    -> 进入候选路线

否则 -> 直接保留 raw 文本
```

这一步不是拒识器，也不会因为触发候选路线就直接清空文本。它只决定是否值得额外运行 MossFormer2 并做一次候选复核。

## 8. 阶段五：MossFormer2 生成候选

实现：`SzyV1Runner.MossFormer2Separator`。

模型：

```text
/home/hustdun/szy_workspace/midea/models/modelscope/
dengcunqin/speech_mossformer2_separation_temporal_16k/simple_model.onnx
```

分离器通过 ONNX Runtime 调用。`moss_provider=auto` 时优先使用可用的 CUDAExecutionProvider，否则使用 CPU。每次分离产生两路候选：

```text
<prefix>_ms0.wav
<prefix>_ms1.wav
```

唤醒音频和识别音频分别分离，并写入 `outputs/szy_v1_cache/separation/`。分离结果只是候选声源，不能直接认为是目标人，也不能仅凭音量或单一声纹分数决定目标源。

## 9. 阶段六：用唤醒文本选择目标 anchor

实现：`src/midea_qiji/moss_candidate.py:select_wake_anchor`。

先对唤醒分离的两路 `wake0`、`wake1` 运行 FireRed，再把两路文本分别与已知 `唤醒文本` 计算字符编辑距离。

规则如下：

| 条件 | anchor | contrast |
|---|---|---|
| 只有 wake0 非空 | 原始完整唤醒音频 | wake1 |
| 只有 wake1 非空 | 原始完整唤醒音频 | wake0 |
| 两路均非空 | 编辑距离更小的一路 | 另一条分离源 |
| 两路均为空 | 原始完整唤醒音频 | 固定 source0 作为对照 |
| 编辑距离相等 | source0 | source1 |

当某一路为空时，使用完整 raw wake 作为 anchor，而不是把空分离源当作注册音频。这是重要的 raw fallback：短唤醒、强重叠或分离失败时，仍然保留目标人的完整原始声纹信息。

唤醒文本只在这里作为“谁说了唤醒词”的条件，不直接作为识别音频的转写结果，也不单独构成拒识条件。

## 10. 阶段七：同次分离差分声纹选择命令源

当 `cmd0` 和 `cmd1` 的 FireRed 转写均非空时，系统对以下四段音频提取两个声纹前端的 embedding：

```text
anchor waveform
contrast waveform
cmd0 waveform
cmd1 waveform
```

对每一个命令候选、每一个声纹前端计算：

```text
difference(candidate)
    = cosine(anchor, candidate)
    - cosine(contrast, candidate)
```

然后把 ERes2NetV2 和 CAM++ 的 difference 相加：

```text
score_ms0 = eres_diff_ms0 + cam_diff_ms0
score_ms1 = eres_diff_ms1 + cam_diff_ms1
```

选择规则：

```text
score_ms0 >= score_ms1 -> 选择 ms0
score_ms0 <  score_ms1 -> 选择 ms1
```

相等时固定选择 `ms0`，保证可复现。

如果 `cmd0` 或 `cmd1` 任一路 FireRed 为空，说明分离候选不完整，系统不选择剩余的一路，而是直接回退 raw command。这避免把“另一路为空”误判为“非空一路一定是目标人”。

## 11. 异常和回退矩阵

| 故障位置 | 最终行为 | 说明 |
|---|---|---|
| 主门控低于阈值 | 空文本 | Reject，不运行 ASR |
| raw FireRed 正常且未触发切换 | raw 文本 | 最低风险主路径 |
| raw FireRed 触发切换，Moss 失败 | raw 文本 | 分离异常不改变原始结果 |
| 唤醒一路为空 | raw wake 作 anchor | 保留完整唤醒声纹 |
| 唤醒两路均为空 | raw wake 作 anchor | 对照源仅用于保持流程确定性 |
| 命令一路为空 | raw 文本 | 不从单一路候选推断目标人 |
| 命令两路均非空 | 选择 ms0/ms1 文本 | 使用同次分离差分声纹 |
| 候选 ASR 或声纹异常 | raw 文本 | 记录 `error` 和 `raw_fallback_error` |
| raw FireRed 也异常 | 空文本 | 记录 `empty_fallback_error` |

这套回退设计的核心是：分离候选只能改善结果，不能夺走 raw fallback 的控制权。

## 12. 输出文件和审计字段

默认输出：

```text
outputs/szy_v1_submission.json
outputs/szy_v1_detail.json
outputs/szy_v1_manifest.json
outputs/szy_v1_checkpoint.json
outputs/szy_v1_cache/separation/
```

### 12.1 submission

提交文件包含每条样本的：

- 稳定 `id`。
- 最终 `content`。
- 正样本的 item CER；负样本 CER 为空。

### 12.2 detail

逐样本 detail 额外记录：

- `route`：`primary_reject`、`raw`、`moss_candidate`、`raw_fallback_error` 等。
- `accepted`、`sv_score`、`sv_accept_threshold`。
- `raw_confidence`、`chars_per_sec`、`command_keyword_score`。
- `switch_triggered`、`selected_candidate`、`selection_reason`。
- 唤醒两路文本和编辑距离选择结果。
- 命令两路文本。
- ERes2NetV2/CAM++ 的 `eres_diff_ms0/ms1`、`cam_diff_ms0/ms1`。
- `command_source_empty`、`error`、`item_duration`。

### 12.3 manifest

manifest 固定记录：

- 配置快照。
- 输入 JSONL 和数据根目录。
- 样本数。
- MossFormer2 模型 SHA256。
- dual MLP 权重 SHA256。
- submission、detail、checkpoint 路径。

它用于确认一次结果到底使用了哪一份模型和配置。

## 13. 指标口径和当前结果

DatasetA 全量规模：正样本 1364 条，负样本 474 条。

指标定义：

```text
CER = 正样本总 Levenshtein 编辑距离 / 正样本归一化参考字数
PR  = 正样本最终非空输出数 / 正样本总数
RR  = 负样本最终空输出数 / 负样本总数
J   = (1 - CER) + RR
```

SZY v1 全量实测：

| 指标 | 原始计数 | 结果 |
|---|---:|---:|
| CER | `3020 / 9512` | `0.3174937` |
| PR | `1199 / 1364` | `0.8790323` |
| RR | `398 / 474` | `0.8396624` |
| J | `(1 - CER) + RR` | `1.5221688` |
| PR + RR | `0.8790323 + 0.8396624` | `1.7186947` |
| 总耗时 | `1106.64 s` | 约 `18.44 min` |

CER 只在正样本上计算，负样本不参与 CER 分母；负样本只通过 RR 评价。正样本被主门控拒绝时，最终文本为空，因此会自然表现为较高的 CER 和较低的 PR，这两个指标必须一起看。

## 14. 为什么当前仍采用这条路线

当前已测试但未批准的分支包括：

1. ULUNAS-TSE 波形替换：external 波形 SI-SNRi 可以提升，但 FireRed 正样本 CER 反而上升，且 TSE 输出出现大量空转写。
2. ULUNAS presence 二次门控：external 和 DatasetA 上未能同时提高 PR 与 RR。
3. 更激进的 TSE 替换：会出现伪影、音素丢失和正样本回归。

因此当前最稳妥的原则不是“分离越多越好”，而是：先用主门控控制目标存在，再让 raw FireRed 作为默认识别路径，只有原始结果不可靠时才调用 MossFormer2 候选，并在候选失败时回退 raw。

## 15. 运行方式

### 15.1 全量运行

```bash
cd /home/hustdun/szy_workspace/midea-qiji
CUDA_VISIBLE_DEVICES=3 ./scripts/run_szy_v1.sh
```

脚本默认使用：

- Python：`/home/hustdun/.conda/envs/yuyin/bin/python`
- `PYTHONPATH=src:sv_backend:vendor`
- ModelScope 本地缓存：`midea/models/modelscope`
- `HF_HUB_OFFLINE=1`
- 配置：`configs/szy_v1_firered.json`

### 15.2 中断续跑

```bash
cd /home/hustdun/szy_workspace/midea-qiji
CUDA_VISIBLE_DEVICES=3 ./scripts/run_szy_v1.sh --resume
```

续跑前应确认 `outputs/szy_v1_checkpoint.json` 存在，且输入 JSONL、配置和模型路径没有改变。若模型或配置改变，应使用新的输出目录和新的 checkpoint，不能混用旧结果。

### 15.3 小规模冒烟

```bash
cd /home/hustdun/szy_workspace/midea-qiji
CUDA_VISIBLE_DEVICES=3 ./scripts/run_szy_v1.sh \
  --limit 20 \
  --output outputs/szy_v1_smoke_submission.json \
  --final-output outputs/szy_v1_smoke_detail.json \
  --checkpoint-output outputs/szy_v1_smoke_checkpoint.json \
  --manifest-output outputs/szy_v1_smoke_manifest.json
```

冒烟结果只能验证依赖、模型加载、字段格式和回退逻辑，不能用于替代全量 CER/PR/RR 验收。

## 16. 不允许的改动

以下改动必须产生新的实验版本，不能直接覆盖当前 SZY v1：

- 修改 dual MLP 阈值。
- 把局部窗口或短唤醒裁剪结果替换完整唤醒参考。
- 去掉 raw command fallback。
- 将任一路 MossFormer2 输出直接作为最终命令。
- 让 `tse_pred_logit`、单一 embedding 分数或音量变化直接决定拒识。
- 在 DatasetA 上重新调阈值、挑选 replacement 样本或选择 checkpoint。
- 把 SI-SNRi、RMS 下降、波形相关性或训练 loss 单独当作端到端成功。

任何新分支都必须在独立 external train/dev 上冻结模型、阈值和选择规则，再用 DatasetA 做一次不可回馈的最终验证。只要 CER、PR、RR 或整体 `J` 未按该分支所属目标改善，就回退本路线。

## 17. 主要风险和后续研究边界

当前主要风险：

1. 高相似干扰人可能通过主门控，导致负样本误接。
2. 短唤醒、远场和两人重叠可能让全局声纹参考不稳定。
3. MossFormer2 输出有伪影，候选文本可能比 raw FireRed 更差。
4. FireRed 置信度不是经过严格校准的目标人置信度，只能作为候选切换特征。

后续优化必须先针对单一问题建立 external 实验：

- 只做拒识的模块，要求 PR/RR 按选择模块口径改善。
- 改变后续转写的模块，要求正样本 micro-CER 降低。
- 同时改变路由和转写的整体模块，要求 `J` 提高且不破坏 raw fallback。

在这些门槛通过前，SZY v1 是当前唯一应保留的主链路。

## 18. 代码和证据索引

| 内容 | 路径 |
|---|---|
| 主流程 | `midea-qiji/src/szy_v1/pipeline.py` |
| 双前端 MLP 门控 | `midea-qiji/src/midea_qiji/gate.py` |
| 唤醒文本选 anchor / 命令候选规则 | `midea-qiji/src/midea_qiji/moss_candidate.py` |
| FireRed 后端和文本归一化 | `midea-qiji/src/midea_qiji/infer.py` |
| 固定配置 | `midea-qiji/configs/szy_v1_firered.json` |
| 运行脚本 | `midea-qiji/scripts/run_szy_v1.sh` |
| 逐条结果 | `midea-qiji/outputs/szy_v1_detail.json` |
| 提交结果 | `midea-qiji/outputs/szy_v1_submission.json` |
| 运行清单 | `midea-qiji/outputs/szy_v1_manifest.json` |
| SZY v1 指标报告 | `midea-qiji/docs/SZY_V1.md` |
| SZY v2 失败分支和回退报告 | `midea-qiji/docs/SZY_V2_EXPERIMENT_REPORT.md` |
| ULUNAS 失败验收 | `midea-qiji/docs/历史档案/任务归档/2026-08-07-ulunas-tse-boundary-v2-e1-g3.md` |
