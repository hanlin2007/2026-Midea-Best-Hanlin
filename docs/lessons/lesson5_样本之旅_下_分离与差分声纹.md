# 第 5 课：一条样本的旅程（下）—— Moss 分离、差分声纹、回退与指标

> 上接第 4 课：`pos/cmd_13` 已触发候选路线（switch=True）。这一课是旅程的最后一段，也是整条
> 链路技术含量最高的一段：**MossFormer2 盲分离 → 唤醒锚点选择 → 差分声纹选命令源 → 回退矩阵 →
> CER/PR/RR 汇总**。涉及文件：`pipeline.py`、`moss_candidate.py`、`gate.py`、`infer.py`。
>
> 先剧透结局：这条样本最终输出"我们的石油"（标签是"打开洗衣服"，全错，CER=1.0）。
> 它是一个**典型的难例**——分离出了幻觉文本，差分声纹"正确地"选了一个"错误的"候选。
> 看懂它为什么错，比看懂它为什么对更有价值。

---

## 1. 第四步（续）：MossFormer2 盲分离

触发切换后，`process()` 调用：

```python
(wake_paths, command_paths) = self._paths_for(sample)
```

`_paths_for` 里对唤醒音频和命令音频**各自**做一次分离，得到 4 个 wav：

```python
wake0, wake1 = 分离(kws_13.wav)   # 唤醒音频 → 两路
cmd0,  cmd1  = 分离(cmd_13.wav)   # 命令音频 → 两路
```

分离核心在 `MossFormer2Separator.separate`（第 2 课读过），算法要点：

```python
separated = self.session.run(None, {self.input_name: waveform[np.newaxis, :]})[0]
# 输出形状 (1, T, 2)：一路输入 → 时间轴 T → 两个分离源
```

### 1.1 什么是"盲分离"（blind source separation）？

想象一段录音里**两个人同时说话**，波形是两人声音的**叠加**。盲分离模型（MossFormer2 是一类）的
任务：**只根据混合信号，把两个声音拆开**，输出两路（源 0、源 1）。

关键限定词是"**盲**"：模型**不知道**哪一路是谁。它只保证"两路内容被分开了"，不保证"源 0 一定是
目标人"。所以分离结果必须被后面几步**重新判定**，不能直接当答案——这正是本课的核心矛盾。

### 1.2 两个工程细节

- **峰归一化 `signal / peak * 0.5`**：把每路最大振幅压到 0.5，两路统一刻度，后面声纹比较才公平。
- **缓存复用**：`if all(path.is_file() for path in outputs): return outputs`——分离结果落盘到
  `outputs/szy_v1_cache/separation/`，中断重跑时跳过已分离的样本。

---

## 2. 第五步：唤醒锚点选择（moss_candidate.py 的 `select_wake_anchor`）

分离出 `wake0`/`wake1` 后，先对两路转写：

```python
wake0, wake1, command0, command1 = self._transcribe(
    self.asr, [wake_paths[0], wake_paths[1], command_paths[0], command_paths[1]])
```

对 `pos/cmd_13`，FireRed 转写结果：

```text
wake0 = "二十四"        wake1 = "嗯等二十分钟"      （都是分离后的幻觉文本，不是"你好科慕"）
command0 = "我们的石油"   command1 = "有人低头"
```

然后：

```python
anchor = select_wake_anchor(sample.wake_text, wake0["text"], wake1["text"])
```

### 2.1 为什么要选锚点？

分离出的两路 `wake0`/`wake1` 里，**只有一路是目标人（说唤醒词的人）**。但模型不告诉我们哪路是。
好在**我们知道唤醒文本**（"你好科慕"）——谁转写出来更接近唤醒词，谁就是目标人。这个"被认定为
目标人"的声纹，就叫 **anchor（锚点）**，后面所有"是不是目标人"的判断都以它为参照。

### 2.2 规则与真实数据

```python
reference = normalize_transcript(wake_text or "")   # "你好科慕"
source0 = str(source0_text or "").strip()           # "二十四"
source1 = str(source1_text or "").strip()           # "嗯等二十分钟"
edit0 = levenshtein(source0, reference)             # 编辑距离（都很大，因为分离出的是幻觉文本）
edit1 = levenshtein(source1, reference)
selected = 0 if edit0 <= edit1 else 1               # 距离更小（或相等）者胜 → 0
tie = not reference or edit0 == edit1
```

`pos/cmd_13` 的 detail 里 `wake_text_tie = true`、`wake_selected_source = 0`，说明 `edit0 == edit1`
（两路离"你好科慕"一样远——因为都是幻觉文本），按规则**平局固定选 source0**。

### 2.3 四种分支（第 1 课的 select_wake_anchor 完整规则回顾）

| 条件 | 锚点 | 对照 | raw 回退 |
|---|---|---|---|
| 只有 source0 非空 | 原始完整唤醒音频 | source1 | ✅ |
| 只有 source1 非空 | 原始完整唤醒音频 | source0 | ✅ |
| 两路都非空 | 编辑距离更小的一路 | 另一路 | ❌ |
| 两路都为空 | 原始完整唤醒音频 | 固定 source0 | ✅ |

**为什么"某一路为空"时要用原始完整唤醒音频当锚点？** 因为空的那路是分离失败的产物（静音/噪声），
而**原始唤醒音频虽含噪声，但完整保留了目标人的全部声纹信息**。短唤醒、强重叠、分离失败时，
原始音反而比残缺的分离源更可靠——这是关键的 raw fallback。

> 注意：`pos/cmd_13` 两路都非空（虽然都是幻觉文本），所以没有走 raw 回退，
> 而是"错误地"把 source0（"二十四"）当成了锚点。**这就是幻觉文本的危害：编辑距离规则
> 在"两路都不像唤醒词"时，只能瞎选一个。**

---

## 3. 第六步：差分声纹选择命令源（本课技术核心）

锚点定下来后，`process()` 读四段波形、算"差分声纹"：

```python
anchor_path = sample.wake_audio if anchor.raw_wake_fallback else wake_paths[anchor.selected_source]
contrast_path = wake_paths[anchor.contrast_source]
anchor_wave = read_wav(anchor_path)         # 锚点（目标人）
contrast_wave = read_wav(contrast_path)     # 对照（非目标人）
candidate_waves = [read_wav(command_paths[0]), read_wav(command_paths[1])]  # cmd0/cmd1

for name, embedder in (("eres", self.gate.eres_embedder), ("cam", self.gate.cam_embedder)):
    values = embedder.extract_waveforms([anchor_wave, contrast_wave, *candidate_waves])
    anchor_embedding, contrast_embedding = values[:2]
    embeddings[name] = [
        float(np.dot(anchor_embedding, candidate) - np.dot(contrast_embedding, candidate))
        for candidate in values[2:]
    ]
```

### 3.1 核心公式：差分（difference）

对每个命令候选 `c` 和每个声纹模型：

```text
diff(c) = cos(anchor, c) − cos(contrast, c)
```

其中 embedding 已 L2 归一化，所以 `np.dot` = 余弦相似度。

### 3.2 为什么是"差分"，而不是"直接比 anchor"？

这是**最值得你停下来想清楚**的设计。直觉版本是：

```text
cos(anchor, c) 大 → c 是目标人
```

但这有问题：**同一个声道、同一段录音里的所有分离源，声纹天然就"像"**（因为都是同一设备、
同一噪声环境、甚至可能都是人声）。直接比 anchor，很可能 cmd0、cmd1 都得分不低，分不出谁是目标人。

**差分的思想**：找一个"非目标人"的参照（contrast，来自唤醒分离的另一路），算"候选离目标人
比离非目标人近多少"：

```text
diff(c) = 候选与目标人的相似度 − 候选与非目标人的相似度
```

- 若 c 是目标人 → 第一项大、第二项小 → diff 大（正数）；
- 若 c 不是目标人 → 两项差不多 → diff 接近 0 甚至为负。

**差分把"环境/设备带来的共性相似度"抵消掉了**，留下"目标人特有的那部分"。这是声纹领域处理
"同声道干扰"的经典思路。

### 3.3 双前端求和 + 平局规则

```python
selected = "ms0" if (embeddings["eres"][0] + embeddings["cam"][0]) >= \
                    (embeddings["eres"][1] + embeddings["cam"][1]) else "ms1"
```

代入 `pos/cmd_13` 的真实差分值：

```text
                 ms0           ms1
eres_diff      0.116318      -0.365473
cam_diff       0.265125      -0.209972
─────────────────────────────────────
求和           0.381443      -0.575445
```

`0.381443 >= -0.575445` → 选 **ms0**。所以最终 `content = command0["text"] = "我们的石油"`。

### 3.4 为什么这条样本错了？

把整条链串起来看：

1. 分离把 cmd_13 拆成了"我们的石油"和"有人低头"——**两路都是幻觉文本**（真实标签是"打开洗衣服"，
   分离后没有任何一路保留了正确的四个字）。
2. 唤醒侧也一样，"二十四"和"嗯等二十分钟"都是幻觉，编辑距离平局瞎选了 source0。
3. 差分声纹在"两个都不对"的候选里，比较"谁更像（幻觉锚点）"，选出了 ms0——**决策逻辑完全按规则
   运行，但输入是垃圾，输出必然是垃圾**。

**这正是这条路线最大的风险**（队长文档第 17 节明说）：MossFormer2 输出有伪影，候选文本可能比
raw FireRed 更差。这也解释了为什么 `raw` 路线 CER 均值只有 0.155，而 `moss_candidate` 路线
CER 均值高达 0.488——**候选路线是被"难例"触发的**（这些样本 raw 本来就不可靠），分离又常常
帮倒忙。但也别忘了：`selected_candidate=orig` 的 64 条说明候选路线有时也会"回头选原始结果"。

---

## 4. 回退矩阵（整条链路的"安全网"）

`process()` 把候选路线整体包在 try/except 里，任何异常都回退：

```python
except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
    try:
        result["content"] = self.asr.transcribe_many([sample.command_audio])[0]
        result["route"] = "raw_fallback_error"      # 回退 raw
    except Exception as raw_error:
        result["error"] += f"; raw_fallback={...}"
        result["route"] = "empty_fallback_error"    # raw 也失败 → 空
```

完整回退矩阵（对应队长文档第 11 节）：

| 故障位置 | 最终行为 | 结果 route |
|---|---|---|
| 主门控低于阈值 | 空文本，不跑 ASR | primary_reject |
| raw FireRed 正常且未触发切换 | raw 文本 | raw |
| raw 触发切换，Moss 失败 | raw 文本 | raw_fallback_error |
| 唤醒一路为空 | 原始唤醒作锚点 | （仍 moss_candidate，wake_raw_fallback=true） |
| 命令一路为空 | raw 文本 | moss_candidate 但 selected_candidate=orig |
| 候选 ASR 或声纹异常 | raw 文本 | raw_fallback_error |
| raw FireRed 也异常 | 空文本 | empty_fallback_error |

**设计铁律**：分离候选只能改善结果，**不能夺走 raw fallback 的控制权**。这是队长在多次失败实验
（TSE 波形替换反而抬高 CER）后得出的血泪结论。

---

## 5. 终点：指标汇总 `summarize`（pipeline.py）

1838 条都跑完后，`main()` 调用 `summarize(results, duration)` 算总分：

```python
errors = sum(levenshtein(normalize_transcript(content), normalize_transcript(label)) for row in positive)
characters = sum(len(normalize_transcript(label)) for row in positive)
nonempty = sum(bool(content) for row in positive)
rejects = sum(not content for row in negative)
final_cer = errors / characters          # 0.3174937
final_pr  = nonempty / len(positive)     # 0.8790323
final_rr  = rejects / len(negative)      # 0.8396624
```

### 5.1 三个指标的口径

| 指标 | 公式 | 只对 | 含义 |
|---|---|---|---|
| CER | 编辑距离和 / 参考字符和 | 正样本 | 转写错多少字 |
| PR | 非空输出数 / 总数 | 正样本 | 召回：目标人在时，我们输出了没有 |
| RR | 空输出数 / 总数 | 负样本 | 拒识：目标人不在时，我们闭嘴了没有 |

竞赛总分 `J = (1 − CER) + RR = 1.5221688`。

### 5.2 真实路由分布（来自 baseline）

```text
raw              870 条    （56%，最低风险的默认路径）
moss_candidate   408 条    （22%，触发候选复核）
primary_reject   560 条    （30%，被主门控拒绝）
```

- 560 条被拒里，既有一堆负样本（正确拒识，贡献 RR），也混着被误杀的正样本（牺牲 CER/PR）。
- **三个指标必须一起看**：单独看 CER 0.317 不低、PR 0.879 也不高——因为门控牺牲了正样本召回去换
  负样本拒识。这是"抗干扰 + 拒识"任务的本质权衡。

---

## 6. 收口：`write_outputs` 与 manifest

```python
write_outputs(results, summary, Path(args.output), Path(args.final_output))
# → szy_v1_submission.json（提交格式）+ szy_v1_detail.json（审计明细）

manifest = { "config": config.__dict__, "moss_model_sha256": _sha256(...),
             "dual_backend_sha256": _sha256(...), ... }
# → szy_v1_manifest.json（本次用了哪份模型、哪个阈值、哪些输入）
```

`submission.json` 里每条只有 `id/content/label/cer`（提交格式）；`detail.json` 里保留全部审计字段
（route、sv_score、差分分数、各源文本……）——你在这三课里看到的所有中间量，都是为了**审计和复现**。

---

## 7. 全课总回顾：一条样本的完整旅程

```text
pos.jsonl 一行
  │ read_many_jsonl（infer.py）→ Sample
  ▼
submission_id（infer.py）→ "pos/cmd_13"
  ▼
DualMLPGate.accept（gate.py）→ sv_score=0.924 ≥ 0.446031 → 放行
  ▼
FireRed 转写 raw cmd（infer.py / fireredasr2）→ confidence=0.596
  │ normalize_transcript → _command_features → should_switch_command
  ▼
switch=True → MossFormer2 分离（pipeline.py）→ wake0/1、cmd0/1
  ▼
FireRed 转写 4 路 → select_wake_anchor（moss_candidate.py）→ anchor=wake0
  ▼
差分声纹（gate.py embedder + numpy）→ eres/cam 双前端求和 → 选 ms0
  ▼
content = "我们的石油"
  ▼
summarize（CER/PR/RR）→ write_outputs → manifest（sha256）
```

### 五个文件的分工（一句话各记一遍）

| 文件 | 一句话 |
|---|---|
| `szy_v1/pipeline.py` | 总指挥：编排流程、断点续跑、异常兜底、写结果 |
| `midea_qiji/gate.py` | 看门大爷：双前端 MLP 判"目标人在不在" |
| `midea_qiji/infer.py` | 工具箱：读样本、归一化、ASR 封装、编辑距离、指标 |
| `midea_qiji/moss_candidate.py` | 决策规则：唤醒锚点 + 切换规则（纯逻辑，可单测） |
| `configs/szy_v1_firered.json` | 冻结配置：所有阈值/路径的唯一来源 |

### 建议的下一步学习路线

1. **读懂 `infer.py` 的 `main()`**（旧基线入口）：它和 `pipeline.py` 的 `main()` 是两套入口，
   对比着看能理解"实验开关怎么设计"。
2. **读 `docs/MOSS_CANDIDATE_REPRODUCTION.md` 和 `docs/SZY_V1.md`**：了解历史缓存回放和失败分支。
3. **跑冒烟测试**：`bash scripts/run_szy_v1.sh --limit 5`，然后打开 `outputs/smoke_detail.json`
   逐字段对照本课讲解。
4. **改一个阈值试试**（在**你自己的新实验**里，别覆盖冻结配置）：把 `should_switch_command` 的
   0.70 改成 0.60，观察 `switches` 数量变化——你会直观感受到"规则参数如何影响路由"。

---

## 8. 思考题（终章）

1. `pos/cmd_13` 的 `wake_text_tie=true` 意味着什么？为什么说"编辑距离规则在两路都是幻觉文本时
   只能瞎选"？
2. 如果 `cmd1` 的转写是空串，`selected_candidate` 会是什么？对照 `choose_command_candidate`
   的"空源回退"分支回答。
3. 为什么"差分声纹"能抵消同声道干扰？用一句话概括它的几何直觉。
4. 主门控阈值调高（如 0.6）会让 PR 和 RR 分别怎么变？为什么竞赛分数 J 需要 PR 和 RR 一起看？
