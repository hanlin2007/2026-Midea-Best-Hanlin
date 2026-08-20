# 第 3 课：一条样本的旅程（上）—— 从 JSONL 到主门控判决

> 前两课是"逐行精读单个文件"；从这一课起换视角：**追着一条真实样本**，看它如何在一个个文件、
> 一个个函数之间穿梭，最终变成一条提交结果。涉及文件：`pipeline.py`、`infer.py`、`gate.py`。
>
> 我们全程使用基线结果里的真实样本 **`pos/cmd_13`**（标签"打开洗衣服"），它的完整记录长这样
> （节选自 `outputs/baseline/szy_v1_detail.json`）：

```json
{
  "id": "pos/cmd_13",
  "content": "我们的石油",              ← 最终输出（识别错了，CER=1.0，典型难例）
  "label": "打开洗衣服",
  "sv_score": "0.924140",               ← 主门控分数（很高，放行）
  "accepted": true,
  "sv_accept_threshold": "0.446031",
  "route": "moss_candidate",            ← 走了候选路线
  "selected_candidate": "ms0",
  "raw_confidence": "0.596000",         ← FireRed 对原始音频不自信
  "chars_per_sec": "1.901141",
  "command_keyword_score": "1.000000",
  "wake_source0_text": "二十四",
  "wake_source1_text": "嗯等二十分钟",
  "command_source0_text": "我们的石油",
  "command_source1_text": "有人低头",
  "eres_diff_ms0": "0.116318", "eres_diff_ms1": "-0.365473",
  "cam_diff_ms0": "0.265125",  "cam_diff_ms1": "-0.209972",
  "error": null
}
```

---

## 0. 全链路调用图（先建立全局地图）

```
pipeline.py                                  其它文件
───────────                                 ──────────
main()
 ├─ load_szy_config(config)                  (pipeline.py 自己)
 ├─ read_many_jsonl(jsonl, data_root)  ───▶  infer.py:read_many_jsonl → 生成 Sample 列表
 ├─ SzyV1Runner(config, ...)                 pipeline.py
 │    ├─ DualMLPGate(...)              ───▶  gate.py:DualMLPGate
 │    ├─ MossFormer2Separator(...)     ───▶  pipeline.py 自己
 │    └─ FireRedRecognizer(...)        ───▶  infer.py:FireRedRecognizer
 └─ for sample in samples:
       runner.process(sample)                pipeline.py:SzyV1Runner.process
         ├─ gate.accept(wake, cmd)     ───▶  gate.py:DualMLPGate.accept → score
         │     ├─ read_wav(...)        ───▶  gate.py:read_wav
         │     ├─ sliding_windows(...) ───▶  gate.py:sliding_windows
         │     └─ _model_features ×2   ───▶  gate.py 内部（embedder 来自 ModelScopeEmbedder）
         ├─ _transcribe(...)           ───▶  infer.py:FireRedRecognizer（底层 fireredasr2）
         ├─ should_switch_command(...) ───▶  moss_candidate.py:should_switch_command
         ├─ separator.separate(...)    ───▶  pipeline.py:MossFormer2Separator
         ├─ select_wake_anchor(...)    ───▶  moss_candidate.py:select_wake_anchor
         └─ 差分声纹选择（内联逻辑）         gate.py 的 embedder + numpy
       summarize() / write_outputs()         pipeline.py 自己
```

**记住这张图**。第 3 课讲"入口 → 主门控"，第 4 课讲"ASR → 切换规则"，第 5 课讲"分离 → 差分选择 →
回退 → 指标"。每课都会标出当前走到了图的哪一步。

---

## 1. 第一步：命令行入口（pipeline.py 的 `main`）

你在终端敲下：

```bash
python -m szy_v1.pipeline --data-root dataset/dataset_test/datasetA/datasetA \
    --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl ... --config configs/szy_v1_firered.json
```

Python 依次做这几件事（对应 `main()` 开头几行）：

```python
args = build_parser().parse_args()        # ① 把命令行参数解析成一个 args 对象
config = load_szy_config(args.config)     # ② 读 JSON 配置 → SzyConfig（含阈值 0.446031...）
samples = read_many_jsonl(args.jsonl, args.data_root)   # ③ 读样本清单 → Sample 列表
runner = SzyV1Runner(config, Path(args.output).parent)  # ④ 组装门控+分离器+ASR（加载模型）
```

- `parse_args()`：argparse 把 `--data-root xxx` 变成 `args.data_root == "xxx"`。它还会检查必填项、
  自动生成 `--help`。
- `load_szy_config`：读 JSON，`fail-closed` 校验未知键（第 2 课讲过），得到**冻结配置**。
  其中 `dual_threshold = 0.4460311856655159` 是本课主角——主门控阈值。
- `read_many_jsonl`：下一步详讲。
- `SzyV1Runner(...)`：**在这里加载三个重型模型**（耗时几十秒，只做一次）。

---

## 2. 第二步：JSONL → Sample（infer.py 的 `read_many_jsonl`）

`pos.jsonl` 里的每一行是一条样本。看看 `pos/cmd_13` 那一行长什么样：

```json
{"id": 13, "唤醒音频": "pos/kws_13.wav", "唤醒文本": "你好科慕",
 "识别音频": "pos/cmd_13.wav", "识别文本": "打开洗衣服"}
```

`read_many_jsonl` 的核心逻辑（已在第 1、2 课零散讲过，这里串起来）：

```python
for line_number, line in enumerate(handle, start=1):
    if not line.strip():            # 跳过空行
        continue
    row = json.loads(line)          # 一行文本 → dict
    wake_relative_path = str(row["唤醒音频"])    # "pos/kws_13.wav"
    command_relative_path = str(row["识别音频"])  # "pos/cmd_13.wav"
    samples.append(Sample(
        raw_id=str(row.get("id", "")),            # "13"
        wake_audio=root / wake_relative_path,     # 绝对路径（root + 相对路径拼接）
        command_audio=root / command_relative_path,
        command_relative_path=command_relative_path,
        label=row.get("识别文本"),                 # "打开洗衣服"（负样本这里是 None）
        wake_text=row.get("唤醒文本") or row.get("wake_text"),  # "你好科慕"
    ))
```

### 几个值得注意的细节

**① JSONL 为什么是"一行一个 JSON"而不是一个大 JSON 数组？**
因为可以**流式处理**：一行一行读，内存里永远只有一行。1838 条还好，如果是几百万条，一次
`json.load` 整个数组会爆内存。这是大数据清单的标准格式。

**② `root / wake_relative_path`**：`Path` 对象的 `/` 运算符就是拼接路径。`root` 是 `--data-root`
传进来的绝对路径。JSONL 里存**相对路径**（`pos/kws_13.wav`），运行时拼成绝对路径——**数据可整体
搬迁，不用改清单**。

**③ 正负样本的区别就在 `label`**：正样本 `label="打开洗衣服"`，负样本 `label=None`。后面所有
"正/负"判断都靠它（`sample.label is not None`）。

---

## 3. 第三步：稳定 ID（infer.py 的 `submission_id`）

进入 `process()` 第一件事：

```python
item_id = submission_id(sample, self.config.id_mode)   # id_mode = "relative_stem"
```

`relative_stem` 模式：

```python
path = Path(sample.command_relative_path)   # Path("pos/cmd_13.wav")
return (path.parent / path.stem).as_posix()  # Path("pos/cmd_13") → "pos/cmd_13"
```

得到 `"pos/cmd_13"`。

### 为什么不用 JSONL 里的 `id`（=13）？

因为 **pos.jsonl 和 neg.jsonl 的 id 各自从 0 开始编**，会撞车（pos 有条 id=13，neg 也有条 id=13）。
直接用作提交 ID 会互相覆盖；用作断点续跑的"已完成"键也会误判。从**相对路径**派生 ID 则天然唯一、
稳定（换个目录跑也不变）。这就是"稳定 ID"的含义。

---

## 4. 第四步：主门控判决（gate.py 的 `DualMLPGate.accept`）

```python
accepted, gate_score = self.gate.accept(sample.wake_audio, sample.command_audio)
```

对 `pos/cmd_13`，这一步得到 `gate_score = 0.924140`，阈值 `0.446031`，于是 `accepted = True`。

下面把 `score()` 的算法**拆到最细**，并对照真实数据讲解。

### 4.1 读波形 + 切滑窗

```python
enroll_wav = read_wav(wake_audio)      # kws_13.wav  → float32 数组（唤醒，当作"注册声纹"）
test_wav = read_wav(command_audio)     # cmd_13.wav  → float32 数组（识别，要判断目标人在不在）
windows = sliding_windows(test_wav)    # 识别音频切成 N 个 1 秒滑窗
```

- `enroll_wav`（唤醒）用**整段**；`test_wav`（识别）既用整段、又切滑窗。
- 为什么唤醒不切窗？唤醒词很短（"你好科慕"约 1 秒），整段就是目标人的声音；识别音频长、混合
  内容多，所以要滑窗捕捉"局部像目标人"的证据（第 1 课讲过）。

### 4.2 每个声纹模型算 16 维特征

```python
features = self._model_features(self.eres_embedder, enroll_wav, test_wav, windows)  # ERes2NetV2 的 16 维
features += self._model_features(self.cam_embedder,  enroll_wav, test_wav, windows)  # CAM++ 的 16 维
```

`_model_features` 内部（第 1 课读过，这里提炼算法）：

```python
# ① 一次性批量提取所有波形的声纹 embedding（1 次 GPU 前向）
embeddings = embedder.extract_waveforms([enroll_wav, test_wav, *windows])
emb_enroll, emb_test = embeddings[0], embeddings[1]      # 唤醒嵌入、识别整段嵌入
win_scores = [cosine_similarity(emb_enroll, w) for w in embeddings[2:]]  # 每个滑窗 vs 唤醒

# ② 16 维特征 = 整段余弦 + 10 个窗口统计 + 时长 + 响度比 + 削波
return [
    cosine_similarity(emb_enroll, emb_test),   # 0. 整段余弦
    *_window_stats(win_scores),                # 1~10. 窗口的 max/mean/std/top3/计数/跨度
    len(enroll_wav)/16000,                     # 11. 唤醒时长(秒)
    len(test_wav)/16000,                       # 12. 识别时长(秒)
    _rms(test_wav)/max(_rms(enroll_wav),1e-6), # 13. 响度比
    _clip_ratio(enroll_wav),                   # 14. 唤醒削波比例
    _clip_ratio(test_wav),                     # 15. 识别削波比例
]
```

**两个声纹模型各跑一遍，拼成 32 维。** ERes2NetV2 和 CAM++ 结构不同、互为补充——这是"多前端
融合"（multi-frontend fusion）的思路：单个模型可能对某种干扰盲，两个模型"投票"更稳。

### 4.3 z-score 标准化

```python
x = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
```

`self.mean` / `self.std` 是训练时统计的 32 维均值/标准差（存在 `backend_dual.pt` 里）。
**为什么必须标准化？** 32 维特征量纲五花八门：余弦在 [-1,1]、时长可能是 0.5~10、响度比 0~20、
削波比例 0~0.05。不标准化，量纲大的特征会"压过"量纲小的。标准化让每维都变成"均值 0、方差 1"，
MLP 才好学。

### 4.4 MLP 前向 + sigmoid

```python
with self.torch.no_grad():
    logit = self.mlp(self.torch.from_numpy(x).unsqueeze(0)).item()
return float(1.0 / (1.0 + np.exp(-logit)))
```

- MLP 结构 `32 → 32 → ReLU → Dropout → 32 → 16 → ReLU → 16 → 1`，输出一个标量 `logit`。
- **sigmoid** 把 logit 压到 (0,1)。直观理解：

```text
logit = -3   →  sigmoid ≈ 0.047   （几乎不可能）
logit =  0   →  sigmoid = 0.5     （半信半疑）
logit = +3   →  sigmoid ≈ 0.953   （几乎肯定）
```

`pos/cmd_13` 的 `sv_score = 0.924140` 对应 logit ≈ 2.50，MLP 相当确定"目标人在说话"。

### 4.5 判决

```python
def accept(self, wake_audio, command_audio):
    score = self.score(wake_audio, command_audio)
    return score >= self.threshold, score   # 0.924140 >= 0.446031 → True
```

阈值 `0.4460311856655159` 是队长**在训练/校准集上选出来的**，不是拍脑袋。它平衡了：

- 阈值太高 → 正样本被误杀（PR 下降）；
- 阈值太低 → 负样本被放进来（RR 下降）。

这是竞赛的核心权衡，课程第 5 课讲指标时会再回来看它。

---

## 5. 对照：一条被拒绝的样本

看 `pos/cmd_18`（标签"关闭灯光"）——它是**正样本，但被主门控拒绝了**：

```json
{"id": "pos/cmd_18", "content": "", "label": "关闭灯光",
 "sv_score": "0.382036", "accepted": false, "route": "primary_reject"}
```

`0.382036 < 0.446031` → 拒绝 → **连 ASR 都不跑**，直接输出空文本：

```python
if not accepted:
    result["item_duration"] = f"{time.perf_counter() - started:.6f}"
    return result      # content 保持 ""
```

### 这条样本暴露了什么？

它 label 是"关闭灯光"（正样本，目标人**确实**在说话），但门控分数只有 0.382，被误杀。这解释了
最终指标里 CER 为什么会偏高：**正样本被主门控拒绝 → 输出空 → 编辑距离 = 整个标签长度 → CER 贡献
巨大**。所以竞赛的 CER 和 PR 必须一起看（第 5 课展开）。

### 门控拒绝后为什么"不跑 ASR"？

三个原因：

1. **省算力**：全量 1838 条里 560 条被拒（30%），这些不跑 ASR 能省大量时间。
2. **防误输出**：ASR 对任何输入都会"吐出点字"（哪怕是噪声也会幻觉出文字）。负样本如果跑了 ASR，
   输出非空，RR 就崩了。
3. **语义正确**：主门控说"目标人不在"，那就该输出空，后面的步骤没有意义。

---

## 6. 本课小结

这一课，样本 `pos/cmd_13` 走完了：**命令行 → 配置 → JSONL → Sample → 稳定 ID → 主门控**。

- 主门控的核心是 **声纹相似度**：用两个模型把音频压成向量，比较"唤醒声纹"与"识别声纹"像不像。
- `pos/cmd_13` 门控分 0.924（放行）；`pos/cmd_18` 门控分 0.382（拒绝）。
- 门控是整个系统**风险最高**的一环：它拒绝了 560 条（含部分正样本，牺牲了 CER/PR），但换来了
  负样本的正确拒识（RR）。

### 思考题

1. 如果阈值从 0.446031 调到 0.60，`pos/cmd_13`（0.924）会怎样？`pos/cmd_18`（0.382）会怎样？
   整体 PR 和 RR 会如何变化？
2. 为什么"两个声纹模型"比"一个声纹模型"更稳？结合第 1 课的"多前端"讲一讲。
3. `submission_id` 有 6 种模式（raw/name/path/relative/relative_stem/stem），如果换成 `raw` 会出
   什么问题？

> 下一课（第 4 课）：样本穿过门控后，进入 FireRed ASR、文本归一化、切换规则。我们会看到
> `pos/cmd_13` 的 `raw_confidence=0.596` 是如何触发候选路线的。
