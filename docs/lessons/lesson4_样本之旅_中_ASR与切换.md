# 第 4 课：一条样本的旅程（中）—— FireRed ASR、文本归一化与切换规则

> 上接第 3 课：`pos/cmd_13` 已经穿过主门控（sv_score 0.924 ≥ 阈值 0.446031，放行）。
> 这一课讲它接下来的三步：**raw FireRed 转写 → 文本归一化 → 二维切换规则**。
> 涉及文件：`pipeline.py`（`process` 主体）、`infer.py`（`FireRedRecognizer`、
> `normalize_transcript`、`levenshtein`）、`moss_candidate.py`（`should_switch_command`）。

---

## 1. 第三步（续）：raw 命令音频 FireRed 转写

门控通过后，`process()` 里第一件事是把**原始命令音频**送去转写：

```python
raw_command = self._transcribe(self.asr, [sample.command_audio])[0]
# → {"text": normalize_transcript("打开洗衣服" 的转写结果), "confidence": 0.596}
```

`_transcribe`（第 2 课讲过）内部调 `asr._model.transcribe(...)`，即 FireRedASR2-AED 模型。

### 1.1 FireRed 是什么？AED 是什么？

FireRedASR 是小红书开源的 ASR（自动语音识别）系统，`AED` = **Attention Encoder-Decoder**
（编码器-注意力-解码器），是语音识别的一种主流架构：

```text
音频 wav
  │
  ├─ ① 提取声学特征：fbank（滤波器组特征）——把波形切成帧，每帧算几十个频带能量
  │      （kaldi_native_fbank 就是干这个的，所以它是 FireRed 的硬依赖）
  ▼
编码器 Encoder（Transformer/Conformer 一类）
  │   把声学特征逐帧编码成"高维表示"
  ▼
解码器 Decoder（自回归生成文本）
     每一步输出一个字，并"注意"（attention）到编码器相关位置
  ▼
文本 "打开洗衣服"
```

竞赛里队长实测 FireRed 的 CER 比 Qwen3-ASR 好约 3.7 个点，所以 SZY v1 全链路用它。

### 1.2 冻结的解码参数（一次看懂就够）

`FireRedRecognizer.__init__` 里传了一串参数，几个关键的：

| 参数 | 值 | 含义 |
|---|---|---|
| `beam_size=1` | 1 | 束搜索宽度=1，即**贪心解码**：每一步只取概率最大的字，不再回退。最快 |
| `aed_length_penalty=0.6` | 0.6 | 长度惩罚 <1 倾向短句，抑制"没完没了地重复" |
| `temperature=1.0` | 1.0 | 采样温度，1.0=不改分布（纯贪心用不到） |
| `softmax_smoothing=1.25` | 1.25 | 输出概率平滑 >1 拉平分布，防止过度自信 |

> 这些参数是队长冻结的，**改动会改变识别结果**，所以复现时必须一模一样——这正是配置和 manifest
> 存在的意义。

### 1.3 为什么叫"raw"（原始）？

因为此时送进去的是**没分离过的原始 cmd.wav**。它的转写结果承担两个角色：

1. 结果可靠 → 直接当最终输出（大部分样本走这条路，`route=raw`）；
2. 结果不可靠 → 作为**回退兜底** + 候选比较的基准。

**raw 永远是保底选项**，这是整条路线"候选只可改善、不可夺权"原则的起点。

---

## 2. 文本归一化：`normalize_transcript`（infer.py，纯 Python 精华）

转写文本和标签在比较前，都要过同一个"归一化"函数，否则"25" 和 "二十五"会被错算成不同。
这是本课 Python 含量最高的一段，逐条拆。

```python
def normalize_transcript(text: str) -> str:
    digits = "零一二三四五六七八九"

    def number_to_chinese(value: int) -> str:
        if value < 10:
            return digits[value]
        if value < 20:
            return "十" + (digits[value % 10] if value % 10 else "")
        if value < 100:
            return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
        if value < 1000:
            remainder = value % 100
            prefix = digits[value // 100] + "百"
            if remainder == 0:
                return prefix
            if remainder < 10:
                return prefix + "零" + digits[remainder]
            return prefix + number_to_chinese(remainder)
        return str(value)

    def convert_number(match):
        return number_to_chinese(int(match.group(0)))

    normalized = re.sub(r"<\|[^|]+?\|>", "", str(text))          # ① 删特殊 token
    normalized = re.sub(r"(\d+)\s*%", r"百分之\1", normalized)     # ② 百分比
    normalized = re.sub(r"百分之\s*(\d+)", lambda m: "百分之" + number_to_chinese(int(m.group(1))), normalized)  # ③ 百分比里的数字转中文
    normalized = re.sub(r"\d+", convert_number, normalized)        # ④ 所有阿拉伯数字转中文
    return re.sub(r"[\s,，。.!！?？:：;；、】【、\"'“”‘’（）()\[\]<>《》|/\\_-]+", "", normalized.lower())  # ⑤ 删空白和标点
```

### 2.1 `number_to_chinese`：整数转中文读法（递归 + 整除取余）

对照几个例子理解每个分支（`//` 整除、`%` 取余）：

```text
5    → digits[5] = "五"                          （<10）
10   → "十" + (digits[0]="" 因为 10%10=0) = "十"   （<20 且个位为 0）
15   → "十" + "五" = "十五"                        （<20）
25   → digits[2]+"十"+"五" = "二十五"              （<100）
30   → "三"+"十"+"" = "三十"                       （<100 个位为 0）
300  → "三百"                                     （<1000 且整百）
305  → "三百"+"零"+"五" = "三百零五"               （<1000 且余数 <10）
315  → "三百" + number_to_chinese(15) = "三百一十五" （递归）
1234 → "1234"（放弃，保留阿拉伯数字）
```

注意 `digits[value]`：**字符串可以像列表一样用下标取字符**（`"零一二..."[5] == "五"`）。

### 2.2 正则五连（`re.sub` 的三个用法）

`re.sub(模式, 替换, 文本)` 把匹配到的部分替换掉。三处用了三种替换写法：

**① 删除特殊 token**

```python
re.sub(r"<\|[^|]+?\|>", "", str(text))
```

模型输出里有 `<|zh|>`、`<|endoftext|>` 这类特殊标记，正则 `<\|[^|]+?\|>` 匹配它们：
`<\|` = 字面 `<|`（`\|` 转义竖线），`[^|]+?` = 一个或多个非竖线字符（非贪婪），`\|>` = 字面 `|>`。
替换成空串 = 删除。

**② 百分比：`30% → 百分之30`**

```python
re.sub(r"(\d+)\s*%", r"百分之\1", normalized)
```

- `(\d+)`：圆括号 = **捕获组**，匹配一串数字并"记住"。
- `\s*`：可有可无的空格。
- 替换串 `r"百分之\1"` 里的 `\1` = **引用第 1 个捕获组**，即那串数字。

**③ 百分比里的数字转中文：`百分之30 → 百分之三十`**

```python
re.sub(r"百分之\s*(\d+)", lambda m: "百分之" + number_to_chinese(int(m.group(1))), normalized)
```

**`re.sub` 的第二参数可以是函数**（回调）：每匹配一处就调用一次，返回值作为替换结果。
`m.group(1)` 取出捕获组里的数字。这是正则 + 函数组合的高级用法。

**④ 所有数字转中文：`25 → 二十五`**

```python
re.sub(r"\d+", convert_number, normalized)
```

同样用回调函数，`convert_number(match)` 里 `match.group(0)` = 整个匹配串。

**⑤ 删空白与标点**

```python
re.sub(r"[\s,，。.!！?？:：;；、】【、\"'“”‘’（）()\[\]<>《》|/\\_-]+", "", normalized.lower())
```

`[...]` 是**字符类**，匹配其中任意一个字符；末尾 `+` 表示"连续的一串"；`\s` 匹配空白。
`.lower()` 顺手转小写（英文标签大小写归一）。

> 为什么要删除标点？因为竞赛的 CER 按"字符"算，标点、空格会让"开灯。"和"开灯"被算成不同。
> 统一删掉，大家站在同一起跑线。

### 2.3 验证（测试文件里有真实断言）

```python
normalize_transcript("空调开到25度，风量30%") == "空调开到二十五度风量百分之三十"
normalize_transcript("设置101分钟")          == "设置一百零一分钟"
```

---

## 3. 命令特征：`_command_features`（pipeline.py）

```python
chars_per_sec, command_keyword_score = _command_features(raw_command["text"], _duration(sample.command_audio))
```

```python
def _command_features(text, duration):
    normalized = normalize_transcript(text)
    return (float(len(normalized) / max(duration, 0.1)),   # chars_per_sec 语速
            float(len(CMD_KW.findall(normalized))))         # 命令关键词数
```

- `_duration(path)` 读音频秒数（`sf.info` 只读头，不加载波形）。
- **`chars_per_sec`（语速）** = 转写字数 / 时长。正常中文说话约 2~5 字/秒。**异常值**（太快/太慢）
  往往说明转写有问题（比如把噪声也"转"出了字）。
- **`command_keyword_score`（关键词数）** = 文本里出现了几个家电指令词（打开/关闭/空调/音量…）。
  用 `CMD_KW.findall(text)` 返回所有匹配，数个数。

`pos/cmd_13` 的这两个值是 `1.901141` 和 `1.000000`：语速偏慢（1.9 字/秒）、有 1 个关键词
（转写"我们的石油"里……等等，其实"石油"不含关键词——但注意 `_command_features` 用的是
**raw 转写**，而 raw 转写是什么我们不知道。关键词数 1.0 说明 raw 转写里有一个词命中了关键词表，
可能是模型把标签"打开洗衣服"里某个词转出来了，也可能转出了别的。这里我们只解释特征本身）。

---

## 4. 切换规则：`should_switch_command`（moss_candidate.py）

```python
switch = should_switch_command(
    confidence=raw_command["confidence"],          # 0.596
    command_keyword_score=command_keyword_score,   # 1.0
    chars_per_sec=chars_per_sec,                   # 1.901
)
```

函数本体（队长冻结的"二维规则"）：

```python
def should_switch_command(*, confidence, command_keyword_score, chars_per_sec):
    if confidence <= 0.70:
        return command_keyword_score >= 0.5 or chars_per_sec >= 3.0 or confidence <= 0.57
    return confidence <= 1.01 and chars_per_sec >= 3.0 and command_keyword_score < 0.5
```

### 4.1 逐分支解读

**分支 A：`confidence <= 0.70`（模型不太自信）**

```text
关键词数 >= 0.5   或   语速 >= 3.0   或   置信度 <= 0.57
→ 满足任一，就进入候选路线
```

直觉：模型不自信时，只要文本里出现了"命令关键词"、或者语速可疑、或者置信度特别低（≤0.57），
就值得花算力做一次候选复核。

**分支 B：`confidence > 0.70`（模型比较自信）**

```text
置信度 <= 1.01   且   语速 >= 3.0   且   关键词数 < 0.5
→ 三者同时满足，才进入候选路线
```

直觉：即使模型自信，如果"语速快得反常（≥3.0）却没有关键词"，也值得复核（可能是模型把噪声
念成了一段飞快的废话）。

### 4.2 代入 pos/cmd_13

```text
confidence = 0.596 ≤ 0.70 → 走分支 A
  keyword_score = 1.0 ≥ 0.5 → 满足 → switch = True ✅
```

所以 `pos/cmd_13` 触发候选路线，`switch_triggered = true`、`route = "moss_candidate"`。

### 4.3 两个容易误解的点

1. **`confidence` 不是"目标人在说话"的概率**，而是 FireRed 对自己转写质量的信心（可能 >1，因为
   它没经过校准）。所以规则里会出现 `confidence <= 1.01` 这种"几乎恒真"的写法——那是历史冻结行为，
   **复现时不能改**。
2. **`switch=True` 不代表会拒识或清空文本**。它只意味着"值得跑一次候选复核"；复核失败照样回退 raw
   文本（第 5 课讲）。这就是"切换"和"拒识"的本质区别。

---

## 5. 附：编辑距离 `levenshtein`（第 5 课要用，这里先打底）

唤醒锚点选择、CER 计算都依赖编辑距离。`infer.py` 里的实现是**动态规划 + 两行滚动数组**：

```python
def levenshtein(left, right):
    if left == right: return 0
    if not left: return len(right)
    if not right: return len(left)
    previous = list(range(len(right) + 1))           # 第 0 行：空串 → right 前 j 个字符
    for i, lc in enumerate(left, start=1):
        current = [i]                                # 每行第 0 列：删掉 left 前 i 个
        for j, rc in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,                     # 插入 rc
                previous[j] + 1,                     # 删除 lc
                previous[j-1] + (lc != rc),          # 替换（相同则 0）
            ))
        previous = current
    return previous[-1]
```

### 手工走一个例子：`"开灯" → "打开"`

```text
        ""  开  灯
  ""    0   1   2      ← 空串到"开"要插1、"开灯"要插2
  打    1   1   2      ← "打"vs"开"替换(1) → 1
  开    2   1   2      ← "打开"vs"开灯"：min(插入=1+1, 删除=1+1, 替换=1+1)=2
```

右下角 = 2，即"开灯"变"打开"要 2 步（其实"打"→"开"替换，"灯"→"开"替换）。

**为什么叫"两行滚动"？** 计算第 i 行只需要第 i-1 行（`previous`）和当前行已算出的 `current[-1]`，
所以不需要存整张表，两个列表就够了——**空间复杂度从 O(n×m) 降到 O(m)**。

`min(a, b, c)` 一次取三个的最小值；`lc != rc` 是布尔值，在算术里 True=1、False=0，正好表示
"字符不同则替换代价 1"。

---

## 6. 本课小结

`pos/cmd_13` 在这一课走完了：**raw FireRed 转写 → 归一化 → 语速/关键词特征 → 切换规则触发**。

- FireRed AED 是主力 ASR（fbank → 编码器 → 解码器），参数被冻结。
- `normalize_transcript` 是纯 Python 精华：递归数字转中文 + 正则五连 + 回调替换。
- `should_switch_command` 是"二维置信度规则"：低置信度 + 有关键词 → 触发候选。
- 编辑距离 `levenshtein` 是动态规划，后面选锚点、算 CER 全靠它。

### 思考题

1. `number_to_chinese(0)` 返回什么？`number_to_chinese(100)` 呢？为什么 10 是"十"而不是"一十"？
2. 如果 confidence=0.58（>0.57）且 keyword=0 且 cps=2.0，`should_switch_command` 返回什么？
   对照测试文件里的断言验证。
3. 为什么 CER 要在**归一化后**的文本上算，而不是在原始文本上算？

> 下一课（第 5 课，最后一课）：`pos/cmd_13` 进入 MossFormer2 盲分离，用唤醒文本选锚点，
> 用差分声纹选命令源，最后一路回退矩阵 + 指标汇总，把整条链路收口。
