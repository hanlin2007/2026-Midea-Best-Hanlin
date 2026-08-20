# 第 1 课：从 list/dict/class 到算法实战 —— 逐行精读 `gate.py`

> 本课对应文件：`src/midea_qiji/gate.py`（已全文加注释，可直接对照阅读）。
> 难度：Python 初学者 → 初级算法。学完你会：读懂一个真实的"声纹门控"算法，
> 顺手掌握 list/dict/class/numpy 在工程里的真实用法。

---

## 0. 先搞清楚：这段代码到底在解决什么问题？

竞赛任务里有两条音频：

- **唤醒音频 kws.wav**：一段多人重叠的语音，其中**目标人说了唤醒词**（如"你好科慕"）。
- **识别音频 cmd.wav**：一段可能多人重叠的命令语音，我们要识别**目标人**说的话。

在花大价钱跑语音识别（ASR）之前，系统先要回答一个问题：

> cmd.wav 里，**到底有没有目标说话人在说话？**

这就是 `gate.py` 的职责——一个"看门大爷"。它输出一个 0~1 的概率：

```
DualMLPGate.accept(kws.wav, cmd.wav)
    → (True,  0.924)   表示"很像目标人，放行去识别"
    → (False, 0.382)   表示"不像目标人，直接拒绝，输出空文本"
```

看门大爷的"看门方法"分三步，正好对应文件里的三块代码：

```
┌─────────────────────────────────────────────────────────────┐
│ 第 1 步：提取声纹特征                                        │
│   用两个声纹模型（ERes2NetV2 + CAM++），把音频压成向量(embedding) │
│   再算"整段余弦 + 滑窗余弦统计 + 时长/能量/削波"共 16 维特征/模型   │
├─────────────────────────────────────────────────────────────┤
│ 第 2 步：拼成 32 维，做 z-score 标准化                        │
├─────────────────────────────────────────────────────────────┤
│ 第 3 步：喂给一个小型神经网络 MLP（裁判），输出概率             │
│   score >= 阈值(0.446031) → 放行；否则拒绝                    │
└─────────────────────────────────────────────────────────────┘
```

好，现在开始逐行读。**强烈建议你边看边打开 `src/midea_qiji/gate.py` 对照。**

---

## 1. 文件头部：`__future__`、import 与常量

### 1.1 模块 docstring（三引号字符串）

```python
"""双前端 MLP 声纹门控（ERes2NetV2 + CAM++ 特征 → MLP 裁判）。 ... """
```

Python 里，一个 `.py` 文件最开头用**三个引号**包起来的大段文字叫 **docstring**（文档字符串）。
它不只是注释——运行 `help(模块名)` 或 `模块名.__doc__` 时能直接读出来。工程代码的 docstring
通常回答三个问题：**这个模块是干什么的、给谁用、怎么用**。养成读 docstring 的习惯，是看懂
别人代码的第一技能。

### 1.2 `from __future__ import annotations`

```python
from __future__ import annotations  # 让类型注解可以"延迟求值"
```

- 这是"未来特性开关"。`__future__` 模块专门用来提前启用**未来版本**才默认打开的特性。
- `annotations` 这个特性让所有**类型注解**（形如 `def f(x: int) -> str:` 里的 `int`、`str`）
  变成"字符串"延迟求值，而不是在定义函数时立刻去判断类型是否存在。
- 好处：可以在注解里写"还没定义的类型"（比如类自己引用自己），还能略微加速模块导入。
- 新手可以先不理解深意，只要记住：**很多现代工程文件第一行（import 之后）都这么写**，照抄即可。

### 1.3 导入模块

```python
from collections import defaultdict  # 默认值字典
from pathlib import Path             # 面向对象的路径工具
from typing import Sequence          # 类型标注：任意可下标遍历的序列
import numpy as np                   # 数值计算库（别名 np 是社区惯例）
```

逐条讲解：

| 写法 | 含义 | 本课里哪里用到 |
|---|---|---|
| `from collections import defaultdict` | 从 `collections` 模块导入 `defaultdict` 这一个东西 | `extract_waveforms` 里按帧数分组 |
| `from pathlib import Path` | 导入 `Path` 类，用它表示文件路径 | 到处都有，如 `read_wav(path)` |
| `from typing import Sequence` | 导入类型标注工具 | 函数签名 `waveforms: Sequence[np.ndarray]` |
| `import numpy as np` | 导入 numpy 并起别名 `np` | 所有数组运算 |

**两种 import 的区别（务必记住）：**

```python
import numpy as np              # 导入整个模块，用 np.xxx 访问
from pathlib import Path        # 只导入模块里的一个名字，直接用 Path(...)
```

### 1.4 常量

```python
SAMPLE_RATE = 16000
FEATURE_KEYS = [ "cos_whole", "n_windows", ... ]   # 16 个特征名的 list
```

- `SAMPLE_RATE = 16000`：**全大写下划线命名 = 常量**。这是 Python 社区约定（不是语法强制）：
  "这个值不应该被修改"。16 kHz 表示一秒钟音频有 16000 个采样点。
- `FEATURE_KEYS` 是一个 **list**，装着 16 个字符串。它其实是"特征顺序的说明书"——后面
  `_model_features()` 返回的 16 个数字，必须按这个顺序一一对应，否则 MLP 权重会对错位。
  **用 list 当"顺序说明书"是工程里很常见的做法**：代码、文档、数据三方靠同一个 list 对齐。

> 顺便学一个知识点：为什么音频采样率是 16000 而不是 44100（CD 音质）？
> 人类语音的主要信息集中在 4 kHz 以下，按奈奎斯特定理 8 kHz 采样就够还原；
> 16 kHz 是语音识别/声纹领域的行业标准（够用、文件小、算得快）。

---

## 2. 读音频：`read_wav`

```python
def read_wav(path: str | Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ValueError(f"unsupported wav format: {path}")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
```

### 逐行讲

**`path: str | Path`** —— 类型注解里的竖线 `|` 表示"或"：参数可以是字符串，也可以是 `Path` 对象。
这是 Python 3.10+ 的语法。

**`import wave`（函数内导入）** —— 为什么 import 写在函数里面？这叫 **lazy import（惰性导入）**。
`wave` 是标准库，导入很快，但团队的习惯是：**只在真正用到的地方才导入重量级模块**（后面你会在
gate.py 里看到 `import torch`、`import modelscope` 都写在函数内部）。好处：别人 `import gate` 时
不会连带加载一堆用不上的重库，模块导入更快、更轻。

**`with wave.open(...) as handle:`** —— `with` 是**上下文管理器**语法，保证文件无论中间是否报错，
离开缩进块时都会自动关闭。读文件的标准姿势就是 `with open(...) as f:`，不要写 `f = open(...)` 然后
忘了 `f.close()`（会泄漏文件句柄）。

**`"rb"`** —— 读二进制（raw binary）。wav 是二进制格式，读出来是 `bytes`。

**格式守卫** ——

```python
if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
    raise ValueError(...)
```

这叫 **fail-fast（快速失败）**：数据不符合预期时，立刻、响亮地报错，而不是静默产出错误结果。
这是工程纪律：**宁可早点崩，不要错得无声无息**。竞赛数据约定所有 wav 都是 16k 单声道 PCM16，
一旦不符就说明数据喂错了。

**`handle.readframes(handle.getnframes())`** —— 读出全部采样帧。此时 `raw` 是 `bytes`（一串字节）。

**`np.frombuffer(raw, dtype=np.int16)`** —— 把字节流按"16 位有符号整数"解读成 numpy 数组。
PCM16 格式里，每个采样点是 2 字节（16 位）的整数。`np.frombuffer` 是零拷贝转换，非常快。

**`.astype(np.float32)`** —— 把整数转成 32 位浮点（深度学习模型都吃浮点）。

**`/ 32768.0`** —— 归一化。int16 取值范围是 [-32768, 32767]，除以 32768 后，波形振幅被缩放到
大约 [-1.0, 1.0]。为什么？**让所有音频在同一个量纲上**，后续计算（点积、统计）数值才稳定。

### 小结这一节你学到的

- `with ... as ...` 读文件
- `raise ValueError` 快速失败
- `bytes → numpy 数组 → float32 → 归一化` 的四步音频读取套路

---

## 3. 余弦相似度：`cosine_similarity`

```python
def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
```

### 数学直觉（不用怕，只有初中数学）

两个向量（这里是两个"声纹嵌入"）的余弦相似度：

```
cos(θ) = (a · b) / (|a| × |b|)
```

- **分子** `a · b`：点积，衡量"两个向量有多同向"。
- **分母** `|a| × |b|`：两个长度相乘，用来**消除长度影响**。
- 结果范围 [-1, 1]：1 = 同方向，0 = 垂直，-1 = 反方向。

**为什么声纹比较用余弦而不是直接算距离？**
因为说话人的声纹主要藏在嵌入向量的**方向**里，而不是长度里。音量大小、句子长短都会改变向量
长度，但同一个人的"声音方向"是稳定的。余弦相似度对长度免疫，正好符合需求。

### 逐行讲

- **`np.linalg.norm(x)`**：numpy 的线性代数（linalg = linear algebra）函数，默认算 L2 范数
  （就是欧氏长度 `sqrt(平方和)`）。
- **`float(...)`**：把 numpy 标量（`np.float32` 之类）转成 Python 原生 `float`。为什么要转？
  因为原生 float 打印好看、能直接进 JSON。**工程里 numpy 算完、交给外部时，常用 `float()` 收口。**
- **`if left_norm <= 1e-8`**：防御性判断。如果某段是静音（波形全 0），范数就是 0，除 0 会产生
  `NaN` 或 `Inf`，污染后续所有统计。`1e-8` 是科学计数法 = 0.00000001，表示"几乎为 0"。
- **`np.dot(left, right)`**：点积。

---

## 4. 滑窗：`sliding_windows`

```python
def sliding_windows(audio, window_seconds=1.0, shift_seconds=0.2):
    window = int(window_seconds * SAMPLE_RATE)   # 16000
    shift = int(shift_seconds * SAMPLE_RATE)     # 3200
    if len(audio) <= window:
        padded = np.zeros(window, dtype=np.float32)
        padded[: len(audio)] = audio
        return [padded]
    segments = []
    for start in range(0, len(audio) - window + 1, shift):
        segments.append(audio[start : start + window])
    tail = len(audio) - (len(segments) - 1) * shift - window
    if tail >= int(0.4 * SAMPLE_RATE):
        segments.append(audio[len(audio) - window :])
    return segments
```

### 为什么需要滑窗？

命令音频可能 5~10 秒，而目标人可能只在其中 1 秒说话（其余是别人或噪声）。如果只对"整段"算一次
相似度，目标人的信号会被稀释。**滑窗**就是把音频切成很多 1 秒小段，每 0.2 秒前进一步，分别打分，
再用统计量捕捉"局部很像"的证据。

```
音频: |==============|  （10 秒 = 160000 点）
窗口: [----]        （1 秒 = 16000 点）
        [----]      （前进 0.2 秒 = 3200 点）
          [----]
            ...
```

### 逐行讲

**`int(window_seconds * SAMPLE_RATE)`** —— 秒换算成采样点数。16000 点 = 1 秒。

**短音频分支**：

```python
if len(audio) <= window:
    padded = np.zeros(window, dtype=np.float32)   # 造一个全 0 的一秒窗口
    padded[: len(audio)] = audio                  # 切片赋值，把音频塞进开头
    return [padded]
```

- `np.zeros(n, dtype=...)`：造一个长度为 n 的全 0 数组。
- `padded[: len(audio)] = audio`：**切片赋值**。`[:k]` 表示"前 k 个位置"，把音频填进去，其余保持 0。
  这叫 **padding（补零）**，是深度学习里让"变长输入对齐成定长"的最常用手段。
- 为什么要补零？——保证至少有一个窗口、且所有窗口等长，后面才能拼成矩阵批量算。

**`range(0, len(audio) - window + 1, shift)`** —— `range(起点, 终点, 步长)`。这里终点是
`len(audio) - window + 1`，保证最后一个窗口的右边界不越界。

**`audio[start : start + window]`** —— 列表/数组**切片**。取 `[start, start+window)` 左闭右开区间，
正好 1 秒。

**尾部处理**（这是工程里最容易被忽略的细节）：

```python
tail = len(audio) - (len(segments) - 1) * shift - window
```

`range` 循环按步长前进，最后往往剩一段"不够一扇窗"的尾巴。这里算出尾巴有多长，如果 >= 0.4 秒
（6400 点）值得再补一个窗口（取音频**末尾**整整 1 秒），太短（< 0.4 秒）就丢弃。**工程细节：
阈值 0.4 秒是权衡——太短的信息量太小，留着反而引入噪声。**

---

## 5. 窗口统计：`_window_stats`

```python
def _window_stats(win_scores: list[float]) -> list[float]:
    scores = np.asarray(win_scores, dtype=np.float64)
    n = len(scores)
    above = scores >= 0.30
    if above.any():
        indices = np.flatnonzero(above)
        span = (indices[-1] - indices[0] + 1) / n
    else:
        span = 0.0
    return [
        float(n), float(scores.max()), float(scores.mean()), float(scores.std()),
        float(np.sort(scores)[-3:].mean()),
        float((scores >= 0.25).sum()), float(above.sum()), float((scores >= 0.35).sum()),
        float(above.mean()), float(span),
    ]
```

输入是一个 list：每个滑窗与唤醒嵌入的余弦相似度。输出是 10 个统计数字。这是把"一堆窗口分数"
压缩成"一小撮特征"的过程——**特征工程（feature engineering）**。

### 逐行讲

**`np.asarray(win_scores, dtype=np.float64)`** —— list → numpy 数组。`asarray` 和 `array` 的区别：
如果已经是数组，`asarray` 不复制（省内存）。`float64` 提高统计精度。

**`above = scores >= 0.30`** —— 这是 numpy 的**布尔掩码（boolean mask）**。`scores >= 0.30` 不是
一个 True/False，而是一个**同样长度的 True/False 数组**：

```python
scores = np.array([0.10, 0.35, 0.42, 0.20])
above  = scores >= 0.30
# above == [False, True, True, False]
```

**`above.any()`** —— 数组里只要有一个 True 就返回 True。

**`np.flatnonzero(above)`** —— 返回所有 True 的**下标**（位置）。

**`span = (indices[-1] - indices[0] + 1) / n`** —— "目标人疑似出现"的区域在时间轴上占多大跨度。
`indices[-1]` 是最后一个 True 的下标，`indices[0]` 是第一个。连续说话 → span 大；只在某一瞬间
像了一下 → span 小。

**`np.sort(scores)[-3:].mean()`** —— 排序后取最后 3 个（最大的 3 个）再平均。`[-3:]` 是负索引切片：
"从倒数第 3 个到末尾"。**"Top-3 均值"是鲁棒统计的常用技巧**：比"最大值"抗噪（防止某个孤立窗口
偶然很高）。

**`(scores >= 0.25).sum()`** —— `True` 在 Python 里可以当 `1` 用，所以对布尔数组 `.sum()` 就是
"数 True 的个数"。**这是 numpy 里数"满足条件元素个数"的惯用法。**

**`above.mean()`** —— 布尔数组的均值 = True 的比例（因为 True=1, False=0）。

### 小结这一节

你看到了一次完整的**特征工程**：原始信号（一堆余弦分数）→ 10 个有物理含义的统计特征。
以后你会在无数 ML 项目里看到类似套路：**原始数据太"大"太"散"，要压缩成机器学习模型能理解的
少量数字。**

---

## 6. 能量与削波：`_rms` 和 `_clip_ratio`

```python
def _rms(audio):
    return float(np.sqrt(np.mean(audio**2) + 1e-12))

def _clip_ratio(audio):
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= 0.999))
```

- **`audio**2`**：numpy 数组的**逐元素平方**（`**` 是幂运算）。`np.mean` 求平均，`np.sqrt` 开方。
  合起来就是 RMS（均方根），衡量"响度"。
- **`+ 1e-12`**：防全 0 静音时 `sqrt(0)=0` 的数值防御（与训练端保持一致）。
- **`_clip_ratio`**：削波比例。削波（clipping）= 录音设备增益过大，本该 >1.0 的振幅被硬切到 1.0，
  波形顶被"削平"。`np.abs(audio) >= 0.999` 标记接近满幅的点，`.mean()` 得到比例。
  削波的音频信息已丢失，MLP 可以学着"削波严重 → 分数可疑"。

> 注意这两个函数名以 `_` 开头。Python 约定：**下划线开头 = 模块内部私有**（"别从外面直接调我"）。
> 不是语法禁止，是"君子协定"。

---

## 7. 声纹嵌入器：`ModelScopeEmbedder`

这是本课第一个 **class**。先记住 class 的三个基本概念：

- **`__init__`**：构造方法，`MyClass(args)` 时自动调用，用来初始化"这个对象身上的数据"。
- **`self`**：指"当前这个对象自己"。类里的方法第一个参数永远是 `self`。
- **方法**：定义在 class 里的函数，通过 `对象.方法()` 调用。

### 7.1 `__init__`

```python
class ModelScopeEmbedder:
    def __init__(self, model, device, batch_size=64):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.batch_size = batch_size
        self.pipeline = pipeline(task=Tasks.speaker_verification,
                                 model=str(model), device=device)
```

- `self.batch_size = batch_size`：把构造参数存到对象上，供其它方法用。
- `pipeline(...)`：ModelScope 的"流水线"对象，封装了模型下载/加载/预处理。这里复用它做"说话人确认"。

### 7.2 `extract_waveforms` —— 本课最值得精读的方法

```python
def extract_waveforms(self, waveforms):
    if not waveforms:
        return []
    import torch
    prepared = self.pipeline.preprocess(list(waveforms))
    model = self.pipeline.model
    extract_feature = None
    for attr in dir(model):
        if attr.endswith("__extract_feature"):
            extract_feature = getattr(model, attr)
            break
    if extract_feature is None:
        raise AttributeError(f"no __extract_feature on {type(model).__name__}")
    features = [extract_feature(w.unsqueeze(0) if w.ndim == 1 else w) for w in prepared]
    by_frame_count = defaultdict(list)
    for index, feature in enumerate(features):
        by_frame_count[int(feature.shape[1])].append(index)
    embeddings = [None] * len(features)
    with torch.inference_mode():
        for indices in by_frame_count.values():
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                feature_batch = torch.cat([features[i] for i in batch_indices], dim=0).to(model.device)
                batch = model.embedding_model(feature_batch).detach().cpu().numpy()
                for index, embedding in zip(batch_indices, batch):
                    embeddings[index] = np.asarray(embedding, dtype=np.float32)
    normalized = []
    for embedding in embeddings:
        if embedding is None:
            raise ValueError("embedder did not return every embedding")
        norm = float(np.linalg.norm(embedding))
        normalized.append((embedding / max(norm, 1e-8)).astype(np.float32))
    return normalized
```

### 逐段讲

**① 空输入早退**

```python
if not waveforms:
    return []
```

`not []` 为 True（空列表是"假值"）。Python 里空 list、空 str、0、None 都是"假"。**先处理边界情况
再走主逻辑**，是函数设计的好习惯。

**② 找到"提取特征"的内部方法**

```python
for attr in dir(model):
    if attr.endswith("__extract_feature"):
        extract_feature = getattr(model, attr)
        break
```

- `dir(obj)`：列出对象所有的属性/方法名（返回 list of str）。
- `attr.endswith("__extract_feature")`：字符串方法，"以...结尾"。
- `getattr(model, attr)`：**按名字取属性**。`getattr(x, "foo")` 等价于 `x.foo`，但名字是字符串时
  只能用 getattr。这是**反射（reflection）**的入门——在运行时动态找方法。
- `break`：找到就跳出循环。

> 为什么 ModelScope 的模型方法叫 `__extract_feature` 这种怪名字（双下划线开头结尾）？
> 双下划线开头结尾（`__xxx__`）是 Python 的"魔法方法"命名习惯，ModelScope 故意起这种名字
> 让普通用户不轻易调用（内部接口）。代码里用"字符串结尾匹配"来找它，是为了兼容不同版本模型。

**③ 列表推导式批量提特征**

```python
features = [extract_feature(w.unsqueeze(0) if w.ndim == 1 else w) for w in prepared]
```

这是**列表推导式（list comprehension）**——用一行写出"对每个元素做处理，收集成新 list"：

```python
# 等价于：
features = []
for w in prepared:
    features.append(extract_feature(w.unsqueeze(0) if w.ndim == 1 else w))
```

`w.unsqueeze(0) if w.ndim == 1 else w` 是**三元表达式**（`A if 条件 else B`）：一维波形就加一个
"批维度"变成二维。

**④ `defaultdict`：自动给默认值的字典**

```python
by_frame_count = defaultdict(list)
for index, feature in enumerate(features):
    by_frame_count[int(feature.shape[1])].append(index)
```

- `enumerate(features)`：遍历的同时给出下标，`for index, feature in ...` 同时解包两个值。
- `defaultdict(list)`：普通 dict 里 `d[新键]` 会报 `KeyError`；defaultdict 里第一次访问新键时
  **自动用 `list()`（空列表）初始化**，于是可以直接 `.append(...)`。
- 这段代码的作用：**按"帧数"（音频长度）把下标分组**。不同长度的音频前端特征维度不同，
  无法拼成一个大矩阵，所以同长度的凑一批。

```python
by_frame_count == { 320: [0, 3, 7],    # 320 帧的音频有这 3 条
                    512: [1, 2], ... } # 512 帧的有这 2 条
```

**⑤ 预分配 + 按位回填**

```python
embeddings = [None] * len(features)
```

`[None] * n` 造一个长度为 n、全是 None 的列表。**预分配槽位，后面按 index 填**——这样结果顺序
永远和输入顺序一致（哪怕内部乱序处理）。

**⑥ 批量推理**

```python
with torch.inference_mode():
    ...
    batch = model.embedding_model(feature_batch).detach().cpu().numpy()
    for index, embedding in zip(batch_indices, batch):
        embeddings[index] = np.asarray(embedding, dtype=np.float32)
```

- `torch.inference_mode()`：推理模式，关闭梯度跟踪，省显存、算得快。
- `model.embedding_model(feature_batch)`：前端特征 → 声纹嵌入。
- `.detach().cpu().numpy()`：三个连续调用——**脱离计算图 → 搬回内存 → 转 numpy**。这是 torch 推理
  结果的"收口三连"。
- `zip(batch_indices, batch)`：把两个序列**配对**成 (下标, 嵌入) 一起遍历。

**⑦ L2 归一化**

```python
norm = float(np.linalg.norm(embedding))
normalized.append((embedding / max(norm, 1e-8)).astype(np.float32))
```

把每个嵌入除以自己的长度，让所有嵌入长度变成 1。**归一化之后，余弦相似度 = 点积**（分母恒为 1），
计算更简单，而且所有比较都在同一尺度。

---

## 8. 裁判：`DualMLPGate`

### 8.1 `__init__` —— 加载裁判

```python
def __init__(self, sv_model, cam_model, backend_path, threshold, device, model_root=None):
    import torch
    from torch import nn
    from midea_qiji.infer import resolve_model_path

    self.threshold = threshold
    self.eres_embedder = ModelScopeEmbedder(resolve_model_path(sv_model, model_root), device)
    self.cam_embedder = ModelScopeEmbedder(resolve_model_path(cam_model, model_root), device)

    bundle = torch.load(str(backend_path), map_location="cpu", weights_only=False)["dual"]
    self.mean = np.asarray(bundle["mean"], dtype=np.float32)
    self.std = np.asarray(bundle["std"], dtype=np.float32)
    self.mlp = nn.Sequential(
        nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )
    renamed = {k.replace("net.", ""): v for k, v in bundle["mlp"].items()}
    self.mlp.load_state_dict(renamed)
    self.mlp.eval()
    self.torch = torch
```

要点：

- **两个 embedder**：ERes2NetV2 和 CAM++ 各一个，`self.eres_embedder` / `self.cam_embedder`。
- **`torch.load(...)`**：读训练时保存的 `.pt` 权重文件。`map_location="cpu"` 先加载到内存。
- **`["dual"]`**：权重文件是个 dict，取 `"dual"` 这个键（里面装着 mean/std/mlp）。
- **`nn.Sequential(...)`**：把若干层**串成一条流水线**，数据从左到右流过。这里的 MLP 结构：

```
输入 32 维 → Linear(32→32) → ReLU → Dropout → Linear(32→16) → ReLU → Linear(16→1) → 输出 1 维
```

  - `nn.Linear(in, out)`：全连接层，`y = x·W^T + b`，把 in 维线性变换到 out 维。
  - `nn.ReLU()`：非线性激活 `max(0, x)`。没有它，多层线性叠加还是线性，等于一层。
  - `nn.Dropout(0.1)`：训练时随机丢 10% 神经元防过拟合（`eval()` 模式下自动失效）。

- **`{k.replace("net.", ""): v for k, v in bundle["mlp"].items()}`**：**字典推导式**。训练时保存的
  键带 `net.` 前缀，这里剥掉再装载。`.items()` 遍历 (键, 值) 对。

- **`self.mlp.load_state_dict(renamed)`**：把权重"灌"进刚搭好的结构。**结构必须和训练时一模一样**，
  否则形状对不上会报错。

- **`self.mlp.eval()`**：切到评估模式（推理用，固定 Dropout 等行为）。

### 8.2 `score()` —— 把两个模型的特征拼起来判分

```python
def score(self, wake_audio, command_audio):
    enroll_wav = read_wav(wake_audio)
    test_wav = read_wav(command_audio)
    windows = sliding_windows(test_wav)
    features = self._model_features(self.eres_embedder, enroll_wav, test_wav, windows)
    features += self._model_features(self.cam_embedder, enroll_wav, test_wav, windows)
    x = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
    with self.torch.no_grad():
        logit = self.mlp(self.torch.from_numpy(x).unsqueeze(0)).item()
    return float(1.0 / (1.0 + np.exp(-logit)))
```

- `features += ...`：list 的 `+=` 就是**原地拼接**。第一个模型 16 维 + 第二个 16 维 = 32 维。
- **z-score 标准化** `(x - mean) / std`：把每维特征缩放到"均值 0、标准差 1"。mean/std 是训练时
  统计的，**推理必须用同一套**，否则 MLP 输入分布对不上。
- `torch.from_numpy(x)`：numpy → torch 张量；`.unsqueeze(0)` 加批维度成 `(1, 32)`；`self.mlp(...)`
  前向推理；`.item()` 取出标量。
- **sigmoid** `1/(1+e^(-logit))`：把 (-∞, +∞) 的 logit 压到 (0, 1)，变成"概率"。

### 8.3 `accept()` —— 判决

```python
def accept(self, wake_audio, command_audio):
    score = self.score(wake_audio, command_audio)
    return score >= self.threshold, score
```

返回一个**元组** `(是否接受, 分数)`。`return a, b` 是 Python 的"多返回值"——本质是打包成一个 tuple，
调用方 `accepted, gate_score = gate.accept(...)` 再**解包**。

---

## 9. 本课 Python 知识点速查表

| 知识点 | 出现位置 | 一句话说明 |
|---|---|---|
| docstring | 文件开头 | `"""..."""` 文档字符串，`help()` 可读 |
| `from __future__` | 文件头 | 提前启用未来特性 |
| `import x as y` / `from x import y` | 头部 | 两种导入方式 |
| 常量命名 | `SAMPLE_RATE` | 全大写 = 常量约定 |
| 类型注解 `a \| b` | 函数签名 | 参数可以是 str 或 Path |
| `with open(...) as f` | read_wav | 上下文管理器，自动关文件 |
| `raise ValueError` | read_wav | 快速失败 |
| 归一化 `/ 32768.0` | read_wav | int16 振幅 → [-1,1] |
| `np.linalg.norm` / `np.dot` | cosine | L2 范数 / 点积 |
| 防御 `<= 1e-8` | cosine | 防除零 |
| `range(start, stop, step)` | sliding | 步长循环 |
| 切片 `a[i:j]` | sliding | 左闭右开 |
| `np.zeros` + 切片赋值 | sliding | 补零 padding |
| 布尔掩码 `a >= 0.3` | _window_stats | 元素级比较出 True/False 数组 |
| `np.flatnonzero` / `.any()` | _window_stats | 找下标 / 判断是否存在 |
| 负索引 `[-3:]` | _window_stats | 取最大的 3 个 |
| `(mask).sum()` / `.mean()` | _window_stats | 数个数 / 算比例 |
| 下划线私有约定 | `_rms` 等 | `_` 开头 = 内部使用 |
| class / `self` / `__init__` | ModelScopeEmbedder | 面向对象三要素 |
| 列表推导式 | extract_waveforms | `[f(x) for x in xs]` |
| 三元表达式 | extract_waveforms | `A if 条件 else B` |
| `defaultdict(list)` | extract_waveforms | 自动初始化值的字典 |
| `enumerate` / `zip` | 多处 | 边遍历边带下标 / 配对遍历 |
| 预分配 `[None]*n` | extract_waveforms | 按位回填保持顺序 |
| `getattr` / `dir` | extract_waveforms | 反射：按名字取属性 |
| 字典推导式 | DualMLPGate | `{k: v for ...}` |
| `nn.Sequential` / `nn.Linear` | DualMLPGate | 搭神经网络 |
| z-score / sigmoid | score | 标准化 / 概率化 |
| 元组打包与解包 | accept | `return a, b` ↔ `x, y = f()` |

---

## 10. 练习题（建议动手写）

1. **读波形**：写一个函数，用 `read_wav` 读任意 wav，打印它的长度、`np.max`、`np.min`，并断言
   最大绝对值 ≤ 1.0（验证归一化）。
2. **滑窗**：对一个 `np.zeros(30000, dtype=np.float32)` 调 `sliding_windows`，数一数返回了多少个窗口，
   用 `len(audio) - window + 1` 手算对照，解释"尾巴"规则。
3. **余弦**：验证 `cosine_similarity(a, a)` ≈ 1.0，`cosine_similarity(a, -a)` ≈ -1.0，其中
   `a = np.array([1.0, 2.0, 3.0])`。
4. **defaultdict**：用普通 dict 和 defaultdict 分别实现"统计字符串里每个字符出现次数"，
   体会区别。
5. **布尔掩码**：对 `np.array([1, 5, 2, 8, 3])`，写出"大于 3 的元素下标"和"大于 3 的比例"
   的 numpy 表达式。

> 答案提示：2 题注意 `range(0, 30000-16000+1, 3200)` 的项数 + 尾巴是否补窗口；4 题普通 dict 需要
> `d.get(c, 0)`。

---

## 11. 小结与下一课预告

这一课你读完了整条流水线的**第一道关卡**：从读 wav → 滑窗 → 余弦特征 → 两个声纹模型的 32 维特征
→ 标准化 → MLP → sigmoid 概率 → 阈值判决。顺带过了一遍 list/dict/class/numpy 的实战用法。

**下一课（第 2 课）** 我们读 `szy_v1/pipeline.py`——它是"总指挥"，会调用本课的 `DualMLPGate`。
你将学到**工程化**的东西：怎么用 dataclass 管理配置、怎么用 argparse 写命令行、怎么在长任务里
断点续跑、怎么用 try/except 做"兜底回退"。第 3~5 课再把一条真实样本从头到尾走一遍，把
gate、infer、moss_candidate、pipeline 四个文件串起来。
