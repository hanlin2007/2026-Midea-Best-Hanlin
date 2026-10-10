"""core：目标说话人语音识别的基础内核（文本 / 音频 / 声纹 / 识别 / 决策规则）。

本包是全仓库分层架构的最底层（"内核层"）：上层代码（pipeline 系统层、
scripts 工具）都可以 import 它，但它自身绝不反向依赖上层。依赖只允许
向下，才能理清"谁被谁用"、避免循环导入，也方便把公共口径收敛到一处
（一处实现、全仓库一致）。

内核层的两条工程约定（写上层代码时同样适用）：
  - 轻量模块只依赖标准库或 numpy——import core.text 绝不拖进 torch；
    重型依赖（torch / modelscope / FireRed 包等）一律延迟到使用处
    （函数体内）再 import，让"只想用文本工具"的场景保持轻快；
  - 本包内含大量"冻结语义"（文本归一化口径、决策阈值、默认配置），
    它们是历史评测数字（CER/PR/RR）的复现基础，改动前必须先跑
    tests/ 里的语义锁测试。

分层职责（子模块按依赖从底向上）：

  - ``text.py``        文本归一化 / 编辑距离 / CER（仅标准库）；
  - ``phoneme.py``     拼音音素相似度与唤醒词别名表（pypinyin 可选）；
  - ``audio.py``       采样率常量与 WAV 读取（numpy / soundfile）；
  - ``model_paths.py`` 仓库内模型路径常量与本地模型解析（仅标准库）；
  - ``samples.py``     样本数据模型与 JSONL 清单读取（Sample / submission_id）；
  - ``nbest.py``       FireRed N-best 候选的保守 MBR 重排序；
  - ``embeddings.py``  声纹 embedding 提取（ModelScope 前端封装）；
  - ``firered.py``     FireRedASR2 识别后端（AED 生产 / LLM 可选）；
  - ``wake_anchor.py`` 唤醒锚点选择的冻结决策规则。

pipeline 系统层与 scripts 工具均构建在本内核之上。

用法：本包是"库"而非入口，按需导入子模块：

    from core.text import normalize_transcript, levenshtein
    from core.model_paths import resolve_model_path
    from core.wake_anchor import select_wake_anchor
"""
# 内核包版本号：冻结语义或接口变化时递增。它只描述本包（内核层），与
# pyproject.toml 中整个发行包的版本号是两回事。
__version__ = "1.0.0"
