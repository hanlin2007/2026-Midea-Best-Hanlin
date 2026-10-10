"""midea_qiji：DatasetA 基线内核（门控 + 识别后端 + 决策规则 + CLI）。

本包是全仓库分层架构的最底层（"内核层"）：上层代码（共享层 szy_common、
阶段层 szy_v1/v2/v4、scripts 工具）都可以 import 它，但它自身绝不反向
依赖上层。依赖只允许向下，才能理清"谁被谁用"、避免循环导入，也方便把
公共口径收敛到一处（一处实现、全仓库一致）。

内核层的两条工程约定（写上层代码时同样适用）：
  - 轻量模块只依赖标准库或 numpy——import midea_qiji.text 绝不拖进
    torch；重型依赖（torch / modelscope / onnxruntime 等）一律延迟到
    使用处（函数体内）再 import，让"只想用文本工具"的场景保持轻快；
  - 本包内含大量"冻结语义"（文本归一化口径、决策阈值、默认配置），
    它们是历史评测数字（CER/PR/RR）的复现基础，改动前必须先跑
    tests/ 里的语义锁测试（test_infer / test_moss_candidate）。

分层职责（子模块按依赖从底向上）：

  - ``text.py``         文本归一化 / 编辑距离 / CER（仅标准库）；
  - ``phoneme.py``      拼音音素相似度与唤醒词别名表（pypinyin 可选）；
  - ``firered_nbest.py`` FireRed N-best 候选的保守 MBR 重排序；
  - ``audio.py``        采样率常量与 WAV 读取（numpy / soundfile）；
  - ``paths.py``        仓库内模型路径常量与本地模型解析（仅标准库）；
  - ``recognizers.py``  Qwen3-ASR / FireRed AED / FireRed LLM / 条件 LoRA
                        四个识别后端（重型依赖全部延迟导入）；
  - ``gate.py``         声纹 embedding 与双前端 MLP 门控；
  - ``moss_candidate.py`` 唤醒锚点选择与指令候选切换的冻结决策规则；
  - ``infer.py``        baseline 编排与 CLI（配置/断点/批量调度/报告）。

SZY 系统管线与 scripts 工具均构建在本内核之上。

用法：本包是"库"而非入口，一般不直接运行顶层包，而是按需导入子模块：

    from midea_qiji.text import normalize_transcript, cer
    from midea_qiji.paths import resolve_model_path
    from midea_qiji.moss_candidate import should_switch_command_timbre

唯一的命令行入口在 infer.py（``python -m midea_qiji.infer``），需要模型
资产；纯文本/规则函数不碰模型，可以直接在测试里调用。
"""
# 内核包版本号：冻结语义或接口变化时递增。它只描述本包（内核层），与
# pyproject.toml 中整个发行包（SZY 系统）的版本号（4.0.0）是两回事。
__version__ = "1.0.0"
