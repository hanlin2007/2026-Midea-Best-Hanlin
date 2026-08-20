"""midea_qiji：DatasetA 基线内核（门控 + 识别后端 + 决策规则 + CLI）。

分层职责（子模块按依赖从底向上）：

  - ``text.py``         文本归一化 / 编辑距离 / CER（仅标准库）；
  - ``audio.py``        采样率常量与 WAV 读取（numpy / soundfile）；
  - ``paths.py``        仓库内模型路径常量与本地模型解析（仅标准库）；
  - ``recognizers.py``  Qwen3-ASR / FireRed AED / FireRed LLM / 条件 LoRA
                        四个识别后端（重型依赖全部延迟导入）；
  - ``gate.py``         声纹 embedding 与双前端 MLP 门控；
  - ``moss_candidate.py`` 唤醒锚点选择与指令候选切换的冻结决策规则；
  - ``infer.py``        baseline 编排与 CLI（配置/断点/批量调度/报告）。

SZY v1/v2/v4 各阶段管线与 scripts 工具均构建在本内核之上。
"""
__version__ = "1.0.0"
