"""midea_qiji 工具包：样本解析、文本归一化、声纹门控、ASR 后端、指标计算。

这个包是"工具箱"，被顶层流水线 szy_v1.pipeline 复用，也自带一条旧版基线
入口（python -m midea_qiji.infer）。核心模块：
  - infer.py         : Sample 数据类、JSONL 读取、文本归一化、编辑距离、ASR 封装、指标报告
  - gate.py          : ERes2NetV2 + CAM++ 双前端 MLP 声纹门控（SZY v1 主门控）
  - moss_candidate.py: MossFormer 候选路线的冻结决策规则（纯逻辑）
"""

__version__ = "1.0.0"
