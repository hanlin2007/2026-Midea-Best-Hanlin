"""szy_v1 包：SZY v1 端到端目标说话人 ASR 主流程（当前最佳路线）。

入口：python -m szy_v1.pipeline（命令行参数见 pipeline.build_parser）。
运行前请先按 docs/SETUP_本地复现指南.md 准备好模型与数据。
"""

__all__ = ["__version__"]  # 控制 from szy_v1 import * 时导出什么（这里只导出版本号）
__version__ = "1.0.0"
