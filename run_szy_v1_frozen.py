#!/usr/bin/env python3
"""SZY v1 冻结门控版一键启动器（消融用）：自动加 src/ 到模块路径并切到仓库根。

与 run_szy_v1.py 完全同构，唯一区别是最后导入的是 szy_v1.pipeline_frozen 的 main。

【用法】（从仓库根目录执行）

   datasetA 前 200 条 pos 的 CER 上限：
     python run_szy_v1_frozen.py --data-root dataset/dataset_test/datasetA/datasetA \
         --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl --limit 200

   datasetX expr / ctrl：
     python run_szy_v1_frozen.py --data-root dataset/dataset_test/datasetX/datasetX_sample_expr \
         --jsonl dataset/dataset_test/datasetX/datasetX_sample_expr/pos.jsonl
     python run_szy_v1_frozen.py --data-root dataset/dataset_test/datasetX/datasetX_sample_ctrl \
         --jsonl dataset/dataset_test/datasetX/datasetX_sample_ctrl/pos.jsonl

   （等价方案：PYTHONPATH=src python -m szy_v1.pipeline_frozen ...）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

from szy_v1.pipeline_frozen import main  # noqa: E402


if __name__ == "__main__":
    main()
