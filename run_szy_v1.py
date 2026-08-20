#!/usr/bin/env python3
"""SZY v1 一键启动器：自动把 src/ 加入模块搜索路径并切到仓库根目录。

================================================================================

【为什么需要这个文件？】

项目采用 src/ 布局：真正的包在 src/midea_qiji 和 src/szy_v1 里。直接执行

    python -m szy_v1.pipeline

会报 ModuleNotFoundError，因为 Python 默认不会去 src/ 里找包（除非你设了
PYTHONPATH=src 或执行了 pip install -e .）

（也可用 bash scripts/run_szy_v1.sh，它内部做了同样的事。）

另一种等价方案（一次性）：pip install -e .  之后 python -m szy_v1.pipeline 也能直接跑。



本启动器做两件事，让你在任何目录都能跑：
  1. sys.path.insert(0, "src")  —— 让 import midea_qiji / import szy_v1 找得到
  2. os.chdir(仓库根)            —— 让配置里的相对路径（model/...、outputs/...）都按仓库根解析

【用法】（等价于 python -m szy_v1.pipeline，但无需任何环境变量）

  冒烟测试：
    python run_szy_v1.py --data-root dataset/dataset_test/datasetA/datasetA \
        --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl \
        --jsonl dataset/dataset_test/datasetA/datasetA/neg.jsonl \
        --limit 5 --output outputs/smoke_submission.json \
        --final-output outputs/smoke_detail.json \
        --checkpoint-output outputs/smoke_checkpoint.json \
        --manifest-output outputs/smoke_manifest.json
dataset/dataset_test/datasetX/
  全量运行：
    python run_szy_v1.py --data-root dataset/dataset_test/datasetA/datasetA 
        --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl \
        --jsonl dataset/dataset_test/datasetA/datasetA/neg.jsonl

    python run_szy_v1.py --data-root dataset/dataset_test/datasetX/datasetX_sample_expr
        --jsonl dataset/dataset_test/datasetX_sample_expr/pos.jsonl \
        --jsonl dataset/dataset_test/datasetX_sample_expr/neg.jsonl
================================================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 定位仓库根目录：本文件就在仓库根，__file__ 的父目录即仓库根
REPO_ROOT = Path(__file__).resolve().parent

# ① 把 src/ 塞到模块搜索路径最前面（优先于其它同名包）
sys.path.insert(0, str(REPO_ROOT / "src"))

# ② 切到仓库根目录：保证 configs/、model/、outputs/ 这些相对路径按仓库根解析
os.chdir(REPO_ROOT)

# 延迟导入：必须在 sys.path 修改之后才能 import 到 szy_v1
from szy_v1.pipeline import main  # noqa: E402  （E402=模块级导入位置告警，这里故意延迟）


if __name__ == "__main__":
    main()
