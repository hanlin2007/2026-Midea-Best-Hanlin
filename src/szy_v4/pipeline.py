"""szy_v4 产线入口：复用 szy_v2 主流程，配置指向微调 .pt 分离器即启用。

szy_v4 与 szy_v2 的全部差异仅在音频前端：配置 ``moss_model`` 指向
``szy_v4.train_mossformer2`` 产出的 .pt 时，szy_v2 运行器自动改用
``szy_v4.separator.MossFormer2TorchSeparator``（见 szy_v2.pipeline 中
按扩展名选择分离器实现的分支）。门控/触发/选源/回退逻辑零改动。

用法见 scripts/run_szy_v4.sh。
"""
from __future__ import annotations

from szy_v2.pipeline import main

if __name__ == "__main__":
    main()
