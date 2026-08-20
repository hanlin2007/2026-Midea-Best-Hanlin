"""midea_qiji.paths：仓库内模型路径常量与本地模型解析。

``resolve_model_path`` 是唯一合法的"模型 id → 本地路径"解析入口：优先直接
路径，其次 --model-root / MODELSCOPE_CACHE，最后 models/modelscope 与
modelscope hub 缓存，并兼容 ModelScope 的 ``. → ___`` 目录转义。

PACKAGE_MODEL_ROOT 指向仓库内的 ``model/`` 目录，FireRed 三件套
（模型 / 源码仓库 / vendor）默认位于其下。

依赖：仅标准库（json 不需要；os / pathlib）。
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_MODEL_ROOT = Path(__file__).resolve().parents[2].parent / "model"
DEFAULT_FIRED_MODEL = str(PACKAGE_MODEL_ROOT / "external" / "FireRedASR2-AED")
DEFAULT_FIRED_REPO = str(PACKAGE_MODEL_ROOT / "external" / "FireRedASR2S")
DEFAULT_FIRED_VENDOR = PACKAGE_MODEL_ROOT / "runtime" / "vendor"


def resolve_model_path(model: str | Path, model_root: str | Path | None) -> Path:
    """把模型 id（如 ``iic/speech_eres2netv2_sv_zh-cn_16k-common``）解析为本地路径。"""
    direct = Path(model)
    if direct.exists():
        return direct.resolve()
    roots = []
    if model_root is not None:
        roots.append(Path(model_root))
    if os.environ.get("MODELSCOPE_CACHE"):
        roots.append(Path(os.environ["MODELSCOPE_CACHE"]))
    roots.extend([Path("models/modelscope"), Path.home() / ".cache" / "modelscope" / "hub"])
    for root in roots:
        for candidate in (
            root / str(model),
            root / str(model).replace(".", "___"),
            root / "models" / str(model),
            root / "models" / str(model).replace(".", "___"),
        ):
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(
        f"local model {model!r} was not found; set --model-root or MODELSCOPE_CACHE"
    )
