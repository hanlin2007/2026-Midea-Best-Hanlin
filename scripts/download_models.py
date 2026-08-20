#!/usr/bin/env python3
"""一键下载 SZY v1 所需的 3 个公开模型（ModelScope），并校验 sha256。

================================================================================
【这个脚本做什么？】

把 configs/szy_v1_firered.json 里指向的 3 个【公开】模型下载到仓库内 model/
目录的正确位置：

  1. iic/speech_eres2netv2_sv_zh-cn_16k-common   （ERes2NetV2 声纹前端）
  2. iic/speech_campplus_sv_zh-cn_16k-common     （CAM++ 声纹前端）
  3. dengcunqin/speech_mossformer2_separation_temporal_16k （MossFormer2 分离器，取 simple_model.onnx）

这 3 个模型都托管在 ModelScope，公开可下。

【还有 3 个资产脚本下不了，必须从队长工作区复制或另行获取】：
  A. dual MLP 门控权重 backend_dual.pt —— 队长训练出来的私有产物（模型根目录
     model/workspace/sv_backend/backend_dual_v2_out/backend_dual.pt）
  B. FireRedASR2-AED 模型 + FireRedASR2S 推理源码 —— 见 SETUP 指南第 5 节
  C. DatasetA 数据集 —— 你本地已有（dataset/ 目录，见 SETUP 指南第 4 节）

【用法】在仓库根目录执行：
  python scripts/download_models.py

【依赖】需要先安装 modelscope（本仓库 requirements.txt 已含）：
  pip install "modelscope>=1.17.0"
下载需要 git + git-lfs（大文件走 LFS 协议）。
================================================================================
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# 定位仓库根目录：本脚本在 <repo>/scripts/ 下，取上一级
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELSCOPE_ROOT = REPO_ROOT / "model" / "workspace" / "midea" / "models" / "modelscope"

# 三个公开模型及其落盘位置（与 configs/szy_v1_firered.json 一致）
MODELS = [
    {
        "id": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "target": MODELSCOPE_ROOT / "iic" / "speech_eres2netv2_sv_zh-cn_16k-common",
        "note": "ERes2NetV2 声纹前端",
    },
    {
        "id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "target": MODELSCOPE_ROOT / "iic" / "speech_campplus_sv_zh-cn_16k-common",
        "note": "CAM++ 声纹前端",
    },
    {
        "id": "dengcunqin/speech_mossformer2_separation_temporal_16k",
        "target": MODELSCOPE_ROOT / "dengcunqin" / "speech_mossformer2_separation_temporal_16k",
        "note": "MossFormer2 分离器（配置里取 simple_model.onnx）",
    },
]

# 来自 baseline manifest 的参考哈希（用于下载后校验文件是否完整一致）
MOSS_EXPECTED_SHA256 = "768f7a8aab85a8f75581bd296c9a53cd3d79d7d274d2a03383ab7b75a4748518"


def sha256(path: Path) -> str:
    """分块计算文件 SHA-256（与 pipeline._sha256 同款，防止大文件撑爆内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model_id: str, target: Path) -> None:
    """用 ModelScope snapshot_download 下载模型到指定目录（已存在则跳过）。"""
    if target.is_dir() and any(target.iterdir()):  # 目录非空 → 认为已下载
        print(f"[跳过] {model_id} 已存在：{target}")
        return
    try:
        from modelscope import snapshot_download  # lazy import：用到才导入
    except ImportError:
        print("[错误] 未安装 modelscope，请先执行：pip install \"modelscope>=1.17.0\"")
        raise SystemExit(1)
    print(f"[下载] {model_id} -> {target}")
    # local_dir 指定落盘目录：ModelScope 会把模型文件直接放进这个目录
    snapshot_download(model_id, local_dir=str(target))
    print(f"[完成] {model_id}")


def main() -> None:
    for item in MODELS:
        download(item["id"], item["target"])

    print("\n========== 校验 ==========")
    # 校验 MossFormer2 的 simple_model.onnx
    moss_onnx = MODELS[2]["target"] / "simple_model.onnx"
    if moss_onnx.is_file():
        got = sha256(moss_onnx)
        ok = got == MOSS_EXPECTED_SHA256
        print(f"moss simple_model.onnx sha256: {got}")
        print(f"  期望值: {MOSS_EXPECTED_SHA256}")
        print(f"  -> {'一致 ✓' if ok else '不一致 ✗（可能下载损坏或版本不同，请重新下载）'}")
    else:
        print(f"[警告] 未找到 {moss_onnx}，请确认 MossFormer2 模型目录结构（应含 simple_model.onnx）")

    print("\n========== 还需要你手动准备的资产 ==========")
    print("以下 3 项脚本无法自动下载，请按 docs/SETUP_本地复现指南.md 第 4~5 节准备：")
    print("  A. dual MLP 门控权重（队长私有产物）:")
    print(f"       model/workspace/sv_backend/backend_dual_v2_out/backend_dual.pt")
    print("       期望 sha256: 58eefbdaa75cf74d143cbb4d88af800c98b1ec1f487438b61928274f72d2ec6b")
    print("  B. FireRedASR2-AED 模型 + FireRedASR2S 源码（model/external/）")
    print("  C. DatasetA 数据集（dataset/dataset_test/datasetA/datasetA）——你本地已有")
    print("\n全部就绪后可运行：python scripts/check_models.py 做一次自检。")
    print("完整流程见：docs/SETUP_本地复现指南.md")


if __name__ == "__main__":
    main()
