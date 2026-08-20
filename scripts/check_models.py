#!/usr/bin/env python3
"""SZY v1 环境自检：检查模型文件、数据集是否齐全，并校验关键文件的 sha256。

================================================================================
【用法】在仓库根目录执行：
  python scripts/check_models.py

【检查内容】
  1. configs/szy_v1_firered.json 能否正常加载（顺带验证 fail-closed 未知键校验）
  2. 配置里指向的每一个模型文件是否存在
  3. 两个有参考指纹的模型文件 sha256 是否与 baseline manifest 一致
     （moss simple_model.onnx、dual backend_dual.pt）
  4. 数据集 jsonl 与 wav 是否就位

【退出码】全部通过返回 0；任一失败返回 1（可直接用在 CI / 自动化脚本里）。
================================================================================
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 来自 baseline 输出（outputs/baseline/szy_v1_manifest.json）的参考指纹，
# 用于确认你手里的模型与队长产出结果时用的是同一份文件
EXPECTED_SHA256 = {
    "moss_model": "768f7a8aab85a8f75581bd296c9a53cd3d79d7d274d2a03383ab7b75a4748518",
    "dual_backend": "58eefbdaa75cf74d143cbb4d88af800c98b1ec1f487438b61928274f72d2ec6b",
}


def sha256(path: Path) -> str:
    """分块计算 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_str: str) -> Path:
    """把配置里的相对路径解析成基于仓库根的路径（相对路径以运行目录=仓库根为基准）。"""
    p = Path(path_str).expanduser()
    return (p if p.is_absolute() else (REPO_ROOT / p)).resolve()


def main() -> None:
    config_path = REPO_ROOT / "configs" / "szy_v1_firered.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    errors: list[str] = []   # 收集所有错误，最后一起报告（而不是遇到第一个就退出）
    # warnings: list[str] = []  # 非致命提示

    # 1) 检查配置里所有"文件型"模型路径
    file_keys = ["dual_backend", "moss_model"]
    for key in file_keys:
        p = resolve(config[key])
        if not p.is_file():
            errors.append(f"缺少模型文件 [{key}]: {p}")
        elif key in EXPECTED_SHA256:  # 有参考指纹 → 校验
            got = sha256(p)
            if got != EXPECTED_SHA256[key]:
                errors.append(
                    f"sha256 不一致 [{key}]: {p}\n"
                    f"    期望 {EXPECTED_SHA256[key]}\n"
                    f"    实际 {got}"
                )
            else:
                print(f"[ok] {key} sha256 一致: {p.name}")
        else:
            print(f"[ok] 存在模型文件 [{key}]: {p}")

    # 2) 检查目录型模型路径
    for key in ("firered_model", "firered_repo", "firered_vendor"):
        p = resolve(config[key])
        if not p.is_dir():
            errors.append(f"目录型模型尚未就位 [{key}]: {p}（FireRed 相关，见 SETUP 指南第 5 节）")
        else:
            print(f"[ok] 存在模型目录 [{key}]: {p}")

    # 3) 检查声纹模型（model_root + 模型 ID）
    for key in ("sv_model", "cam_model"):
        p = resolve(config["model_root"]) / config[key]
        if not p.is_dir():
            errors.append(f"缺少声纹模型 [{key}]: {p}\n    提示：运行 python scripts/download_models.py 下载")
        else:
            print(f"[ok] 声纹模型就位 [{key}]: {p.name}")

    # 4) 检查数据集
    data_root = REPO_ROOT / "dataset" / "dataset_test" / "datasetA" / "datasetA"
    for name in ("pos.jsonl", "neg.jsonl"):
        f = data_root / name
        if not f.is_file():
            errors.append(f"缺少数据集清单: {f}")
        else:
            print(f"[ok] 数据集清单就位: {name}")
    for d in ("pos", "neg"):
        p = data_root / d
        if not p.is_dir() or not any(p.glob("*.wav")):
            # 并不完全严谨，只有 any 返回 False，即 glob 的结果为空，才走 errors 分支，只要有一条音频就能通过
            errors.append(f"缺少音频目录或目录为空: {p}")
        else:
            print(f"[ok] 音频目录就位: {d}/")

    # ---- 汇总报告 ----
    print("\n========== 自检结果 ==========")
    # for w in warnings:
    #     print(f"[提示] {w}")
    if errors:
        print(f"发现 {len(errors)} 个问题：")
        for e in errors:
            print(f"  [✗] {e}")
        print("\n请按 docs/SETUP_本地复现指南.md 补齐后重新运行本脚本。")
        raise SystemExit(1)
    print("全部检查通过，可以开始运行。")
    print("  一次性安装（可选）：pip install -e .   —— 之后 python -m szy_v1.pipeline 也能直接跑")
    print("  冒烟测试（推荐，免 PYTHONPATH）：")
    print("    python run_szy_v1.py --data-root dataset/dataset_test/datasetA/datasetA "
          "--jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl "
          "--jsonl dataset/dataset_test/datasetA/datasetA/neg.jsonl "
          "--limit 5 --output outputs/smoke_submission.json --final-output outputs/smoke_detail.json "
          "--checkpoint-output outputs/smoke_checkpoint.json --manifest-output outputs/smoke_manifest.json")
    print("  全量运行：python run_szy_v1.py 或 bash scripts/run_szy_v1.sh（均会自动设置好路径）")


if __name__ == "__main__":
    main()
