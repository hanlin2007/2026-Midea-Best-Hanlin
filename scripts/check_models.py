#!/usr/bin/env python3
"""模型资产完整性自检：按 model/MANIFEST.json 校验资产 + 发现 model/ 外残留权重。

校验内容：canonical 资产存在、兼容软链指向正确、目录字节数/文件数一致、
文件 sha256 一致、外部依赖软链存在；随后扫描工作区，报告未纳入 model/
管理的残留权重文件。哈希/目录统计/排除规则与资产搬迁工具
relocate_workspace_models.py 共用同一实现，避免双份口径漂移。

用法（仓库根目录）：
  python scripts/check_models.py
"""
from __future__ import annotations

import json
import sys

from relocate_workspace_models import (
    WEIGHT_SCAN_ROOTS,
    WEIGHT_SUFFIXES,
    directory_stats,
    is_excluded,
    sha256,
    workspace_root,
)


def main() -> None:
    root = workspace_root()
    manifest_path = root / "model" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for asset in manifest["assets"]:
        original = root / asset["original_path"]
        canonical = root / asset["canonical_path"]
        if not canonical.exists():
            errors.append(f"missing canonical asset: {canonical}")
            continue
        if not original.is_symlink():
            errors.append(f"original path is not a compatibility link: {original}")
        elif original.resolve() != canonical.resolve():
            errors.append(f"compatibility link points elsewhere: {original}")
        if canonical.is_dir():
            size, count = directory_stats(canonical)
            if size != asset["bytes"] or count != asset["file_count"]:
                errors.append(f"directory stat mismatch: {canonical}")
        else:
            if canonical.stat().st_size != asset["bytes"]:
                errors.append(f"file size mismatch: {canonical}")
            if asset["sha256"] and sha256(canonical) != asset["sha256"]:
                errors.append(f"file hash mismatch: {canonical}")
    for dependency in manifest["external_dependencies"]:
        link = root / dependency["link"]
        if not link.is_symlink() or not link.exists():
            errors.append(f"missing external dependency link: {link}")
    residual = []
    for relative_root in WEIGHT_SCAN_ROOTS:
        scan_root = root / relative_root
        if not scan_root.exists():
            continue
        paths = (scan_root,) if scan_root.is_file() else scan_root.rglob("*")
        for path in paths:
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            relative = path.relative_to(root)
            if not is_excluded(relative):
                residual.append(str(relative))
    if residual:
        errors.append(f"residual non-linked model files outside model/: {residual}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"asset_count": manifest["asset_count"], "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
