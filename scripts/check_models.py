#!/usr/bin/env python3
"""Verify the model registry and detect residual model files outside model/."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from relocate_workspace_models import (
    EXCLUDED_PARTS,
    EXCLUDED_TOP_LEVEL,
    WEIGHT_SCAN_ROOTS,
    WEIGHT_SUFFIXES,
    workspace_root,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_stats(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
            count += 1
    return total, count


def is_excluded(relative: Path) -> bool:
    if not relative.parts or relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    return any(part in EXCLUDED_PARTS or part.startswith(".venv-") for part in relative.parts)


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
