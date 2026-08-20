#!/usr/bin/env python3
"""Move workspace model assets into ../model and leave compatibility links."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


WEIGHT_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".hdf5", ".joblib", ".model", ".onnx",
    ".pb", ".pt", ".pth", ".safetensors", ".tflite",
}
EXCLUDED_TOP_LEVEL = {"dataset", "dataset_probe", "external_data", "midea-best", "model"}
EXCLUDED_PARTS = {".git", ".vendor_transformers"}
WEIGHT_SCAN_ROOTS = (
    "exp_sep/moss_anchor_external_v1",
    "midea/outputs",
    "midea/outputs3",
    "midea/tmp",
    "sv_asr",
    "sv_backend",
    "sv_finetune",
    "sv_probe",
    "sv_selector/selector.pt",
    "sv_tse",
    "sv_firered",
)
BUNDLE_PATHS = (
    "midea/models",
    "midea/outputs3/qwen_target_lora/checkpoint-1250",
    "midea/outputs3/qwen_target_lora/checkpoint-1275",
    "midea/outputs3/qwen_target_lora/final_adapter",
    "midea/outputs3/qwen_target_lora_smoke/checkpoint-10",
    "midea/outputs3/qwen_target_lora_smoke/final_adapter",
    "sv_asr/out_full/best_adapter",
    "sv_asr/out_full/last_adapter",
    "sv_asr/out_probe/best_adapter",
    "sv_asr/out_probe/last_adapter",
    "sv_firered/out/best",
    "sv_firered/out/last",
)
EXTERNAL_LINKS = {
    "external/FireRedASR2-AED": "/home/hustdun/models/FireRedASR2-AED",
    "external/FireRedASR2S": "/home/hustdun/zhy_workspace/FireRedASR2S",
    "runtime/vendor": "vendor",
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_excluded(relative: Path) -> bool:
    if not relative.parts or relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    return any(part in EXCLUDED_PARTS or part.startswith(".venv-") for part in relative.parts)


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_model_files(root: Path) -> Iterable[Path]:
    for relative_root in WEIGHT_SCAN_ROOTS:
        scan_root = root / relative_root
        if not scan_root.exists():
            continue
        paths = (scan_root,) if scan_root.is_file() else scan_root.rglob("*")
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if is_excluded(relative):
                continue
            if path.suffix.lower() in WEIGHT_SUFFIXES:
                yield path


def directory_stats(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
            count += 1
    return total, count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_assets(root: Path) -> list[Path]:
    bundles = [root / relative for relative in BUNDLE_PATHS]
    missing = [str(path.relative_to(root)) for path in bundles if not path.exists()]
    if missing:
        raise FileNotFoundError(f"declared model bundles missing: {missing}")

    files = [
        path for path in iter_model_files(root)
        if not any(is_under(path, bundle) for bundle in bundles)
    ]
    return sorted([*bundles, *files], key=lambda path: str(path.relative_to(root)))


def record_for(source: Path, destination: Path, root: Path) -> dict[str, object]:
    relative = source.relative_to(root)
    if destination.is_dir():
        size, file_count = directory_stats(destination)
        digest = None
        kind = "directory"
    else:
        size = destination.stat().st_size
        file_count = 1
        digest = sha256(destination)
        kind = "file"
    return {
        "original_path": str(relative),
        "canonical_path": str(destination.relative_to(root)),
        "kind": kind,
        "bytes": size,
        "file_count": file_count,
        "sha256": digest,
    }


def ensure_external_links(root: Path) -> list[dict[str, str]]:
    model_root = root / "model"
    records = []
    for relative, source_value in EXTERNAL_LINKS.items():
        link = model_root / relative
        source = (root / source_value).resolve() if not source_value.startswith("/") else Path(source_value)
        if not source.exists():
            raise FileNotFoundError(f"external runtime dependency missing: {source}")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != source.resolve():
                raise FileExistsError(f"external link conflicts with expected target: {link}")
        else:
            link.symlink_to(os.path.relpath(source, start=link.parent), target_is_directory=source.is_dir())
        records.append({"link": str(link.relative_to(root)), "target": str(source)})
    return records


def plan(root: Path) -> list[dict[str, object]]:
    model_root = root / "model" / "workspace"
    result = []
    for source in build_assets(root):
        destination = model_root / source.relative_to(root)
        if source.is_symlink():
            status = "already_linked" if destination.exists() and source.resolve() == destination.resolve() else "conflict"
        elif destination.exists() or destination.is_symlink():
            status = "conflict"
        else:
            status = "ready"
        result.append({
            "source": str(source.relative_to(root)),
            "destination": str(destination.relative_to(root)),
            "kind": "directory" if source.is_dir() else "file",
            "status": status,
        })
    return result


def apply(root: Path) -> dict[str, object]:
    entries = plan(root)
    conflicts = [entry for entry in entries if entry["status"] == "conflict"]
    if conflicts:
        raise RuntimeError(f"migration conflicts: {conflicts}")
    records = []
    for entry in entries:
        source = root / str(entry["source"])
        destination = root / str(entry["destination"])
        if entry["status"] == "ready":
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            source.symlink_to(
                os.path.relpath(destination, start=source.parent),
                target_is_directory=destination.is_dir(),
            )
        records.append(record_for(source, destination, root))

    external = ensure_external_links(root)
    manifest = {
        "schema": "szy_workspace_model_registry/v1",
        "workspace": str(root),
        "asset_count": len(records),
        "assets": records,
        "external_dependencies": external,
    }
    manifest_path = root / "model" / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="move assets after printing the exact plan")
    args = parser.parse_args()
    root = workspace_root()
    if args.apply:
        manifest = apply(root)
        print(json.dumps({"asset_count": manifest["asset_count"], "manifest": "model/MANIFEST.json"}, ensure_ascii=False))
    else:
        print(json.dumps(plan(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
