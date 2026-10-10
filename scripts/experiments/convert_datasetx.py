#!/usr/bin/env python3
"""把 datasetX 的 metadata.jsonl 转成 datasetA 兼容的 pos.jsonl + neg.jsonl。

每条 metadata 生成 1 条样本，音频路径指向 mix/mix_*.wav；输出目录下
pos/ neg/ 以符号链接指向真实数据目录。

用法：
    python scripts/experiments/convert_datasetx.py \
        --datasetx-root /path/to/output_30000 \
        --output-dir dataset/datasetX
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(datasetx_root: Path, output_dir: Path) -> None:
    metadata_path = datasetx_root / "metadata.jsonl"

    pos_rows = []
    neg_rows = []

    with metadata_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            group_id = d["group_id"]
            split = d["label"]  # "pos" or "neg"
            wake_text = d["wake_text"]
            cmd_text = d["cmd_text"] if d["cmd_text"] else None

            group_dir = f"group_{group_id:04d}"

            sample = {
                "id": f"{split}_{group_id}",
                "唤醒音频": f"{split}/{group_dir}/mix/mix_kws.wav",
                "唤醒文本": wake_text,
                "识别音频": f"{split}/{group_dir}/mix/mix_cmd.wav",
                "识别文本": cmd_text,
            }

            if split == "pos":
                pos_rows.append(sample)
            else:  # neg
                neg_rows.append(sample)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_path = output_dir / "pos.jsonl"
    with pos_path.open("w", encoding="utf-8") as f:
        for row in pos_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    neg_path = output_dir / "neg.jsonl"
    with neg_path.open("w", encoding="utf-8") as f:
        for row in neg_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Create symlink to actual data
    for split in ["pos", "neg"]:
        link = output_dir / split
        target = datasetx_root / split
        if link.exists():
            link.unlink()
        link.symlink_to(target)

    print(f"Converted {len(pos_rows)} positive samples → {pos_path}")
    print(f"Converted {len(neg_rows)} negative samples → {neg_path}")
    print(f"Created symlinks: {output_dir}/pos → {datasetx_root}/pos")
    print(f"Created symlinks: {output_dir}/neg → {datasetx_root}/neg")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert datasetX metadata to pos/neg jsonl")
    ap.add_argument("--datasetx-root", required=True, help="datasetX 生成输出目录（含 metadata.jsonl 与 pos/ neg/）")
    ap.add_argument("--output-dir", required=True, help="转换结果输出目录（如 dataset/datasetX）")
    args = ap.parse_args()
    convert(Path(args.datasetx_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
