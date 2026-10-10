#!/usr/bin/env bash
# SZY 生产管线 + dasynth 微调 MossFormer2 分离前端（torch .pt）。
# 直接可跑，环境变量见文件内注释。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${repo}/model"
python_bin="${PYTHON_BIN:-python}"

cd "${repo}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
PYTHONPATH="${repo}/src:${model_root}/runtime/vendor" \
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${model_root}/workspace/midea/models/modelscope}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
"${python_bin}" -u -m pipeline.cli \
  --data-root "${repo}/dataset/dataset_test/datasetA/datasetA" \
  --jsonl "${repo}/dataset/dataset_test/datasetA/datasetA/pos.jsonl" \
          "${repo}/dataset/dataset_test/datasetA/datasetA/neg.jsonl" \
  --config "${repo}/configs/pipeline.json" \
  --output "${repo}/outputs/submission.json" \
  --final-output "${repo}/outputs/detail.json" \
  --checkpoint-output "${repo}/outputs/checkpoint.json" \
  --manifest-output "${repo}/outputs/manifest.json" \
  "$@"
