#!/usr/bin/env bash
# szy_v4：szy_v2 全管线 + dasynth 微调 MossFormer2 分离前端（torch .pt）。
# 与 run_szy_v2.sh 的唯一差异在配置（分离器与独立缓存目录）。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${repo}/model"
python_bin="${PYTHON_BIN:-python}"

cd "${repo}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
PYTHONPATH="${repo}/src:${model_root}/runtime/vendor" \
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${model_root}/workspace/midea/models/modelscope}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
"${python_bin}" -u -m szy_v4.pipeline \
  --data-root "${repo}/dataset/dataset_test/datasetA/datasetA" \
  --jsonl "${repo}/dataset/dataset_test/datasetA/datasetA/pos.jsonl" \
          "${repo}/dataset/dataset_test/datasetA/datasetA/neg.jsonl" \
  --config "${repo}/configs/szy_v4_firered.json" \
  --output "${repo}/outputs/szy_v4_submission.json" \
  --final-output "${repo}/outputs/szy_v4_detail.json" \
  --checkpoint-output "${repo}/outputs/szy_v4_checkpoint.json" \
  --manifest-output "${repo}/outputs/szy_v4_manifest.json" \
  "$@"
