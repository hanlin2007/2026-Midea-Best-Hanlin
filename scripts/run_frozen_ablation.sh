#!/usr/bin/env bash
# =============================================================================
# 冻结门控消融：跳过 MLP 主门控，只放行 pos 样本进入 MossFormer2 + 差分声纹链路。
# 依次对三个数据切分跑一遍，最后打印各自 CER 上限。
#   datasetA 前 200 条 pos / datasetX_sample_expr / datasetX_sample_ctrl
#
# 用法（在仓库根目录或任意目录执行，脚本会自动定位仓库根）：
#   bash scripts/run_frozen_ablation.sh
#
# 前置：先激活你的 conda 环境（含 torch/modelscope/onnxruntime 等依赖），例如：
#   conda activate szy_v1
#
# 环境变量覆盖（与 scripts/run_szy_v1.sh 一致）：
#   PYTHON_BIN    : Python 解释器（默认 PATH 里的 python）
#   SZY_DEVICE    : 计算设备（默认 cuda:0）
#   CUDA_VISIBLE_DEVICES : 用哪张 GPU（默认 0）
# =============================================================================
set -euo pipefail   # e=出错即停；u=未定义变量报错；o pipefail=管道任一环失败都算失败

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${repo}/model"
python_bin="${PYTHON_BIN:-python}"
device="${SZY_DEVICE:-cuda:0}"

cd "${repo}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${repo}/src"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${model_root}/workspace/midea/models/modelscope}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

run_one() {
  local name="$1"; shift
  echo ""
  echo "============================================================"
  echo "[frozen] ${name}"
  echo "============================================================"
  "${python_bin}" -u -m szy_v1.pipeline_frozen "$@"
}

# ① datasetA 前 200 条 pos —— CER 上限（主对照）
run_one "datasetA 前200 pos" \
  --data-root "${repo}/dataset/dataset_test/datasetA/datasetA" \
  --jsonl "${repo}/dataset/dataset_test/datasetA/datasetA/pos.jsonl" \
  --limit 200 \
  --config "${repo}/configs/szy_v1_firered.json" \
  --output "${repo}/outputs/frozen_datasetA_200_submission.json" \
  --final-output "${repo}/outputs/frozen_datasetA_200_detail.json" \
  --checkpoint-output "${repo}/outputs/frozen_datasetA_200_checkpoint.json" \
  --manifest-output "${repo}/outputs/frozen_datasetA_200_manifest.json"

# ② datasetX expr（70 pos）—— 实验组
run_one "datasetX expr" \
  --data-root "${repo}/dataset/dataset_test/datasetX/datasetX_sample_expr" \
  --jsonl "${repo}/dataset/dataset_test/datasetX/datasetX_sample_expr/pos.jsonl" \
  --config "${repo}/configs/szy_v1_firered.json" \
  --output "${repo}/outputs/frozen_datasetX_expr_submission.json" \
  --final-output "${repo}/outputs/frozen_datasetX_expr_detail.json" \
  --checkpoint-output "${repo}/outputs/frozen_datasetX_expr_checkpoint.json" \
  --manifest-output "${repo}/outputs/frozen_datasetX_expr_manifest.json"

# ③ datasetX ctrl（70 pos）—— 对照组
run_one "datasetX ctrl" \
  --data-root "${repo}/dataset/dataset_test/datasetX/datasetX_sample_ctrl" \
  --jsonl "${repo}/dataset/dataset_test/datasetX/datasetX_sample_ctrl/pos.jsonl" \
  --config "${repo}/configs/szy_v1_firered.json" \
  --output "${repo}/outputs/frozen_datasetX_ctrl_submission.json" \
  --final-output "${repo}/outputs/frozen_datasetX_ctrl_detail.json" \
  --checkpoint-output "${repo}/outputs/frozen_datasetX_ctrl_checkpoint.json" \
  --manifest-output "${repo}/outputs/frozen_datasetX_ctrl_manifest.json"

echo ""
echo "[frozen] 三个切分全部完成，汇总 CER 见各自输出末尾的 final_cer。"
