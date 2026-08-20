#!/usr/bin/env bash
# =============================================================================
# SZY v1 运行脚本（已从队长工作区绝对路径改为仓库内相对路径，可整体搬迁）
#
# 用法（在仓库根目录执行，或任意目录执行，脚本会自动定位仓库根）：
#   ./scripts/run_szy_v1.sh                      # 全量运行
#   ./scripts/run_szy_v1.sh --resume             # 断点续跑
#   ./scripts/run_szy_v1.sh --limit 20 --output outputs/smoke_submission.json \
#        --final-output outputs/smoke_detail.json \
#        --checkpoint-output outputs/smoke_checkpoint.json \
#        --manifest-output outputs/smoke_manifest.json   # 冒烟测试
#
# 环境变量覆盖：
#   PYTHON_BIN    : Python 解释器（默认用 PATH 里的 python）
#   SZY_DEVICE    : 计算设备（默认 cuda:0）
#   CUDA_VISIBLE_DEVICES : 用哪张 GPU（默认 0）
#   MODELSCOPE_CACHE     : ModelScope 缓存（默认仓库内 model/.../modelscope）
#
# 前置条件：先完成 docs/SETUP_本地复现指南.md 的模型与数据准备。
# Windows 用户：请在 Git Bash 里运行本脚本，或用 PowerShell 执行等价命令
#   （见 SETUP 指南第 6 节），本脚本的 bash 语法 PowerShell 不认。
# =============================================================================
set -euo pipefail   # e=出错即停；u=未定义变量报错；o pipefail=管道任一环失败都算失败

# 定位仓库根目录：本脚本位于 <repo>/scripts/，取上一级
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 模型根目录（仓库内 model/）
model_root="${repo}/model"
# Python 解释器：优先用环境变量 PYTHON_BIN，否则用 PATH 里的 python
python_bin="${PYTHON_BIN:-python}"
# 计算设备：优先用环境变量 SZY_DEVICE，否则 cuda:0
device="${SZY_DEVICE:-cuda:0}"

cd "${repo}"   # 切到仓库根目录（所有相对路径都以它为准）

# 导出运行环境变量，然后启动主流程（python -m szy_v1.pipeline）
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONPATH="${repo}/src" \
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${model_root}/workspace/midea/models/modelscope}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
"${python_bin}" -u -m szy_v1.pipeline \
  --data-root "${repo}/dataset/dataset_test/datasetA/datasetA" \
  --jsonl "${repo}/dataset/dataset_test/datasetA/datasetA/pos.jsonl" \
          "${repo}/dataset/dataset_test/datasetA/datasetA/neg.jsonl" \
  --config "${repo}/configs/szy_v1_firered.json" \
  --output "${repo}/outputs/szy_v1_submission.json" \
  --final-output "${repo}/outputs/szy_v1_detail.json" \
  --checkpoint-output "${repo}/outputs/szy_v1_checkpoint.json" \
  --manifest-output "${repo}/outputs/szy_v1_manifest.json" \
  "$@"   # 把脚本自己的额外参数原样透传给 python（如 --resume、--limit）
