#!/usr/bin/env bash
# 比赛推理启动脚本。所有路径均可通过同名环境变量覆盖。
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_CACHE="${WORKSPACE_DIR}/model/modelscope"
# 不继承宿主机 PYTHONPATH，防止意外引用其他工作区的可编辑源码。
export PYTHONPATH="${WORKSPACE_DIR}/src:${WORKSPACE_DIR}/vendor:${WORKSPACE_DIR}/third_party/FireRedASR2S/examples_infer/asr"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-${WORKSPACE_DIR}/config/runtime.json}"
DATA_ROOT="${DATA_ROOT:-/data}"
TEST_JSON="${TEST_JSON:-${DATA_ROOT}/test.jsonl}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_DIR}/model}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-${WORKSPACE_DIR}/outputs/result.json}"
PROFILE="${PROFILE:-0}"

command=("${PYTHON_BIN}" "${WORKSPACE_DIR}/run_test.py"
  --config "${CONFIG}"
  --data-root "${DATA_ROOT}"
  --test-json "${TEST_JSON}"
  --model-path "${MODEL_PATH}"
  --device "${DEVICE}"
  --output "${OUTPUT}")

if [[ "${PROFILE}" == "1" ]]; then
  command+=(--profile)
fi

printf '[demo] 执行:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
