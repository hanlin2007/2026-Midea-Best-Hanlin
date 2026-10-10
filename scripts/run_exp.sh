#!/usr/bin/env bash
# 命名实验运行入口：把一次评测的所有产物收进 outputs/<NAME>/ 并自动生成报告。
#
# 与 scripts/run.sh 的分工：
#   run.sh     —— 固定输出到 outputs/ 根（submission/detail/...），日常快速验证；
#   run_exp.sh —— 每次运行一个独立目录（outputs/<NAME>/<NAME>_*.json + _report.txt），
#                 实验对比、断点续跑、报告归档都用它。
#
# 用法（仓库根目录）：
#   bash scripts/run_exp.sh <NAME> [--dataset A|X] [--config F] [--limit N] [--resume] [--gpu G]
#
#   <NAME>      运行名；缺省自动生成 run_YYYYMMDD_HHMMSS
#   --dataset   数据集：A（默认，dataset_test/datasetA/datasetA）或 X（dataset/datasetX）
#   --config    配置基准（默认 configs/pipeline.json）；会复制一份并把
#               separation_cache 改指 outputs/<NAME>/cache/separation 后使用
#   --limit N   分层抽样 N 条冒烟（正负比例与全集一致）
#   --resume    从 outputs/<NAME>/<NAME>_checkpoint.json 续跑
#   --gpu G     CUDA_VISIBLE_DEVICES（默认取环境变量，否则 1）
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

NAME=""
DATASET="A"
CONFIG_ARG=""
LIMIT_ARG=""
RESUME_ARG=""
GPU="${CUDA_VISIBLE_DEVICES:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --config) CONFIG_ARG="$2"; shift 2 ;;
    --limit)  LIMIT_ARG="$2"; shift 2 ;;
    --resume) RESUME_ARG=1; shift ;;
    --gpu)    GPU="$2"; shift 2 ;;
    -*)       echo "unknown flag: $1" >&2; exit 2 ;;
    *)        if [[ -z "$NAME" ]]; then NAME="$1"; shift; else echo "unexpected: $1" >&2; exit 2; fi ;;
  esac
done

if [[ -z "$NAME" ]]; then
  NAME="run_$(date +%Y%m%d_%H%M%S)"
  echo "no name given, using: $NAME"
fi

out_dir="${repo}/outputs/${NAME}"
mkdir -p "${out_dir}"

cfg_base="${CONFIG_ARG:-${repo}/configs/pipeline.json}"
cfg_tmp="${out_dir}/${NAME}_config.json"

# 复制配置并把分离缓存指到本次运行的目录（同名重跑可命中缓存，
# 不同实验之间互不污染）
"${python_bin}" - "$cfg_base" "${NAME}" "${cfg_tmp}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
c['separation_cache'] = 'outputs/' + sys.argv[2] + '/cache/separation'
json.dump(c, open(sys.argv[3], 'w', encoding="utf-8"), indent=2)
PY

if [[ "${DATASET}" == "A" ]]; then
  data_root="${repo}/dataset/dataset_test/datasetA/datasetA"
else
  data_root="${repo}/dataset/dataset${DATASET}"
fi

extra=()
[[ -n "$LIMIT_ARG" ]] && extra+=(--limit "$LIMIT_ARG")
[[ -n "$RESUME_ARG" ]] && extra+=(--resume)

cd "${repo}"
CUDA_VISIBLE_DEVICES="${GPU}" \
PYTHONPATH="${repo}/src:${repo}/model/runtime/vendor" \
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${repo}/model/workspace/midea/models/modelscope}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
"${python_bin}" -u -m pipeline.cli \
  --data-root "${data_root}" \
  --jsonl "${data_root}/pos.jsonl" "${data_root}/neg.jsonl" \
  --config "${cfg_tmp}" \
  --output "${out_dir}/${NAME}_submission.json" \
  --final-output "${out_dir}/${NAME}_detail.json" \
  --checkpoint-output "${out_dir}/${NAME}_checkpoint.json" \
  --manifest-output "${out_dir}/${NAME}_manifest.json" \
  ${extra:+"${extra[@]}"}
