#!/usr/bin/env bash
# SZY epoch 扫描：训练（mossft_sweep，out_mossft_sweep/ep{N}.pt）过程中，
# 每出现一个偶数 epoch 快照就在空闲 GPU 上跑一次 DatasetA 全量评估。
# 三张评估 GPU 各负责一组 epoch，组内串行：
#   GPU0: ep2  ep8  ep14 ep20
#   GPU1: ep4  ep10 ep16
#   GPU3: ep6  ep12 ep18
# 结果：outputs/sweep/ep{N}_detail.json + ep{N}.log；分离缓存各自独立。
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd "${repo}/.." && pwd)"
model_root="${workspace}/model"
python_bin="${PYTHON_BIN:-/home/hustdun/.conda/envs/yuyin/bin/python}"
ckpt_dir="${workspace}/dataset_dasynth/out_mossft_sweep"
out="${repo}/outputs/sweep"
data_root="${workspace}/dataset/dataset_test/datasetA/datasetA"
mkdir -p "${out}"

wait_stable() {  # 等文件存在且大小 5 秒内不再变化（torch.save 写完）
  local f="$1"
  while [[ ! -f "$f" ]]; do sleep 60; done
  local s1 s2
  s1=$(stat -c %s "$f")
  while true; do sleep 5; s2=$(stat -c %s "$f"); [[ "$s1" == "$s2" ]] && break; s1="$s2"; done
}

run_one() {  # $1=epoch $2=gpu
  local ep="$1" gpu="$2"
  local cfg="${out}/ep${ep}.json"
  "${python_bin}" - <<EOF
import json
c = json.load(open("${repo}/configs/pipeline_firered.json"))
c["moss_model"] = "${ckpt_dir}/ep${ep}.pt"
c["separation_cache"] = "outputs/szy_cache_sweep/ep${ep}/separation"
json.dump(c, open("${cfg}", "w"), ensure_ascii=False, indent=2)
EOF
  cd "${repo}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONPATH="${repo}/src:${model_root}/runtime/vendor" \
  MODELSCOPE_CACHE="${model_root}/workspace/midea/models/modelscope" \
  HF_HUB_OFFLINE=1 \
  "${python_bin}" -u -m szy.cli \
    --data-root "${data_root}" \
    --jsonl "${data_root}/pos.jsonl" "${data_root}/neg.jsonl" \
    --config "${cfg}" \
    --output "${out}/ep${ep}_submission.json" \
    --final-output "${out}/ep${ep}_detail.json" \
    --checkpoint-output "${out}/ep${ep}_checkpoint.json" \
    --manifest-output "${out}/ep${ep}_manifest.json" \
    > "${out}/ep${ep}.log" 2>&1
  echo "[worker gpu${gpu}] ep${ep} eval done"
}

worker() {  # $1=gpu，其余=epoch 列表
  local gpu="$1"; shift
  local ep
  for ep in "$@"; do
    echo "[worker gpu${gpu}] waiting ep${ep}.pt ..."
    wait_stable "${ckpt_dir}/ep${ep}.pt"
    echo "[worker gpu${gpu}] start ep${ep}"
    run_one "${ep}" "${gpu}"
  done
}

worker 0 2 8 14 20 &
worker 1 4 10 16 &
worker 3 6 12 18 &
wait
echo "ALL SWEEP EVALS DONE"
