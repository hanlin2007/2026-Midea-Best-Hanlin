#!/usr/bin/env bash
# GPU1 补跑器：等预训练消融（pretrained_detail.json）结束后，再依次评估 ep10、ep16。
# （用户要求：消融跑完之前不在 GPU1 上启动 ep10 评估。）
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd "${repo}/.." && pwd)"
model_root="${workspace}/model"
python_bin="${PYTHON_BIN:-/home/hustdun/.conda/envs/yuyin/bin/python}"
ckpt_dir="${workspace}/dataset_dasynth/out_mossft_sweep"
out="${repo}/outputs/sweep_v4"
data_root="${workspace}/dataset/dataset_test/datasetA/datasetA"

# 1) 等消融结束
while [[ ! -f "${out}/pretrained_detail.json" ]]; do sleep 60; done
echo "[gpu1-rest] ablation done, proceeding"

wait_stable() {
  local f="$1"
  while [[ ! -f "$f" ]]; do sleep 60; done
  local s1 s2
  s1=$(stat -c %s "$f")
  while true; do sleep 5; s2=$(stat -c %s "$f"); [[ "$s1" == "$s2" ]] && break; s1="$s2"; done
}

run_one() {
  local ep="$1"
  local cfg="${out}/ep${ep}.json"
  "${python_bin}" - <<EOF
import json
c = json.load(open("${repo}/configs/szy_v4_firered.json"))
c["moss_model"] = "${ckpt_dir}/ep${ep}.pt"
c["separation_cache"] = "outputs/szy_v4_cache_sweep/ep${ep}/separation"
json.dump(c, open("${cfg}", "w"), ensure_ascii=False, indent=2)
EOF
  cd "${repo}"
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="${repo}/src:${model_root}/runtime/vendor" \
  MODELSCOPE_CACHE="${model_root}/workspace/midea/models/modelscope" \
  HF_HUB_OFFLINE=1 \
  "${python_bin}" -u -m szy_v4.pipeline \
    --data-root "${data_root}" \
    --jsonl "${data_root}/pos.jsonl" "${data_root}/neg.jsonl" \
    --config "${cfg}" \
    --output "${out}/ep${ep}_submission.json" \
    --final-output "${out}/ep${ep}_detail.json" \
    --checkpoint-output "${out}/ep${ep}_checkpoint.json" \
    --manifest-output "${out}/ep${ep}_manifest.json" \
    > "${out}/ep${ep}.log" 2>&1
  echo "[gpu1-rest] ep${ep} eval done"
}

for ep in 10 16; do
  echo "[gpu1-rest] waiting ep${ep}.pt ..."
  wait_stable "${ckpt_dir}/ep${ep}.pt"
  run_one "${ep}"
done
echo "GPU1 REST EVALS DONE"
