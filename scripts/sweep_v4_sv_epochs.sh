#!/usr/bin/env bash
# SV 重训逐 2 epoch 评估 watcher：out_sv_v4sep/ep{N}.pt 一出现就做全链路重放评估。
# 每个 epoch：新 valid 特征（pre 缓存 + ft 现算）→ 重训 logistic → valid 定阈
# → DatasetA 全链路重放 → 结果追加 outputs/sweep_v4/sv_sweep_results.jsonl。
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ckpt_dir="/home/hustdun/szy_workspace/dataset_dasynth/out_sv_v4sep"
py="/home/hustdun/.conda/envs/yuyin/bin/python"

freest_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}

wait_stable() {
  local f="$1"
  while [[ ! -f "$f" ]]; do
    # 链条日志宣告训练结束/失败 且 文件不出 → 放弃等待
    if grep -qE "^\[chain\] (train FAILED|TRAIN DONE)" /tmp/v4_chain.log 2>/dev/null; then
      echo "[sv-sweep] chain ended, stop waiting $f"; exit 0
    fi
    sleep 60
  done
  local s1 s2
  s1=$(stat -c %s "$f")
  while true; do sleep 5; s2=$(stat -c %s "$f"); [[ "$s1" == "$s2" ]] && break; s1="$s2"; done
}

for ep in 2 4 6 8 10 12 14 16 18 20; do
  echo "[sv-sweep] waiting ep${ep}.pt ..."
  wait_stable "${ckpt_dir}/ep${ep}.pt"
  # 已评估过则跳过（续跑安全）
  if grep -q "\"svv4_ep${ep}\"" "${repo}/outputs/sweep_v4/sv_sweep_results.jsonl" 2>/dev/null; then
    echo "[sv-sweep] ep${ep} already evaluated, skip"; continue
  fi
  gpu=$(freest_gpu)
  echo "[sv-sweep] eval ep${ep} on gpu ${gpu}"
  cd "${repo}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="${repo}/src" \
  "${py}" scripts/replay_v4_full.py --mode eval \
    --ft-ckpt "${ckpt_dir}/ep${ep}.pt" --tag "svv4_ep${ep}" --device cuda:0 \
    >> "${repo}/outputs/sweep_v4/sv_sweep.log" 2>&1
  echo "[sv-sweep] ep${ep} done"
done
echo "SV SWEEP ALL DONE"
