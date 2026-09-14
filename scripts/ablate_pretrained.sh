#!/usr/bin/env bash
# 消融：ClearerVoice 预训练 torch 分离器（零 dasynth 微调）跑 DatasetA 全量。
# 用于分解系统收益来源 = 模型容量(ONNX 3.97M -> torch 55.7M) vs dasynth 微调。
# 等扫描训练（GPU2）结束后在该卡上运行。
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd "${repo}/.." && pwd)"
model_root="${workspace}/model"
python_bin="${PYTHON_BIN:-/home/hustdun/.conda/envs/yuyin/bin/python}"
out="${repo}/outputs/sweep"
data_root="${workspace}/dataset/dataset_test/datasetA/datasetA"
mkdir -p "${out}"

# 等扫描训练结束（进程消失且 ep20.pt 已产出）
while pgrep -f "szy.train" > /dev/null; do sleep 60; done
sleep 30

cfg="${out}/pretrained.json"
"${python_bin}" - <<EOF
import json
c = json.load(open("${repo}/configs/pipeline_firered.json"))
c["moss_model"] = "${workspace}/exp_sep/clearvoice_ckpt/MossFormer2_SS_16K/last_best_checkpoint.pt"
c["separation_cache"] = "outputs/szy_cache_pretrained/separation"
json.dump(c, open("${cfg}", "w"), ensure_ascii=False, indent=2)
EOF

cd "${repo}"
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH="${repo}/src:${model_root}/runtime/vendor" \
MODELSCOPE_CACHE="${model_root}/workspace/midea/models/modelscope" \
HF_HUB_OFFLINE=1 \
"${python_bin}" -u -m szy.cli \
  --data-root "${data_root}" \
  --jsonl "${data_root}/pos.jsonl" "${data_root}/neg.jsonl" \
  --config "${cfg}" \
  --output "${out}/pretrained_submission.json" \
  --final-output "${out}/pretrained_detail.json" \
  --checkpoint-output "${out}/pretrained_checkpoint.json" \
  --manifest-output "${out}/pretrained_manifest.json" \
  > "${out}/pretrained.log" 2>&1
echo "PRETRAINED ABLATION DONE"
