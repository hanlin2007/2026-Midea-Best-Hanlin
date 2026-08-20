# Midea Best：SZY v4 生产版

本仓库保留远端工作区当前采用的 SZY v4 目标说话人语音指令识别主线。v4 使用
DatasetA/合成域适配的 MossFormer2 分离前端，并复用 v2/v1 的稳定推理组件；旧 MLP
后端门控、旧 MossFormer2 ONNX 权重和历史运行输出已移除。

## 当前生产成果

- 生产路线：SZY v4（SZY v2 全管线 + dasynth 微调 MossFormer2）。
- 生产 MossFormer2：`model/szy_v4/mossformer2_dasynth_sweep_ep6.pt`。
- DatasetA 质量分：`0.8679`；完整实验口径见
  `docs/CURRENT_BEST_PIPELINE_SZY_V4.md`。
- v1 至 v4 的设计与实验文档均保留在 `docs/`。

## 本地资产布局

```text
midea_qiji_best_hanlin/
├── configs/szy_v4_firered.json
├── dataset/dataset_test/datasetA/datasetA/
├── model/
│   ├── external/FireRedASR2-AED/
│   ├── external/FireRedASR2S/
│   ├── runtime/vendor/
│   ├── workspace/midea/models/modelscope/iic/
│   └── szy_v4/
│       ├── ep6_datasetA_best.pt
│       ├── gate_fusion.json
│       └── mossformer2_dasynth_sweep_ep6.pt
├── src/{midea_qiji,szy_v1,szy_v2,szy_v4}/
└── third_party/mossformer2/
```

`src/szy_v1` 和 `src/szy_v2` 是 v4 的运行依赖，不代表保留旧版入口。大型模型、
DatasetA 和运行输出由 `.gitignore` 排除，不进入 Git 历史。

## 运行与验证

```bash
conda activate yuyin
bash scripts/run_szy_v4.sh --limit 5

PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

默认读取仓库内的 `dataset/` 与 `model/`。也可以通过 `PYTHON_BIN`、
`CUDA_VISIBLE_DEVICES` 和 `MODELSCOPE_CACHE` 覆盖运行环境。

## 文档

- `docs/CURRENT_BEST_PIPELINE_SZY_V1.md`
- `docs/CURRENT_BEST_PIPELINE_SZY_V2.md`
- `docs/CURRENT_BEST_PIPELINE_SZY_V3_EXPERIMENTAL.md`
- `docs/CURRENT_BEST_PIPELINE_SZY_V4.md`
- `docs/SZY_V1.md`
