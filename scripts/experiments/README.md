# 离线实验工具（scripts/experiments/）

这里的脚本都**不参与生产管线**，它们读取 `outputs/` 里已经跑出来的
`detail.json` / 分离缓存，做离线分析、阈值重扫与数据转换。共同前提：
先通过 `scripts/run.sh` 或 `scripts/run_exp.sh` 跑出一次完整评测。

| 脚本 | 用途 | 输入 |
|---|---|---|
| `sweep_threshold.py` | 重扫融合门控阈值（PR/RR/CER/Score 随阈值变化） | detail.json |
| `analyze_nbest.py` | 分析 N-best 重排序的覆盖率与置信度分布 | detail.json |
| `measure_q.py` | 给被门控拒绝的正样本补跑 ASR，测"若接受了 CER 是多少"（q） | detail.json + 分离缓存 + GPU |
| `gate_v2.py` | 门控 v2 实验框架：滑窗特征 + Score 代价加权 + 5 折 CV | detail.json + 滑窗统计 + （可选）measured_q.json |
| `gate_v2_ablate.py` | gate_v2 的消融：增益是真信号还是长度痕迹 | 同上 |
| `gate_v2_diag.py` | gate_v2 的诊断：AUC 升但 Score 降时定位问题 | 同上 |
| `convert_datasetx.py` | datasetX metadata → datasetA 兼容的 pos/neg.jsonl | datasetX 生成目录 |

依赖说明：`gate_v2*.py` 需要 scikit-learn（gbdt 模型另需 lightgbm）；
`measure_q.py` 需要 GPU 与完整模型资产；其余脚本纯 CPU。
