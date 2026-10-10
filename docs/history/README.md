# 阶段演进历史（存档）

这里保存系统从 v1 到 v4 各阶段的设计文档与实验记录，**仅供理解演进
脉络与历史决策**，内容描述的代码形态已被现役系统取代。现役系统
（LSZ v4.5，Score 0.8937）的文档在 `docs/` 上级目录：

- `docs/CURRENT_BEST_PIPELINE.md` — 现役管线算法、阈值与运行方式
- `docs/OPTIMIZATION_SUMMARY.md` — 从 v4 基线到 0.8937 的逐项优化实验

## 各阶段文档

| 文档 | 阶段 | 要点 |
|---|---|---|
| `SZY_V1.md` | v1 补充说明 | 早期双 MLP 门控 + ONNX 分离器的运行边界 |
| `CURRENT_BEST_PIPELINE_SZY_V1.md` | v1 | 双前端 MLP 门控 + MossFormer 候选的完整流程 |
| `MOSS_CANDIDATE_REPRODUCTION.md` | v1 时代 | MossFormer2 候选路线的复现实验 |
| `CURRENT_BEST_PIPELINE_SZY_V2.md` | v2 | 融合门控取代双 MLP 门控 |
| `CURRENT_BEST_PIPELINE_SZY_V3_EXPERIMENTAL.md` | v3（实验） | 未采用的实验分支 |
| `CURRENT_BEST_PIPELINE_SZY_V4.md` | v4（Score 0.8679） | dasynth 域适配微调分离器 + 融合门控 + 音色触发规则 |
| `MIDEA_BEST_STRUCTURE.md` | v1 时代 | 早期目录结构记录（已完全过时，仅存档） |

## 演进主线（一句话版）

v1 双 MLP 门控 → v2 线性融合门控 → v4 dasynth 微调分离器 + 音色规则
（0.8679）→ v4.5 拼音锚点 + N-best + ELM + 共享 AED + 声学单双人路由
（0.8937）。每一步的动机与数字见 `docs/OPTIMIZATION_SUMMARY.md`。
