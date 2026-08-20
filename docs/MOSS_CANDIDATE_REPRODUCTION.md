# MossFormer 候选路线复现说明

本目录保存历史 MossFormer 候选选择结果的缓存回放工具。它用于离线评估，
不是生产配置。主门控仍然是 `ERes2NetV2 + CAM++ 双 MLP`；该工具只会为
已经被主门控接受的正样本重新选择转写文本。

回放过程分为三个彼此独立的阶段：

1. 固定输入：从缓存读取双 MLP 的逐条结果、MossFormer 候选波形、FireRed
   候选转写、FireRed 置信度特征和唤醒候选源的转写。
2. 决策：使用唤醒文本选择唤醒候选源；当候选源为空时回退到原始唤醒音频。
   固定的二维置信度规则决定是否进入命令候选选择。差分 ERes 分数只在
   `ms0` 和 `ms1` 之间选择；任一命令候选源为空时保留 `orig`。
3. 指标：`report.json` 在与基线完全一致的 DatasetA 全量逐条结果范围内
   计算 CER、PR 和 RR。负样本不发生改变，因此 RR 必须与冻结基线的 RR
   相同。

请在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 scripts/run_moss_candidate_reproduce.sh --strict
```

输出写入 `outputs/moss_candidate_reproduction/`：

- `decisions.jsonl`：每个主门控接受的正样本对应一条可审计的候选选择决策；
- `report.json`：基线与候选路线的聚合指标，以及缓存覆盖范围。

任务中断后可使用 `--resume` 续跑。每处理一批记录后，决策缓存会以原子方式
重写。默认预期 CER 为 `0.3191`，与历史冻结缓存实验一致。若结果不一致，
应排查缓存或模型版本差异；不得在 DatasetA 上调整该路线的阈值。

默认实现保留历史 embedding 的调用顺序，因为合并调用会改变少量处于 GPU
数值边界上的候选源选择。`--fast-batched-embeddings` 只用于诊断，不能用于
声明严格复现结果。
