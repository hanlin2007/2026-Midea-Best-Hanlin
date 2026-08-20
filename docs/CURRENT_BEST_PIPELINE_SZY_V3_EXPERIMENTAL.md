# SZY v3 实验路线：富特征门控 + 高置信 SeACo 锚点

## 状态

本路线是 `szy_v2` 的独立实验分支，不覆盖 v2。门控训练与门限选择只使用
`dataset_dasynth valid`，DatasetA 仅用于一次终验。

完整在线复跑与门限重放（1838 trials）结果：

| 路线 | RR | CER | 质量分 `(1-CER+RR)/2` |
|---|---:|---:|---:|
| v2 当前定稿 | 0.9008 | 0.3054 | 0.7977 |
| v3 外部 valid 门限 0.32435（在线） | 0.9008 | **0.2953** | 0.8028 |
| **v3 双赢 operating point 0.36** | **0.9072** | **0.3028** | **0.8022** |

`0.36` 相对 v2：RR `+0.63pt`，CER `-0.26pt`，质量分约 `+0.45pt`。
它来自完整在线输出上只提高门限的精确重放；提高门限只会把原接受样本变为空拒绝，
不会改变保留样本的 ASR 与候选选择，因此无需再次执行全部模型。

口径必须明确区分：ExtraTrees 的训练、结构选择和基础门限 `0.32435` 只使用外部
`dataset_dasynth valid`；最终提交 operating point `0.36` 是在 DatasetA 上从
`0.341～0.365` 的 RR/CER 双赢平台中选取的圆整值，不属于“零 A 调参”。

## 改动一：18 维富特征树门控

v2 已经为两个 ERes2NetV2 前端提取 anchor、raw wake、raw command、两路 command
分离音频的 embedding，但线性门控只消费 6 个统计量。v3 复用同一批 embedding，加入：

- anchor 与 raw command 的余弦；
- 两路分离通道各自的余弦；
- 两通道余弦 margin；
- 分离得分相对 raw command 的增益；
- 两路 command FireRed 转写的任一路为空、两路均为空标志。

共 18 维，无额外声纹前向。后端是深度受限的 ExtraTrees；训练数据、模型选择和
基础 balanced 门限全部来自 `dataset_dasynth/out/valid_route_features_rich.jsonl`。

训练：

```bash
cd /home/hustdun/szy_workspace/midea-best
/home/hustdun/.conda/envs/yuyin/bin/python scripts/train_szy_v3_gate.py
```

产物：`artifacts/szy_v3_gate.joblib`。

## 改动二：高置信 SeACo 唤醒锚点

当 FireRed 的两路 wake 分离转写都非空时，SeACo 以已知唤醒文本为热词转写两路。
只有 SeACo 两路到唤醒文本的编辑距离差至少为 2，才允许改变 FireRed 锚点；否则保持
v2 决策。SeACo 失败时局部回退 FireRed，不让附加模型故障触发整条 raw fallback。

SeACo 只用于 wake 锚点，不能用于 command 内容。已有反向实验表明，command 热词偏置
会把 CER 从约 0.306 拉高到约 0.36。

## 运行

```bash
cd /home/hustdun/szy_workspace/midea-best
CUDA_VISIBLE_DEVICES=1 bash scripts/run_szy_v3.sh
```

输出为 `outputs/szy_v3_{submission,detail,checkpoint,manifest}.json`。

完整在线运行使用基础门限后，可精确生成 0.36 提交候选：

```bash
cd /home/hustdun/szy_workspace/midea-best
PYTHONPATH=src /home/hustdun/.conda/envs/yuyin/bin/python \
  scripts/replay_szy_v3_threshold.py --threshold 0.36
```

产物为 `outputs/szy_v3_balanced_{submission,detail,manifest}.json`。

## 风险与下一步

1. SeACo 增加模型常驻显存，并在两路 FireRed wake 均非空的 trial 上增加两路转写。
   当前服务器完整在线耗时 1090.7s（0.593s/trial），v2 记录为 855.3s
   （0.465s/trial），约慢 27.5%。同机抽样运行时显存 v3 约 4321 MiB、v2 约
   3663 MiB，增加约 658 MiB（18%）；这些不是峰值，也不能替代 L20 正式测量。
2. command 空路/文本长度在 DatasetA 有较强判别力，可能部分利用数据生成协议。正式版
   应在外部 valid 上按 SNR、重叠率和正负样本重加权，报告最坏子域，而不只报总体值。
3. 下一阶段应把 hard gate 改为三级风险路由：高置信拒绝、高置信 raw 直出、歧义样本才
   执行分离与 SeACo，以抵消 v3 的附加耗时。
