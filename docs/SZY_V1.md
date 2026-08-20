# SZY v1 正式 MossFormer2 路由

## 路由

```text
唤醒音频 + 识别音频
  -> v2 dual MLP 主门控
  -> 主门控拒绝：输出空
  -> 主门控接受：FireRed 原始命令转写与置信度
  -> 二维置信度/文本规则未触发：保留原始命令
  -> 规则触发：MossFormer2 分离唤醒音频和命令音频
       唤醒两源 FireRed 转写 + 唤醒文本编辑距离选锚点
       同次分离的锚点/对照差分声纹选择命令源
       任一命令源空转写：回退原始命令
```

实现入口是 `src/szy_v1/pipeline.py`。它从原始 DatasetA JSONL 和 WAV 读取数据，不使用 `exp_sep` 的历史转写或分离缓存；分离 WAV 只作为可复用的中间缓存。每处理一条样本就写 checkpoint，支持 `--resume`。

## 运行

```bash
cd /home/hustdun/szy_workspace/midea-qiji
CUDA_VISIBLE_DEVICES=3 ./scripts/run_szy_v1.sh
```

中断后续跑：

```bash
CUDA_VISIBLE_DEVICES=3 ./scripts/run_szy_v1.sh --resume
```

配置在 `configs/szy_v1_firered.json`。最终产物为：

- `outputs/szy_v1_submission.json`：提交格式结果
- `outputs/szy_v1_detail.json`：逐样本路由、置信度、差分分数和汇总
- `outputs/szy_v1_manifest.json`：模型 SHA256、配置和输入记录
- `outputs/szy_v1_checkpoint.json`：断点续跑状态

## 本次全量结果

DatasetA 共 1838 条（正 1364、负 474），FireRed，双 MLP 阈值 `0.4460311857`：

| 指标 | 结果 |
|---|---:|
| CER | `3020 / 9512 = 0.3174937` |
| PR（正样本非空输出） | `1199 / 1364 = 0.8790323` |
| RR（负样本空输出） | `398 / 474 = 0.8396624` |
| 总耗时 | `1106.64 s` |

本次运行没有分离异常。`CER` 使用 `normalize_transcript` 后的字符口径；PR/RR 只判断输出是否为空，与 CER 分开统计。
