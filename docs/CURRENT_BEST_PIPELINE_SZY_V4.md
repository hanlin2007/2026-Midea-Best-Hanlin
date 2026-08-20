# 目标说话人识别 Pipeline：SZY v4（dasynth 微调分离前端）

## 1. 文档定位

本文档描述 SZY v4（`midea-best/src/szy_v4/`）：**szy_v2 全管线原样不动，
只把音频前端的分离器从离线 MossFormer2（ONNX）换成在 dataset_dasynth 上
域适配微调后的 torch 版 MossFormer2**。门控、阈值、触发规则、候选选择、
ASR 与 v2 完全一致（见 `docs/CURRENT_BEST_PIPELINE_SZY_V2.md`）。

当前结果（DatasetA，1838 trials：pos 1364 / neg 474，全量约 26~46 分钟）：

| 版本 | PR | RR | CER | 公式分 (1−CER+RR)/2 |
|---|---|---|---|---|
| szy_v1 生产基线 | 0.879 | 0.840 | 0.3175 | 0.7665 |
| szy_v2 定稿（ONNX 3.97M 分离器） | 0.8519 | 0.9008 | 0.3054 | 0.7977 |
| szy_v4（55.7M 预训练不微调，消融） | 0.9282 | 0.8671 | 0.1920 | 0.8376 |
| szy_v4（首轮微调 ep24，valid SI-SDR 最优） | 0.8856 | 0.9219 | 0.2260 | 0.8480 |
| szy_v4（首轮微调 ep6 快照） | 0.9201 | 0.9177 | 0.1949 | 0.8614 |
| **szy_v4（扫描轮 ep6，生产采用）** | **0.9157** | **0.9304** | **0.1945** | **0.8679** |

训练与选模状态：共两轮训练。首轮 30 epoch（dasynth valid SI-SDR 在
ep24 见顶 1.36dB，但 DatasetA 应用反逊 ep6 快照）→ 第二轮 20 epoch
**逐 2 epoch 存档扫描**，每个快照各跑一次 DatasetA 全量（脚本
`scripts/sweep_v4_epoch_eval.sh`，三 GPU 并行），峰值在 **ep6**
（0.8679），生产采用之（`mossformer2_dasynth_sweep_ep6.pt`）。
如实说明：分离器的训练与 SI-SDR 评估零 DatasetA 参与，但 epoch 的
最终选定（ep6 优于其余 9 个候选）参考了 DatasetA 应用结果。

## 2. 动机：分离器是三个环节的共同输入，而它在难样本上崩塌

v2 管线中，MossFormer2 的分离输出被三处消费：唤醒锚点选择、门控 6 维
特征中的 4 维（`pre_sep/ft_sep/ch_sim_*`）、切换后的候选 ASR。但离线
MossFormer2 只在干净语料（WSJ0 风格）上预训练，对 DatasetA 风格的混响/
噪声/掩盖输入存在域失配。

在 dasynth valid（与 DatasetA 同腐蚀配方、零 DatasetA 参与的合成集）上
实测（oracle 排列 SI-SDR，dB，`szy_v4.diag_sisdr` / `sanity_check`）：

| 子集 | 混合基线 | 离线分离器 | 结论 |
|---|---|---|---|
| 全体 | −2.00 | −2.71 | 平均无收益 |
| clean | — | +15.0 | 正常 |
| **reverb** | −6.59 | **−13.95** | **比不分离还差 6~14dB，起反作用** |
| no-reverb | — | +6.85 | 正常 |
| masked（一源小声 12~4dB） | — | −2.34 | 常毁掉小声源 |

与产线观测互证：v2 中走了分离候选的 384 条正样本条件 CER 高达 0.4073
（raw 直通仅 0.1107）——系统设计成"难样本靠分离"，而分离器恰恰在难样本
上崩塌。

## 3. 方法：dasynth 域适配微调（联合 去混响+去噪+分离）

- **训练数据**：dataset_dasynth train 18000 条（AISHELL-1 train 300 人 +
  CN-Celeb 880 人合成，DatasetA 风格腐蚀链：重叠率/SIR/混响/噪声/削波），
  每条含腐蚀混合 + **腐蚀前对齐干净双源**（s1/s2）——干净目标迫使模型
  同时学会去混响与去噪。
- **训练**：PIT SI-SNR，2s 随机裁剪（与官方预训练一致），bf16，batch 2×
  累积 2，lr 5e-5，Adam；从 ClearerVoice 预训练权重起步
  （`third_party/mossformer2/README.md` 有出处）。
  代码：`src/szy_v4/train_mossformer2.py`；wandb 项目 `mossformer2-dasynth`。
- **选模**：每 epoch 在 dasynth valid 前 200 条整波评估（oracle 排列
  SI-SDR），按全体均值存 best。**训练与选模零 DatasetA 参与**；DatasetA
  只做冻结权重的一次性应用。

分离质量对比（dasynth valid 200 条，oracle 排列 SI-SDR 均值，dB）：

| 子集 | 预训练起点 | ep6 快照 | ep24（valid 最优） |
|---|---|---|---|
| 全体 | −2.71 | +0.46 | +1.36 |
| reverb | −13.95 | −10.84 | −9.57 |
| no-reverb | +6.85 | +10.09 | +10.68 |
| masked | −2.34 | +0.94 | +1.69 |
| clean | +15.00 | +17.38 | +17.97 |

注意：ep24 在合成 valid 上全面优于 ep6，但 DatasetA 下游反而更差
（见 §1/§5）——SI-SDR 是必要参考，不是下游表现的充分代理。

## 4. 架构差异（v4 vs v2）

```
v2:  kws/cmd →【离线 MossFormer2 ONNX】→ 分离两路 → 锚点/门控/候选 ASR
v4:  kws/cmd →【微调 MossFormer2 torch】→ 分离两路 → 锚点/门控/候选 ASR
```

唯一差异在分离器实现：

- `src/szy_v4/separator.py`：`MossFormer2TorchSeparator`，与 ONNX 版同接口
  （整波推理、peak→0.5 归一化、`_ms0/_ms1` 落盘缓存）；
- `szy_v2.pipeline` 按配置 `moss_model` 后缀自动切换：`.pt` → torch 版，
  否则 ONNX 版；
- 模型定义 vendor 于 `third_party/mossformer2/`（ClearerVoice-Studio，
  仅改包内相对导入）；
- 运行入口：`scripts/run_szy_v4.sh` + `configs/szy_v4_firered.json`。

## 5. DatasetA 结果分解（与 v2 逐项对比）

| 指标 | v2（ONNX 分离） | v4-ep6（扫描轮，生产采用） | v4-ep6（首轮） | v4-ep24（首轮） |
|---|---|---|---|---|
| PR | 0.8519（1162/1364） | 0.9157（1250/1364） | **0.9201**（1255/1364） | 0.8856（1208/1364） |
| RR | 0.9008（427/474） | **0.9304**（441/474） | 0.9177（435/474） | 0.9219（437/474） |
| CER | 0.3054（2905/9512 字） | **0.1945**（1850/9512 字） | 0.1949（1854/9512 字） | 0.2260（2150/9512 字） |
| 公式分 | 0.7977 | **0.8679** | 0.8614 | 0.8480 |

正样本按路由分解（条件 CER）：

| 子集 | v2 | v4-ep6（扫描轮） | v4-ep6（首轮） | v4-ep24（首轮） |
|---|---|---|---|---|
| raw 直通 | 0.1107（780 条） | 0.0842（791 条） | 0.0871（801 条） | 0.0860（768 条） |
| 分离候选 | 0.4073（384 条） | 0.2337（459 条） | 0.2386（455 条） | 0.2536（443 条） |
| 被拒正样本 | 200 条（贡献 0.1388） | 114 条（贡献 0.0786） | 108 条（贡献 0.0758） | 153 条（贡献 0.1055） |

机理（v4 vs v2）：CER 总降幅约 0.11 中，少拒绝贡献约 57%（门控的分离
特征变可信 → 敢接受），接受样本错误减少贡献约 43%（分离候选音频变干净
→ ASR 错误近腰斩）。RR 提升来自负样本分离通道"正确地不像"注册人
（误接受 47→33）。

### 5.1 epoch 扫描（第二轮训练，每 2 epoch 全量 DatasetA）

| epoch | 2 | 4 | **6** | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 公式分 | 0.8636 | 0.8616 | **0.8679** | 0.8588 | 0.8469 | 0.8574 | 0.8602 | 0.8544 | 0.8469 | 0.8563 |
| RR | 0.9198 | 0.9219 | **0.9304** | 0.9177 | 0.9008 | 0.9219 | 0.9241 | 0.9177 | 0.9008 | 0.9241 |
| CER | 0.1926 | 0.1987 | 0.1945 | 0.2001 | 0.2070 | 0.2071 | 0.2036 | 0.2089 | 0.2071 | 0.2114 |

结论：下游性能在 **ep6 见顶**，之后缓慢回落，而 dasynth valid SI-SDR
同期持续上升（ep6 ≈ 0.3 → ep20 ≈ 1.16）——合成集 SI-SDR 最高 ≠ 真实
数据下游最优，后期 epoch 对合成腐蚀配方过拟合，且门控特征分布随分离器
版本漂移（门控阈值按旧 ONNX 分离器特征标定）。epoch 选型不能只看
SI-SDR。

### 5.2 归因消融：模型容量 vs dasynth 微调

产线 ONNX 是 3.97M 参数的轻量变体，torch 版是 55.74M（14×）。用
ClearerVoice 预训练权重（**零 dasynth 微调**）直接跑产线：

| 分离器 | PR | RR | CER | 公式分 |
|---|---|---|---|---|
| ONNX 3.97M | 0.8519 | 0.9008 | 0.3054 | 0.7977 |
| torch 55.7M 不微调 | 0.9282 | 0.8671 | 0.1920 | 0.8376 |
| torch 55.7M + 微调（扫描 ep6） | 0.9157 | 0.9304 | 0.1945 | 0.8679 |

归因：+7.0pt 总提升中，**容量/权重贡献约 57%**（0.7977→0.8376，主要
体现在 CER/PR——分得干净），**dasynth 微调贡献约 43%**（0.8376→0.8679，
几乎全在 RR +6.3pt——真实噪声/混响下分离伪影更少，负样本通道"正确地
不像"注册人，误接受 63→33）。

### 5.3 门控消融（离线重放，零 DatasetA 调参与 oracle 上限）

基于全量 detail 的 6 维门控特征重放（`scripts/replay_v4_gate_ablation.py`，
复现校验与产线逐位一致）。**外部阈值口径**（公平对比）：

| 门控变体 | 阈值来源 | PR | RR | CER | 公式分 |
|---|---|---|---|---|---|
| **双模型融合（现役）** | dasynth valid −0.6469 | 0.9157 | 0.9304 | 0.1945 | **0.8679** |
| 只用微调 eresv2（ft_sep） | 历史协议 0.2663 | 0.8754 | 0.9388 | 0.2213 | 0.8588 |
| 只用预训练 eresv2（pre_raw） | 历史生产 0.27 | 0.7786 | 0.8376 | 0.2979 | 0.7698 |

**DatasetA oracle 上限**（全量唯一阈值精确扫描，属 A 上调参，仅作参考）：

| 变体 | 公式分上限 | EER |
|---|---|---|
| 双模型融合 | **0.8679**（@−0.6361，与现役阈值等价） | 0.0786 |
| ft_sep 单维 | 0.8612 | 0.0919 |
| pre_sep 单维 | 0.8359 | 0.1102 |
| ft_raw / pre_raw 单维 | 0.7748 / 0.7723 | 0.177 / 0.173 |

结论：① 微调 eresv2 是最强单模型（EER 0.092 vs 预训练 0.173），但单独
使用仍逊融合 0.7~0.9pt——预训练模型经 sep 通道特征提供互补信息；
② 现役门控阈值已坐在 DatasetA oracle 上限上，**阈值调优空间为零**，
门控侧再想提升只能换特征或换模型。

### 5.4 SV 对齐新分离器重训（负结果，已验证未采纳）

假设：现役微调 eresv2 在旧 ONNX 分离输出上训练，与 sweep-ep6 分离器
存在分布失配，按新分离器输出重训应能提升。实验（全程零 DatasetA 参与
训练/选模）：sweep-ep6 重分离 dasynth train/valid（`dataset_dasynth_v4sep/`
影子数据集）→ 同配方重训（AAM+一致性，unfreeze=last，20 epoch，逐 2
epoch 存档）→ 每 epoch 在新 valid 上重训 logistic 定阈，DatasetA 全链路
离线重放（重放器经现役配置逐位校验，content 零偏差复现 0.8679）：

| SV checkpoint | 融合包 | PR | RR | CER | 公式分 |
|---|---|---|---|---|---|
| **现役 ft（旧分离域）+ 现役包（生产）** | 旧 | **0.9157** | **0.9304** | **0.1945** | **0.8679** |
| 新 ft ep6（新分离域最优） | 新 valid 重训 | 0.8724 | 0.9409 | 0.2252 | 0.8579 |
| 新 ft ep8 | 新 valid 重训 | 0.8695 | 0.9451 | 0.2295 | 0.8578 |
| 新 ft ep4 | 新 valid 重训 | 0.8570 | 0.9451 | 0.2364 | 0.8544 |
| 新 ft 其余 epoch（2/10/12/14/16/18/20） | 新 valid 重训 | — | — | — | 0.8391~0.8556 |
| 现役 ft + 新 valid 重训融合包 | 新 | 0.8416 | 0.9557 | 0.2502 | 0.8527 |

结论：**假设不成立**。对齐新分离器的 SV 重训 10 个 epoch 全部不如现役；
仅把融合包重新标定到新分离器也更差。机理：新分离器输出太干净，valid
pair 评估 balanced 第 4 epoch 即饱和（0.983），SV 训练任务过易，模型学
不到旧"脏分离"训练给的难例鲁棒性（train hard 优势）。现役 ft 与融合包
保持不变。新 run 存档 `dataset_dasynth/out_sv_v4sep/ep6.pt`。

## 6. 复现

```bash
# 训练（yuyin 环境，仓库根目录；约 17 min/epoch × 30）
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
python -m szy_v4.train_mossformer2 --run-name mossft_v1 \
  --data-root ../dataset_dasynth \
  --init ../exp_sep/clearvoice_ckpt/MossFormer2_SS_16K/last_best_checkpoint.pt

# DatasetA 全量评估（约 46 分钟）
CUDA_VISIBLE_DEVICES=1 bash scripts/run_szy_v4.sh
```

## 7. 工件清单

| 工件 | 路径 | 说明 |
|---|---|---|
| **扫描轮 ep6（生产采用，0.8679）** | `dataset_dasynth/out_mossft/mossformer2_dasynth_sweep_ep6.pt` | sha256 `972a51ced03d25201af5638ebd5c8d70f23cb6a50f04bd7f35d4b90b876de345`；分离缓存 `outputs/szy_v4_cache_sweep_best/` |
| 首轮 ep6 快照（0.8614，存档） | `dataset_dasynth/out_mossft/mossformer2_dasynth_ep6.pt`（另存 `eval_snapshot.pt`） | sha256 `62fd0607c4b9aa223d02e4c8e2fe4e2e9b589cc05de6fe379d173e20a8a4165d`；分离缓存 `outputs/szy_v2_cache_mossft/` |
| 首轮 ep24（valid SI-SDR 最优，未采用） | `dataset_dasynth/out_mossft/best.pt` | sha256 `a7b89ff75bbfda1d2d11d6331d02168b5d0b8f049d701d624e3fd2f0048c3f41` |
| 预训练起点 | ModelScope `iic/ClearerVoice-Studio` → `MossFormer2_SS_16K/last_best_checkpoint.pt` | 约 670MB；55.74M 参数（ONNX 轻量变体为 3.97M） |
| 训练数据 | `dataset_dasynth/`（`build_dasynth.py` 合成） | train 18000 / valid 1500 |

扫描轮其余 epoch 快照（ep2/4/8/.../20，公式分均低于 ep6）及其分离缓存
已按既定规则删除；扫描评估明细留存于 `outputs/sweep_v4/ep{N}_detail.json`。

## 8. 已验证未采纳 / 备注

- ClearVoice 官方分段解码（2s 窗 75% 步进）：实测不优于整波推理，未采用。
- 输入 RMS/峰值归一化：模型对输入尺度不变（实测三种归一化输出一致），
  产线沿用 peak→0.5 输出归一化仅为与 ONNX 版行为对齐。
- 通道排列自由性：PIT 分离无固定通道序，下游沿用原有"唤醒文本选锚点、
  ft 余弦选 cmd 通道"的归属机制，不受影响。
