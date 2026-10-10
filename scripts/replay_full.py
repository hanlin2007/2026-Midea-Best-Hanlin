#!/usr/bin/env python3
"""SZY 全链路离线重放器：给定 ft SV checkpoint，精确重放整条 SZY 决策链。

用途：
  1) validate：用现役 ft（ep6_datasetA_best）+ 现役融合包 + -0.6469 重放，
     必须逐位复现官方 0.8679（校验重放器正确性）；
  2) eval：对重训的 SV 快照（out_sv_v4sep/ep{N}.pt），先在新分离器的
     dasynth valid 上重训 logistic、valid 定阈（零 DatasetA），再重放
     DatasetA 全链路出公式分。

重放内容：唤醒锚点（沿用 detail 存的选择）→ 6 维门控特征（重算 SV embedding）
→ 融合门控 → 音色触发（sv_diff 重算）→ ft 直接余弦选源 → content。
文本/置信度用 build_replay_assets.py 的资产，不重跑 ASR/分离。

重放（replay）设计思想——昂贵的语音管线的省钱工程手法：
SZY 管线的一次全量运行里，真正贵的是 ASR 转写（FireRed 前向）与语音分离，
而"决策链"（门控、选源、切换）只依赖几个便宜的量：SV 余弦、转写文本、
置信度。重放的做法是：第一次全量运行时把贵的结果落盘（detail.json 里的
gate_features/sv_score/content + build_replay_assets.py 补的文本），
之后每次实验（换 SV 快照、换融合权重、换阈值）只重算 SV embedding 和
决策本身，不再碰 ASR 和分离。于是"N 次实验 = N 次数小时"变成
"一次数小时 + N 次数分钟"。validate 模式的意义：用已知官方分数 0.8679
反向校验重放器本身没有写错——重放器不可信，之后所有 eval 数字都不可信。

用法（yuyin，仓库根目录）：
  PYTHONPATH=src python scripts/replay_full.py --mode validate
  PYTHONPATH=src python scripts/replay_full.py --mode eval \
    --ft-ckpt ../dataset_dasynth/out_sv_v4sep/ep6.pt --tag sv_ep6
说明：需要 GPU（重算 SV embedding）；--tag 非空时把本次结果以一行 JSON
追加到 outputs/sweep_v4/sv_sweep_results.jsonl（多次实验的汇总日志）。

依赖：midea_qiji（gate/text/paths/moss_candidate）、szy.helpers、
numpy、torch（在函数内延迟 import）。重型依赖 torch 不放到模块顶部，
保证本脚本 import 阶段轻快、报错时机更晚（真用到才报缺 torch）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---- 手动扩展模块搜索路径（必须在 import 仓库包之前）----
# 脚本位于 scripts/，不在包内。直接执行时 sys.path 只有 scripts/ 与
# 标准库，找不到 src/ 下的 midea_qiji、szy 包，所以先手动注入：
# parents[1] = scripts/ 上一级 = 仓库根；_WS 再上一级 = 工作区根
# （model/、dataset/、dataset_dasynth/ 等资产所在处）。
# insert(0, ...) 插到搜索路径最前 = 最高优先级，确保 import 到的是
# 仓库内版本而非环境里可能存在的同名旧包。必须放在所有 import
# 仓库包语句之前，否则 import 时搜索路径里还没有 src。
_REPO = Path(__file__).resolve().parents[1]
_WS = _REPO.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_WS / "model" / "runtime" / "vendor"))

# noqa: E402 = 告诉 linter "import 不在文件顶是有意的，别报"。
from core.embeddings import ModelScopeEmbedder, read_wav  # noqa: E402
from core.wake_anchor import should_switch_command_timbre  # noqa: E402
from core.model_paths import resolve_model_path  # noqa: E402
from core.text import levenshtein, normalize_transcript  # noqa: E402
from pipeline.helpers import command_features, duration_seconds  # noqa: E402

# 路径常量集中定义：输入（detail/文本/分离缓存/融合包/SV 快照）
# 与输出（sweep 目录）一眼可见。
DATASETA = _WS / "dataset" / "dataset_test" / "datasetA" / "datasetA"
DETAIL = _REPO / "outputs" / "sweep_v4" / "ep6_detail.json"
TEXTS = _REPO / "outputs" / "sweep_v4" / "ep6_replay_texts.json"
CACHE = _REPO / "outputs" / "szy_cache_sweep_best" / "separation"
SWEEP_OUT = _REPO / "outputs" / "sweep_v4"
V4SEP = _WS / "dataset_dasynth_v4sep"
# 现役融合门控包：mean/std/weight/bias（logistic 融合的参数）。
FUSION = _WS / "dataset_dasynth" / "out" / "gate_fusion.json"
# 现役（生产）ft SV 快照：validate 模式用它复现官方分数。
FT_CURRENT = _WS / "dataset_dasynth" / "out" / "ep6_datasetA_best.pt"
# 基础说话人验证模型（ModelScope 模型 id）；微调快照是其 encoder 权重。
SV_MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
# ModelScope 模型缓存根（resolve_model_path 会在此目录下查找本地副本）。
MODEL_ROOT = _WS / "model" / "workspace" / "midea" / "models" / "modelscope"

# 融合门控的 6 维特征列顺序：全脚本（含下游 x 向量拼接、valid 训练）
# 都必须与这个顺序保持一致，否则融合分数会张冠李戴。
FEATURES = ["pre_raw", "ft_raw", "pre_sep", "ft_sep", "ch_sim_pre", "ch_sim_ft"]


# ---------------------------------------------------------------- 样本与资产

def load_samples() -> list[dict]:
    # 把重放所需的一切样本侧信息拼成统一结构：
    # detail.json（门控/唤醒决策/历史 accepted 与 content）+ 文本资产 +
    # datasetA 的 jsonl 清单（音频路径与标注）三方按 uid 对齐。
    # detail 先转成 {id: record} 字典：后面按 uid 查表是 O(1)，
    # 比每次线性扫列表快；jsonl 逐行读，内存友好。
    detail = {r["id"]: r for r in json.load(DETAIL.open(encoding="utf-8"))["results"]}
    texts = json.load(TEXTS.open(encoding="utf-8"))
    samples = []
    # pos.jsonl / neg.jsonl：JSON Lines 格式，每行一个独立 JSON 对象。
    # 逐行读而不是 json.load 整读：文件大时省内存；且天然支持"读到哪
    # 算哪"的流式处理。
    for jsonl_name, stype in (("pos.jsonl", "positive"), ("neg.jsonl", "negative")):
        for line in (DATASETA / jsonl_name).open(encoding="utf-8"):
            if not line.strip():
                continue  # 跳过空行（文件末尾常有多余换行）
            row = json.loads(line)
            # uid = 子集目录/文件名（去扩展名）：与 detail 的 id 口径一致。
            uid = f"{row['识别音频'].split('/')[0]}/{Path(row['识别音频']).stem}"
            d = detail[uid]
            t = texts[uid]
            # 分离缓存命名：uid 里的 "/" 换成 "_"（与分离器落盘规则一致）。
            item = uid.replace("/", "_")
            samples.append({
                "id": uid,
                "is_pos": stype == "positive",
                "wake_audio": DATASETA / row["唤醒音频"],
                "cmd_audio": DATASETA / row["识别音频"],
                # 正样本有标注指令文本（CER 的参考），先过一遍口径归一化；
                # 负样本无标注（应当拒绝），label 为空串。
                "label": normalize_transcript(row["识别文本"] or "") if stype == "positive" else "",
                # 以下字段直接从 detail 沿用历史运行的选择——重放"沿用
                # 落盘的决策"而不是重新决策，这正是 replay 的前提：
                # 唤醒锚点选哪个通道、是否回退原始唤醒音频，这些是分离
                # 阶段的产物，重放不重跑分离，所以照搬 detail。
                "wake_raw_fallback": bool(d["wake_raw_fallback"]),
                "wake_selected_source": int(d["wake_selected_source"]),
                "command_source_empty": bool(d["command_source_empty"]),
                # 锚点音频：若历史运行回退到原始唤醒音频则用原音频；
                # 否则用分离缓存里被选中的唤醒通道。
                "anchor_path": (DATASETA / row["唤醒音频"]) if d["wake_raw_fallback"]
                               else CACHE / f"{item}_wake_ms{int(d['wake_selected_source'])}.wav",
                "cmd0_path": CACHE / f"{item}_command_ms0.wav",
                "cmd1_path": CACHE / f"{item}_command_ms1.wav",
                "raw_text": t["raw_text"],
                "raw_confidence": float(t["raw_confidence"]),
                "cmd0_text": t["cmd0_text"],
                "cmd1_text": t["cmd1_text"],
                # 历史运行的 content 与 accepted 留作重放正确性校验
                # （validate 模式逐样本比对，见 replay 的 content_mismatch）。
                "stored_content": d.get("content", ""),
                "stored_accepted": bool(d["accepted"]),
            })
    return samples


# ---------------------------------------------------------------- SV 特征

def build_embedder(ft_ckpt: Path | None, device: str) -> ModelScopeEmbedder:
    # 组装 SV embedding 提取器。torch 是重型依赖（几秒到十几秒的 import +
    # 显存初始化），放进函数内 import：本模块 import 时不需要 torch，
    # 只在真正要建 embedder 时才加载——报"缺 torch"的时机也被推迟到
    # 用到的那一刻，错误信息更贴近真实原因。
    import torch

    # 基础 ERes2NetV2 模型（预训练声纹模型），resolve_model_path 会在
    # MODEL_ROOT 下定位模型本地副本（找不到再走下载逻辑）。
    emb = ModelScopeEmbedder(resolve_model_path(SV_MODEL, MODEL_ROOT), device)
    if ft_ckpt is not None:
        # 载入微调快照：map_location="cpu" 保证无论快照当初在哪个 GPU
        # 上保存，都能先加载到 CPU（目标设备可能没有足够的空闲显存）。
        # weights_only=False：历史快照是完整 pickle 保存的字典
        # （含非张量元数据），若按新 torch 默认的 weights_only=True
        # 加载会直接报错。
        ckpt = torch.load(str(ft_ckpt), map_location="cpu", weights_only=False)
        # 用微调后的 encoder 权重覆盖基础模型的 embedding 网络——
        # 基础模型 + 微调权重 = "ft SV"。
        emb.pipeline.model.embedding_model.load_state_dict(ckpt["encoder"])
    return emb


def dataseta_sv_features(samples: list[dict], tag: str, ckpt: Path | None, device: str,
                         cache: bool) -> dict[str, dict[str, float]]:
    """每样本 6 维门控特征 + ft 逐通道余弦/ft_rawcmd。带 npz 缓存（pre 用）。

    （扩充）pre（未微调基础模型）的特征与"用哪个 ft 快照"无关，可以安全
    缓存复用；ft 特征每换一个快照就要重算。所以调用方对 pre 传 cache=True、
    对 ft 传 cache=False：多次 eval 实验共享一份 pre 缓存，省下大量 GPU 时间。
    缓存载体 .npz 是 numpy 的压缩数组容器。
    """
    cache_path = SWEEP_OUT / f"dataseta_{tag}_feats.npz"
    if cache and cache_path.is_file():
        # 命中缓存：直接解包。值当初是以"JSON 字节流"形式存成 uint8
        # 数组的（见下方 np.savez），所以读回时 bytes(z[k]).decode("utf-8")
        # 再 json.loads 还原成 {特征名: float} 字典。
        # 选 JSON 字节而不是 pickle：可读、可迁移、无 pickle 反序列化风险。
        z = np.load(cache_path, allow_pickle=True)
        return {k: json.loads(bytes(z[k]).decode("utf-8")) for k in z.files}
    emb = build_embedder(ckpt, device)
    out: dict[str, dict[str, float]] = {}
    for s in samples:
        # 一次读入 5 条音频（锚点/原始唤醒/原始指令/两分离指令通道），
        # extract_waveforms 批量前向——一次调用算完，避免 5 次模型调用
        # 的重复开销（GPU 打包前向比逐条快得多）。
        waves = [read_wav(s["anchor_path"]), read_wav(s["wake_audio"]), read_wav(s["cmd_audio"]),
                 read_wav(s["cmd0_path"]), read_wav(s["cmd1_path"])]
        e = emb.extract_waveforms(waves)
        e_anchor, e_rwake, e_rcmd, e_c0, e_c1 = e

        def cos(a, b) -> float:
            # embedding 已 L2 归一化（模长为 1），点积即余弦相似度。
            # 局部函数：只在本函数内使用的小工具，就近定义最清晰。
            return float(np.dot(a, b))

        # "分离"特征 = 锚点与两分离通道余弦的较大者；若历史记录显示
        # 分离通道为空（分离失败），则用原始指令音频顶替（退化回退，
        # 与生产管线的历史行为一致）。
        candidates = [cos(e_anchor, e_c0), cos(e_anchor, e_c1)]
        if s["command_source_empty"]:
            candidates.append(cos(e_anchor, e_rcmd))
        out[s["id"]] = {
            f"{tag}_raw": cos(e_rwake, e_rcmd),
            f"{tag}_sep": max(candidates),
            f"ch_sim_{tag}": cos(e_c0, e_c1),
            # 下面三个余弦供重放阶段做"ft 直接余弦选源"和 sv_diff：
            # 锚点 vs 两分离通道、锚点 vs 原始指令。
            f"{tag}_cos_cmd0": cos(e_anchor, e_c0),
            f"{tag}_cos_cmd1": cos(e_anchor, e_c1),
            f"{tag}_rawcmd": cos(e_anchor, e_rcmd),
        }
    if cache:
        # 写缓存：np.savez 要求值都是数组，所以把每样本的 {特征: float}
        # 字典序列化成 JSON 字节串，再 frombuffer 成 uint8 数组存进去。
        np.savez(cache_path, **{k: np.frombuffer(json.dumps(v).encode(), dtype=np.uint8)
                                for k, v in out.items()})
    return out


# ---------------------------------------------------------------- valid 侧

def load_valid_common() -> tuple[list[dict], dict, dict]:
    # valid 侧三份元数据：
    # valid.jsonl          逐条记录（id、两个说话人 source 音频与起止采样点等）；
    # valid_mossformer2.jsonl  分离器输出映射（s1/s2 分别来自 mos0/mos1 哪个通道）；
    # valid_moss_text.json 两通道转写文本（build_replay_assets.py 的产物）。
    # 列表推导里带 if l.strip() 过滤空行：jsonl 文件常见末尾空行。
    records = [json.loads(l) for l in (V4SEP / "metadata" / "valid.jsonl").open(encoding="utf-8") if l.strip()]
    # sep_map 转成 {id: record} 字典，方便按 id 查分离映射。
    sep_map = {r["id"]: r for r in
               (json.loads(l) for l in (V4SEP / "metadata" / "valid_mossformer2.jsonl").open(encoding="utf-8") if l.strip())}
    moss_text = json.load((V4SEP / "metadata" / "valid_moss_text.json").open(encoding="utf-8"))
    return records, sep_map, moss_text


def valid_trials(records: list[dict], sep_map: dict) -> list[dict]:
    # 构造说话人验证风格的"试验对"：每条 usable 记录 × 2 个说话人侧，
    # 各配一条正样本试验（同一记录，label=1）和一条负样本试验
    # （随机抽另一条、且说话人不重复的记录，label=0）。
    # 这是给 logistic 定阈造的训练数据：正负各半，覆盖"同人/不同人"
    # 两种决策场景。
    usable = []
    for r in records:
        sep = sep_map.get(r["id"])
        if sep is None:
            continue  # 没有分离映射的记录不可用（无法构造通道特征）
        # 搞清 s1/s2 各对应哪个分离通道（mos0/mos1），
        # 依据是分离映射里的 {key}_reference 字段。
        mos_of = {"s1": None, "s2": None}
        for key in ("mos0", "mos1"):
            ref = sep[f"{key}_reference"]
            if ref in ("s1", "s2"):
                mos_of[ref] = key
        # 两个说话人都要有对应通道才算可用，否则跳过。
        if mos_of["s1"] is None or mos_of["s2"] is None:
            continue
        r["_mos_of"] = mos_of
        usable.append(r)
    # default_rng(0)：numpy 新一代随机数接口，固定种子保证每次运行
    # 抽出完全相同的负样本——实验结果可复现。用 default_rng 而不是
    # 全局 np.random.seed：不污染全局随机状态，其他代码不受影响。
    rng = np.random.default_rng(0)
    trials = []
    for r in usable:
        for side in ("1", "2"):
            spk = r[f"speaker_{side}"]
            trials.append({"enroll": r, "side": side, "test": r, "label": 1})
            # 抽负样本：随机选一条记录，要求测试说话人与注册说话人不同
            # （保证"真负"），且不是同一条记录；不满足就重抽。
            # 这是一个"随机 + 条件校验"的 while 循环，条件在集合很小
            # 时理论上可能多转几圈，但实际数据下很快命中。
            while True:
                b = usable[int(rng.integers(len(usable)))]
                if spk != b["speaker_1"] and spk != b["speaker_2"] and b["id"] != r["id"]:
                    break
            trials.append({"enroll": r, "side": side, "test": b, "label": 0})
    return trials


def valid_sv_features(tag: str, ckpt: Path | None, device: str, cache: bool) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X[N,3]: raw/sep/ch_sim, y[N])。pre 侧带 npz 缓存。

    （扩充）X 的三列与 datasetA 侧 6 维特征不同——valid 侧只有 pre/ft 各
    3 维，训练时拼成 6 列（见 main 的 x_tr 拼接）。y 是正负标签。
    """
    cache_path = SWEEP_OUT / f"valid_{tag}_feats.npz"
    if cache and cache_path.is_file():
        # pre 特征与 ft 快照无关，缓存复用（同 datasetA 侧思路）。
        z = np.load(cache_path)
        return z["X"], z["y"]
    records, sep_map, moss_text = load_valid_common()
    trials = valid_trials(records, sep_map)
    emb = build_embedder(ckpt, device)

    # 唯一化 enroll/test 波形：
    # 同一波形会在多条 trial 里反复出现（同一注册人配多个测试），
    # 如果每条 trial 都重算一遍 embedding，GPU 时间成倍浪费。
    # 先收集"去重后的键"，各算一次 embedding，再按索引取用。
    enroll_keys = sorted({(t["enroll"]["id"], t["side"]) for t in trials})
    test_ids = sorted({t["test"]["id"] for t in trials})
    # 键 -> 行号 的映射表：后面按 (id, side) / id 直接定位 embedding 行。
    e_index = {k: i for i, k in enumerate(enroll_keys)}
    t_index = {k: i for i, k in enumerate(test_ids)}
    waves: list[np.ndarray] = []
    # 注册波形：从说话人的 source 音频里按记录的起止采样点切出他本人的
    # 语音段（而不是整段混音）——注册必须用"纯"的说话人片段。
    for eid, side in enroll_keys:
        r = next(r for r in records if r["id"] == eid)
        w = read_wav(V4SEP / r[f"source_{side}_audio"])
        waves.append(w[r[f"source_{side}_start_sample"]: r[f"source_{side}_start_sample"] + r[f"source_{side}_num_samples"]])
    # 测试波形：混音 + 两个分离通道，各算一遍 embedding。
    for mid in test_ids:
        waves.append(read_wav(V4SEP / "valid" / "mix" / f"{mid}.wav"))
    for mid in test_ids:
        waves.append(read_wav(V4SEP / "valid" / "mos0" / f"{mid}.wav"))
    for mid in test_ids:
        waves.append(read_wav(V4SEP / "valid" / "mos1" / f"{mid}.wav"))
    # 全部波形打包一次前向。结果布局约定：
    # [0, n_e) = 注册；[n_e, n_e+n_t) = 混音；之后各 n_t 为 mos0/mos1。
    embs = emb.extract_waveforms(waves)

    n_e, n_t = len(enroll_keys), len(test_ids)

    def cos(a, b) -> float:
        return float(np.dot(a, b))

    rows, labels = [], []
    for t in trials:
        j = t_index[t["test"]["id"]]
        # 分离通道是否"空"（转写文本为空串）——空通道视为分离失败，
        # 与 datasetA 侧一样把 raw 混音加进候选（历史语义一致）。
        texts = moss_text.get(t["test"]["id"], {})
        ms0_empty = not texts.get("ms0", "").strip()
        ms1_empty = not texts.get("ms1", "").strip()
        e_enroll = embs[e_index[(t["enroll"]["id"], t["side"])]]
        e_mix = embs[n_e + j]
        e_ms0 = embs[n_e + n_t + j]
        e_ms1 = embs[n_e + 2 * n_t + j]
        raw = cos(e_enroll, e_mix)
        candidates = [cos(e_enroll, e_ms0), cos(e_enroll, e_ms1)]
        if ms0_empty or ms1_empty:
            candidates.append(raw)
        # 每条 trial 一行特征：[raw, sep(两通道较大者), ch_sim(两通道间)]。
        rows.append([raw, max(candidates), cos(e_ms0, e_ms1)])
        labels.append(t["label"])
    # 显式转 float32 / int64：保证下游 torch 张量转换类型正确，
    # 也是落盘缓存体积更小的类型。
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    if cache:
        np.savez(cache_path, X=X, y=y)
    return X, y


# ---------------------------------------------------------------- logistic

def train_logistic(x_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    # 手工实现 logistic 回归（torch 自动微分 + Adam）+ 训练集上扫描定阈。
    # 不用 sklearn 的原因：1) 少一个依赖；2) 定阈口径（balanced 精度
    # 最大化）必须与历史实现完全一致，自己写才能逐位复现。
    import torch

    # 固定随机种子：Adam 的初始化/更新是确定性的，但种子固定后每次
    # 运行结果逐位一致，实验可复现（论文级实验的基本素养）。
    torch.manual_seed(0)
    # 特征标准化（z-score）：不同特征量纲不同（如 raw 余弦在 0.5 附近、
    # ch_sim 可能接近 1），不标准化的话梯度更新会被大量纲特征主导。
    # 分母 + 1e-8 是防除零的数值保险（某列方差恰为 0 时）。
    # 注意 mean/std 会被存进返回字典：推理侧必须用同一套统计量
    # 标准化新数据，否则训练/推理口径不一致。
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-8
    xs = torch.from_numpy((x_tr - mean) / std)
    yt = torch.from_numpy(y_tr).float()
    # 单层线性模型：z = x @ w + b，requires_grad=True 让 torch 记录
    # 计算图、自动求梯度（不需要手推反向传播公式）。
    w = torch.zeros(x_tr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    # BCEWithLogitsLoss = sigmoid + 二分类交叉熵的融合实现，数值上比
    # "先手动 sigmoid 再 BCE"更稳（避免 log(0) 这类溢出）。
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(3000):
        # 固定步数的训练循环三步曲：清梯度 -> 前向+损失 -> 反向+更新。
        # 每次迭代前 zero_grad 清空累积梯度，否则梯度会跨迭代累加。
        opt.zero_grad()
        loss = loss_fn(xs @ w + b, yt)
        loss.backward()
        opt.step()
    # 训练结束后在训练集上扫描定阈：no_grad 关掉梯度记录（此时不需要
    # 反向传播，省内存也更快）。
    with torch.no_grad():
        s = ((xs @ w + b)).numpy()
    # 阈值只在"数据中实际出现的分数值"里选：相邻两个分数值之间没有
    # 样本，阈值取中间任何位置决策结果都一样，所以唯一值扫描 = 精确穷举。
    best = None
    for t in np.unique(s):
        accept = s >= t
        pos = y_tr == 1
        # pa：正样本召回（positive accuracy）；nr：负样本拒绝率。
        # 分母 max(..., 1) 防御"某类样本数为 0"的除零。
        pa = float((accept & pos).sum() / max(pos.sum(), 1))
        nr = float((~accept & ~pos).sum() / max((~pos).sum(), 1))
        # balanced = (pa + nr) / 2：对正负类一视同仁的评价口径，
        # 避免"全拒绝也有高准确率"的类别不平衡陷阱。
        bal = (pa + nr) / 2
        if best is None or bal > best[0]:
            best = (bal, float(t))
    # 返回完整"融合包"：标准化统计量 + 权重/偏置 + 选出的阈值 + 该阈值的
    # balanced 分数（后者只是记录，供报告参考）。
    return {"mean": mean, "std": std, "w": w.detach().numpy(), "b": float(b.detach()[0]),
            "threshold": best[1], "valid_balanced": best[0]}


# ---------------------------------------------------------------- 重放

def replay(samples, fused: dict[str, float], fts: dict[str, dict], threshold: float) -> dict:
    # 重放核心：纯 Python 决策循环，不碰任何模型。输入全部是预先算好的
    # 量（融合分数、ft 余弦特征、转写文本），所以换阈值/换融合包都只是
    # 改输入参数，重跑这个循环只需几秒。
    total_err = 0        # 全部正样本的编辑错误数（CER 分子）
    total_chars = 0      # 全部正样本的标注字符数（CER 分母）
    n_pos_acc = 0        # 正样本中输出非空（被"接受且给出内容"）的个数
    n_pos = 0
    n_neg_rej = 0        # 负样本中被拒绝（空输出）的个数
    n_neg = 0
    content_mismatch = 0  # 与历史 detail 的 content 不一致的样本数
    for s in samples:
        # 门控只做一件事：融合分过阈值就接受，否则拒绝。
        accept = fused[s["id"]] >= threshold
        if s["is_pos"]:
            n_pos += 1
        else:
            n_neg += 1
        if not accept:
            if s["is_pos"]:
                # 拒绝正样本：输出视为空。按竞赛口径，空输出对整条
                # 标注文本全错——CER 分子加整条长度。
                total_chars += len(s["label"])
                total_err += len(s["label"])
                # 交叉校验：历史运行此样本是接受的，而重放却拒绝
                # ——重放与历史不一致（对 validate 模式意味着重放器有 bug）。
                if s["stored_accepted"]:
                    content_mismatch += 1
            else:
                n_neg_rej += 1  # 正确拒绝
            continue
        # ---- 接受分支：走"选源"决策链（与生产管线同款规则）----
        f = fts[s["id"]]
        # sv_diff：音色触发特征 = 锚点与分离通道的最大余弦 - 锚点与原始
        # 混音的余弦。物理含义：目标嗓音在分离通道里是否比在混音里更突出。
        sv_diff = max(f["ft_cos_cmd0"], f["ft_cos_cmd1"]) - f["ft_rawcmd"]
        # command_features 返回 (字速, 关键词命中数)；duration_seconds 只读
        # 音频文件头取时长（不解码，几乎零成本）。
        cps, kw = command_features(s["raw_text"], duration_seconds(s["cmd_audio"]))
        # 音色触发规则（冻结的历史决策函数，见 moss_candidate 模块注释）：
        # 是否"换用分离通道的转写"。
        switch = should_switch_command_timbre(
            confidence=s["raw_confidence"], command_keyword_score=kw, sv_diff=sv_diff)
        if switch and not s["command_source_empty"]:
            # 切换 + 分离通道可用：按 ft 余弦直接选更贴锚点的通道。
            sel = "ms0" if f["ft_cos_cmd0"] >= f["ft_cos_cmd1"] else "ms1"
        elif switch:
            sel = "orig"
        else:
            sel = "orig"
        # 按选定的来源取最终 content：原始转写或某个分离通道的转写。
        content = {"orig": s["raw_text"], "ms0": s["cmd0_text"], "ms1": s["cmd1_text"]}[sel]
        if s["is_pos"]:
            if content.strip():
                n_pos_acc += 1
            # CER 口径：归一化后的编辑距离 / 归一化标注长度。
            total_chars += len(s["label"])
            total_err += levenshtein(normalize_transcript(content), s["label"])
            # 与历史 content 逐样本比对（归一化后比，避免空白差异误报）。
            if normalize_transcript(content) != normalize_transcript(s["stored_content"]):
                content_mismatch += 1
    # max(..., 1) 防除零（如样本全为负时 n_pos=0）。
    pr = n_pos_acc / max(n_pos, 1)
    rr = n_neg_rej / max(n_neg, 1)
    cer = total_err / max(total_chars, 1)
    # round(..., 4)：输出四位小数，与官方报告口径一致。
    # 公式分 = (1 - CER + RR) / 2：竞赛官方公式，PR 不直接进分。
    return {"PR": round(pr, 4), "RR": round(rr, 4), "CER": round(cer, 4),
            "formula": round((1 - cer + rr) / 2, 4), "content_mismatch": content_mismatch}


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser()
    # choices 限定两种模式：validate（复现官方分）/ eval（新快照评测）。
    ap.add_argument("--mode", choices=["validate", "eval"], required=True)
    # type=Path：argparse 直接把参数值转成 Path 对象，省去手动转换。
    ap.add_argument("--ft-ckpt", type=Path, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    samples = load_samples()
    print(f"samples={len(samples)}", flush=True)

    if args.mode == "validate":
        # pre 特征（基础模型）缓存复用；ft 用现役生产快照。
        # 两个模型各自跑一遍 6 维特征。
        pre = dataseta_sv_features(samples, "pre", None, args.device, cache=True)
        ft = dataseta_sv_features(samples, "ft", FT_CURRENT, args.device, cache=True)
        # 现役融合包（生产管线训好的 logistic 参数）原样载入。
        bundle = json.load(FUSION.open(encoding="utf-8"))
        mean = np.asarray(bundle["mean"], dtype=np.float32)
        std = np.asarray(bundle["std"], dtype=np.float32)
        weight = np.asarray(bundle["weight"], dtype=np.float32)
        bias = float(bundle["bias"])
        # 生产阈值（历史标定值，不是本脚本算出来的）：用它重放必须
        # 复现官方公式分 0.8679，否则重放器有 bug。
        thr = -0.6469
    else:
        # eval：必须提供重训的 SV 快照（assert 在程序入口处把
        # "参数缺了"这类配置错误尽早暴露，报错信息清晰）。
        assert args.ft_ckpt is not None
        pre = dataseta_sv_features(samples, "pre", None, args.device, cache=True)
        # ft 特征不缓存：每个新快照都是全新特征，缓存了也没人复用。
        ft = dataseta_sv_features(samples, "ft", args.ft_ckpt, args.device, cache=False)
        # valid 侧重训 logistic（pre 缓存 + ft 现算）。
        # valid_sv_features 返回 (X[N,3], y[N])：pre 与 ft 各 3 维
        # （raw/sep/ch_sim），按 FEATURES 的列序交错拼成 6 维训练矩阵——
        # 列顺序必须与 datasetA 侧融合时的 x 向量完全一致。
        xp, yv = valid_sv_features("pre", None, args.device, cache=True)
        xf, _ = valid_sv_features("ft", args.ft_ckpt, args.device, cache=False)
        x_tr = np.concatenate([xp[:, [0]], xf[:, [0]], xp[:, [1]], xf[:, [1]],
                               xp[:, [2]], xf[:, [2]]], axis=1)
        bundle_t = train_logistic(x_tr, yv)
        mean, std, weight, bias = bundle_t["mean"], bundle_t["std"], bundle_t["w"], bundle_t["b"]
        thr = bundle_t["threshold"]
        print(f"[valid] logistic threshold={thr:.4f} balanced={bundle_t['valid_balanced']:.4f}", flush=True)

    # 融合门控分：6 维特征 -> 标准化 -> 加权求和 + 偏置。
    # 注意运算优先级：先 (x - mean) / std 得标准化向量，@ 是矩阵乘法
    # （这里就是点积），再加 bias。广播机制让每个样本的 6 维数组同时算。
    fused: dict[str, float] = {}
    for s in samples:
        uid = s["id"]
        x = np.array([pre[uid]["pre_raw"], ft[uid]["ft_raw"], pre[uid]["pre_sep"],
                      ft[uid]["ft_sep"], pre[uid]["ch_sim_pre"], ft[uid]["ch_sim_ft"]],
                     dtype=np.float32)
        fused[uid] = float((x - mean) / std @ weight + bias)
    m = replay(samples, fused, ft, thr)
    # ensure_ascii=False：中文标签等原样输出，日志可读。
    print(json.dumps({"tag": args.tag, "threshold": thr, **m}, ensure_ascii=False), flush=True)
    if args.tag:
        # 追加一行到 jsonl 结果日志：open("a") 追加模式，每次实验只加
        # 一行，不用读旧文件、不用重写全文件——多次实验的累计日志
        # 天然有序、可逐行 grep 分析。每行一个 JSON 对象 = 增量写出。
        with (SWEEP_OUT / "sv_sweep_results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": args.tag, "threshold": float(thr), **m},
                                ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
