#!/usr/bin/env python3
"""v4 全链路离线重放器：给定 ft SV checkpoint，精确重放整条 szy_v4 决策链。

用途：
  1) validate：用现役 ft（ep6_datasetA_best）+ 现役融合包 + -0.6469 重放，
     必须逐位复现官方 0.8679（校验重放器正确性）；
  2) eval：对重训的 SV 快照（out_sv_v4sep/ep{N}.pt），先在新分离器的
     dasynth valid 上重训 logistic、valid 定阈（零 DatasetA），再重放
     DatasetA 全链路出公式分。

重放内容：唤醒锚点（沿用 detail 存的选择）→ 6 维门控特征（重算 SV embedding）
→ 融合门控 → 音色触发（sv_diff 重算）→ ft 直接余弦选源 → content。
文本/置信度用 build_v4_replay_assets.py 的资产，不重跑 ASR/分离。

用法（yuyin，仓库根目录）：
  PYTHONPATH=src python scripts/replay_v4_full.py --mode validate
  PYTHONPATH=src python scripts/replay_v4_full.py --mode eval \
    --ft-ckpt ../dataset_dasynth/out_sv_v4sep/ep6.pt --tag svv4_ep6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_WS = _REPO.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_WS / "model" / "runtime" / "vendor"))

from midea_qiji.gate import ModelScopeEmbedder, read_wav  # noqa: E402
from midea_qiji.moss_candidate import should_switch_command_timbre  # noqa: E402
from midea_qiji.paths import resolve_model_path  # noqa: E402
from midea_qiji.text import levenshtein, normalize_transcript  # noqa: E402
from szy_common.helpers import command_features, duration_seconds  # noqa: E402

DATASETA = _WS / "dataset" / "dataset_test" / "datasetA" / "datasetA"
DETAIL = _REPO / "outputs" / "sweep_v4" / "ep6_detail.json"
TEXTS = _REPO / "outputs" / "sweep_v4" / "ep6_replay_texts.json"
CACHE = _REPO / "outputs" / "szy_v4_cache_sweep_best" / "separation"
SWEEP_OUT = _REPO / "outputs" / "sweep_v4"
V4SEP = _WS / "dataset_dasynth_v4sep"
FUSION = _WS / "dataset_dasynth" / "out" / "gate_fusion.json"
FT_CURRENT = _WS / "dataset_dasynth" / "out" / "ep6_datasetA_best.pt"
SV_MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
MODEL_ROOT = _WS / "model" / "workspace" / "midea" / "models" / "modelscope"

FEATURES = ["pre_raw", "ft_raw", "pre_sep", "ft_sep", "ch_sim_pre", "ch_sim_ft"]


# ---------------------------------------------------------------- 样本与资产

def load_samples() -> list[dict]:
    detail = {r["id"]: r for r in json.load(DETAIL.open(encoding="utf-8"))["results"]}
    texts = json.load(TEXTS.open(encoding="utf-8"))
    samples = []
    for jsonl_name, stype in (("pos.jsonl", "positive"), ("neg.jsonl", "negative")):
        for line in (DATASETA / jsonl_name).open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = f"{row['识别音频'].split('/')[0]}/{Path(row['识别音频']).stem}"
            d = detail[uid]
            t = texts[uid]
            item = uid.replace("/", "_")
            samples.append({
                "id": uid,
                "is_pos": stype == "positive",
                "wake_audio": DATASETA / row["唤醒音频"],
                "cmd_audio": DATASETA / row["识别音频"],
                "label": normalize_transcript(row["识别文本"] or "") if stype == "positive" else "",
                "wake_raw_fallback": bool(d["wake_raw_fallback"]),
                "wake_selected_source": int(d["wake_selected_source"]),
                "command_source_empty": bool(d["command_source_empty"]),
                "anchor_path": (DATASETA / row["唤醒音频"]) if d["wake_raw_fallback"]
                               else CACHE / f"{item}_wake_ms{int(d['wake_selected_source'])}.wav",
                "cmd0_path": CACHE / f"{item}_command_ms0.wav",
                "cmd1_path": CACHE / f"{item}_command_ms1.wav",
                "raw_text": t["raw_text"],
                "raw_confidence": float(t["raw_confidence"]),
                "cmd0_text": t["cmd0_text"],
                "cmd1_text": t["cmd1_text"],
                "stored_content": d.get("content", ""),
                "stored_accepted": bool(d["accepted"]),
            })
    return samples


# ---------------------------------------------------------------- SV 特征

def build_embedder(ft_ckpt: Path | None, device: str) -> ModelScopeEmbedder:
    import torch

    emb = ModelScopeEmbedder(resolve_model_path(SV_MODEL, MODEL_ROOT), device)
    if ft_ckpt is not None:
        ckpt = torch.load(str(ft_ckpt), map_location="cpu", weights_only=False)
        emb.pipeline.model.embedding_model.load_state_dict(ckpt["encoder"])
    return emb


def dataseta_sv_features(samples: list[dict], tag: str, ckpt: Path | None, device: str,
                         cache: bool) -> dict[str, dict[str, float]]:
    """每样本 6 维门控特征 + ft 逐通道余弦/ft_rawcmd。带 npz 缓存（pre 用）。"""
    cache_path = SWEEP_OUT / f"dataseta_{tag}_feats.npz"
    if cache and cache_path.is_file():
        z = np.load(cache_path, allow_pickle=True)
        return {k: json.loads(bytes(z[k]).decode("utf-8")) for k in z.files}
    emb = build_embedder(ckpt, device)
    out: dict[str, dict[str, float]] = {}
    for s in samples:
        waves = [read_wav(s["anchor_path"]), read_wav(s["wake_audio"]), read_wav(s["cmd_audio"]),
                 read_wav(s["cmd0_path"]), read_wav(s["cmd1_path"])]
        e = emb.extract_waveforms(waves)
        e_anchor, e_rwake, e_rcmd, e_c0, e_c1 = e

        def cos(a, b) -> float:
            return float(np.dot(a, b))

        candidates = [cos(e_anchor, e_c0), cos(e_anchor, e_c1)]
        if s["command_source_empty"]:
            candidates.append(cos(e_anchor, e_rcmd))
        out[s["id"]] = {
            f"{tag}_raw": cos(e_rwake, e_rcmd),
            f"{tag}_sep": max(candidates),
            f"ch_sim_{tag}": cos(e_c0, e_c1),
            f"{tag}_cos_cmd0": cos(e_anchor, e_c0),
            f"{tag}_cos_cmd1": cos(e_anchor, e_c1),
            f"{tag}_rawcmd": cos(e_anchor, e_rcmd),
        }
    if cache:
        np.savez(cache_path, **{k: np.frombuffer(json.dumps(v).encode(), dtype=np.uint8)
                                for k, v in out.items()})
    return out


# ---------------------------------------------------------------- valid 侧

def load_valid_common() -> tuple[list[dict], dict, dict]:
    records = [json.loads(l) for l in (V4SEP / "metadata" / "valid.jsonl").open(encoding="utf-8") if l.strip()]
    sep_map = {r["id"]: r for r in
               (json.loads(l) for l in (V4SEP / "metadata" / "valid_mossformer2.jsonl").open(encoding="utf-8") if l.strip())}
    moss_text = json.load((V4SEP / "metadata" / "valid_moss_text.json").open(encoding="utf-8"))
    return records, sep_map, moss_text


def valid_trials(records: list[dict], sep_map: dict) -> list[dict]:
    usable = []
    for r in records:
        sep = sep_map.get(r["id"])
        if sep is None:
            continue
        mos_of = {"s1": None, "s2": None}
        for key in ("mos0", "mos1"):
            ref = sep[f"{key}_reference"]
            if ref in ("s1", "s2"):
                mos_of[ref] = key
        if mos_of["s1"] is None or mos_of["s2"] is None:
            continue
        r["_mos_of"] = mos_of
        usable.append(r)
    rng = np.random.default_rng(0)
    trials = []
    for r in usable:
        for side in ("1", "2"):
            spk = r[f"speaker_{side}"]
            trials.append({"enroll": r, "side": side, "test": r, "label": 1})
            while True:
                b = usable[int(rng.integers(len(usable)))]
                if spk != b["speaker_1"] and spk != b["speaker_2"] and b["id"] != r["id"]:
                    break
            trials.append({"enroll": r, "side": side, "test": b, "label": 0})
    return trials


def valid_sv_features(tag: str, ckpt: Path | None, device: str, cache: bool) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X[N,3]: raw/sep/ch_sim, y[N])。pre 侧带 npz 缓存。"""
    cache_path = SWEEP_OUT / f"valid_{tag}_feats.npz"
    if cache and cache_path.is_file():
        z = np.load(cache_path)
        return z["X"], z["y"]
    records, sep_map, moss_text = load_valid_common()
    trials = valid_trials(records, sep_map)
    emb = build_embedder(ckpt, device)

    # 唯一化 enroll/test 波形
    enroll_keys = sorted({(t["enroll"]["id"], t["side"]) for t in trials})
    test_ids = sorted({t["test"]["id"] for t in trials})
    e_index = {k: i for i, k in enumerate(enroll_keys)}
    t_index = {k: i for i, k in enumerate(test_ids)}
    waves: list[np.ndarray] = []
    for eid, side in enroll_keys:
        r = next(r for r in records if r["id"] == eid)
        w = read_wav(V4SEP / r[f"source_{side}_audio"])
        waves.append(w[r[f"source_{side}_start_sample"]: r[f"source_{side}_start_sample"] + r[f"source_{side}_num_samples"]])
    for mid in test_ids:
        waves.append(read_wav(V4SEP / "valid" / "mix" / f"{mid}.wav"))
    for mid in test_ids:
        waves.append(read_wav(V4SEP / "valid" / "mos0" / f"{mid}.wav"))
    for mid in test_ids:
        waves.append(read_wav(V4SEP / "valid" / "mos1" / f"{mid}.wav"))
    embs = emb.extract_waveforms(waves)

    n_e, n_t = len(enroll_keys), len(test_ids)

    def cos(a, b) -> float:
        return float(np.dot(a, b))

    rows, labels = [], []
    for t in trials:
        j = t_index[t["test"]["id"]]
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
        rows.append([raw, max(candidates), cos(e_ms0, e_ms1)])
        labels.append(t["label"])
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    if cache:
        np.savez(cache_path, X=X, y=y)
    return X, y


# ---------------------------------------------------------------- logistic

def train_logistic(x_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    import torch

    torch.manual_seed(0)
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-8
    xs = torch.from_numpy((x_tr - mean) / std)
    yt = torch.from_numpy(y_tr).float()
    w = torch.zeros(x_tr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(3000):
        opt.zero_grad()
        loss = loss_fn(xs @ w + b, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = ((xs @ w + b)).numpy()
    best = None
    for t in np.unique(s):
        accept = s >= t
        pos = y_tr == 1
        pa = float((accept & pos).sum() / max(pos.sum(), 1))
        nr = float((~accept & ~pos).sum() / max((~pos).sum(), 1))
        bal = (pa + nr) / 2
        if best is None or bal > best[0]:
            best = (bal, float(t))
    return {"mean": mean, "std": std, "w": w.detach().numpy(), "b": float(b.detach()[0]),
            "threshold": best[1], "valid_balanced": best[0]}


# ---------------------------------------------------------------- 重放

def replay(samples, fused: dict[str, float], fts: dict[str, dict], threshold: float) -> dict:
    total_err = 0
    total_chars = 0
    n_pos_acc = 0
    n_pos = 0
    n_neg_rej = 0
    n_neg = 0
    content_mismatch = 0
    for s in samples:
        accept = fused[s["id"]] >= threshold
        if s["is_pos"]:
            n_pos += 1
        else:
            n_neg += 1
        if not accept:
            if s["is_pos"]:
                total_chars += len(s["label"])
                total_err += len(s["label"])
                if s["stored_accepted"]:
                    content_mismatch += 1
            else:
                n_neg_rej += 1
            continue
        f = fts[s["id"]]
        sv_diff = max(f["ft_cos_cmd0"], f["ft_cos_cmd1"]) - f["ft_rawcmd"]
        cps, kw = command_features(s["raw_text"], duration_seconds(s["cmd_audio"]))
        switch = should_switch_command_timbre(
            confidence=s["raw_confidence"], command_keyword_score=kw, sv_diff=sv_diff)
        if switch and not s["command_source_empty"]:
            sel = "ms0" if f["ft_cos_cmd0"] >= f["ft_cos_cmd1"] else "ms1"
        elif switch:
            sel = "orig"
        else:
            sel = "orig"
        content = {"orig": s["raw_text"], "ms0": s["cmd0_text"], "ms1": s["cmd1_text"]}[sel]
        if s["is_pos"]:
            if content.strip():
                n_pos_acc += 1
            total_chars += len(s["label"])
            total_err += levenshtein(normalize_transcript(content), s["label"])
            if normalize_transcript(content) != normalize_transcript(s["stored_content"]):
                content_mismatch += 1
    pr = n_pos_acc / max(n_pos, 1)
    rr = n_neg_rej / max(n_neg, 1)
    cer = total_err / max(total_chars, 1)
    return {"PR": round(pr, 4), "RR": round(rr, 4), "CER": round(cer, 4),
            "formula": round((1 - cer + rr) / 2, 4), "content_mismatch": content_mismatch}


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "eval"], required=True)
    ap.add_argument("--ft-ckpt", type=Path, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    samples = load_samples()
    print(f"samples={len(samples)}", flush=True)

    if args.mode == "validate":
        pre = dataseta_sv_features(samples, "pre", None, args.device, cache=True)
        ft = dataseta_sv_features(samples, "ft", FT_CURRENT, args.device, cache=True)
        bundle = json.load(FUSION.open(encoding="utf-8"))
        mean = np.asarray(bundle["mean"], dtype=np.float32)
        std = np.asarray(bundle["std"], dtype=np.float32)
        weight = np.asarray(bundle["weight"], dtype=np.float32)
        bias = float(bundle["bias"])
        thr = -0.6469
    else:
        assert args.ft_ckpt is not None
        pre = dataseta_sv_features(samples, "pre", None, args.device, cache=True)
        ft = dataseta_sv_features(samples, "ft", args.ft_ckpt, args.device, cache=False)
        # valid 侧重训 logistic（pre 缓存 + ft 现算）
        xp, yv = valid_sv_features("pre", None, args.device, cache=True)
        xf, _ = valid_sv_features("ft", args.ft_ckpt, args.device, cache=False)
        x_tr = np.concatenate([xp[:, [0]], xf[:, [0]], xp[:, [1]], xf[:, [1]],
                               xp[:, [2]], xf[:, [2]]], axis=1)
        bundle_t = train_logistic(x_tr, yv)
        mean, std, weight, bias = bundle_t["mean"], bundle_t["std"], bundle_t["w"], bundle_t["b"]
        thr = bundle_t["threshold"]
        print(f"[valid] logistic threshold={thr:.4f} balanced={bundle_t['valid_balanced']:.4f}", flush=True)

    fused: dict[str, float] = {}
    for s in samples:
        uid = s["id"]
        x = np.array([pre[uid]["pre_raw"], ft[uid]["ft_raw"], pre[uid]["pre_sep"],
                      ft[uid]["ft_sep"], pre[uid]["ch_sim_pre"], ft[uid]["ch_sim_ft"]],
                     dtype=np.float32)
        fused[uid] = float((x - mean) / std @ weight + bias)
    m = replay(samples, fused, ft, thr)
    print(json.dumps({"tag": args.tag, "threshold": thr, **m}, ensure_ascii=False), flush=True)
    if args.tag:
        with (SWEEP_OUT / "sv_sweep_results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": args.tag, "threshold": float(thr), **m},
                                ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
