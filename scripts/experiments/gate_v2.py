#!/usr/bin/env python3
"""门控 v2 离线实验框架：加滑窗特征 + 按 Score 代价重训 + 5 折 CV。

设计要点：
  1. 训练目标不是准确率，而是 Score=(1-CER+RR)/2 的代价加权
     - 拒 1 个负样本  → +0.5/474
     - 接受正样本 i   → +0.5*(len_i - e_i)/TOTAL_CHARS
     两者构成 3.11:1 的不对称代价，样本权重直接取这个增益。
  2. 评估直接算 Score，不看 AUC；被拒正样本的 CER 未知，用 q 做敏感性分析
  3. 阈值在训练折上选、测试折上用（nested），避免阈值本身过拟合

关键恒等式（使阈值扫描 O(n log n) 而非 O(600n)）：
    Score(accept) = Score(全拒) + sum_{i in accept} gain_i
  其中 gain_i 就是上面的代价权重（负样本为负值）。

数据全部离线，不占 GPU：
  outputs/<run>/<run>_detail.json    6 个原始 gate_features + 实际转写
  outputs/sliding_gate_verify.json   滑窗统计量（对原始命令音频，不依赖分离）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]  # 仓库根（本文件在 scripts/experiments/ 下）
NEG_TOTAL = 474


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def _f(v, dv=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return dv


SEP_FREE = ["pre_raw", "ft_raw"]
ORIG6 = ["pre_raw", "ft_raw", "pre_sep", "ft_sep", "ch_sim_pre", "ch_sim_ft"]
SLIDE4 = ["max_sim", "mean_sim", "std_sim", "min_sim"]
LOW3 = ["low025", "low030", "low035"]
EXTRA = ["n_windows", "wake_raw_fallback"]

FEATURE_SETS = {
    "orig6": ORIG6,
    "orig6+slide4": ORIG6 + SLIDE4,
    "orig6+slide7": ORIG6 + SLIDE4 + LOW3,
    "orig6+all": ORIG6 + SLIDE4 + LOW3 + EXTRA,
    # 激进版：跳过分离后可用的特征（不含任何 *_sep / ch_sim_*）
    "raw2": SEP_FREE,
    "raw2+slide4": SEP_FREE + SLIDE4,
    "raw2+slide7": SEP_FREE + SLIDE4 + LOW3,
    "raw2+all": SEP_FREE + SLIDE4 + LOW3 + EXTRA,
}
ALL_KEYS = ORIG6 + SLIDE4 + LOW3 + EXTRA


def load(detail: Path, sliding: Path, measured: Path | None = None,
         measured_key: str = "err_raw") -> list[dict]:
    det = json.loads(detail.read_text(encoding="utf-8"))["results"]
    sld = {r["id"]: r for r in json.loads(sliding.read_text(encoding="utf-8"))}
    # measure_q.py 补跑的被拒正样本真实错字：有了它就不必再假设 q
    meas = {}
    if measured is not None and measured.exists():
        for m in json.loads(measured.read_text(encoding="utf-8")):
            if measured_key == "err_oracle":
                meas[m["id"]] = min(m["err_raw"], m["err_ms0"], m["err_ms1"])
            else:
                meas[m["id"]] = m[measured_key]
    rows = []
    for r in det:
        sid = r.get("id")
        if sid not in sld:
            continue
        typ = r.get("sample_type")
        if typ not in ("positive", "negative"):
            continue
        if typ == "positive" and not r.get("label"):
            continue
        gf = r.get("gate_features") or {}
        s = sld[sid]
        label = r.get("label") or ""
        content = r.get("content") or ""
        cur_acc = bool(r.get("accepted")) and bool(content)
        rows.append({
            "id": sid,
            "y": 1 if typ == "positive" else 0,
            "ref_len": len(label),
            # 当前已接受样本的真实编辑距离；被拒样本未知（用 q 估）
            "err_known": (lev(content, label) if (typ == "positive" and cur_acc)
                          else (meas.get(sid) if typ == "positive" else None)),
            "cur_accept": cur_acc,
            "sv_score": _f(r.get("sv_score")),
            "feat": {
                "pre_raw": _f(gf.get("pre_raw")), "ft_raw": _f(gf.get("ft_raw")),
                "pre_sep": _f(gf.get("pre_sep")), "ft_sep": _f(gf.get("ft_sep")),
                "ch_sim_pre": _f(gf.get("ch_sim_pre")), "ch_sim_ft": _f(gf.get("ch_sim_ft")),
                "max_sim": _f(s.get("max_sim")), "mean_sim": _f(s.get("mean_sim")),
                "std_sim": _f(s.get("std_sim")), "min_sim": _f(s.get("min_sim")),
                "low025": _f(s.get("low_sim_ratio_025")),
                "low030": _f(s.get("low_sim_ratio_030")),
                "low035": _f(s.get("low_sim_ratio_035")),
                "n_windows": _f(s.get("n_windows")),
                "wake_raw_fallback": 1.0 if r.get("wake_raw_fallback") else 0.0,
            },
        })
    return rows


class Task:
    """把 rows 拍成 numpy，提供 gain / Score 的向量化计算。"""

    def __init__(self, rows):
        self.rows = rows
        self.y = np.array([r["y"] for r in rows])
        self.ref_len = np.array([r["ref_len"] for r in rows], dtype=np.float64)
        self.err = np.array([np.nan if r["err_known"] is None else r["err_known"]
                             for r in rows], dtype=np.float64)
        self.known = np.isfinite(self.err)
        self.X = np.array([[r["feat"][k] for k in ALL_KEYS] for r in rows], dtype=np.float64)
        if not np.isfinite(self.X).all():
            med = np.nanmedian(np.where(np.isfinite(self.X), self.X, np.nan), axis=0)
            bad = np.where(~np.isfinite(self.X))
            self.X[bad] = med[bad[1]]
        self.col = {k: i for i, k in enumerate(ALL_KEYS)}

    def cols(self, keys):
        return self.X[:, [self.col[k] for k in keys]]

    def gain(self, q: float, idx=None):
        """accept 第 i 条相对『全拒』带来的 Score 增益（负样本为负）。"""
        i = slice(None) if idx is None else idx
        y, rl, err, kn = self.y[i], self.ref_len[i], self.err[i], self.known[i]
        e = np.where(kn, err, q * rl)
        tot = rl[y == 1].sum()
        n_neg = int((y == 0).sum())
        g = np.where(y == 1, 0.5 * np.maximum(rl - e, 0.0) / tot, -0.5 / n_neg)
        base = 0.5 * 1.0  # 全拒时 CER=1, RR=1 → Score = (1-1+1)/2
        return g, base

    def score(self, accept, q: float, idx=None):
        g, base = self.gain(q, idx)
        i = slice(None) if idx is None else idx
        y, rl, err, kn = self.y[i], self.ref_len[i], self.err[i], self.known[i]
        a = np.asarray(accept, dtype=bool)
        e = np.where(kn, err, q * rl)
        tot = rl[y == 1].sum()
        n_neg = int((y == 0).sum())
        cer = (np.where(a, e, rl)[y == 1].sum()) / tot
        rr = int(((~a) & (y == 0)).sum()) / n_neg
        pr = int((a & (y == 1)).sum()) / int((y == 1).sum())
        return {"PR": pr, "RR": rr, "CER": cer, "Score": (1 - cer + rr) / 2,
                "check": base + g[a].sum()}

    def best_threshold(self, s, q, idx):
        """O(n log n)：按分数降序累加 gain，峰值处即最优阈值。"""
        g, _ = self.gain(q, idx)
        o = np.argsort(-s, kind="stable")
        cum = np.cumsum(g[o])
        k = int(np.argmax(cum))
        if cum[k] <= 0:
            return float(s.max()) + 1.0     # 全拒更优
        return float(s[o][k])


def run_cv(task, keys, q, seed=0, folds=5, model="lr"):
    X = task.cols(keys)
    y = task.y
    oof = np.zeros(len(y))
    accept = np.zeros(len(y), dtype=bool)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w, _ = task.gain(q, tr)
        w = np.abs(w)
        w = w / w.mean()
        if model == "lr":
            clf = LogisticRegression(C=1.0, max_iter=4000)
            clf.fit(Xtr, y[tr], sample_weight=w)
            s_tr, s_te = clf.decision_function(Xtr), clf.decision_function(Xte)
        else:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=8, max_depth=3,
                                     learning_rate=0.05, min_child_samples=30,
                                     subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1)
            clf.fit(Xtr, y[tr], sample_weight=w)
            s_tr, s_te = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]
        t = task.best_threshold(s_tr, q, tr)   # 阈值只在训练折上选
        oof[te] = s_te
        accept[te] = s_te >= t
    return oof, accept


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    o = np.argsort(s)
    rk = np.empty(len(s), float)
    rk[o] = np.arange(1, len(s) + 1)
    a, b = (y == 1).sum(), (y == 0).sum()
    return (rk[y == 1].sum() - a * (a + 1) / 2) / (a * b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", default="outputs/no_conditional_full/no_conditional_full_detail.json")
    ap.add_argument("--sliding", default="outputs/sliding_gate_verify.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--measured", default=None,
                    help="measure_q.json 路径；给了就用真实错字，q 只影响仍未知的样本")
    ap.add_argument("--measured-key", default="err_raw",
                    choices=["err_raw", "err_ms0", "err_ms1", "err_oracle"])
    args = ap.parse_args()

    rows = load(ROOT / args.detail, ROOT / args.sliding,
                (ROOT / args.measured) if args.measured else None, args.measured_key)
    task = Task(rows)
    n_unknown = sum(1 for r in rows if r["y"] == 1 and r["err_known"] is None)
    print(f"正样本中错字未知(仍需假设 q)的: {n_unknown}")
    y = task.y
    print(f"样本 {len(rows)}  正 {int(y.sum())} / 负 {int((1-y).sum())}  "
          f"参考总字符 {int(task.ref_len[y == 1].sum())}")

    sv = np.array([r["sv_score"] for r in rows])
    base_acc = np.array([r["cur_accept"] for r in rows])
    b = task.score(base_acc, 0.08)
    print("\n=== 基线（现有 6 特征线性门控 @ -0.6469）===")
    print(f"  PR={b['PR']:.4f} RR={b['RR']:.4f} CER={b['CER']:.4f} "
          f"Score={b['Score']:.4f}  AUC={auc(y, sv):.4f}")

    print("\n=== 长度偏置核验 ===")
    for k in ("n_windows", "max_sim", "mean_sim", "std_sim", "min_sim", "ft_raw", "ft_sep"):
        v = task.cols([k])[:, 0]
        a = auc(y, v)
        flip = "  (方向翻转)" if a < 0.5 else ""
        print(f"  单特征 {k:18s} AUC={max(a, 1-a):.4f}{flip}")
    nw = task.cols(["n_windows"])[:, 0]
    for k in SLIDE4:
        v = task.cols([k])[:, 0]
        print(f"  corr(n_windows, {k:9s}) = {np.corrcoef(nw, v)[0, 1]:+.4f}")

    QS = (0.08, 0.30, 0.50, 0.80)
    print(f"\n=== 5 折 CV × {args.seeds} seeds（阈值在训练折内选）× q 敏感性 ===")
    head = " ".join(f"{'q=' + str(q):>10s}" for q in QS)
    print(f"  {'特征集':<14s} {'模型':<5s} {'AUCoof':>8s} {head}")
    results = {}
    for name, keys in FEATURE_SETS.items():
        for model in ("lr", "gbdt"):
            scores, aucs = [], []
            for q in QS:
                per = []
                for sd in range(args.seeds):
                    oof, acc = run_cv(task, keys, q, seed=sd, model=model)
                    per.append(task.score(acc, q)["Score"])
                    if q == QS[0]:
                        aucs.append(auc(y, oof))
                scores.append(float(np.mean(per)))
            results[(name, model)] = scores
            body = " ".join(f"{s:10.4f}" for s in scores)
            print(f"  {name:<14s} {model:<5s} {np.mean(aucs):8.4f} {body}")

    print(f"\n  基线 Score = {b['Score']:.4f}")
    best = max(results.items(), key=lambda kv: min(kv[1]))
    print(f"  最稳健(q 全区间最小值最大): {best[0][0]} / {best[0][1]}  "
          f"min_q={min(best[1]):.4f}  Δ={min(best[1]) - b['Score']:+.4f}")

    # 全量拟合一版线性权重，看各特征贡献
    print("\n=== 全量拟合的线性权重（orig6+slide7, q=0.08, 仅供观察）===")
    keys = FEATURE_SETS["orig6+slide7"]
    X = task.cols(keys)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    w, _ = task.gain(0.08)
    w = np.abs(w) / np.abs(w).mean()
    clf = LogisticRegression(C=1.0, max_iter=4000).fit((X - mu) / sd, y, sample_weight=w)
    for k, c in sorted(zip(keys, clf.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f"  {k:12s} {c:+8.4f}")


if __name__ == "__main__":
    main()
