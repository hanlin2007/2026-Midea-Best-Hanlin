#!/usr/bin/env python3
"""消融：增益是真信号还是长度痕迹？收益来自拒负样本还是救正样本？"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_v2 import load, Task, auc, ROOT, ORIG6, SLIDE4, LOW3  # noqa: E402

SEEDS = 5
QS = (0.08, 0.30, 0.50, 0.80)

SETS = {
    "orig6":                    ORIG6,
    "orig6+slide7":             ORIG6 + SLIDE4 + LOW3,
    "orig6+slide7+wakefb":      ORIG6 + SLIDE4 + LOW3 + ["wake_raw_fallback"],
    "orig6+slide7+nwin":        ORIG6 + SLIDE4 + LOW3 + ["n_windows"],
    "orig6+all":                ORIG6 + SLIDE4 + LOW3 + ["n_windows", "wake_raw_fallback"],
    "orig6+wakefb":             ORIG6 + ["wake_raw_fallback"],
    "orig6+nwin":               ORIG6 + ["n_windows"],
    "orig6+nwin+wakefb":        ORIG6 + ["n_windows", "wake_raw_fallback"],
}


def cv(task, keys, q, seed, model="gbdt"):
    X = task.cols(keys)
    y = task.y
    oof = np.zeros(len(y))
    accept = np.zeros(len(y), dtype=bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w, _ = task.gain(q, tr)
        w = np.abs(w) / np.abs(w).mean()
        if model == "lr":
            clf = LogisticRegression(C=1.0, max_iter=4000).fit(Xtr, y[tr], sample_weight=w)
            s_tr, s_te = clf.decision_function(Xtr), clf.decision_function(Xte)
        else:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=8, max_depth=3,
                                     learning_rate=0.05, min_child_samples=30,
                                     subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1)
            clf.fit(Xtr, y[tr], sample_weight=w)
            s_tr, s_te = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]
        t = task.best_threshold(s_tr, q, tr)
        oof[te] = s_te
        accept[te] = s_te >= t
    return oof, accept


def main():
    rows = load(ROOT / "outputs/no_conditional_full/no_conditional_full_detail.json",
                ROOT / "outputs/sliding_gate_verify.json")
    task = Task(rows)
    y = task.y
    cur = np.array([r["cur_accept"] for r in rows])
    base = task.score(cur, 0.08)
    print(f"基线 Score={base['Score']:.4f}  PR={base['PR']:.4f} RR={base['RR']:.4f}\n")

    print("=== 消融（gbdt, nested 阈值, 5 seeds）===")
    head = " ".join(f"{'q=' + str(q):>9s}" for q in QS)
    print(f"  {'特征集':<22s} {'AUCoof':>8s} {head}")
    keep = {}
    for name, keys in SETS.items():
        sc, ac = [], []
        for q in QS:
            per = []
            for s in range(SEEDS):
                oof, acc = cv(task, keys, q, s)
                per.append(task.score(acc, q)["Score"])
                if q == QS[0]:
                    ac.append(auc(y, oof))
                    if s == 0:
                        keep[name] = acc
            sc.append(float(np.mean(per)))
        print(f"  {name:<22s} {np.mean(ac):8.4f} " + " ".join(f"{v:9.4f}" for v in sc))

    print("\n=== 收益来源分解（q=0.08, seed 0）===")
    print(f"  {'特征集':<22s} {'多拒负':>7s} {'漏拒负':>7s} {'救回正':>7s} {'误拒正':>7s} "
          f"{'ΔRR':>8s} {'ΔCER':>8s}")
    for name, acc in keep.items():
        neg = y == 0
        pos = y == 1
        more_rej_neg = int((neg & cur & ~acc).sum())
        less_rej_neg = int((neg & ~cur & acc).sum())
        saved_pos = int((pos & ~cur & acc).sum())
        lost_pos = int((pos & cur & ~acc).sum())
        s = task.score(acc, 0.08)
        print(f"  {name:<22s} {more_rej_neg:7d} {less_rej_neg:7d} {saved_pos:7d} "
              f"{lost_pos:7d} {s['RR'] - base['RR']:+8.4f} {s['CER'] - base['CER']:+8.4f}")

    print("\n=== n_windows 到底在编码什么 ===")
    nw = task.cols(["n_windows"])[:, 0]
    for nm, m in (("正样本", y == 1), ("负样本", y == 0)):
        v = nw[m]
        print(f"  {nm} n_windows: mean={v.mean():.2f} p25={np.percentile(v,25):.0f} "
              f"p50={np.percentile(v,50):.0f} p75={np.percentile(v,75):.0f}")
    print(f"  → 窗长 0.8s / 步长 0.4s，n_windows≈(时长-0.4)/0.4，即命令音频时长")
    wf = task.cols(["wake_raw_fallback"])[:, 0]
    for nm, m in (("正样本", y == 1), ("负样本", y == 0)):
        print(f"  {nm} wake_raw_fallback=1 占比: {wf[m].mean():.4f}")


if __name__ == "__main__":
    main()
