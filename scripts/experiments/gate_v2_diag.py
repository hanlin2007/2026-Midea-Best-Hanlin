#!/usr/bin/env python3
"""诊断：AUC 上升但 Score 下降，问题在模型还是在阈值传递？

对每个配置报三个数：
  AUCoof        out-of-fold 排序质量
  Score@nested  阈值在训练折上选、测试折上用（真实可部署）
  Score@oracle  阈值直接在 OOF 分数上取最优（上界，不可部署）
再对比 加权 / 不加权 训练。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_v2 import load, Task, FEATURE_SETS, auc, ROOT  # noqa: E402

Q = 0.08
SEEDS = 3


def cv(task, keys, model, weighted, seed):
    X = task.cols(keys)
    y = task.y
    oof = np.zeros(len(y))
    accept = np.zeros(len(y), dtype=bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if weighted:
            w, _ = task.gain(Q, tr)
            w = np.abs(w)
            w = w / w.mean()
        else:
            w = None
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
        t = task.best_threshold(s_tr, Q, tr)
        oof[te] = s_te
        accept[te] = s_te >= t
    t_or = task.best_threshold(oof, Q, np.arange(len(y)))
    return (auc(y, oof),
            task.score(accept, Q)["Score"],
            task.score(oof >= t_or, Q)["Score"])


def main():
    rows = load(ROOT / "outputs/no_conditional_full/no_conditional_full_detail.json",
                ROOT / "outputs/sliding_gate_verify.json")
    task = Task(rows)
    y = task.y
    base = task.score(np.array([r["cur_accept"] for r in rows]), Q)
    print(f"基线 Score={base['Score']:.4f}  (阈值 -0.6469, 无 CV, 同分布内)\n")

    print(f"  {'特征集':<14s} {'模型':<5s} {'权重':<6s} {'AUCoof':>8s} "
          f"{'@nested':>9s} {'@oracle':>9s} {'oracle-nested':>14s}")
    for name in ("orig6", "orig6+slide4", "orig6+all", "raw2+all"):
        for model in ("lr", "gbdt"):
            for weighted in (True, False):
                a, sn, so = np.mean([cv(task, FEATURE_SETS[name], model, weighted, s)
                                     for s in range(SEEDS)], axis=0)
                tag = "cost" if weighted else "plain"
                print(f"  {name:<14s} {model:<5s} {tag:<6s} {a:8.4f} "
                      f"{sn:9.4f} {so:9.4f} {so - sn:14.4f}")

    # 阈值稳定性：各折选出来的阈值分散度
    print("\n=== 阈值在各折之间的分散度（orig6+all / gbdt / cost）===")
    X = task.cols(FEATURE_SETS["orig6+all"])
    import lightgbm as lgb
    for seed in range(SEEDS):
        ts, accs = [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            w, _ = task.gain(Q, tr)
            w = np.abs(w) / np.abs(w).mean()
            clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=8, max_depth=3,
                                     learning_rate=0.05, min_child_samples=30,
                                     subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1)
            clf.fit(Xtr, y[tr], sample_weight=w)
            s_tr = clf.predict_proba(Xtr)[:, 1]
            s_te = clf.predict_proba(Xte)[:, 1]
            t_tr = task.best_threshold(s_tr, Q, tr)
            t_te = task.best_threshold(s_te, Q, te)
            ts.append((t_tr, t_te))
            accs.append(task.score(s_te >= t_tr, Q, te)["Score"] -
                        task.score(s_te >= t_te, Q, te)["Score"])
        print(f"  seed {seed}: 训练折阈值 {[f'{a:.3f}' for a, _ in ts]}")
        print(f"          测试折最优 {[f'{b:.3f}' for _, b in ts]}")
        print(f"          用错阈值的代价 {[f'{d:+.4f}' for d in accs]}")


if __name__ == "__main__":
    main()
