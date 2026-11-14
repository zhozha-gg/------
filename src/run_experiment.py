"""Обучение детекторов аномалий на окнах и оценка.

Запуск (из корня проекта):
    python src/run_experiment.py results/bruteforce
Необязательные параметры:
    --train-cap 10000   сколько нормальных окон брать для SVM и LOF (они медленные)
    --test-cap 150000   сколько тестовых окон оценивать (иначе очень долго)
    --seed 0
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ---------- данные ----------

def load(res_dir, split):
    d = np.load(Path(res_dir) / f"windows_{split}.npz", allow_pickle=False)
    return d["X"].astype(np.float32), d["y"], d["rec"]


def per_record_stride(X, y, rec, cap):
    """Равномерно прореживает окна внутри каждой записи, оставляя >=1 окно."""
    n = len(y)
    if n <= cap:
        return X, y, rec
    step = int(np.ceil(n / cap))
    starts = np.r_[0, np.flatnonzero(np.diff(rec)) + 1]
    ends = np.r_[starts[1:], n]
    idx = np.concatenate([np.arange(s, e, step) for s, e in zip(starts, ends)])
    return X[idx], y[idx], rec[idx]


# ---------- метрики ----------

def window_metrics(y, score, thr):
    pred = (score > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


AGGREGATORS = ("max", "p95", "frac")


def record_scores(score, rec, thr_w, how):
    """Оценка записи. max зависит от числа окон (длины записи), p95 и frac нет."""
    s = pd.Series(score)
    if how == "max":
        return s.groupby(rec).max()
    if how == "p95":
        return s.groupby(rec).quantile(0.95)
    if how == "frac":                       # доля окон выше порога окон
        return (s > thr_w).groupby(rec).mean()
    raise ValueError(how)


def record_labels(y, rec):
    return pd.Series(y).groupby(rec).max()


# ---------- модели ----------

def make_models(seed):
    return {
        "IsolationForest": IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
    }


def anomaly_score(model, X):
    # в sklearn больше score_samples = нормальнее, поэтому берём минус
    return -model.score_samples(X)


# ---------- основной запуск ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res_dir")
    ap.add_argument("--train-cap", type=int, default=10000)
    ap.add_argument("--test-cap", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.res_dir)

    Xtr, ytr, rtr = load(out, "training")
    Xva, yva, rva = load(out, "validation")
    Xte, yte, rte = load(out, "test")
    print(f"Окон: train {len(ytr)}, val {len(yva)}, test {len(yte)}")

    # обучающая подвыборка (только норма: в training атак нет)
    cap_idx = rng.choice(len(Xtr), size=min(a.train_cap, len(Xtr)), replace=False)
    Xfit = Xtr[cap_idx]
    Xva, yva, rva = per_record_stride(Xva, yva, rva, a.test_cap)
    Xte, yte, rte = per_record_stride(Xte, yte, rte, a.test_cap)
    print(f"Оцениваем: val {len(yva)}, test {len(yte)} окон "
          f"(доля атак в test: {yte.mean():.3f})")

    scaler = StandardScaler().fit(Xtr)        # масштаб по всему train
    Xfit, Xva_s, Xte_s = (scaler.transform(v) for v in (Xfit, Xva, Xte))

    rows, curves = [], {}
    for name, model in make_models(a.seed).items():
        t = time.time()
        model.fit(Xfit)
        t_fit = time.time() - t

        t = time.time()
        s_va = anomaly_score(model, Xva_s)
        s_te = anomaly_score(model, Xte_s)
        t_pred = time.time() - t

        # порог по окнам: 99-й процентиль оценок НОРМАЛЬНЫХ окон validation
        thr_w = np.percentile(s_va, 99)
        m = window_metrics(yte, s_te, thr_w)
        rows.append({"model": name, "level": "window", "fit_s": t_fit,
                     "predict_s": t_pred, **m})

        # уровень записей: порог = 95-й процентиль оценок нормальных записей validation
        rl_te = record_labels(yte, rte)
        for how in AGGREGATORS:
            rs_va = record_scores(s_va, rva, thr_w, how)
            rs_te = record_scores(s_te, rte, thr_w, how)
            thr_r = np.percentile(rs_va.values, 95)
            mr = window_metrics(rl_te.values, rs_te.values, thr_r)
            rows.append({"model": name, "level": f"record_{how}", "fit_s": t_fit,
                         "predict_s": t_pred, **mr})

        fpr, tpr, _ = roc_curve(yte, s_te)
        curves[name] = (fpr, tpr, m["roc_auc"])
        np.savez_compressed(out / f"scores_test_{name}.npz",
                            score=s_te, y=yte, rec=rte, thr_window=thr_w)
        print(f"{name}: готово (fit {t_fit:.1f}s, predict {t_pred:.1f}s)")

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    show = ["model", "level", "precision", "recall", "f1", "fpr",
            "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
    print("\n" + table[show].round(3).to_string(index=False))

    plt.figure(figsize=(5, 5))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC, уровень окон")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out / "roc_window.png", dpi=150)
    print("\nСохранено:", out / "metrics.csv", "и", out / "roc_window.png")


if __name__ == "__main__":
    main()