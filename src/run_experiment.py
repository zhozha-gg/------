"""Обучение детекторов аномалий на окнах и оценка.

Запуск (из корня проекта):
    python src/run_experiment.py results/bruteforce
Необязательные параметры:
    --train-cap 10000   сколько нормальных окон брать для SVM и LOF (они медленные)
    --test-cap 150000   сколько тестовых окон оценивать (иначе очень долго)
    --seeds 3           сколько раз повторить обучение с разными seed (среднее и разброс)
    --part all|dev|final  какая половина записей test используется (см. ниже)

Протокол: записи test делятся пополам (по записям, поровну норма и атака, фиксированное
разбиение). На половине dev выбираем W, агрегатор, параметры. Итог считаем ОДИН РАЗ на final.
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


def test_part_mask(y, rec, part, split_seed=12345):
    """Маска окон test: dev, final или all. Разбиение по записям, одинаковое при каждом запуске."""
    if part == "all":
        return np.ones(len(y), dtype=bool)
    rl = pd.Series(y).groupby(rec).max()
    rng = np.random.default_rng(split_seed)
    dev_ids = []
    for lab in (0, 1):
        ids = rl.index[rl.values == lab].to_numpy().copy()
        rng.shuffle(ids)
        dev_ids.extend(ids[: len(ids) // 2])
    in_dev = np.isin(rec, dev_ids)
    return in_dev if part == "dev" else ~in_dev


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

class GMMDetector:
    """Гауссова смесь: оценка плотности нормального поведения.
    Число компонент и тип ковариации выбираются по BIC на обучающих (нормальных) окнах.
    Оценка аномальности = -log p(x)."""

    def __init__(self, seed=0, ks=(1, 2, 4, 8, 16), covs=("diag", "full"), reg=1e-3):
        self.seed, self.ks, self.covs, self.reg = seed, ks, covs, reg

    def fit(self, X):
        best = None
        for cov in self.covs:
            for k in self.ks:
                if k >= len(X):
                    continue
                g = GaussianMixture(n_components=k, covariance_type=cov,
                                    reg_covar=self.reg, max_iter=200,
                                    random_state=self.seed)
                try:
                    g.fit(X)
                except ValueError:
                    continue
                bic = g.bic(X)
                if best is None or bic < best[0]:
                    best = (bic, g, k, cov)
        _, self.model_, self.k_, self.cov_ = best
        return self

    def score_samples(self, X):
        return self.model_.score_samples(X)       # log-правдоподобие: больше = нормальнее


def make_models(seed):
    return {
        "IsolationForest": IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
        "GMM": GMMDetector(seed=seed),
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
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--part", choices=["all", "dev", "final"], default="all")
    a = ap.parse_args()
    out = Path(a.res_dir)

    Xtr, ytr, rtr = load(out, "training")
    Xva, yva, rva = load(out, "validation")
    Xte, yte, rte = load(out, "test")
    print(f"Окон: train {len(ytr)}, val {len(yva)}, test {len(yte)}")

    m = test_part_mask(yte, rte, a.part)
    Xte, yte, rte = Xte[m], yte[m], rte[m]
    print(f"Часть test: {a.part}, записей {len(np.unique(rte))}, окон {len(yte)}")
    if a.part == "final":
        print("ВНИМАНИЕ: final используйте один раз для итоговой конфигурации.")

    Xva, yva, rva = per_record_stride(Xva, yva, rva, a.test_cap)
    Xte, yte, rte = per_record_stride(Xte, yte, rte, a.test_cap)
    print(f"Оцениваем: val {len(yva)}, test {len(yte)} окон "
          f"(доля атак в test: {yte.mean():.3f})")

    scaler = StandardScaler().fit(Xtr)
    Xva_s, Xte_s = scaler.transform(Xva), scaler.transform(Xte)
    rl_te = record_labels(yte, rte)

    rows, curves = [], {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        cap_idx = rng.choice(len(Xtr), size=min(a.train_cap, len(Xtr)), replace=False)
        Xfit = scaler.transform(Xtr[cap_idx])

        for name, model in make_models(seed).items():
            t = time.time()
            model.fit(Xfit)
            t_fit = time.time() - t

            t = time.time()
            s_va = anomaly_score(model, Xva_s)
            s_te = anomaly_score(model, Xte_s)
            t_pred = time.time() - t

            thr_w = np.percentile(s_va, 99)
            mw = window_metrics(yte, s_te, thr_w)
            rows.append({"seed": seed, "model": name, "level": "window",
                         "fit_s": t_fit, "predict_s": t_pred, **mw})

            for how in AGGREGATORS:
                rs_va = record_scores(s_va, rva, thr_w, how)
                rs_te = record_scores(s_te, rte, thr_w, how)
                thr_r = np.percentile(rs_va.values, 95)
                mr = window_metrics(rl_te.values, rs_te.values, thr_r)
                rows.append({"seed": seed, "model": name, "level": f"record_{how}",
                             "fit_s": t_fit, "predict_s": t_pred, **mr})

            if seed == 0:
                fpr, tpr, _ = roc_curve(yte, s_te)
                curves[name] = (fpr, tpr, mw["roc_auc"])
                np.savez_compressed(out / f"scores_test_{name}.npz",
                                    score=s_te, y=yte, rec=rte, thr_window=thr_w)
            extra = f", k={model.k_}, cov={model.cov_}" if name == "GMM" else ""
            print(f"seed {seed}, {name}: готово (fit {t_fit:.1f}s, predict {t_pred:.1f}s{extra})")

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "metrics_seeds.csv", index=False, encoding="utf-8-sig")
    keys = ["model", "level"]
    num = [c for c in raw.columns if c not in keys + ["seed"]]
    mean = raw.groupby(keys, sort=False)[num].mean().reset_index()
    std = raw.groupby(keys, sort=False)[num].std().fillna(0).reset_index()
    mean.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    std.to_csv(out / "metrics_std.csv", index=False, encoding="utf-8-sig")

    show = ["model", "level", "precision", "recall", "f1", "fpr",
            "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
    print(f"\nСреднее по {a.seeds} seed:")
    print(mean[show].round(3).to_string(index=False))
    if a.seeds > 1:
        print("\nРазброс (std) по seed:")
        print(std[["model", "level", "f1", "roc_auc"]].round(3).to_string(index=False))

    plt.figure(figsize=(5, 5))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC, уровень окон ({a.part})")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out / "roc_window.png", dpi=150)
    print("\nСохранено:", out / "metrics.csv", ", metrics_std.csv, metrics_seeds.csv, roc_window.png")


if __name__ == "__main__":
    main()