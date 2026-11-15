"""Обучает итоговую GMM на нормальных данных и сохраняет всё нужное для разбора новых записей.

Запуск (из корня проекта):
    python src/train_final.py results/bruteforce_W400 models/bruteforce_gmm.joblib
    python src/train_final.py results/cve_2012_2122_W400 models/cve_2012_2122_gmm.joblib

Число компонент k и тип ковариации выбираются по BIC на обучающих окнах (как в gmm_study.py).
Порог окна = 99-й перцентиль оценок -log p(x) на нормальных окнах validation,
порог записи = 95-й перцентиль по записям validation (оценка записи = p95 оценок её окон).
Тестовые данные здесь не используются.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def record_p95(scores, rec):
    return np.array([np.percentile(scores[rec == r], 95) for r in np.unique(rec)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res_dir")
    ap.add_argument("out_model")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--train-cap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = Path(a.res_dir)

    tr = np.load(res / "windows_training.npz", allow_pickle=False)
    va = np.load(res / "windows_validation.npz", allow_pickle=False)
    Xtr, Xva, rva = tr["X"], va["X"], va["rec"]
    vocab = [str(v) for v in tr["vocab"]]
    W, S = int(tr["W"]), int(tr["S"])

    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(Xtr), size=min(a.train_cap, len(Xtr)), replace=False)
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr[idx])

    best = None
    for cov in ("diag", "full"):
        for k in [int(x) for x in a.ks.split(",")]:
            if k >= len(Z):
                continue
            g = GaussianMixture(k, covariance_type=cov, reg_covar=1e-3, max_iter=200,
                                random_state=a.seed).fit(Z)
            bic = g.bic(Z)
            print(f"cov={cov:4s} k={k:3d} BIC={bic:.0f}")
            if best is None or bic < best[0]:
                best = (bic, cov, k, g)
    _, cov, k, g = best
    print(f"\nВыбрано по BIC: cov={cov}, k={k}")

    s_va = -g.score_samples(sc.transform(Xva))
    thr_window = float(np.percentile(s_va, 99))
    thr_record = float(np.percentile(record_p95(s_va, rva), 95))
    print(f"Порог окна (99-й перцентиль validation): {thr_window:.2f}")
    print(f"Порог записи (95-й перцентиль p95 по validation): {thr_record:.2f}")

    out = Path(a.out_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(vocab=vocab, W=W, S=S, scaler=sc, gmm=g, k=k, cov=cov,
                     thr_window=thr_window, thr_record=thr_record,
                     train_mean_share=Xtr.mean(0)), out)
    print("Сохранено:", out)


if __name__ == "__main__":
    main()