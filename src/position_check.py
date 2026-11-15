"""Проверка: не объясняется ли обнаружение положением окна в записи.

Окна атаки лежат во второй части записи, а нормальные окна есть и там, и там. Если оценка
аномальности растёт просто со временем от начала записи, модель могла бы "обнаруживать"
позицию, а не атаку. Проверка: сравниваем атакующие и нормальные окна с одинаковым
номером окна в записи (корзины по номеру) и считаем ROC-AUC внутри каждой корзины.

Запуск (из корня проекта), только dev:
    python src/position_check.py results/bruteforce_W400 --k 16 --cov full
    python src/position_check.py results/cve_2012_2122_W400 --k 8 --cov full
(k и cov возьмите из строки "Выбор по BIC" в gmm_study.py)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from run_experiment import load, per_record_stride, test_part_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res_dir")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cov", default="full")
    ap.add_argument("--part", choices=["dev", "final"], default="dev")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--train-cap", type=int, default=10000)
    ap.add_argument("--allow-final", action="store_true")
    a = ap.parse_args()
    if a.part == "final" and not a.allow_final:
        raise SystemExit("используйте dev (или --allow-final)")
    out = Path(a.res_dir)

    Xtr, _, _ = load(out, "training")
    Xte, yte, rte = load(out, "test")
    m = test_part_mask(yte, rte, a.part)
    Xte, yte, rte = Xte[m], yte[m], rte[m]
    # номер окна внутри записи (окна в файле идут по порядку записи)
    starts = np.r_[0, np.flatnonzero(np.diff(rte)) + 1]
    pos = np.concatenate([np.arange(e - s) for s, e in
                          zip(starts, np.r_[starts[1:], len(rte)])])

    sc = StandardScaler().fit(Xtr)
    idx = np.random.default_rng(0).choice(len(Xtr), size=min(a.train_cap, len(Xtr)),
                                          replace=False)
    g = GaussianMixture(a.k, covariance_type=a.cov, reg_covar=1e-3, max_iter=200,
                        random_state=0).fit(sc.transform(Xtr[idx]))
    s = -g.score_samples(sc.transform(Xte))

    rec_att = pd.Series(yte).groupby(rte).max()
    normal_rec = np.isin(rte, rec_att.index[rec_att.values == 0])
    rho, p = spearmanr(pos[normal_rec], s[normal_rec])
    print(f"Нормальные записи: корреляция Спирмена (номер окна, оценка) = {rho:.3f} (p={p:.3g})")
    print("  близко к 0: оценка не растёт со временем; сильная положительная: подозрительно\n")

    print(f"Окон: нормальных {int((yte == 0).sum())}, атакующих {int((yte == 1).sum())}; "
          f"номер окна атаки: от {pos[yte == 1].min()} до {pos[yte == 1].max()}, "
          f"медиана {int(np.median(pos[yte == 1]))}")
    edges = np.unique(np.quantile(pos[yte == 1], np.linspace(0, 1, a.bins + 1)))
    rows, w_num, w_den = [], 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        mk = (pos >= lo) & ((pos <= hi) if last else (pos < hi))
        ya, sa = yte[mk], s[mk]
        n1, n0 = int((ya == 1).sum()), int((ya == 0).sum())
        auc = roc_auc_score(ya, sa) if n1 and n0 else np.nan
        rows.append(dict(window_index=f"{int(lo)}-{int(hi)}", attack=n1, normal=n0,
                         auc_inside_bin=auc))
        if n1 and n0:
            w_num += auc * (n1 + n0); w_den += n1 + n0
    t = pd.DataFrame(rows)
    print("\nROC-AUC внутри корзин по номеру окна (сравниваем окна с одинаковой позицией):")
    print(t.round(3).to_string(index=False))
    print(f"\nВзвешенный AUC внутри корзин: {w_num / max(w_den, 1):.3f}; "
          f"общий AUC окон: {roc_auc_score(yte, s):.3f}")
    print("Если AUC внутри корзин остаётся близким к общему, положение окна не объясняет результат.")
    t.to_csv(out / "position_check.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()