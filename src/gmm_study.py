"""Исследование GMM как основного метода (только на части dev).

Запуск (из корня проекта):
    python src/gmm_study.py results/bruteforce_W400
    python src/gmm_study.py results/cve_2012_2122_W400

Что считается:
  1. Сетка k (1..64) x тип ковариации (diag, full): BIC на обучающих окнах,
     правдоподобие на validation, ROC-AUC на dev (окна и записи, агрегатор p95).
     Выбор модели идёт по BIC. AUC только показывает чувствительность, по нему не выбираем.
  2. Чувствительность к порогу: перцентиль 95, 99, 99.5, 99.9 по оценкам validation.
  3. Компоненты выбранной модели: вес и самые частые системные вызовы.
  4. Вклад признаков: какие системные вызовы сильнее всего отличают атакующие окна
     от нормальных с точки зрения модели.

Часть final по умолчанию запрещена, чтобы не потратить единственный итоговый прогон.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from run_experiment import (load, per_record_stride, record_labels,
                            record_scores, test_part_mask, window_metrics)

COVS = ("diag", "full")


def contributions(g, Z):
    """Вклад каждого признака в расстояние Махаланобиса до самой вероятной компоненты.
    Сумма по признакам равна квадрату расстояния. Возвращает массив (n, d)."""
    comp = g.predict(Z)
    out = np.zeros_like(Z)
    for j in np.unique(comp):
        m = comp == j
        d = Z[m] - g.means_[j]
        if g.covariance_type == "full":
            out[m] = d * (d @ g.precisions_[j])
        else:                                   # diag: precisions_ = 1 / дисперсия
            out[m] = d * d * g.precisions_[j]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res_dir")
    ap.add_argument("--part", choices=["dev", "final"], default="dev")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--train-cap", type=int, default=10000)
    ap.add_argument("--test-cap", type=int, default=150000)
    ap.add_argument("--allow-final", action="store_true")
    a = ap.parse_args()
    if a.part == "final" and not a.allow_final:
        raise SystemExit("final тратить нельзя: исследование GMM делается на dev. "
                         "Если точно нужно, добавьте --allow-final.")
    out = Path(a.res_dir)
    ks = [int(k) for k in a.ks.split(",")]

    vocab = list(np.load(out / "windows_training.npz", allow_pickle=False)["vocab"])
    Xtr, _, _ = load(out, "training")
    Xva, yva, rva = load(out, "validation")
    Xte, yte, rte = load(out, "test")
    m = test_part_mask(yte, rte, a.part)
    Xte, yte, rte = Xte[m], yte[m], rte[m]
    Xte, yte, rte = per_record_stride(Xte, yte, rte, a.test_cap)
    Xva, yva, rva = per_record_stride(Xva, yva, rva, a.test_cap)
    rl = record_labels(yte, rte)
    print(f"Часть {a.part}: окон test {len(yte)} (доля атак {yte.mean():.3f}), "
          f"записей {len(rl)} (атакующих {int(rl.sum())})")

    sc = StandardScaler().fit(Xtr)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=min(a.train_cap, len(Xtr)), replace=False)
    Zfit, Zva, Zte = sc.transform(Xtr[idx]), sc.transform(Xva), sc.transform(Xte)

    # ---------- 1. сетка k x ковариация ----------
    rows, models = [], {}
    for cov in COVS:
        for k in ks:
            if k >= len(Zfit):
                continue
            g = GaussianMixture(k, covariance_type=cov, reg_covar=1e-3,
                                max_iter=200, random_state=0)
            t = time.time()
            try:
                g.fit(Zfit)
            except ValueError as e:
                print(f"cov={cov} k={k}: пропуск ({e})")
                continue
            s_te = -g.score_samples(Zte)
            rs = record_scores(s_te, rte, 0.0, "p95")
            rows.append(dict(cov=cov, k=k, bic=g.bic(Zfit), val_loglik=g.score(Zva),
                             auc_window=roc_auc_score(yte, s_te),
                             auc_record_p95=roc_auc_score(rl.values,
                                                          rs.reindex(rl.index).values),
                             fit_s=time.time() - t))
            models[(cov, k)] = g
            print(f"cov={cov:4s} k={k:3d}  BIC={rows[-1]['bic']:.0f}  "
                  f"AUC окон {rows[-1]['auc_window']:.3f}  ({rows[-1]['fit_s']:.1f} с)")
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "gmm_study_grid.csv", index=False, encoding="utf-8-sig")
    best_row = grid.loc[grid["bic"].idxmin()]
    cov_b, k_b = best_row["cov"], int(best_row["k"])
    edge = k_b == max(ks)
    print("\n=== 1. Сетка k x ковариация ===")
    print(grid.round(3).to_string(index=False))
    print(f"\nВыбор по BIC: cov={cov_b}, k={k_b}"
          + ("  (край сетки: BIC ещё убывает, k можно увеличить)" if edge else ""))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for cov, color in zip(COVS, ("tab:blue", "tab:red")):
        d = grid[grid["cov"] == cov]
        ax[0].plot(d.k, d.bic, "o-", color=color, label=cov)
        ax[1].plot(d.k, d.auc_window, "o-", color=color, label=f"{cov}, окна")
        ax[1].plot(d.k, d.auc_record_p95, "s--", color=color, label=f"{cov}, записи p95")
    for x in ax:
        x.set_xscale("log", base=2); x.set_xlabel("число компонент k")
    ax[0].set_ylabel("BIC (меньше лучше)"); ax[1].set_ylabel("ROC-AUC")
    ax[0].set_title("BIC на обучающих окнах"); ax[1].set_title(f"ROC-AUC на {a.part}")
    ax[0].legend(); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "gmm_study_bic_auc.png", dpi=150); plt.close()

    # ---------- 2. чувствительность к порогу ----------
    g = models[(cov_b, k_b)]
    s_va = -g.score_samples(Zva)
    s_te = -g.score_samples(Zte)
    th_rows = []
    for p in (95, 99, 99.5, 99.9):
        thr = np.percentile(s_va, p)
        r = window_metrics(yte, s_te, thr)
        th_rows.append(dict(level="window", percentile=p, **{c: r[c] for c in
                            ("precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn")}))
    rs_va = record_scores(s_va, rva, 0.0, "p95")
    rs_te = record_scores(s_te, rte, 0.0, "p95").reindex(rl.index)
    for p in (90, 95, 99):
        thr = np.percentile(rs_va.values, p)
        r = window_metrics(rl.values, rs_te.values, thr)
        th_rows.append(dict(level="record_p95", percentile=p, **{c: r[c] for c in
                            ("precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn")}))
    th = pd.DataFrame(th_rows)
    th.to_csv(out / "gmm_study_thresholds.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 2. Чувствительность к порогу (GMM cov={cov_b}, k={k_b}) ===")
    print("перцентиль берётся по оценкам validation (только нормальные данные)")
    print(th.round(3).to_string(index=False))

    # ---------- 3. компоненты ----------
    orig = np.clip(sc.inverse_transform(g.means_), 0, None)
    comp = []
    print(f"\n=== 3. Компоненты: вес и 5 самых частых вызовов (доля в окне) ===")
    for j in np.argsort(-g.weights_):
        top = np.argsort(-orig[j])[:5]
        desc = ", ".join(f"{vocab[i]}={orig[j, i]:.2f}" for i in top)
        comp.append(dict(component=int(j), weight=g.weights_[j], top5=desc))
        print(f"  вес {g.weights_[j]:.3f}: {desc}")
    pd.DataFrame(comp).to_csv(out / "gmm_study_components.csv", index=False,
                              encoding="utf-8-sig")

    # ---------- 4. вклад признаков ----------
    C = contributions(g, Zte)
    att, nor = C[yte == 1], C[yte == 0]
    # медиана, а не среднее: у редких вызовов с почти нулевой дисперсией несколько окон
    # дают огромные значения и целиком определяют среднее
    diff = np.median(att, axis=0) - np.median(nor, axis=0)
    order = np.argsort(-diff)[:10]
    pres_a = (Xte[yte == 1] > 0).mean(0)
    pres_n = (Xte[yte == 0] > 0).mean(0)
    contrib = pd.DataFrame({
        "syscall": [vocab[i] for i in order],
        "median_contrib_attack": np.median(att, axis=0)[order],
        "median_contrib_normal": np.median(nor, axis=0)[order],
        "difference": diff[order],
        "share_attack_mean": Xte[yte == 1].mean(0)[order],
        "share_normal_mean": Xte[yte == 0].mean(0)[order],
        "present_in_attack_windows": pres_a[order],
        "present_in_normal_windows": pres_n[order]})
    contrib.to_csv(out / "gmm_study_contrib.csv", index=False, encoding="utf-8-sig")
    print("\n=== 4. Какие вызовы сильнее всего повышают аномальность атакующих окон ===")
    print("(медианный вклад в расстояние Махаланобиса, в стандартизованных единицах;"
          " share_* - средняя доля вызова в окне; present_* - в какой доле окон вызов есть)")
    print(contrib.round(3).to_string(index=False))
    plt.figure(figsize=(6, 4))
    plt.barh(contrib["syscall"][::-1], contrib["difference"][::-1], color="tab:red")
    plt.xlabel("вклад в аномальность: атака минус норма")
    plt.title("Вызовы, отличающие атакующие окна")
    plt.tight_layout(); plt.savefig(out / "gmm_study_contrib.png", dpi=150); plt.close()

    print("\nСохранено в", out, ": gmm_study_grid.csv, gmm_study_thresholds.csv, "
          "gmm_study_components.csv, gmm_study_contrib.csv, gmm_study_bic_auc.png, "
          "gmm_study_contrib.png")


if __name__ == "__main__":
    main()