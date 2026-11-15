"""График оценки аномальности окон по времени для нормальной и атакующей записи.

Запуск (из корня проекта):
    python src/plot_record.py models/bruteforce_gmm.joblib NORMAL.zip ATTACK.zip results/bruteforce_record_scores.png

NORMAL.zip и ATTACK.zip: пути к двум записям из test (нормальной и из normal_and_attack).
Показано: оценка -log p(x) каждого окна, порог окна и (для атакующей записи) момент начала эксплойта.
Шкала по вертикали symlog: оценки нормы и атаки отличаются на порядки.
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features_windows import exploit_ns, read_record


def window_scores(model, path):
    meta, ts, calls = read_record(path)
    vocab, W, S = model["vocab"], model["W"], model["S"]
    idx = {c: i for i, c in enumerate(vocab[:-1])}
    unk = len(vocab) - 1
    seq = np.array([idx.get(c, unk) for c in calls])
    starts = list(range(0, len(seq) - W + 1, S))
    if not starts:
        raise SystemExit(f"{Path(path).name}: запись короче окна ({len(seq)} вызовов при W={W})")
    X = np.array([np.bincount(seq[s:s + W], minlength=unk + 1) / W for s in starts])
    scores = -model["gmm"].score_samples(model["scaler"].transform(X))
    t = (ts[starts] - ts[0]) / 1e9                      # секунды от начала записи
    att = exploit_ns(meta)
    t_att = None if att is None else (att - ts[0]) / 1e9
    return t, scores, t_att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("normal_zip")
    ap.add_argument("attack_zip")
    ap.add_argument("out_png")
    a = ap.parse_args()
    model = joblib.load(a.model)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, path, title in ((axes[0], a.normal_zip, "Нормальная запись"),
                            (axes[1], a.attack_zip, "Запись с атакой")):
        t, s, t_att = window_scores(model, path)
        above = s > model["thr_window"]
        ax.plot(t, s, "-", color="tab:blue", lw=1)
        ax.scatter(t[above], s[above], color="tab:red", s=14, zorder=3,
                   label="окно выше порога")
        ax.axhline(model["thr_window"], color="k", ls="--", lw=1, label="порог окна")
        if t_att is not None:
            ax.axvline(t_att, color="tab:orange", ls=":", lw=2, label="начало эксплойта")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("время от начала записи, с")
        ax.set_title(f"{title}\n{Path(path).name}: выше порога {above.mean():.0%} окон", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("оценка аномальности -log p(x)")
    plt.tight_layout()
    Path(a.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out_png, dpi=150)
    print("Сохранено:", a.out_png)


if __name__ == "__main__":
    main()