"""Baseline для раздела 3.3: STIDE и правило по длительности записи.

Запуск (из корня проекта; сначала нужны results/<name>_W<W>/windows_*.npz):
    python src/baselines.py data/Bruteforce_CWE-307/Bruteforce_CWE-307 results/bruteforce_W400 --part dev
    python src/baselines.py data/CVE-2012-2122 results/cve_2012_2122_W400 --part dev

STIDE (Forrest et al., 1996): из нормальных обучающих записей собирается база всех
n-грамм подряд идущих системных вызовов. Оценка окна = доля n-грамм окна, которых нет в базе.
Длительность: оценка записи = |длительность - медиана длительности обучающих записей|.
Это нарочно примитивный baseline: он показывает, сколько даёт одна лишь длина записи
(именно из-за неё в первой версии признаков была утечка).

Окна, метки, разбиение dev/final и пороги такие же, как в run_experiment.py:
порог окна = 99-й перцентиль оценок validation, порог записи = 95-й.
Параметры baseline фиксированы заранее (n=6), подбирать их по test нельзя.
final используйте один раз, после проверки на dev.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features_windows import exploit_ns, get_archives, read_record
from run_experiment import (AGGREGATORS, load, record_labels, record_scores,
                            test_part_mask, window_metrics)

BASE = 64          # основание кода n-граммы; словарь короче 64 вызовов


def ngram_codes(seq, n):
    """Целочисленный код каждой n-граммы seq[i:i+n]. Для словаря < 64 и n <= 9 влезает в int64."""
    m = len(seq) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    code = np.zeros(m, dtype=np.int64)
    for j in range(n):
        code = code * BASE + seq[j:j + m]
    return code


def record_info(path, vocab_index, unk):
    meta, ts, calls = read_record(path)
    seq = np.array([vocab_index.get(c, unk) for c in calls], dtype=np.int64)
    return seq, ts, exploit_ns(meta)


def split_windows(res_dir, data_dir, split, db, n, W, S, vocab_index, unk):
    """Оценки STIDE для окон выбранной выборки в том же порядке, что в windows_<split>.npz,
    и длительность каждой записи. Возвращает (score, y, rec, durations)."""
    d = np.load(Path(res_dir) / f"windows_{split}.npz", allow_pickle=False)
    names, y_ref, rec_ref = list(d["names"]), d["y"], d["rec"]
    path_of = {p.name: p for p in get_archives(data_dir, split)}

    scores, dur = [], {}
    for rid, name in enumerate(names):
        seq, ts, attack_ns = record_info(path_of[name], vocab_index, unk)
        dur[rid] = (ts[-1] - ts[0]) / 1e9
        codes = ngram_codes(seq, n)
        bad = (~np.isin(codes, db)).astype(np.float64)        # 1 = n-граммы нет в базе
        csum = np.r_[0.0, np.cumsum(bad)]
        for s in range(0, len(seq) - W + 1, S):
            t0, t1 = ts[s], ts[s + W - 1]
            if attack_ns is not None and t1 >= attack_ns and t0 < attack_ns:
                continue                                       # граничное окно, как в features_windows
            lo, hi = s, s + W - n + 1                          # n-граммы целиком внутри окна
            scores.append((csum[hi] - csum[lo]) / max(hi - lo, 1))
    scores = np.array(scores)
    if len(scores) != len(y_ref):
        raise SystemExit(f"{split}: число окон не совпало ({len(scores)} против {len(y_ref)}). "
                         "Проверьте, что W, S и словарь те же, что при построении windows_*.npz.")
    return scores, y_ref, rec_ref, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("res_dir")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--part", choices=["all", "dev", "final"], default="dev")
    a = ap.parse_args()
    out = Path(a.res_dir)
    n = a.n

    meta = np.load(out / "windows_training.npz", allow_pickle=False)
    vocab = [str(v) for v in meta["vocab"]]
    W, S = int(meta["W"]), int(meta["S"])
    vocab_index = {c: i for i, c in enumerate(vocab[:-1])}      # последний элемент - UNK
    unk = len(vocab) - 1
    if len(vocab) > BASE:
        raise SystemExit("словарь длиннее 63 вызовов, увеличьте BASE")
    print(f"W={W}, S={S}, словарь {len(vocab) - 1}, n-грамма n={n}, часть test: {a.part}")

    # ---- база n-грамм по training ----
    data_dir = a.data_dir
    tr = np.load(out / "windows_training.npz", allow_pickle=False)
    path_of = {p.name: p for p in get_archives(data_dir, "training")}
    parts, dur_tr = [], []
    for name in tr["names"]:
        seq, ts, _ = record_info(path_of[str(name)], vocab_index, unk)
        parts.append(np.unique(ngram_codes(seq, n)))
        dur_tr.append((ts[-1] - ts[0]) / 1e9)
    db = np.unique(np.concatenate(parts))
    print(f"База: {len(db)} различных {n}-грамм по {len(tr['names'])} обучающим записям")
    med_dur = float(np.median(dur_tr))

    s_va, y_va, r_va, dur_va = split_windows(out, data_dir, "validation", db, n, W, S, vocab_index, unk)
    s_te, y_te, r_te, dur_te = split_windows(out, data_dir, "test", db, n, W, S, vocab_index, unk)

    m = test_part_mask(y_te, r_te, a.part)
    s_te, y_te, r_te = s_te[m], y_te[m], r_te[m]
    rl = record_labels(y_te, r_te)
    print(f"test ({a.part}): окон {len(y_te)} (доля атак {y_te.mean():.3f}), "
          f"записей {len(rl)} (атакующих {int(rl.sum())})")

    rows = []
    thr_w = np.percentile(s_va, 99)
    mw = window_metrics(y_te, s_te, thr_w)
    rows.append(dict(baseline=f"STIDE n={n}", level="window", **mw))
    for how in AGGREGATORS:
        rv = record_scores(s_va, r_va, thr_w, how)
        rt = record_scores(s_te, r_te, thr_w, how).reindex(rl.index)
        mr = window_metrics(rl.values, rt.values, np.percentile(rv.values, 95))
        rows.append(dict(baseline=f"STIDE n={n}", level=f"record_{how}", **mr))

    # ---- правило по длительности записи ----
    dv = np.array([abs(dur_va[r] - med_dur) for r in sorted(dur_va)])
    dt = np.array([abs(dur_te[r] - med_dur) for r in rl.index])
    md = window_metrics(rl.values, dt, np.percentile(dv, 95))
    rows.append(dict(baseline="длительность записи", level="record", **md))

    res = pd.DataFrame(rows)
    res.to_csv(out / f"baselines_{a.part}.csv", index=False, encoding="utf-8-sig")
    show = ["baseline", "level", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
            "tp", "fp", "fn", "tn"]
    print("\nBaseline (те же пороги и разбиение, что у моделей):")
    print(res[show].round(3).to_string(index=False))
    att_d = np.array([dur_te[r] for r in rl.index[rl.values == 1]])
    nor_d = np.array([dur_te[r] for r in rl.index[rl.values == 0]])
    print(f"\nДлительность записей, с: норма медиана {np.median(nor_d):.1f} "
          f"(мин {nor_d.min():.1f}, макс {nor_d.max():.1f}); "
          f"атака медиана {np.median(att_d):.1f} (мин {att_d.min():.1f}, макс {att_d.max():.1f})")
    print("Сохранено:", out / f"baselines_{a.part}.csv")


if __name__ == "__main__":
    main()