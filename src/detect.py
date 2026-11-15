"""Разбор записи (или папки с записями) LID-DS обученной моделью.

Запуск (из корня проекта):
    python src/detect.py models/bruteforce_gmm.joblib путь/к/записи.zip
    python src/detect.py models/bruteforce_gmm.joblib путь/к/папке --limit 20
    python src/detect.py models/bruteforce_gmm.joblib запись.zip --csv results/one_record_windows.csv

Для каждой записи выводится: число окон, доля окон выше порога, оценка записи (p95 оценок окон),
вердикт и системные вызовы, чья доля в аномальных окнах сильнее всего отличается от обычной.
Это офлайн-анализ уже записанной трассы, не мониторинг в реальном времени.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features_windows import read_record


def windows_of(path, vocab, W, S):
    _, ts, calls = read_record(path)
    idx = {c: i for i, c in enumerate(vocab[:-1])}
    unk = len(vocab) - 1
    seq = np.array([idx.get(c, unk) for c in calls])
    X = [np.bincount(seq[s:s + W], minlength=unk + 1) / W
         for s in range(0, len(seq) - W + 1, S)]
    return np.array(X, dtype=np.float32), len(seq)


def analyse(model, path, csv=None):
    vocab, W, S = model["vocab"], model["W"], model["S"]
    X, n_calls = windows_of(path, vocab, W, S)
    print(f"\n=== {Path(path).name} ===")
    if len(X) == 0:
        print(f"Запись короче одного окна ({n_calls} вызовов при W={W}), разбор невозможен.")
        return None
    Z = model["scaler"].transform(X)
    s = -model["gmm"].score_samples(Z)
    flag = s > model["thr_window"]
    rec_score = float(np.percentile(s, 95))
    verdict = "АНОМАЛИЯ" if rec_score > model["thr_record"] else "норма"
    print(f"Вызовов: {n_calls}, окон: {len(X)}")
    print(f"Окон выше порога: {int(flag.sum())} ({flag.mean():.1%}); "
          f"порог окна {model['thr_window']:.1f}")
    print(f"Оценка записи (p95): {rec_score:.1f}, порог записи {model['thr_record']:.1f}")
    print(f"Вердикт: {verdict}")
    if flag.any():
        dev = Z[flag].mean(0)                       # на сколько стандартных отклонений сдвинуты доли
        order = np.argsort(-np.abs(dev))[:5]
        print("Вызовы, отличающиеся в аномальных окнах (доля в аномальных окнах / обычная доля):")
        for i in order:
            print(f"  {vocab[i]:<18} {X[flag][:, i].mean():.3f} / {model['train_mean_share'][i]:.3f}")
    if csv:
        pd.DataFrame(dict(window=np.arange(len(s)), score=s, above_threshold=flag)).to_csv(
            csv, index=False)
        print("Оценки окон сохранены:", csv)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--csv")
    a = ap.parse_args()
    model = joblib.load(a.model)
    p = Path(a.path)
    files = [p] if p.is_file() else sorted(p.rglob("*.zip"))[:a.limit]
    if not files:
        raise SystemExit("zip-архивов не найдено")
    for f in files:
        analyse(model, f, a.csv if len(files) == 1 else None)


if __name__ == "__main__":
    main()