"""Прогон по размерам окна W: признаки + модели + сводная таблица.

Запуск (из корня проекта):
    python src/sweep.py data/Bruteforce_CWE-307/Bruteforce_CWE-307 bruteforce 20 100 400 --part dev --seeds 3
Результат: results/sweep_<name>.csv и таблица в консоли.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def run(cmd):
    print("\n>>", " ".join(map(str, cmd)))
    subprocess.run([sys.executable, *map(str, cmd)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("name")
    ap.add_argument("widths", nargs="*", type=int)
    ap.add_argument("--part", default="all", choices=["all", "dev", "final"])
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    data_dir, name = a.data_dir, a.name
    widths = a.widths or [20, 100, 400]
    frames = []
    for W in widths:
        out = Path("results") / f"{name}_W{W}"
        run([HERE / "features_windows.py", data_dir, out, W, max(W // 2, 1)])
        run([HERE / "run_experiment.py", out, "--part", a.part, "--seeds", a.seeds])
        df = pd.read_csv(out / "metrics.csv", encoding="utf-8-sig")
        df.insert(0, "W", W)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.insert(0, "scenario", name)
    path = Path("results") / f"sweep_{name}_{a.part}.csv"
    all_df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\nROC-AUC по W (строки: модель и уровень, столбцы: W)")
    pivot = all_df.pivot_table(index=["model", "level"], columns="W",
                               values="roc_auc").round(3)
    print(pivot.to_string())
    print("\nСохранено:", path)


if __name__ == "__main__":
    main()