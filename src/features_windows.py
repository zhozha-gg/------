"""Окна системных вызовов + разметка по времени эксплойта.

Запуск:
    python features_windows.py data/Bruteforce_CWE-307/Bruteforce_CWE-307 results/bruteforce
    python features_windows.py <папка_сценария> <папка_результата> [W] [S]
"""
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

# Колонки строки .sc (как в вашем features.py)
TS_COL, SYSCALL_COL, DIR_COL = 0, 5, 6


# ---------- чтение записи ----------

def find_archive_file(archive, extension):
    files = [n for n in archive.namelist()
             if n.endswith(extension) and "__MACOSX" not in n]
    if not files:
        raise ValueError(f"Файл {extension} не найден в архиве")
    return files[0]


def read_record(archive_path):
    """Возвращает (metadata, времена[нс], имена вызовов) только по входам '>'."""
    with ZipFile(archive_path, "r") as archive:
        meta = json.loads(
            archive.read(find_archive_file(archive, ".json")).decode("utf-8"))
        sc_text = archive.read(
            find_archive_file(archive, ".sc")).decode("utf-8", errors="replace")

    ts, calls = [], []
    for line in sc_text.splitlines():
        p = line.split()
        if len(p) < 7 or p[DIR_COL] != ">":
            continue
        try:
            ts.append(int(p[TS_COL]))
        except ValueError:
            continue
        calls.append(p[SYSCALL_COL])

    order = np.argsort(ts, kind="stable")
    ts = np.array(ts, dtype=np.int64)[order]
    calls = [calls[i] for i in order]
    return meta, ts, calls


def exploit_ns(meta):
    events = meta.get("time", {}).get("exploit", [])
    return int(events[0]["absolute"] * 1_000_000_000) if events else None


def get_archives(data_dir, split):
    return sorted((Path(data_dir) / split).rglob("*.zip"))


# ---------- словарь только по training ----------

def build_vocabulary(data_dir):
    vocab = set()
    for path in get_archives(data_dir, "training"):
        _, _, calls = read_record(path)
        vocab.update(calls)
    return sorted(vocab)


# ---------- окна и метки ----------

def make_windows(ts, calls, vocab, W=20, S=10, attack_ns=None):
    unk = len(vocab)
    idx = {c: i for i, c in enumerate(vocab)}
    seq = np.array([idx.get(c, unk) for c in calls])
    X, y = [], []
    for s in range(0, len(seq) - W + 1, S):
        t0, t1 = ts[s], ts[s + W - 1]
        if attack_ns is None or t1 < attack_ns:
            label = 0
        elif t0 >= attack_ns:
            label = 1
        else:
            continue                      # окно на границе пропускаем
        X.append(np.bincount(seq[s:s + W], minlength=unk + 1) / W)
        y.append(label)
    if not X:
        return np.empty((0, unk + 1)), np.empty(0, dtype=int)
    return np.array(X), np.array(y)


# ---------- обработка выборки ----------

def process_split(data_dir, split, vocab, out_dir, W, S):
    archives = get_archives(data_dir, split)
    print(f"\n{split.upper()}: архивов {len(archives)}")

    X_parts, y_parts, rec_parts, names = [], [], [], []
    skipped = []

    for path in archives:
        try:
            meta, ts, calls = read_record(path)
        except Exception as error:
            skipped.append((path.name, str(error)))
            continue

        attack_ns = exploit_ns(meta)

        # Запись из папки normal_and_attack обязана иметь время эксплойта,
        # иначе все её окна молча получили бы метку 0.
        if path.parent.name == "normal_and_attack" and attack_ns is None:
            skipped.append((path.name, "нет времени эксплойта"))
            continue

        X, y = make_windows(ts, calls, vocab, W, S, attack_ns)
        if len(X) == 0:
            skipped.append((path.name, "нет окон"))
            continue

        rec_id = len(names)
        names.append(path.name)
        X_parts.append(X)
        y_parts.append(y)
        rec_parts.append(np.full(len(y), rec_id))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    rec = np.concatenate(rec_parts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"windows_{split}.npz",
        X=X.astype(np.float32), y=y, rec=rec,
        names=np.array(names), vocab=np.array(vocab + ["UNK"]),
        W=W, S=S)

    print("Записей:", len(names), "| окон:", len(y),
          "| метки:", dict(Counter(y.tolist())))
    if skipped:
        print("Пропущено:", len(skipped))
        for n, e in skipped[:10]:
            print("  ", n, ":", e)


def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    S = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    vocab = build_vocabulary(data_dir)
    print("Словарь по training:", len(vocab), "системных вызовов")

    for split in ("training", "validation", "test"):
        process_split(data_dir, split, vocab, out_dir, W, S)


if __name__ == "__main__":
    main()