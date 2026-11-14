import json
import numpy as np
from zipfile import ZipFile

TS_COL, SYSCALL_COL, DIR_COL = 0, 5, 6   # как в вашем features.py
# Колонку thread id найдите по выводу inspect_data.py (для окон по потокам, опционально)

def read_record(zip_path):
    with ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if "__MACOSX" not in n]
        meta = json.loads(z.read(next(n for n in names if n.endswith(".json"))).decode("utf-8"))
        sc = z.read(next(n for n in names if n.endswith(".sc"))).decode("utf-8", errors="replace")
    ts, calls = [], []
    for line in sc.splitlines():
        p = line.split()
        if len(p) < 7 or p[DIR_COL] != ">":      # только входы в вызов
            continue
        try:
            ts.append(int(p[TS_COL]))
        except ValueError:
            continue
        calls.append(p[SYSCALL_COL])
    order = np.argsort(ts, kind="stable")
    return meta, np.array(ts, dtype=np.int64)[order], [calls[i] for i in order]

def exploit_ns(meta):
    ev = meta.get("time", {}).get("exploit", [])
    return int(ev[0]["absolute"] * 1e9) if ev else None