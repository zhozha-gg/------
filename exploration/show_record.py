"""Показывает метаданные времени и первые строки трассы одной записи.

Запуск (из корня проекта):
    python exploration/show_record.py data/Bruteforce_CWE-307/Bruteforce_CWE-307/test/normal_and_attack/aggressive_nightingale_1018.zip
"""
import json
import sys
from zipfile import ZipFile

path = sys.argv[1]

with ZipFile(path) as z:
    names = [n for n in z.namelist() if "__MACOSX" not in n]
    meta = json.loads(z.read(next(n for n in names if n.endswith(".json"))).decode("utf-8"))
    sc = z.read(next(n for n in names if n.endswith(".sc"))).decode("utf-8", errors="replace")

print("=== metadata['time'] ===")
print(json.dumps(meta.get("time"), indent=2, ensure_ascii=False))

print("\n=== остальные ключи metadata (кроме time) ===")
for k, v in meta.items():
    if k != "time":
        print(k, ":", str(v)[:200])

print("\n=== первые 5 строк .sc ===")
for line in sc.splitlines()[:5]:
    print(line)
    print("   поля:", list(enumerate(line.split()))[:10])