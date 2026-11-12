import json
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

DATA_DIR = Path(
    "data/Bruteforce_CWE-307/Bruteforce_CWE-307"
)

archive_path = (
    DATA_DIR
    / "test"
    / "normal_and_attack"
    / "aggressive_nightingale_1018.zip"
)

with ZipFile(archive_path, "r") as archive:

    json_file = next(
        name for name in archive.namelist()
        if name.endswith(".json")
        and "__MACOSX" not in name
    )

    sc_file = next(
        name for name in archive.namelist()
        if name.endswith(".sc")
    )

    metadata = json.loads(
        archive.read(json_file).decode("utf-8")
    )

    lines = archive.read(sc_file).decode(
        "utf-8", errors="replace"
    ).splitlines()

attack_time = metadata["time"]["exploit"][0]["absolute"]
attack_ns = int(attack_time * 1_000_000_000)

events = []

for line in lines:
    parts = line.split()

    if len(parts) < 7:
        continue

    timestamp = int(parts[0])
    syscall = parts[5]
    direction = parts[6]

    if direction == ">":
        events.append((timestamp, syscall, line))

events.sort(key=lambda x: x[0])

before = [e for e in events if e[0] < attack_ns]
after = [e for e in events if e[0] >= attack_ns]

print("=" * 60)
print("ВРЕМЕННАЯ ПРОВЕРКА АТАКИ")
print("=" * 60)

print("Всего вызовов (> или входов):", len(events))
print("До начала атаки:", len(before))
print("После начала атаки:", len(after))

if events:
    print(
        "Начало трассы:",
        (events[0][0] - attack_ns) / 1e9,
        "сек относительно атаки"
    )

    print(
        "Конец трассы:",
        (events[-1][0] - attack_ns) / 1e9,
        "сек относительно атаки"
    )

print("\nПервые 15 вызовов ПОСЛЕ начала атаки:")

for timestamp, syscall, line in after[:15]:
    delta = (timestamp - attack_ns) / 1e9
    print(f"{delta:+.6f} сек | {syscall} | {line}")

print("\nПоследние 15 вызовов ПЕРЕД началом атаки:")

for timestamp, syscall, line in before[-15:]:
    delta = (timestamp - attack_ns) / 1e9
    print(f"{delta:+.6f} сек | {syscall} | {line}")

print("\nКоличество вызовов в интервалах:")

intervals = [
    (-10, -5),
    (-5, -2),
    (-2, -1),
    (-1, 0),
    (0, 1),
    (1, 2),
    (2, 5),
    (5, 10)
]

for start, end in intervals:
    count = sum(
        1 for timestamp, _, _ in events
        if start <= (timestamp - attack_ns) / 1e9 < end
    )

    print(f"{start:>3} ... {end:>3} сек: {count}")