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

attack_events = metadata.get("time", {}).get("exploit", [])

if not attack_events:
    print("В JSON не найдено время атаки.")
    raise SystemExit

attack_time = attack_events[0]["absolute"]
attack_ns = int(attack_time * 1_000_000_000)

print("Архив:", archive_path.name)
print("Время атаки (секунды):", attack_time)
print("Время атаки (наносекунды):", attack_ns)

directions = Counter()
syscalls = Counter()

nearby_events = []

for line in lines:
    parts = line.split()

    if len(parts) < 7:
        continue

    timestamp = int(parts[0])
    syscall = parts[5]
    direction = parts[6]

    directions[direction] += 1

    if direction == ">":
        syscalls[syscall] += 1

    # События в пределах 2 секунд от начала атаки
    if abs(timestamp - attack_ns) <= 2_000_000_000:
        nearby_events.append((timestamp, line))

print("\nНаправления событий:", dict(directions))
print("Количество уникальных системных вызовов:", len(syscalls))

print("\nТоп-15 системных вызовов по входам (>):")

for name, count in syscalls.most_common(15):
    print(f"{name}: {count}")

print("\nСобытия за 2 секунды до и после начала атаки:")

for timestamp, line in sorted(nearby_events)[:30]:
    difference = (timestamp - attack_ns) / 1_000_000_000
    print(f"{difference:+.6f} сек | {line}")

print("\nВсего событий в выбранном временном окне:",
      len(nearby_events))