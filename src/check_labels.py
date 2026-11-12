
import json
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

DATA_DIR = Path(
    "data/Bruteforce_CWE-307/Bruteforce_CWE-307"
)

splits = ["training", "validation", "test"]

print("=" * 65)
print("ПРОВЕРКА РАЗМЕТКИ DATASET")
print("=" * 65)

for split in splits:
    archives = sorted((DATA_DIR / split).glob("*.zip"))

    exploit_counts = Counter()
    time_exploit_counts = Counter()
    examples = []

    for archive_path in archives:
        try:
            with ZipFile(archive_path, "r") as archive:
                json_file = next(
                    name for name in archive.namelist()
                    if name.endswith(".json")
                    and "__MACOSX" not in name
                )

                metadata = json.loads(
                    archive.read(json_file).decode("utf-8")
                )

                exploit = metadata.get("exploit", "MISSING")
                time_exploit = metadata.get(
                    "time", {}
                ).get("exploit", "MISSING")

                exploit_counts[str(exploit)] += 1

                if isinstance(time_exploit, list):
                    state = (
                        "empty"
                        if len(time_exploit) == 0
                        else "filled"
                    )
                else:
                    state = str(time_exploit)

                time_exploit_counts[state] += 1

                if exploit is True or (
                    isinstance(time_exploit, list)
                    and len(time_exploit) > 0
                ):
                    if len(examples) < 5:
                        examples.append({
                            "file": archive_path.name,
                            "exploit": exploit,
                            "time.exploit": time_exploit
                        })

        except Exception as error:
            print("Ошибка:", archive_path.name, error)

    print(f"\n{'=' * 50}")
    print("ВЫБОРКА:", split.upper())
    print("=" * 50)

    print("Всего архивов:", len(archives))
    print("Поле exploit:", dict(exploit_counts))
    print("Поле time.exploit:", dict(time_exploit_counts))

    print("\nПримеры записей с признаками атаки:")

    for example in examples:
        print(example)


# Дополнительно изучаем структуру системных вызовов
print("\n" + "=" * 65)
print("ПРОВЕРКА ФОРМАТА SC")
print("=" * 65)

sample_archive = next(
    (DATA_DIR / "training").glob("*.zip")
)

with ZipFile(sample_archive, "r") as archive:
    sc_file = next(
        name for name in archive.namelist()
        if name.endswith(".sc")
    )

    text = archive.read(sc_file).decode(
        "utf-8", errors="replace"
    )

    directions = Counter()
    syscalls = Counter()
    malformed = 0

    for line in text.splitlines():
        parts = line.split()

        if len(parts) < 7:
            malformed += 1
            continue

        syscall = parts[5]
        direction = parts[6]

        syscalls[syscall] += 1
        directions[direction] += 1

    print("Архив:", sample_archive.name)
    print("Направления событий:", dict(directions))
    print("Количество уникальных системных вызовов:",
          len(syscalls))
    print("Первые 20 системных вызовов:")

    for name, count in syscalls.most_common(20):
        print(f"{name}: {count}")

    print("Строк с нестандартной структурой:", malformed)

print("\nПРОВЕРКА ЗАВЕРШЕНА")
