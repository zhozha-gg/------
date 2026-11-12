import json
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

DATA_DIR = Path(
    "data/Bruteforce_CWE-307/Bruteforce_CWE-307"
)

# Training и validation
for split in ["training", "validation"]:
    archives = list((DATA_DIR / split).rglob("*.zip"))

    print(f"\n{split.upper()}: {len(archives)} архивов")

    labels = Counter()

    for path in archives:
        with ZipFile(path, "r") as archive:
            json_files = [
                name for name in archive.namelist()
                if name.endswith(".json")
                and "__MACOSX" not in name
            ]

            if not json_files:
                continue

            metadata = json.loads(
                archive.read(json_files[0]).decode("utf-8")
            )

            labels[str(metadata.get("exploit", "MISSING"))] += 1

    print("Значения exploit:", dict(labels))


# Test: учитываем названия вложенных папок
test_dir = DATA_DIR / "test"

print("\nTEST")

folder_counts = Counter()
metadata_counts = Counter()
examples = {}

for path in test_dir.rglob("*.zip"):
    folder = path.parent.name
    folder_counts[folder] += 1

    with ZipFile(path, "r") as archive:
        json_files = [
            name for name in archive.namelist()
            if name.endswith(".json")
            and "__MACOSX" not in name
        ]

        if not json_files:
            continue

        metadata = json.loads(
            archive.read(json_files[0]).decode("utf-8")
        )

        exploit = metadata.get("exploit", "MISSING")
        time_exploit = metadata.get("time", {}).get(
            "exploit", "MISSING"
        )

        metadata_counts[(folder, str(exploit))] += 1

        if folder not in examples:
            examples[folder] = {
                "file": path.name,
                "exploit": exploit,
                "time.exploit": time_exploit
            }

print("Количество архивов по папкам:", dict(folder_counts))
print("\nЗначения exploit по папкам:")

for key, count in metadata_counts.items():
    print(f"{key}: {count}")

print("\nПримеры метаданных:")
for folder, example in examples.items():
    print(folder, example)