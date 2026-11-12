
import csv
import json
from pathlib import Path
from zipfile import ZipFile
from collections import Counter


# ============================================================
# ПУТИ
# ============================================================

DATA_DIR = Path(
    "data/Bruteforce_CWE-307/Bruteforce_CWE-307"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Выходные файлы
TRAIN_CSV = RESULTS_DIR / "features_train.csv"
VAL_CSV = RESULTS_DIR / "features_validation.csv"
TEST_CSV = RESULTS_DIR / "features_test.csv"


# ============================================================
# ПОИСК ФАЙЛОВ ВНУТРИ ZIP
# ============================================================

def find_archive_file(archive, extension):
    files = [
        name for name in archive.namelist()
        if name.endswith(extension)
        and "__MACOSX" not in name
    ]

    if not files:
        raise ValueError(
            f"Файл {extension} не найден в архиве"
        )

    return files[0]


# ============================================================
# ИЗВЛЕЧЕНИЕ МЕТАДАННЫХ И СИСТЕМНЫХ ВЫЗОВОВ
# ============================================================

def extract_record(archive_path):
    with ZipFile(archive_path, "r") as archive:

        json_file = find_archive_file(archive, ".json")
        sc_file = find_archive_file(archive, ".sc")

        metadata = json.loads(
            archive.read(json_file).decode("utf-8")
        )

        sc_text = archive.read(sc_file).decode(
            "utf-8",
            errors="replace"
        )

    syscall_counts = Counter()
    timestamps = []

    malformed = 0

    for line in sc_text.splitlines():
        parts = line.split()

        if len(parts) < 7:
            malformed += 1
            continue

        try:
            timestamp = int(parts[0])
        except ValueError:
            malformed += 1
            continue

        direction = parts[6]

        # Считаем только входы в системные вызовы.
        # Одна операция не должна учитываться дважды.
        if direction != ">":
            continue

        syscall = parts[5]

        syscall_counts[syscall] += 1
        timestamps.append(timestamp)

    if not timestamps:
        raise ValueError(
            "В трассе не найдено системных вызовов"
        )

    # Временные отметки Sysdig представлены в наносекундах.
    duration = (
        max(timestamps) - min(timestamps)
    ) / 1_000_000_000

    duration = max(duration, 0.000001)

    total_calls = sum(syscall_counts.values())

    return {
        "metadata": metadata,
        "syscall_counts": syscall_counts,
        "total_calls": total_calls,
        "duration": duration,
        "calls_per_second": total_calls / duration,
        "malformed_lines": malformed
    }


# ============================================================
# ОПРЕДЕЛЕНИЕ МЕТКИ
# ============================================================

def get_label(split, archive_path):
    if split in ("training", "validation"):
        return 0

    # Для тестовой выборки метку определяем по папке.
    folder = archive_path.parent.name

    if folder == "normal":
        return 0

    if folder == "normal_and_attack":
        return 1

    raise ValueError(
        f"Неизвестная папка тестовой записи: {folder}"
    )


# ============================================================
# ПОДГОТОВКА СПИСКА АРХИВОВ
# ============================================================

def get_archives(split):
    split_dir = DATA_DIR / split

    return sorted(split_dir.rglob("*.zip"))


# ============================================================
# ПОСТРОЕНИЕ ПРИЗНАКОВ
# ============================================================

def make_feature_row(
    archive_path,
    split,
    vocabulary
):
    record = extract_record(archive_path)

    counts = record["syscall_counts"]

    total = record["total_calls"]
    duration = record["duration"]

    row = {
        "archive": archive_path.name,
        "label": get_label(split, archive_path),
        "total_syscalls": total,
        "duration_seconds": duration,
        "calls_per_second": record["calls_per_second"]
    }

    # Частота каждого известного системного вызова
    for syscall in vocabulary:
        count = counts.get(syscall, 0)

        row[f"syscall_{syscall}"] = (
            count / duration
        )

    # Вызовы, которые не встречались в training
    unknown_count = sum(
        count
        for syscall, count in counts.items()
        if syscall not in vocabulary
    )

    row["syscall_UNKNOWN"] = (
        unknown_count / duration
    )

    row["unknown_syscalls_count"] = unknown_count

    return row


# ============================================================
# СОЗДАНИЕ СЛОВАРЯ СИСТЕМНЫХ ВЫЗОВОВ ПО TRAINING
# ============================================================

def build_vocabulary():
    vocabulary = set()

    archives = get_archives("training")

    print("\nФормирование словаря системных вызовов...")

    for archive_path in archives:
        record = extract_record(archive_path)

        vocabulary.update(
            record["syscall_counts"].keys()
        )

    vocabulary = sorted(vocabulary)

    print(
        "Уникальных системных вызовов в training:",
        len(vocabulary)
    )

    return vocabulary


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

def save_csv(rows, output_path, vocabulary):

    base_columns = [
        "archive",
        "label",
        "total_syscalls",
        "duration_seconds",
        "calls_per_second"
    ]

    syscall_columns = [
        f"syscall_{name}"
        for name in vocabulary
    ]

    other_columns = [
        "syscall_UNKNOWN",
        "unknown_syscalls_count"
    ]

    columns = (
        base_columns
        + syscall_columns
        + other_columns
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# ОБРАБОТКА ВЫБОРКИ
# ============================================================

def process_split(split, vocabulary, output_path):

    archives = get_archives(split)

    print("\n" + "=" * 60)
    print("ОБРАБОТКА:", split.upper())
    print("Архивов найдено:", len(archives))
    print("=" * 60)

    rows = []
    errors = []

    for index, archive_path in enumerate(
        archives,
        start=1
    ):
        try:
            row = make_feature_row(
                archive_path,
                split,
                vocabulary
            )

            rows.append(row)

        except Exception as error:
            errors.append(
                (archive_path.name, str(error))
            )

        if index % 50 == 0 or index == len(archives):
            print(
                f"Обработано: {index}/{len(archives)}"
            )

    save_csv(rows, output_path, vocabulary)

    labels = Counter(
        row["label"] for row in rows
    )

    print("\nУспешно обработано:", len(rows))
    print("Распределение меток:", dict(labels))
    print("Ошибок:", len(errors))
    print("CSV сохранён:", output_path)

    if errors:
        print("\nПервые ошибки:")

        for name, error in errors[:10]:
            print(name, ":", error)

    return rows


# ============================================================
# ЗАПУСК
# ============================================================

def main():

    print("=" * 60)
    print("ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ СИСТЕМНЫХ ВЫЗОВОВ")
    print("=" * 60)

    vocabulary = build_vocabulary()

    train_rows = process_split(
        "training",
        vocabulary,
        TRAIN_CSV
    )

    val_rows = process_split(
        "validation",
        vocabulary,
        VAL_CSV
    )

    test_rows = process_split(
        "test",
        vocabulary,
        TEST_CSV
    )

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)

    print("Training:", len(train_rows))
    print("Validation:", len(val_rows))
    print("Test:", len(test_rows))

    print("\nРезультаты находятся в папке results.")


if __name__ == "__main__":
    main()
