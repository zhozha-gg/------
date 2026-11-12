
import json
from pathlib import Path
from zipfile import ZipFile

DATA_DIR = Path(
    "data/Bruteforce_CWE-307/Bruteforce_CWE-307"
)

# Берём первую запись из training
archives = sorted((DATA_DIR / "training").glob("*.zip"))

if not archives:
    raise FileNotFoundError("Архивы training не найдены")

archive_path = archives[0]

print("ИЗУЧАЕМ ЗАПИСЬ:", archive_path.name)

with ZipFile(archive_path, "r") as archive:

    print("\n" + "=" * 60)
    print("1. JSON — МЕТАДАННЫЕ")
    print("=" * 60)

    json_file = next(
        name for name in archive.namelist()
        if name.endswith(".json")
    )

    raw_json = archive.read(json_file).decode(
        "utf-8", errors="replace"
    )

    try:
        metadata = json.loads(raw_json)
        print(json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ))
    except json.JSONDecodeError:
        print(raw_json[:5000])

    print("\n" + "=" * 60)
    print("2. SC — СИСТЕМНЫЕ ВЫЗОВЫ")
    print("=" * 60)

    sc_file = next(
        name for name in archive.namelist()
        if name.endswith(".sc")
    )

    raw_sc = archive.read(sc_file).decode(
        "utf-8", errors="replace"
    )

    lines = raw_sc.splitlines()

    print("Всего строк:", len(lines))
    print("\nПервые 30 строк:\n")

    for line in lines[:30]:
        print(line)

    print("\n" + "=" * 60)
    print("3. RES — ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ")
    print("=" * 60)

    res_file = next(
        name for name in archive.namelist()
        if name.endswith(".res")
    )

    raw_res = archive.read(res_file).decode(
        "utf-8", errors="replace"
    )

    print(raw_res[:3000])

print("\nИЗУЧЕНИЕ ЗАПИСИ ЗАВЕРШЕНО")
