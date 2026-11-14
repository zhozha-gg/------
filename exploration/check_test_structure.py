from pathlib import Path

DATA_DIR = Path(
    "data/Bruteforce_CWE-307/Bruteforce_CWE-307"
)

test_dir = DATA_DIR / "test"

print("Путь к test:", test_dir.resolve())
print("Папка существует:", test_dir.exists())

print("\nСодержимое папки test:")

for item in sorted(test_dir.iterdir()):
    if item.is_dir():
        print(f"[ПАПКА] {item.name}")
    else:
        print(f"[ФАЙЛ] {item.name}")

print("\nПоиск ZIP во всех вложенных папках:")

archives = list(test_dir.rglob("*.zip"))

print("Всего найдено ZIP:", len(archives))

print("\nПервые 20 архивов:")

for archive in archives[:20]:
    print(archive.relative_to(DATA_DIR))