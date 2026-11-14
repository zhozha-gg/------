import sys
from zipfile import ZipFile

with ZipFile(sys.argv[1]) as z:
    name = next(n for n in z.namelist() if n.endswith(".sc") and "__MACOSX" not in n)
    lines = z.read(name).decode("utf-8", errors="replace").splitlines()

col2, col4, diff = set(), set(), 0
for line in lines:
    p = line.split()
    if len(p) < 7:
        continue
    col2.add(p[2]); col4.add(p[4])
    diff += p[2] != p[4]

print("уникальных в колонке 2:", len(col2))
print("уникальных в колонке 4:", len(col4))
print("строк, где колонки 2 и 4 различаются:", diff)