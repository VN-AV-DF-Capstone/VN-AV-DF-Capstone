"""Export một part ra data/exports/<dataset>/<dataset>-partN (thư mục, chưa zip): `07_export.py 1`.

Zip để lên Drive: `08_archive.py part 1` (giống raw).
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "preparation"))
from _run import run

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        raise SystemExit("Usage: 07_export.py <part>")
    part = sys.argv.pop(1)
    run(["export", "--part", part])
