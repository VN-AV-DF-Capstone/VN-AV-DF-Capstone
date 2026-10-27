"""Gom speaker đã duyệt đủ thành part (20 nam + 20 nữ × 15-20 clip), chia split trong part."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "preparation"))
from _run import run

if __name__ == "__main__":
    run(["parts"])
