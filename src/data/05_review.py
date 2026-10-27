"""Duyệt theo speaker ở http://127.0.0.1:8001 : chọn 15-20 clip + gán giới cho từng speaker."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "preparation"))
from _run import run

if __name__ == "__main__":
    run(["review"])
