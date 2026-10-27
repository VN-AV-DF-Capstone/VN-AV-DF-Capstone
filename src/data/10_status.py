"""Xem toàn cảnh: batch, clip, speaker, part, dung lượng còn trống."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "preparation"))
from _run import run

if __name__ == "__main__":
    run(["status"])
