"""Tải một batch: `02_download.py` (batch đầu còn việc) hoặc `02_download.py 3`.

Ctrl+C rồi chạy lại để tải tiếp; video lỗi mạng được thử lại.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "preparation"))
from _run import run

if __name__ == "__main__":
    run(["download"], batch_step=True)
