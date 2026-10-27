"""Đóng gói và đánh dấu đã lên Drive (xoá bản ở máy: 09_clean.py).

    08_archive.py raw 3              ZIP raw batch 3 (sau khi đã cut + identify xong)
    08_archive.py raw 3 --uploaded   đánh dấu: zip raw batch 3 đã lên Drive
    08_archive.py part 1             ZIP part 1 (sau khi 07_export xong)
    08_archive.py part 1 --uploaded  đánh dấu: zip part 1 đã lên Drive
Nhớ up cả thư mục data/v003/ledger/ lên Drive mỗi lần.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "preparation"))
from _run import run

if __name__ == "__main__":
    run(["archive"])
