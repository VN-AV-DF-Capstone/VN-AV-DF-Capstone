"""Đọc data/v003/inputs/urls.csv (url, speakers = 1 | n, note) → chọn video → chia batch raw.

Lần đầu chạy sẽ tạo file mẫu urls.csv. Thêm URL rồi chạy lại: chỉ xử lý dòng mới / dòng lỗi.
Danh sách từng batch: data/v003/raw/batch_XXX/videos.csv. Thêm --cookies-from-browser firefox nếu
YouTube chặn.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "preparation"))
from _run import run

if __name__ == "__main__":
    run(["plan"])
