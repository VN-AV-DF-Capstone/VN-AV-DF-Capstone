"""Tên dataset và PART đang dùng cho generation / training ở local.

Pipeline real v003 (src/data) tự chia part: mỗi part 20 nam + 20 nữ × 20 clip,
split train/validation/test gán sẵn trong part (xem configs/data.yaml, mục parts).
Part N được export ra data/exports/<DATASET_NAME>/<DATASET_NAME>-partN.

PART dưới đây chỉ chọn part mà generation/review/finalize ở local đọc. Kaggle: chọn PART ngay
trong notebook (notebook ghi đè cấu hình trong phiên chạy).
"""

DATASET_NAME = "vn-av-df-data"  # Code generation / training / notebook đang dùng tên này.
PART = 1


def part_name(number=None):
    """Part là gói dữ liệu bổ sung, không phải số phiên bản của thí nghiệm."""
    number = PART if number is None else number
    if type(number) is not int or number < 1:
        raise ValueError("PART must be a positive integer")
    return f"{DATASET_NAME}-part{number}"
