"""Đường dẫn chung của các bước real v003 (chạy từ bất kỳ thư mục nào; cwd = gốc repo).

DATA: workspace (ledger, raw, pool, registry, archive). Thử luật cắt mới trên workspace riêng:
    set VNAV_DATA=data/v003_trial   (PowerShell: $env:VNAV_DATA="data/v003_trial")
Part export ra data/exports/<DATASET_NAME>/<DATASET_NAME>-partN (generation đọc từ đó).
configs/data.yaml và checkpoints/media/ nằm ở gốc repo.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from settings import DATASET_NAME  # noqa: E402, F401

DATA = os.environ.get("VNAV_DATA", "data/v003")
EXPORTS = "data/exports"
