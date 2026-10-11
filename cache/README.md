# Cache local

Preview, phép đo chất lượng và feature có thể tạo lại. Không coi cache cũ là đầu vào mới. Mỗi lần trích feature ghi `run.json` với hash checkpoint, manifest, media và preprocessing; chỉ dùng lại cache khi các hash này khớp. Không commit nội dung cache.

| Thư mục | Nội dung |
|---|---|
| `features/dataset_v1/wav2lip_multi_001_crf23_32/` | Feature hai nhánh AV-HuBERT của run Wav2Lip (1,1 GB); `y_001`, `02_check_cues.py` và `02_fuse.py` đọc |
| `features/dataset_v1/wav2lip_multi_001_crf23_32_fused/` | Feature sau Transformer hợp nhất (1,1 GB); model căn chỉnh, đo độ lệch và `peak_sync` đọc |
| `features/dataset_v1/real_smoke_001/` | Smoke một clip thật ngày 16/09, giữ làm bằng chứng |
| `musetalk_setup/` | Log cài MuseTalk và khóa weights |
| `pip_musetalk/`, `huggingface/`, `numba_*`, `musetalk_yapf/` | Cache tải/biên dịch cho generator; xóa được, lần chạy sau sẽ tải hoặc tạo lại |
| `previews/` | Preview ROI cho review (`PREVIEWS` trong settings), hiện rỗng |
