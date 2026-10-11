# Dữ liệu theo phiên bản: dataset_v1 và dataset_v2

`dataset_v1` là bộ sơ bộ đã cắt, review và chia split trên một người; từ 09/10/2026 được ghép với clip của người 2 (part1) thành split nhiều người cho run Wav2Lip. Nguồn mở rộng hiện ở `dataset_v2`. `src/data/preparation/settings.py` đang đặt `DATASET_VERSION = 'dataset_v2'` (đã commit), nên các bước 01–05 và review chạy cho V2; generator, nén, feature có settings riêng và vẫn dùng V1. Tên phiên bản không chứng nhận dataset đã hoàn tất. Bản AVSP-Net cũ vẫn ở backup/archive.

```text
data/
├── sources/<dataset_version>/
│   ├── videos.csv                  # URL video/playlist đã chọn
│   ├── selected_videos.csv         # Bước 01: chuẩn hóa và mở rộng playlist
│   └── rights.csv                  # Tùy chọn, khi kiểm tra giấy phép
├── raw/<dataset_version>/<download_run>/
│   ├── <video_id>.mp4              # Bước 02: video gốc có audio
│   ├── download_sources.csv        # Khóa danh sách nguồn của đợt tải
│   └── download_results.csv        # Trạng thái thành công/lỗi từng video
├── real/<dataset_version>/
│   └── <thư mục batch do cutter tạo>/
│       ├── media/*.mp4             # Bước 03: clip real đã cắt
│       ├── accepted_clips.csv
│       ├── rejected_windows.csv
│       ├── video_status.csv
│       └── run_summary.json        # Cùng config, checksum và log của cutter
├── manifests/<dataset_version>/
│   ├── clips.csv                   # Bước 04: clip + nguồn, chưa review
│   ├── reviews/
│   │   ├── assignments/            # assignment_<reviewer>.csv
│   │   ├── merged/                 # Nhãn đã gộp và mục cần phân xử
│   │   └── exports/clips/<reviewer>/  # review_<reviewer>.csv; media gom cho reviewer nếu cần
│   ├── reviewed_clips.csv          # Clip giữ lại; bổ sung speaker_id trước split
│   ├── real_splits.csv             # Bước 05: train/val/test và group_id
│   ├── real_splits_multi_001.csv   # Bước 07 (chỉ V1): V1 + part1, kèm .json provenance
│   └── generators/<run_id>/        # Plan, candidates.csv, summary, log của generator
├── generated/<dataset_version>/<run_id>/     # Video real/fake của từng run generator
├── compressed/<dataset_version>/<run_id>/    # Bước 06: media nén, variants.csv, media_pairs.csv
└── rights_evidence/                # Hồ sơ giấy phép khi nhóm bổ sung
```

## Hiện có trên máy (11/10/2026)

| Đường dẫn | Nội dung | Ai đọc |
|---|---|---|
| `sources/dataset_v1/`, `sources/dataset_v2/` | `videos.csv`, `selected_videos.csv`; V2 có 1.954 video đã chọn | Bước 01–02 |
| `raw/dataset_v1/download_001/` | Video gốc của V1, khoảng 3,4 GB | Cắt lại, `export_reviewed_clips` |
| `raw/dataset_v2/download_001/` | Khoảng 182 GB. Lần ghi cuối 26/09: 966 downloaded, 80 failed, 908 pending, 15 file `.part` | Bước 02 (tải tiếp), 03 khi tải đủ |
| `real/dataset_v1/cut_cut_001/` | 2.252 clip real của V1 (4,6 GB). Tên lặp `cut_` do cutter ghép với `CUT_RUN` | Các manifest V1 ghi đường dẫn tuyệt đối tới đây: không đổi tên |
| `manifests/dataset_v1/` | `clips.csv` (2.252), `reviewed_clips.csv` (1.143 keep), `real_splits.csv` (một người: 786/192/165), `real_splits_multi_001.csv` (1.718 clip, 11 người), `generators/`, `reviews/` | Bước 05, 07, generator |
| `generated/dataset_v1/wav2lip_multi_001/` | 946 cặp real/fake 2 giây (1,6 GB), chưa review bằng mắt | Bước 06 → feature → train/eval |
| `generated/dataset_v1/wav2lip_gpu_batch_004/` | Run thử cũ: 33 cặp 4 giây | Không bước nào đọc |
| `generated/dataset_v1/quality_10x2s_002/` | Bước 07 của generator: 10 clip train, Wav2Lip và MuseTalk cạnh nhau, chờ review | Chỉ để xem |
| `generated/dataset_v1/musetalk_smoke_5x2s_001/` | Smoke MuseTalk 5 clip | `preparation/musetalk_smoke.py` |
| `compressed/dataset_v1/wav2lip_multi_001_crf23_32/` | 3.784 file CRF 23/32 và `media_pairs.csv` (1.892 dòng), 1,2 GB | `src/features/01_extract.py` |
| `rights_evidence/` | Rỗng | Khi nhóm bổ sung hồ sơ |

`reviews/exports/` của V1 đã rỗng: media xuất cho reviewer đã xóa sau khi gộp. Nhãn đã gộp ở `reviews/merged/review_labels_v4.csv` (1.143 keep, 1.109 reject). Các `review_<reviewer>.csv` gốc chỉ còn trong lịch sử Git, trước commit `b27d891` (từ đó `data/` không còn được Git theo dõi).

## Clip của người 2 (part1)

852 clip đã cắt của người 2 nằm ở **root repo**: `vn-av-df-data-part1/vn-av-df-data-part1/` (`clips/`, `logs/`, `review.csv`), khoảng 2,3 GB. Thư mục này bị ignore như `data/`. `review.csv` có 575 keep, 222 reject, 55 uncertain, thuộc 10 người (`speaker_01`–`speaker_11`, không có `speaker_04`). Bước 07 (`src/data/07_add_part1.py`) chỉ lấy 575 clip keep.

`real_splits_multi_001.csv` ghi đường dẫn tuyệt đối tới các clip part1, và plan của `wav2lip_multi_001` khóa SHA-256 của manifest này. Nếu di chuyển part1 (ví dụ vào `real/dataset_v1/part1/`), sửa `PART1_REVIEW` trong bước 07 và ghi ra manifest mới (ví dụ `real_splits_multi_002.csv`) cho run sau; không ghi đè `_001`. Nén, feature và `experiments/` không đọc đường dẫn part1.

## Run Wav2Lip đang dùng

| Bước | Số lượng |
|---|---|
| `real_splits_multi_001.csv` | V1: 1.143 clip, một người (`spk_001`), tất cả ở train. Part1: train 225 (speaker_03/06/09), val 179 (speaker_01/07/10), test 171 (speaker_02/05/08/11) |
| Plan `wav2lip_multi_001` | `MAX_PER_SPLIT = 600`: lấy đủ part1, train thêm 375 clip V1; 950 dòng |
| Generator | 946 cặp; 4 lỗi đều ở train (2 V1, 2 part1), Wav2Lip không tìm thấy mặt |
| Nén | 946 × 2 clip × CRF 23/32, không lỗi |
| Feature | 1.843 mẫu dùng được, 49 lỗi (44 thiếu landmark, 5 nhiều mặt ở frame đầu; 24 lỗi từ video G5DQYAQQ5VM) |

Không đặt nguồn vào `data/dataset/_V2`: các file chạy dựng đường dẫn từ `data/sources|raw|real|manifests/<dataset_version>`. Với nguồn mới trong V2, chỉnh `videos.csv` rồi chạy bước 01. `raw/dataset_v2/download_001/download_sources.csv` đang khóa danh sách 1.954 video; nếu `selected_videos.csv` thay đổi, đặt `DOWNLOAD_RUN` mới thay vì trộn với đợt tải đang dở. Giữ file `.part` để tiếp tục tải. Xem [kiểm kê dọn dẹp](../docs/reports/2026-09-23_WORKSPACE_CLEANUP_AUDIT.md) trước khi xóa output generator V1; một số mục trong kiểm kê đó đã được dọn.

Giống repo cũ ở trình tự thu thập → tải → cắt → review → chia dữ liệu → sinh fake → nén. Khác ở việc thư mục tách theo loại dữ liệu, có phiên bản; không còn bốn loại pseudo-fake làm cấu trúc trung tâm. Thứ tự chạy nằm trong `src/data/01_...07_...` và [bảng lệnh](../src/README.md), không cần nhớ thứ tự qua tên thư mục dữ liệu.

Nhóm nguồn vẫn giữ trong cột tier (podcast, presentation, lecture hoặc nhóm do bạn chọn); không mất tính đa dạng khi media đặt chung thư mục. Chống leakage vẫn theo người/nguồn. Bản nén/fake kế thừa split của real.

Feature tính lại nằm ở `cache/`, weights ở `weights/`, kết quả train/eval ở `experiments/`. Media, hồ sơ riêng và manifest làm việc không commit. Hướng dẫn bắt đầu: [pipeline](../src/data/README.md).
