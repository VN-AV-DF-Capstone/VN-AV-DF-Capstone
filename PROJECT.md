# PROJECT.md — trạng thái hiện hành

Cập nhật 27/10/2026. Repo chính chuyển sang code **v003** thống nhất của nhóm (nguồn: `linhxm/vn-av-df-forensics` @ `b017a27`, đã gồm các bản sửa khi sinh fake trên Kaggle). Code trước đó giữ ở tag `pre-v003`.

## Phiên bản

| Phiên bản | Thời gian | Nội dung | Ở đâu |
|---|---|---|---|
| pre-v003 | 09/2026 - 26/10/2026 | dataset_v1 + part1 của người 2, Wav2Lip, AV-HuBERT Base, y_001 / align_002 / peak_sync_001 | tag `pre-v003`; kết quả ở [báo cáo 11/10](docs/reports/2026-10-11_WAV2LIP_SYNC_RESULTS.md), `experiments/` |
| v001 | 16/09 - 10/10/2026 | Pseudo-fake FATE, so sánh ô 2×2 | `src/models/v001/` |
| v002 | 11/10 - 14/10/2026 | MechFake + TriCue | `src/models/v002/` |
| v003 | từ 18/10/2026 | Dữ liệu real mới, sáu ô, 6 generator + 7 nguồn giọng AI, baseline + M1 / M2 | `src/data/`, `src/generators/`, `src/models/v003/`, `src/vn_av_df/` |

Model của pre-v003 (y_001, align_002, peak_sync_001) không đưa sang v003: baseline v003 đã có detector cùng loại (BL-M `avh_align`, BL-M/sup `avh_align_sup`) và điểm khớp không cần train (`speechforensics`, dùng trong demo khi chưa có checkpoint).

## Bảng tiến trình v003

| Hạng mục | Trạng thái |
|---|---|
| Dữ liệu real (`src/data/`) | Part1: 800 clip, 40 người (20 nam + 20 nữ), train 480 / validation 120 / test 200, xuất 20/10/2026. Theo [docs/v003_data.md](docs/v003_data.md) (22/10): 3 part = 120 người × 20 clip; mới sinh fake cho part1. |
| Sinh fake (`src/generators/`) | Part1 xong 25/10/2026: 7.622 mẫu. 56/1.800 job lip-sync bị bỏ: 39 không thấy mặt, 17 lỗi generator chưa chạy lại. Code hiện tại đã có đường dự phòng cho các lỗi đó (Diff2Lip batch nhỏ hơn, S3FD cho VideoReTalking, crop dự phòng cho IP-LAP), chưa chạy lại. |
| Trích feature, train, test (`src/models/v003/`) | Code có; chưa có kết quả lưu trong repo. |
| Demo (`src/vn_av_df/demo/`, `notebooks/demo.ipynb`) | Code có; cần checkpoint đã train. |

## Quy tắc giữ nguyên

- `src/vn_av_df/generation/lipsync.py` và các worker generator được băm vào provenance (`adapter_sha256`); `src/vn_av_df/training/*.py` được băm vào run. Sửa nội dung thì bản ghi / run cũ không còn khớp.
- `src/data/preparation/vn_av_data/data/cut.py`, `jumpcut.py`, `vad.py`, `features/face.py`, `features/syncnet.py` và luật cắt trong `configs/data.yaml` thuộc cut signature: đổi thì bước 03 chặn clip đã cắt. `cut.py` không chạy `ruff format` vì lý do này.
- Thư mục Output trên Kaggle giữ tên `vn-av-df-forensics/` và đường dẫn `generation/plans`, `cache/`, `runs/` như cũ, để Output các phiên trước dùng tiếp được. Part đã chốt nay ở `data/vn-av-df-data-partN`; khi gộp Output cũ, code tự tìm thêm ở `datasets/vn-av-df-data/`.
- Notebook clone nhánh mặc định của repo này: thay đổi chỉ tới Kaggle sau khi đã merge vào `main`.

## Việc kế tiếp

1. Chạy lại 17 job lip-sync lỗi của part1 (chờ quyết định của nhóm).
2. Trích feature và train baseline + M1 / M2 trên part1, test một lần.
3. Sinh fake cho part2, part3.
