# Run của hướng lip-sync mới

Các run trên Wav2Lip ngày 09–11/10/2026; ý nghĩa và số liệu ở [báo cáo 11/10](../docs/reports/2026-10-11_WAV2LIP_SYNC_RESULTS.md).

| Run | Nội dung |
|---|---|
| `y_001/` | Detector phân loại; `cue_check/` cho thấy nó bắt vết hình của Wav2Lip, không dùng tiếng |
| `align_001/` | Model căn chỉnh tiếng–môi chỉ học từ clip thật; `evaluation_shift2/` là lần chấm lại với `--max-shift 2` |
| `align_002/` | Như `align_001` nhưng train với `--resync 5` |
| `av_offset/` | Độ lệch tiếng–hình của từng clip (`04_check_av_offset.py`) |
| `peak_sync_001/` | Điểm khớp tiếng–môi không cần train (`05_evaluate_peak_sync.py`) |

Git giữ các file CSV kết quả; `best.pt` chỉ nằm trên máy đã chạy (`*.pt` bị ignore). Không ghi đè run đã hoàn tất: chạy lại thì đặt tên mới.

Thí nghiệm AVSP-Net cũ đã chuyển vào docs/archives/legacy_avsp/experiments. Smoke test tổng hợp chỉ chứng minh plumbing; không được ghi thành kết quả accuracy.
