# Chỉ mục tài liệu

## Hiện hành (v003)

- [PROJECT](../PROJECT.md): trạng thái, phiên bản và việc tiếp theo.
- [PIPELINE](../PIPELINE.md): cách chạy sinh fake, trích feature, train, test (local và Kaggle).
- [Dữ liệu real](../src/data/README.md) và [sổ tay chạy từng lệnh](../src/data/RUNBOOK.md).
- [v003_data](v003_data.md): pipeline dữ liệu real, số người / clip, độ lệch tiếng-hình.
- [v003_generation](v003_generation.md): sáu ô, generator và giọng AI, số mẫu mỗi part.
- [v003_model](v003_model.md): phương pháp, baseline, model đề xuất, cách đánh giá.
- [Model theo phiên bản](../src/models/README.md): v001, v002, v003.

Ba file `v003_*.md` giữ nguyên bản của nhóm. Đường dẫn trong đó theo repo `vn-av-df-forensics`: `data_pipeline/` nay là `src/data/` (workspace `data/v003/`, export `data/exports/`), `generation/0x` là `src/generators/0x`, `training/0x` là `src/models/v003/0x`.

## Trước v003

Code ở tag `pre-v003`. Các tài liệu dưới đây mô tả hướng X/Y, `dataset_v1` / `dataset_v2` và run Wav2Lip; lệnh và đường dẫn trong đó không còn chạy trên nhánh hiện tại.

- [Hướng nghiên cứu](research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md), [Data protocol](research/DATA_PROTOCOL.md), [Kế hoạch pilot](planning/KE_HOACH_PILOT_LIP_SYNC_VI.md), [Phân công](planning/TEAM_PLAN.md).
- [Báo cáo chuyển đổi 13/09/2026](reports/2026-09-13_REPO_RESET.md), [Kiểm kê dọn workspace 23/09/2026](reports/2026-09-23_WORKSPACE_CLEANUP_AUDIT.md).
- [Kết quả Wav2Lip 11/10/2026](reports/2026-10-11_WAV2LIP_SYNC_RESULTS.md): detector, model căn chỉnh và điểm khớp tiếng–môi trên dữ liệu thật; số liệu ở [experiments/](../experiments/README.md).

## Lịch sử

[AVSP-Net/pseudo-fake](archives/legacy_avsp/README.md) chứa báo cáo cũ và bằng chứng gốc. Không dùng các con số, nhãn, URL hoặc lệnh trong archive như trạng thái hiện hành.
