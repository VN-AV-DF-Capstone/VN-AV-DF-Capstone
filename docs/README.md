# Chỉ mục tài liệu

## Hiện hành (v003)

- [PROJECT](../PROJECT.md): trạng thái, phiên bản và việc tiếp theo.
- [PIPELINE](../PIPELINE.md): cách chạy sinh fake, trích feature, train, test (local và Kaggle).
- [Dữ liệu real](../src/data/README.md) và [sổ tay chạy từng lệnh](../src/data/RUNBOOK.md).
- [v003_data](v003_data.md): pipeline dữ liệu real, số người / clip, độ lệch tiếng-hình.
- [v003_generation](v003_generation.md): sáu ô, generator và giọng AI, số mẫu mỗi part.
- [v003_model](v003_model.md): phương pháp, baseline, model đề xuất, cách đánh giá.
- [Model theo phiên bản](../src/models/README.md): v001-v008.
- Báo cáo của Linh: [v004](v004_report.md) (M3, nhãn mismatch đo được) và [v005](v005_report.md) (M3-R, video nén; v006 của Linh = cùng đội hình, seed 43, đang train), bảng sinh tự động ở `v004_results/`, `v005_results/`.
- Đề xuất detector [v007 · M6 và v008 · M7](v007-v008_de-xuat-detector.md) và [v008 sau review](v008_sau-review.md) (của MN; chưa train; trước 12/11 là v006 và v007; trước 11/11 là v005 · M5 và v006 · M6).
- Báo cáo kết quả [v001](reports/2026-10-15_V001_REPORT.md) (FATE, part1) và [v002](reports/2026-10-15_V002_REPORT.md) (MechFake + TriCue), tiếng Anh, của Linh.

Ba file `v003_*.md` giữ nguyên bản của nhóm. Đường dẫn trong đó theo repo `vn-av-df-forensics`: `data_pipeline/` nay là `src/data/` (workspace `data/v003/`, export `data/exports/`), `generation/0x` là `src/generators/0x`, `training/0x` là `src/models/v003/0x`.

## Trước v003

Code ở tag `pre-v003`. Các tài liệu dưới đây mô tả hướng X/Y, `dataset_v1` / `dataset_v2` và run Wav2Lip; lệnh và đường dẫn trong đó không còn chạy trên nhánh hiện tại.

- [Hướng nghiên cứu](research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md), [Data protocol](research/DATA_PROTOCOL.md), [Kế hoạch pilot](planning/KE_HOACH_PILOT_LIP_SYNC_VI.md), [Phân công](planning/TEAM_PLAN.md).
- [Kết quả Wav2Lip 11/10/2026](reports/2026-10-11_WAV2LIP_SYNC_RESULTS.md): detector, model căn chỉnh và điểm khớp tiếng–môi trên dữ liệu thật; số liệu ở [experiments/](../experiments/README.md).

## Lịch sử

[AVSP-Net/pseudo-fake](archives/legacy_avsp/README.md) chứa báo cáo cũ và bằng chứng gốc. Không dùng các con số, nhãn, URL hoặc lệnh trong archive như trạng thái hiện hành.
