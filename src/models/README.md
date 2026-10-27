# Model theo phiên bản

| Phiên bản | Thư mục | Thời gian | Nguồn (repo @ commit) | Nội dung chính |
|---|---|---|---|---|
| v001 | [v001/](v001/) | 16/09 - 10/10/2026 | `VN-AV-DF-Capstone/vn-av-consistency-deepfake` @ `4c239de` | Pseudo-fake FATE, encoder PE-AV / FATE adapter / AV-HuBERT / DINOv2, so sánh theo ô 2×2 |
| v002 | [v002/](v002/) | 11/10 - 14/10/2026 | `VN-AV-DF-Capstone/vn-av-consistency-deepfake` @ `1779279` | Bỏ FATE, thêm MechFake (pseudo-fake rẻ) và TriCue |
| v003 | [v003/](v003/) + [`src/vn_av_df/`](../vn_av_df/) | từ 18/10/2026 | `linhxm/vn-av-df-forensics` @ `b017a27` | Bản đang dùng: baseline BL-V `lipinc_v2`, BL-A `ssl_aasist`, BL-M `avh_align`, BL-M/sup `avh_align_sup`, BL-Ens; đề xuất M1 `trihead_v2`, M2 `avh_align_plus` + ablation ([docs/v003_model.md](../../docs/v003_model.md)) |

## v003 (đang dùng)

Script các bước nằm ở `v003/` (00 tải encoder, 01 trích feature, 02 train, 03 test, 04 báo cáo, 05 đo thời gian); code model nằm trong package chung `src/vn_av_df/training/`. Cách chạy: [PIPELINE.md](../../PIPELINE.md).

## v001, v002 (lưu để tham khảo)

- Là bản chụp repo cũ tại commit ghi ở bảng (chỉ khác ở `ruff format`), **bỏ `data_pipeline/`** vì phần dữ liệu real đã thay bằng `src/data/` (v003). Không bảo trì, không chạy trong test của repo này.
- Báo cáo kết quả: [v001](../../docs/reports/2026-10-15_V001_REPORT.md), [v002](../../docs/reports/2026-10-15_V002_REPORT.md). Kết quả v001 trong báo cáo chạy bằng nhánh `archive/v001-part1-results` của repo nguồn, không phải `4c239de`.
- Mỗi thư mục có `settings.py`, `src/vn_av_df/` và `tests/` riêng, trùng tên package với v003. Muốn chạy lại đúng như cũ thì clone repo nguồn và checkout commit ở bảng, không chạy lẫn với code v003.
