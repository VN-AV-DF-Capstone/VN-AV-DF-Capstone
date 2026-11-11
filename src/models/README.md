# Model theo phiên bản

| Phiên bản | Thư mục | Thời gian | Nguồn (repo @ commit) | Nội dung chính |
|---|---|---|---|---|
| v001 | [v001/](v001/) | 16/09 - 10/10/2026 | `VN-AV-DF-Capstone/vn-av-consistency-deepfake` @ `4c239de` | Pseudo-fake FATE, encoder PE-AV / FATE adapter / AV-HuBERT / DINOv2, so sánh theo ô 2×2 |
| v002 | [v002/](v002/) | 11/10 - 14/10/2026 | `VN-AV-DF-Capstone/vn-av-consistency-deepfake` @ `1779279` | Bỏ FATE, thêm MechFake (pseudo-fake rẻ) và TriCue |
| v003 | [v003/](v003/) + [`src/vn_av_df/`](../vn_av_df/) | từ 18/10/2026 | `linhxm/vn-av-df-forensics` @ `b017a27` | Bản đang dùng: baseline BL-V `lipinc_v2`, BL-A `ssl_aasist`, BL-M `avh_align`, BL-M/sup `avh_align_sup`, BL-Ens; đề xuất M1 `trihead_v2`, M2 `avh_align_plus` + ablation ([docs/v003_model.md](../../docs/v003_model.md)) |
| v004 | [v004/](v004/) + [`src/vn_av_df/`](../vn_av_df/) | từ 03/11/2026 | `linhxm/vn-av-df-forensics` @ `36942ca` | Nhãn mismatch đo được, M3 `trihead_v4` (+ ablation `_no_prosody`, `_no_phi`, `_abs`, `_ref`, `_no_type`), BL-A2 `xlsr_frozen`, `speechforensics`; giữ M1, M2 và baseline v003 |
| v005 | [v005/](v005/) + [`src/vn_av_df/`](../vn_av_df/) | từ 11/11/2026 | `linhxm/vn-av-df-forensics` @ `a1d0dd2` | Bản đang dùng: M3-R `trihead_v5` (M3 gọn, học thêm bản nén 480p / 360p theo cặp + loss nhất quán, sham là mẫu âm, cổng độ phân giải) (+ ablation `_no_pair`, `_quality`, `_clean`); đã train trên Kaggle, kết quả ở [docs/v005_report.md](../../docs/v005_report.md) |
| v006 | [v006/](v006/) + [`src/vn_av_df/`](../vn_av_df/) | đổi 11/11/2026, chưa train | đề xuất của MN ([docs](../../docs/v006-v007_de-xuat-detector.md) §0b) | M6 = nền M3-R `trihead_v5` + P4, P6 (`evaluate.py`, mọi model) + P5 (chưa làm). `trihead_v3` (M1 + P1-P4, đề xuất cũ) còn trong `models.SPECS`, không trong đội hình |
| v007 | [v007/](v007/) + [`src/vn_av_df/`](../vn_av_df/) | code 03/11/2026, chưa train | đề xuất của MN ([docs](../../docs/v007_sau-review.md)) | M7 `visyl` theo âm tiết tiếng Việt (+ `visyl_no_phon`, `visyl_uniform_no_phon`, `visyl_shuffled_tags`), cache `syll` |

## v006, v007 (đề xuất của MN, chưa train)

Script ở `v006/`, `v007/` giống v005 nhưng gọi `bootstrap("v006" | "v007")` (`settings.VERSIONS`): đội hình, seed và thư mục run riêng (`runs/train_all_v006|v007`); cách train giữ như v005 của Linh (nhãn mismatch đo được, dịch tiếng −2…+5, học thêm bản nén c480 / c360, chọn checkpoint theo val AUC trung bình bản sạch và bản nén). Trước 11/11 hai bản này tên là v005 · M5 và v006 · M6.

- v006 · M6 (MN chốt 11/11): nền M3-R `trihead_v5` + P4 (`postprocess_p4`), P6 (`localization_p6`) trong `evaluate.py`, tính cho **mọi** model; P5 (front-end giọng) chưa có encoder. Nấc 0 không cần train: chép run `trihead_v5_seed42` của v005 vào `runs/train_all_v006/` rồi chạy `v006/03_evaluate.py`. Lý do bỏ P1-P3: bảng §0b của [đề xuất](../../docs/v006-v007_de-xuat-detector.md).
- v007 · M7: `ViSyl` trong `models.py` (đường gộp, tức đầu visual, chỉ thấy hồ sơ khớp đã dời về độ trễ chung của clip; độ trễ chung chỉ vào đầu mismatch; như M3-R: sham là mẫu âm của đầu visual, tắt ngẫu nhiên nhóm lớp XLS-R), `features/syllables.py` (luật chính tả → thẻ khẩu hình, PhoWhisper DTW). Cần thêm cache `syll`: `v007/01_prepare.py --encoders syll`; bản nén c480 / c360 cũng trích `syll` (ASR trên tiếng đã nén).

## v005 (đang dùng)

Script ở `v005/`: như v004 (cả `01a_syncnet.py`, `01b_sync_labels.py`). Code model ở `src/vn_av_df/training/` (`models.py` lớp `TriHeadV5`, `stage.py` chép cache sang đĩa local trên Kaggle; biến thể nén trong `prepare.py`, `data.py`, `train.py`, `evaluate.py`). Biến thể ở `settings.VARIANTS`: c480, c360 (train + test), pr, w360, c288 (chỉ đánh giá). Kết quả: [docs/v005_report.md](../../docs/v005_report.md), [docs/v004_report.md](../../docs/v004_report.md).

## v004

Script `v004/` vẫn chạy được nhưng dùng `settings.py` hiện tại (cấu hình v005). Muốn chạy lại đúng v004 thì checkout `f28a57b`. Khác v003 cần biết: chọn checkpoint theo val AUC (v003: val loss), `audio_lag_aug` [−2, +5] (v003: [−1, +2]), XLS-R 5 nhóm lớp (cache `xlsr` của v003 phải trích lại).

## v003

Script `v003/` vẫn chạy được nhưng dùng `settings.py` hiện tại (cấu hình v005). Muốn chạy lại đúng v003 thì checkout commit trước khi gộp v004 (`2b3a70c`).

Script các bước nằm ở `v003/` (00 tải encoder, 01 trích feature, 02 train, 03 test, 04 báo cáo, 05 đo thời gian); code model nằm trong package chung `src/vn_av_df/training/`. Cách chạy: [PIPELINE.md](../../PIPELINE.md).

## v001, v002 (lưu để tham khảo)

- Là bản chụp repo cũ tại commit ghi ở bảng (chỉ khác ở `ruff format`), **bỏ `data_pipeline/`** vì phần dữ liệu real đã thay bằng `src/data/` (v003). Không bảo trì, không chạy trong test của repo này.
- Báo cáo kết quả: [v001](../../docs/reports/2026-10-15_V001_REPORT.md), [v002](../../docs/reports/2026-10-15_V002_REPORT.md). Kết quả v001 trong báo cáo chạy bằng nhánh `archive/v001-part1-results` của repo nguồn, không phải `4c239de`.
- Mỗi thư mục có `settings.py`, `src/vn_av_df/` và `tests/` riêng, trùng tên package với v003. Muốn chạy lại đúng như cũ thì clone repo nguồn và checkout commit ở bảng, không chạy lẫn với code v003.
