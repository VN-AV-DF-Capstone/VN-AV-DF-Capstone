# Model theo phiên bản

| Phiên bản | Thư mục | Thời gian | Nguồn (repo @ commit) | Nội dung chính |
|---|---|---|---|---|
| v001 | [v001/](v001/) | 16/09 - 10/10/2026 | `VN-AV-DF-Capstone/vn-av-consistency-deepfake` @ `4c239de` | Pseudo-fake FATE, encoder PE-AV / FATE adapter / AV-HuBERT / DINOv2, so sánh theo ô 2×2 |
| v002 | [v002/](v002/) | 11/10 - 14/10/2026 | `VN-AV-DF-Capstone/vn-av-consistency-deepfake` @ `1779279` | Bỏ FATE, thêm MechFake (pseudo-fake rẻ) và TriCue |
| v003 | [v003/](v003/) + [`src/vn_av_df/`](../vn_av_df/) | từ 18/10/2026 | `linhxm/vn-av-df-forensics` @ `b017a27` | Bản đang dùng: baseline BL-V `lipinc_v2`, BL-A `ssl_aasist`, BL-M `avh_align`, BL-M/sup `avh_align_sup`, BL-Ens; đề xuất M1 `trihead_v2`, M2 `avh_align_plus` + ablation ([docs/v003_model.md](../../docs/v003_model.md)) |
| v005 | [v005/](v005/) + [`src/vn_av_df/`](../vn_av_df/) | code 03/11/2026, chưa train | đề xuất của MN ([docs](../../docs/v005-v006_de-xuat-detector.md) §2) | M5 `trihead_v3` = M1 + P1-P4 (+ ablation cộng dồn `trihead_v3_p1`, `trihead_v3_p123`); P4, P6 trong `evaluate.py` |
| v006 | [v006/](v006/) + [`src/vn_av_df/`](../vn_av_df/) | code 03/11/2026, chưa train | đề xuất của MN ([docs](../../docs/v006_sau-review.md)) | M6 `visyl` theo âm tiết tiếng Việt (+ `visyl_no_phon`, `visyl_uniform_no_phon`, `visyl_shuffled_tags`), cache `syll` |
| v004 | [v004/](v004/) + [`src/vn_av_df/`](../vn_av_df/) | từ 03/11/2026 | `linhxm/vn-av-df-forensics` @ `36942ca` | Bản đang dùng: nhãn mismatch đo được, M3 `trihead_v4` (+ ablation `_no_prosody`, `_no_phi`, `_abs`, `_ref`, `_no_type`), BL-A2 `xlsr_frozen`, `speechforensics`; giữ M1, M2 và baseline v003 |

## v005, v006 (đã có code, chưa train)

Script ở `v005/`, `v006/` giống v003 nhưng gọi `bootstrap("v005" | "v006")` (`settings.VERSIONS`): đội hình riêng, run ở `runs/train_part1_v005|v006`, protocol của M1-v003 để so cùng điều kiện (nhãn mismatch theo cách tạo, dịch tiếng −1…+2, chọn theo val loss, lr cố định, XLS-R 3 nhóm lớp như cache v003). v006 cần thêm cache `syll`: `src/models/v006/01_prepare.py --encoders syll` (PhoWhisper trên tiếng từng mẫu).

- v005: `TriHeadV3` trong `models.py` (P1 `PureSyncTrunk`, P2 `VisualTrunk`, P3 mỗi đầu một nhánh, P4 bỏ FiLM). P4 (`postprocess_p4`) và P6 (`localization_p6`) tính trong `evaluate.py` cho **mọi** model, để M1-v003 chấm lại (`03_evaluate.py --force`) cùng hậu xử lý. Chưa có: P5 (front-end giọng, chờ zero-shot), cổng G1-G7 trong `report.py`.
- v006: `ViSyl` trong `models.py`, `features/syllables.py` (luật chính tả → thẻ, đơn vị âm tiết), pseudo-fake theo đơn vị trong `data.replace_audio`. Chưa có: bước 0 (0b' chọn bộ căn, 0c' pilot khép môi), hình học môi và kỳ vọng theo người, phép đảo thẻ lúc suy luận, cổng G1-G8.

## v004 (đang dùng)

Script ở `v004/`: như v003 và thêm `01a_syncnet.py` (đo SyncNet ở local, CPU) và `01b_sync_labels.py` (nhãn mismatch đo được). Code model ở `src/vn_av_df/training/` (`models.py` lớp `TriHeadV4`, `sync_labels.py`, `syncnet_measure.py`, `prosody.py`); cấu hình `settings.py` (`MODELS`, `ABLATIONS`, `TRAIN_GROUPS`, `TRAIN_SESSIONS`, `training.mismatch_labels`). Tài liệu thiết kế `RESEARCH_WORKLOG/v004_model.md` của Linh chưa có trên git.

Khác v003 cần biết: chọn checkpoint theo val AUC (v003: val loss), `audio_lag_aug` [−2, +5] (v003: [−1, +2]), XLS-R 5 nhóm lớp (cache `xlsr` của v003 phải trích lại), W&B project `vn-av-df-v004`.

## v003

Script `v003/` vẫn chạy được nhưng dùng `settings.py` hiện tại (cấu hình v004). Muốn chạy lại đúng v003 thì checkout commit trước khi gộp v004 (`2b3a70c`).

Script các bước nằm ở `v003/` (00 tải encoder, 01 trích feature, 02 train, 03 test, 04 báo cáo, 05 đo thời gian); code model nằm trong package chung `src/vn_av_df/training/`. Cách chạy: [PIPELINE.md](../../PIPELINE.md).

## v001, v002 (lưu để tham khảo)

- Là bản chụp repo cũ tại commit ghi ở bảng (chỉ khác ở `ruff format`), **bỏ `data_pipeline/`** vì phần dữ liệu real đã thay bằng `src/data/` (v003). Không bảo trì, không chạy trong test của repo này.
- Báo cáo kết quả: [v001](../../docs/reports/2026-10-15_V001_REPORT.md), [v002](../../docs/reports/2026-10-15_V002_REPORT.md). Kết quả v001 trong báo cáo chạy bằng nhánh `archive/v001-part1-results` của repo nguồn, không phải `4c239de`.
- Mỗi thư mục có `settings.py`, `src/vn_av_df/` và `tests/` riêng, trùng tên package với v003. Muốn chạy lại đúng như cũ thì clone repo nguồn và checkout commit ở bảng, không chạy lẫn với code v003.
