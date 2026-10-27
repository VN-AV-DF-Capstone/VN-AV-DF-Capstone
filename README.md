# VN-AV-DF-Capstone

Benchmark và detector **lip-sync deepfake trong video tiếng Việt**: người nói thật, sáu ô kết hợp hình / tiếng, generator và giọng AI chưa thấy chỉ xuất hiện ở test. Từ 27/10/2026 repo chính dùng code **v003** thống nhất của nhóm; code trước đó nằm ở tag `pre-v003`.

## Thiết kế sáu ô

| | Tiếng gốc | Tiếng thật bị ghép | Giọng AI |
|---|---|---|---|
| **Video thật** | ① real | ② sham (cả câu + một đoạn) | ③ giọng AI: track A (cả câu), B (một cụm, ghép vào); test thêm C (inpainting) |
| **Video đã lip-sync** | ④ source | ⑤ donor | ⑥ cả hai (môi theo track B) |

- **Real:** mỗi part 600-800 clip (20 nam + 20 nữ × 15-20 clip, 4-7 giây, 720p, đã sửa lệch tiếng-hình), chia train / validation / test theo người.
- **Generator:** Wav2Lip, MuseTalk 1.5, IP-LAP, VideoReTalking cho cả ba split; LatentSync 1.5 và Diff2Lip **chỉ ở test**.
- **Giọng AI:** VieNeu-TTS, F5-TTS tiếng Việt, OmniVoice, Seed-VC cho cả ba split; VoxCPM2, viXTTS và F5 inpainting **chỉ ở test**.
- **Detector:** baseline BL-V `lipinc_v2`, BL-A `ssl_aasist`, BL-M `avh_align`, BL-M/sup `avh_align_sup`, BL-Ens; đề xuất M1 `trihead_v2`, M2 `avh_align_plus` và ablation. Chi tiết: [docs/v003_model.md](docs/v003_model.md).

## Trạng thái (27/10/2026)

- **Real part1:** 800 clip, 40 người (train 480 / validation 120 / test 200), xuất 20/10/2026.
- **Fake part1:** 7.622 mẫu, sinh trên Kaggle và máy local, chốt 25/10/2026. 56/1.800 job lip-sync bị bỏ: 39 không thấy mặt, 17 lỗi generator chưa chạy lại. Báo cáo: `reports/2026-10-25_bao-cao-fake-v003.docx` trong thư mục chung của project.
- **Train / test:** chưa có kết quả lưu trong repo này.
- Test tự động dùng media tổng hợp, generator và ASR giả; qua test không chứng minh detector chính xác.

## Bắt đầu

Môi trường conda `vn-av-df` (Python 3.11), chạy mọi lệnh từ thư mục gốc repo:

```powershell
conda activate vn-av-df
python 00_install.py    # cài src/data/preparation + package vn_av_df (PROFILE trong file)
python -m pytest        # tests/ + tests/data
```

- Dữ liệu real (URL YouTube → part sạch): [src/data/README.md](src/data/README.md), từng lệnh ở [src/data/RUNBOOK.md](src/data/RUNBOOK.md).
- Sinh fake, trích feature, train, test, báo cáo (local hoặc Kaggle): [PIPELINE.md](PIPELINE.md).
- Môi trường worker (generator, TTS, AV-HuBERT): [environments/README.md](environments/README.md).

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/data/` | Bước 00-10 của dữ liệu real (plan, tải, cắt, nhận người, duyệt, chia part, export, archive, dọn, trạng thái); code hỗ trợ ở `preparation/` (package `vn_av_data`) |
| `src/generators/` | Bước 00-04 sinh fake: setup, plan, phiên giọng, phiên lip-sync, finalize |
| `src/models/` | `v003/` bước 00-05 (encoder, feature, train, test, báo cáo, đo thời gian); `v001/`, `v002/` là code cũ lưu để tham khảo ([README](src/models/README.md)) |
| `src/vn_av_df/` | Package chung: `generation/` (ô, plan, giọng, adapter generator), `training/` (dữ liệu, model, train, test, báo cáo), `features/`, `demo/`, `tracking.py` (W&B) |
| `configs/` | `data.yaml` (luật cắt, part, model nhận diện) và mẫu `urls.example.csv` |
| `notebooks/` | `generate`, `prepare`, `train`, `demo` cho Kaggle, sinh từ `tools/make_notebooks.py`; `LA_draft/` là bản nháp riêng |
| `environments/`, `tools/` | Môi trường worker; công cụ Kaggle và sinh notebook |
| `tests/` | Test end-to-end và từng phần trên dữ liệu tổng hợp; `tests/data/` cho dữ liệu real |
| `docs/` | Thiết kế v003 (`v003_data.md`, `v003_generation.md`, `v003_model.md`), báo cáo và tài liệu cũ ([chỉ mục](docs/README.md)) |
| `data/`, `datasets/`, `cache/`, `runs/`, `weights/`, `checkpoints/` | Dữ liệu, feature, kết quả train và weight ở máy; không commit |
| `experiments/` | CSV kết quả trước v003 (Wav2Lip, 09-11/10/2026) |

## Trước v003

Pipeline cũ (dataset_v1, Wav2Lip, AV-HuBERT Base, y_001, align_002, peak_sync_001) nằm ở tag `pre-v003`. Kết quả của nó trên Wav2Lip ở [báo cáo 11/10](docs/reports/2026-10-11_WAV2LIP_SYNC_RESULTS.md) và `experiments/`; đó là kết quả sơ bộ trên một generator, không áp dụng cho dữ liệu v003.

## Dữ liệu và đạo đức

Chỉ dùng video của nhóm hoặc đã được phép. Video gốc và mẫu fake của người thật **không phân phối lại**, chỉ dùng cho nghiên cứu. Không commit media, weight, `.env`, cookie hay dữ liệu riêng tư.
