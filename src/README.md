# Vai trò các thư mục source

`src/data/` chứa toàn bộ luồng dữ liệu: file chạy có số, cấu hình và code xử lý. Đã gộp thư mục pipeline vào đây.

Để tải, mở `data/02_download.py` rồi Run. Ngoài cùng data chỉ có các file Python có số; code hỗ trợ và cấu hình nằm trong `data/preparation/`, gồm `settings.py` và nhánh quality. Không cần chạy lần lượt các file hỗ trợ.

| Thư mục | Vai trò | Hiện trạng |
|---|---|---|
| [data/](data/README.md) | Các bước chạy 01–07, settings và logic thu thập, tải, cắt, manifest, split, nén, quality | Có code |
| [tools/review/](tools/review/README.md) | Xem/nghe clip, phân công và gộp review | Có code và file chạy có số |
| [generators/](generators/README.md) | Tạo lip-sync fake | Wav2Lip/MuseTalk đã chạy các batch xem chất lượng trên V1; output còn chờ review, chưa nối vào dataset huấn luyện |
| [features/](features/README.md) | Chuẩn bị audio/vùng miệng, trích đặc trưng cho model | 01_extract: loader Base pre-fusion, preprocessing, cache cho training; đã chạy một clip thật với pretrained/landmark thật. 02_fuse: thêm Transformer hợp nhất lên cache có sẵn |
| [models/](models/README.md) | Detector nhận audio/visual features | Có detector.py, alignment.py (căn chỉnh tiếng–môi) và 01_smoke_test.py |
| [training/](training/README.md) | Học nhãn và consistency cho X/Y | Loader paired features, train/validation theo epoch, best checkpoint; 02_train_alignment chỉ học trên real; kiểm chứng giả lập |
| [evaluation/](evaluation/README.md) | Dự đoán và đánh giá | 01_evaluate: AUC/EER/accuracy theo split và mức nén; 02_check_cues: bỏ modality và ghép tiếng lệch; 03_evaluate_alignment cho model căn chỉnh; 04_check_av_offset: độ lệch tiếng–hình từng clip; 05_evaluate_peak_sync: điểm không cần train; check_shortcuts.py |

Luồng: dữ liệu đã chia split → tạo fake → tạo các bản nén → chuẩn bị audio/visual → train detector → đánh giá. Các bước đã nối với nhau bằng đường dẫn mặc định (bảng dưới) và mới chạy thử bằng media tổng hợp; chưa chạy trên dữ liệu thật, chưa có fake đã duyệt hay kết quả detector.

## Luồng fake → train → kết quả (V1 + part1)

Chạy từ root repo bằng `vn_av_df`, theo thứ tự. Bước 2–4 chạy lâu và in tiến độ; nếu bị ngắt, chạy lại đúng lệnh đó để làm tiếp, phần đã xong được giữ. Muốn lưu log, thêm `2>&1 | Tee-Object -FilePath <ten>.log`.

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1 | `python src/data/07_add_part1.py` | `data/manifests/dataset_v1/real_splits_multi_001.csv`: V1 vào train, người của part1 chia train/val/test |
| 2 | `python src/generators/00_check_setup.py`, rồi `01_prepare.py`, `02_generate.py` | Cặp real/fake 2 giây cho cả ba split, tối đa `MAX_PER_SPLIT` clip mỗi split |
| 3 | `python src/data/06_compress.py` | Mỗi clip nén CRF 23 và 32, kèm `media_pairs.csv` |
| 4 | `python src/features/01_extract.py` | Feature AV-HuBERT và `pairs.csv` ở `cache/features/dataset_v1/wav2lip_multi_001_crf23_32/` |
| 5 | `python src/training/01_train.py --manifest cache/features/dataset_v1/wav2lip_multi_001_crf23_32/pairs.csv --out experiments/y_001 --epochs 30` | Baseline Y; `best.pt` chọn theo loss val |
| 6 | `python src/evaluation/01_evaluate.py --run experiments/y_001` | AUC, EER, accuracy theo split và mức nén; ngưỡng chọn trên val |
| 7 | `python src/evaluation/02_check_cues.py --run experiments/y_001` | Train lại chỉ hình/chỉ tiếng và chấm hình real ghép tiếng real khác cùng người: model bắt vết generator hay độ khớp môi |
| 8 | `python src/features/02_fuse.py` | Đặc trưng AV-HuBERT đầy đủ (chỉ tiếng, chỉ hình) ở `cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused/`; in kiểm tra độ khớp theo độ lệch frame |
| 9 | `python src/training/02_train_alignment.py --manifest cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused/pairs.csv --out experiments/align_001` | Model căn chỉnh tiếng–môi, chỉ học từ clip real |
| 10 | `python src/evaluation/03_evaluate_alignment.py --run experiments/align_001` | AUC/EER real so với fake Wav2Lip, và hình real ghép tiếng real khác |
| 11 | `python src/evaluation/05_evaluate_peak_sync.py --out experiments/peak_sync_001` | Cùng bảng như bước 10 cho điểm không cần train (độ khớp tốt nhất trong ±15 frame) |

Bước 8–11 không cho model thấy fake khi train, nên model không thể học vết hình riêng của Wav2Lip. Fake chỉ dùng để chọn ngưỡng trên val và đánh giá.

X dùng cùng lệnh bước 5 với `--consistency-weight` lớn hơn 0 và `--out` mới. Kết quả bước 6 là kết quả sơ bộ: fake chưa review bằng mắt, val chỉ có 3 người và test 4 người. Để chạy pipeline data V2, đặt `DATASET_VERSION = 'dataset_v2'` trong `data/preparation/settings.py`; generator có cấu hình V1 riêng.

Video ở `data/`, đặc trưng tính sẵn ở `cache/`, weights tải về ở `weights/`, đầu ra train/eval ở `experiments/`. Các đường dẫn này nằm ở root repo, không nằm trong `src/`.
