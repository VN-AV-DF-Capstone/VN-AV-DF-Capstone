# Training — huấn luyện detector

Trạng thái: có train_step, loader cache feature, padding batch, vòng train/validation
và CLI `01_train.py` lưu best.pt cùng epochs.csv. Đã chạy trên feature thật của
`wav2lip_multi_001`: `experiments/y_001` (Y), `align_001` và `align_002` (model căn chỉnh).
X chưa chạy. CLI chạy CPU.

CSV đầu vào: `sample_id,label,split,feature_path,paired_feature_path`. Path tương đối
với CSV; split dùng train/val/test; real=0, fake=1. sample_id phải chỉ rõ clip
(phân biệt bản real và bản fake) và cửa sổ thời gian. Mỗi cache .pt chứa audio
`[T,Da]`, visual `[T,Dv]`, sample_id, label, split, view_id, extractor_id, timeline_id.
extractor_id phải ghi nhận hash checkpoint, mode extraction và preprocessing;
timeline_id ghi nhận cùng timestamps. Hai view phải khác view_id và cùng metadata
còn lại. Đây là kiểm tra metadata, không tự chứng minh data không có leakage.

Chạy từ root repo SAU KHI có split và feature thật:

`src/features/01_extract.py` tạo `pairs.csv` theo đúng schema này. Hướng dẫn chuẩn bị checkpoint/media ở [features](../features/README.md). Các đường dẫn ví dụ dưới đây dùng output mặc định của 01_extract.

```powershell
python src/training/01_train.py --manifest cache/features/dataset_v1/wav2lip_multi_001_crf23_32/pairs.csv --out experiments/y_001 --epochs 30 --consistency-weight 0
python src/training/01_train.py --manifest cache/features/dataset_v1/wav2lip_multi_001_crf23_32/pairs.csv --out experiments/x_001 --epochs 30 --consistency-weight 1
```

Sau đó chấm điểm bằng `python src/evaluation/01_evaluate.py --run experiments/y_001`.

Trọng số 1 chỉ là ví dụ. Cùng seed/dữ liệu/batch/epoch cho X và Y. Chương trình
không đọc feature test; chọn checkpoint bằng supervised loss validation cho cả
hai phương pháp. Cần kiểm tra speaker/source-disjoint bằng pipeline data trước;
loader không nhận diện cùng người từ sample_id. Không dùng reviewed_clips.csv
trực tiếp: đó chưa phải manifest real/fake với feature và split. Chạy --help để xem
tham số; không bấm Run trống rồi kỳ vọng chương trình tự tìm dữ liệu.

Y dùng `consistency_weight=0`; X dùng trọng số lớn hơn 0. Cả hai đều học nhãn real=0/fake=1 trên cả hai view. X cộng thêm lỗi chênh lệch score giữa hai view. Trọng số X phải chọn bằng validation, không lấy giá trị trong smoke làm cấu hình nghiên cứu.

Batch có audio, visual, paired_audio, paired_visual, lengths, labels, sample_ids và paired_sample_ids. Mỗi ID phải xác định cùng clip và cửa sổ; cặp chỉ khác mức nén, không phải real ghép với fake. Feature phải được trích thật từ từng view bằng cùng cấu hình. Hàm kiểm tra ID/hình dạng, nhưng không thể xác minh nguồn gốc media chỉ từ tensor.

Chứa cách đọc manifest/đặc trưng thành batch, vòng lặp train, cách tính lỗi, cập nhật model, validation và lưu checkpoint. Phần khuyến khích dự đoán nhất quán giữa các bản nén của phương pháp X nằm trong `steps.py`.

Nhận split đã khóa từ luồng data; không tự chia ngẫu nhiên lại clip. X/Y dùng cùng dữ liệu, bản nén và ngân sách train. Chỉ chọn cấu hình/ngưỡng trên validation, không dùng final test để chọn model.

Model được import từ `src/models/`. Log, cấu hình, kết quả validation và checkpoint ghi vào `experiments/<run_id>/`, không vào source. File chạy có số: `01_train.py` (Y/X) và `02_train_alignment.py` (model căn chỉnh).

`02_train_alignment.py` train model căn chỉnh tiếng–môi (`src/models/alignment.py`) **chỉ trên clip real**, theo ý tưởng AVH-Align. Dòng fake trong manifest không được đọc khi train; epoch được giữ là epoch có độ lệch thấp nhất trên clip real của val. Dùng với đặc trưng từ `src/features/02_fuse.py`:

```powershell
python src/training/02_train_alignment.py --manifest cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused/pairs.csv --out experiments/align_001
```

Thêm `--resync R` để trước khi train, mỗi clip được cắt cho tiếng và hình khớp nhau ở độ lệch tốt nhất trong ±R frame, ước lượng từ đặc trưng thô (`src/features/sync.py`), cùng một quy tắc cho real và fake ở mọi split. Video real thu ở nhiều nơi lệch nhau vài frame cố định; không cắt thì model có thể học một độ lệch quen thuộc làm dấu hiệu của real. `03_evaluate_alignment.py` tự đọc R từ checkpoint và cắt y như vậy, cả với cặp hình–tiếng ghép lệch:

```powershell
python src/training/02_train_alignment.py --manifest cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused/pairs.csv --out experiments/align_002 --resync 5
```

`epochs.csv` ghi độ lệch trung bình của train và val. Khi tiếng và môi không có thông tin thời gian chung, giá trị khoảng log(2K+1), khoảng 3,04 với K=10 mặc định; càng thấp càng khớp.
