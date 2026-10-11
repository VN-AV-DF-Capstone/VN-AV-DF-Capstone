# Kết quả phát hiện lip-sync trên Wav2Lip (09–11/10/2026)

Báo cáo của phần người 1 (AV-HuBERT + fake Wav2Lip). Mọi số liệu dưới đây chạy trên máy RTX 4050 của nhóm với dữ liệu thật, trừ khi ghi là kiểm thử giả lập. Fake **chưa được review bằng mắt**, val chỉ có 3 người và test 4 người, nên đây là kết quả sơ bộ.

## Tóm tắt

- Detector phân loại thông thường (`y_001`) đạt test AUC 1,000 nhưng thực chất bắt vết mờ của Wav2Lip, không dùng tiếng: chỉ nhìn hình vẫn đạt 1,000, còn hình thật ghép tiếng sai thì không nhận ra (AUC 0,50).
- Chuyển sang đo độ khớp tiếng–môi bằng AV-HuBERT đầy đủ, không cho model thấy fake khi train. Lần đầu (`align_001`) hỏng trên test (AUC 0,416) vì mỗi video thật có một độ lệch tiếng–hình cố định khác nhau, còn mọi fake Wav2Lip lệch đúng +2 frame; model đã học "lệch +1 là thật".
- Sau khi căn lại từng clip về độ lệch tốt nhất của chính nó, `align_002` đạt test AUC 0,793. Một điểm không cần train (`peak_sync_001`) còn tốt hơn: val 0,890, test 0,875. Cả hai bắt được 100% hình thật ghép tiếng của clip khác, tức là chúng thật sự đo độ khớp tiếng–môi.
- Chọn theo val (cách hợp lệ duy nhất), `peak_sync_001` là phương pháp chính hiện tại.
- Câu hỏi còn mở: fake có điểm khớp thấp hơn có thể một phần do miệng bị mờ chứ không chỉ do sai khẩu hình. Chỉ fake từ generator khác (MuseTalk) mới trả lời được.

## Dữ liệu

| Bước | Kết quả |
|---|---|
| Split | `data/manifests/dataset_v1/real_splits_multi_001.csv`: V1 (spk_001) vào train; 10 người của part1 (575 clip đã duyệt) chia theo người: train speaker_03/06/09, val speaker_01/07/10, test speaker_02/05/08/11. Hai người chung kênh nằm cùng split. |
| Fake | Wav2Lip GAN (bản upstream, chưa fine-tune), run `wav2lip_multi_001`: 946/950 cặp real/fake 2 giây (4 clip train không tìm thấy mặt), khoảng 28 giây mỗi cặp. Tiếng thay vào lấy từ clip khác của cùng người, cùng split, không trùng thời gian. |
| Nén | Mỗi clip có hai bản CRF 23 và CRF 32, không lỗi. |
| Feature | AV-HuBERT Base, miệng cắt bằng dlib: 1.843 dòng dùng được (train 574/575, val 177/178, test 169/170 real/fake), 49 lỗi (24 từ video G5DQYAQQ5VM). Cache ở `cache/features/dataset_v1/wav2lip_multi_001_crf23_32/`. |

## Các bước đã làm và vì sao

1. **Detector phân loại `y_001`.** Train real/fake trên feature nhánh riêng (chưa qua Transformer hợp nhất). Test AUC 1,000, EER 0,015.
2. **Kiểm tra dấu hiệu (`02_check_cues.py`).** Train lại chỉ với hình: test AUC vẫn 1,000; chỉ với tiếng: khoảng 0,50. Hình thật ghép tiếng của clip khác bị gọi là fake 1,5%, gần bằng clip thật (1,8%). Kết luận: `y_001` là bộ bắt vết hình của Wav2Lip, không phải detector lip-sync. Nhóm quyết định đổi hướng (phương án B).
3. **AV-HuBERT đầy đủ (`02_fuse.py`).** Thêm Transformer hợp nhất lên cache có sẵn, không phải trích lại 9 giờ (khoảng 12 phút CPU). Mỗi clip có feature chỉ-tiếng và chỉ-hình. Kiểm tra: hình frame t giống nhất với tiếng frame t+1 ở clip thật (cosine 0,233) và t+2 ở fake (0,172), đỉnh rõ, nên Transformer nạp đúng.
4. **Model căn chỉnh chỉ học từ clip thật (`align_001`).** Theo ý tưởng AVH-Align (CVPR 2025), không phải bản tái lập: với mỗi frame hình, model phải chọn đúng frame tiếng trong ±10 frame. Fake chỉ dùng để chọn ngưỡng trên val và đánh giá. Val AUC 0,923, nhưng test AUC 0,416 và báo nhầm 70% clip thật.
5. **Tìm nguyên nhân (`04_check_av_offset.py`).** Đo độ lệch tiếng–hình từng clip, không dùng model. Clip thật của train chủ yếu lệch +1 frame (328/574) hoặc +2 (163); test trải từ −2 đến +3, chỉ 51/169 ở +1; mọi fake Wav2Lip lệch +2. Clip thật của 9VEIR1iI2-k (−1) và U67mf4sRM_Y (0) bị gọi là fake; 36Jx8VhIfl0 (+2) trùng độ lệch với fake. Việc fake lệch +2 có thể do cửa sổ âm thanh của Wav2Lip; đây là suy luận, chưa kiểm chứng.
6. **Cho phép lệch lúc chấm (`--max-shift 2`).** Test AUC lên 0,661 nhưng vẫn báo nhầm 50% clip thật: model vẫn chấm tốt nhất ở độ lệch đã học.
7. **Căn lại trước khi train (`--resync 5`, `align_002`).** Mỗi clip được cắt về độ lệch tốt nhất của chính nó trong ±5 frame, ước lượng từ feature thô, cùng một quy tắc cho real, fake và mọi split. Val 0,826, test 0,793: val và test giờ gần nhau, nên lối tắt độ lệch đã mất. Model overfit sau epoch 7 vì train ít người.
8. **Điểm không cần train (`05_evaluate_peak_sync.py`, `peak_sync_001`).** Điểm fake là số âm của độ giống tiếng–hình cao nhất trong ±15 frame. Tham số ±15 được đặt từ lúc viết công cụ ở bước 5, không chỉnh theo kết quả.

## Bảng kết quả

Ngưỡng là điểm EER trên val, giữ nguyên cho test. "Tiếng ghép lệch" là hình thật ghép tiếng của clip thật khác cùng người: lệch tiếng nhưng không có vết generator.

| Run | Val AUC | Test AUC | Test: báo nhầm thật | Test: bỏ sót fake | Tiếng ghép lệch (test) |
|---|---|---|---|---|---|
| `y_001` phân loại | 1,000 | 1,000 | 0% (CRF23), 3,6% (CRF32) | — | AUC 0,50, không bắt được |
| `align_001` | 0,923 | 0,416 | 69,8% | — | 100% bị gọi là fake |
| `align_001` + `--max-shift 2` | 0,918 | 0,661 | 50,0% | 27,4% | 100% |
| `align_002` (`--resync 5`) | 0,826 | 0,793 | 19,8% | 33,8% | 100%, AUC 1,000 |
| `peak_sync_001` (không train) | 0,890 | 0,875 | 15,7% | 26,5% | 100%, AUC 1,000 |

Với `peak_sync_001`, nén mạnh làm giảm kết quả: test AUC 0,899 ở CRF 23 và 0,852 ở CRF 32. Bảng đầy đủ theo từng mức nén nằm trong `metrics.csv` của từng run ở `experiments/`.

## Cần ghi rõ khi viết báo cáo

- Tập test Wav2Lip đã được xem nhiều lần và các bước 6–7 được thêm sau khi thấy kết quả test, nên test của `align_001 --max-shift 2` và `align_002` là kết quả sau khi đã xem test. `peak_sync_001` ít bị ảnh hưởng hơn vì không train và tham số đặt trước. Phép thử sạch là một generator chưa dùng (MuseTalk) với ngưỡng lấy từ val Wav2Lip.
- Fake chưa review bằng mắt. Theo lần kiểm tra ngày 09–10/10, mặt Wav2Lip mờ hơn ảnh thật 2–4 lần, nên vết mờ là một yếu tố gây nhiễu.
- AV-HuBERT được pretrain trên tiếng Anh; kết quả trên cho thấy nó vẫn nhận ra tiếng ghép lệch ở tiếng Việt, nhưng chưa so với model khác.
- Model căn chỉnh chỉ là phiên bản đơn giản theo ý tưởng AVH-Align, không phải bản tái lập bài báo.
- Val 3 người, test 4 người: khoảng tin cậy rộng; chưa tính.

## Code đã thêm (nhánh `claude/project-thread-qetcz4`)

| File | Việc |
|---|---|
| `src/data/07_add_part1.py` | Gộp V1 và part1 thành split nhiều người. |
| `src/generators/preparation/wav2lip.py`, `settings.py` | Wav2Lip lập plan cho cả train/val/test; chạy bằng `01_prepare.py`, `02_generate.py` có sẵn. |
| `src/data/06_compress.py`, `preparation/compress.py` | Nén CRF 23 và 32 cho mỗi clip real/fake, kèm `media_pairs.csv`. |
| `src/features/01_extract.py`, `extraction.py` | Trích feature AV-HuBERT từ `media_pairs.csv`, chạy tiếp được khi bị ngắt. |
| `src/features/02_fuse.py`, `src/features/avhubert.py` | Thêm Transformer hợp nhất lên cache có sẵn. |
| `src/features/sync.py` | Độ giống tiếng–hình theo độ lệch, căn lại clip, điểm không cần train. |
| `src/training/02_train_alignment.py`, `dataset.py` | Train model căn chỉnh chỉ trên clip thật (`--resync`). `y_001` dùng `01_train.py` có sẵn. |
| `src/models/alignment.py` | Model căn chỉnh tiếng–môi (`max_shift`). |
| `src/evaluation/01_evaluate.py` … `05_evaluate_peak_sync.py`, `metrics.py`, `sync_report.py`, `mismatch.py` | Đánh giá, kiểm tra dấu hiệu, đo độ lệch, bảng chung cho các điểm khớp tiếng–môi. |

Tất cả có test bằng dữ liệu giả lập (`python -m unittest discover -s tests -q`); test chỉ chứng minh code chạy đúng, không phải độ chính xác.

## Chạy lại

```powershell
python src/evaluation/05_evaluate_peak_sync.py --out experiments/peak_sync_002
python src/training/02_train_alignment.py --manifest cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused/pairs.csv --out experiments/align_003 --resync 5
python src/evaluation/03_evaluate_alignment.py --run experiments/align_003
```

Các bước trước đó (split, tạo fake, nén, trích feature, 02_fuse) ở bảng lệnh của [src/README.md](../../src/README.md). Không ghi đè run cũ: mỗi lần chạy dùng tên mới.

## Việc tiếp theo

1. Tạo fake MuseTalk cho đúng các clip val/test (cùng tiếng thay vào), chấm bằng ngưỡng val Wav2Lip. Repo hiện chỉ có bước MuseTalk thử 10 clip train, nên cần thêm bước mới; MuseTalk thuộc phần người 2 trong kế hoạch nhóm.
2. Review bằng mắt một mẫu fake Wav2Lip.
3. Nếu cần tách ảnh hưởng của vết mờ: làm mờ miệng clip thật rồi chấm lại.
