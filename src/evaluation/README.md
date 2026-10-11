# Evaluation — đánh giá model

Hiện có `check_shortcuts.py`: thử dự đoán bằng metadata của media để phát hiện dấu hiệu phụ. Đây chưa phải chương trình đánh giá detector audio–visual.

Có thêm `predict.py`: hàm predict_batch nhận model và feature audio/visual, trả fake_score và nhãn dự đoán. Ngưỡng mặc định 0.5 chỉ để kiểm tra luồng code.

`01_evaluate.py --run experiments/<run_id>` nạp `best.pt`, kiểm tra hash `pairs.csv` khớp lúc train, chấm điểm val và test trên từng mức nén. Ngưỡng là điểm EER trên val (gộp hai mức nén) và giữ nguyên cho test. `metrics.csv` có AUC, EER, accuracy, balanced accuracy, tỷ lệ báo nhầm real và bỏ sót fake theo split × mức nén; `predictions.csv` có điểm từng mẫu. Số của val tại ngưỡng chọn trên chính val là lạc quan. Chưa tách theo generator đã thấy/chưa thấy hay theo người. So sánh X/Y trên cùng protocol.

`02_check_cues.py --run experiments/<run_id>` kiểm tra model dựa vào dấu hiệu nào, chỉ dùng feature đã lưu: (1) train lại cùng cấu hình nhưng xóa một modality (`visual_only`, `audio_only`); (2) chấm model của run trên hình real ghép tiếng của một clip real khác cùng người, cùng split, không trùng thời gian (lệch tiếng nhưng không có vết generator). Mỗi model dùng ngưỡng EER của val. Kết quả ở `experiments/<run_id>/cue_check/`; đây là chẩn đoán, không phải kết quả chính.

`03_evaluate_alignment.py --run experiments/<run_id>` đánh giá run của `src/training/02_train_alignment.py`. Điểm fake là độ lệch tiếng–môi; ngưỡng là điểm EER trên val, giữ nguyên cho test. Ngoài real so với fake, chương trình chấm hình real ghép tiếng của một clip real khác cùng người (cách ghép giống 02_check_cues, ở `mismatch.py`). Model căn chỉnh hoạt động đúng thì phải gọi phần lớn các cặp này là fake. Kết quả ở `experiments/<run_id>/evaluation/`.

`04_check_av_offset.py --out experiments/av_offset/<ten>.csv` ước lượng độ lệch tiếng–hình của từng clip từ đặc trưng AV-HuBERT đầy đủ (`02_fuse.py`), không dùng model đã train. Độ lệch k nghĩa là hình ở frame t khớp nhất với tiếng ở frame t+k (40 ms mỗi frame). Bảng in trung vị theo từng video nguồn, tách real và fake, để thấy video nào có clip real bị lệch tiếng. Đây là chẩn đoán dữ liệu: nếu đặt quy tắc lọc theo độ lệch, quy tắc phải cố định cho mọi split và loại cả real lẫn fake của cùng nguồn; không chọn clip theo điểm detector hay kết quả test.

Ngưỡng được chọn trên validation và giữ cố định khi đánh giá final test. Dự đoán, chỉ số và cấu hình đánh giá nằm trong `experiments/<run_id>/`, không trong source. Không dùng chẩn đoán metadata làm kết quả của detector chính.
