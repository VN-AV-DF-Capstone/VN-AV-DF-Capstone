# Models — cấu trúc detector

Trạng thái: có detector tối thiểu trong `detector.py`; hai nhánh AV-HuBERT đã nạp weights và kiểm tra tổng hợp ở `src/features/`. Chưa train trên dữ liệu thật.

Mở `01_smoke_test.py` rồi Run để kiểm tra cả model, một vài bước train Y/X và dự đoán bằng feature giả lập. Test này không đọc video, không tải weights và không tạo kết quả nghiên cứu.

Đầu vào: audio `[batch, thời_gian, số_đặc_trưng_audio]`, visual `[batch, thời_gian, số_đặc_trưng_visual]` và lengths là số bước thời gian hợp lệ. Hai luồng phải được căn cùng cửa sổ và thời gian từ trước. Model chiếu mỗi luồng qua một lớp Linear, ghép chúng, chạy GRU và tạo một logit cho fake. Padding cuối chuỗi được bỏ qua. Đổi logit sang score bằng sigmoid; score chưa được hiệu chỉnh thành xác suất đáng tin cậy.

Chứa detector kết hợp đặc trưng audio và hình ảnh, xử lý thông tin theo thời gian và tạo điểm dự đoán real/fake. Bộ nạp hai nhánh AV-HuBERT đóng băng và preprocessing nằm trong `src/features/`; xem [hướng dẫn extractor](../features/README.md). AV-HuBERT vẫn là ứng viên cần kiểm chứng.

Baseline Y và phương pháp X dùng cùng cấu trúc detector. Theo hướng hiện tại, khác biệt chính của X là thêm yêu cầu nhất quán giữa các bản nén trong lúc huấn luyện; phần đó thuộc `src/training/`, không cần tạo hai kiến trúc khác nhau chỉ vì có tên X/Y.

`alignment.py` là model căn chỉnh tiếng–môi theo ý tưởng AVH-Align (Smeu và cộng sự, CVPR 2025), không phải bản tái lập bài báo. Mỗi luồng đi qua một MLP nhỏ. Với frame hình t, các frame tiếng t−K..t+K của cùng clip là ứng viên và frame t là đáp án đúng. Đầu ra là độ lệch trung bình của clip: −log xác suất chọn đúng frame tiếng khớp. Model chỉ học trên real, nên điểm cao nghĩa là tiếng và môi không khớp, không phải "giống Wav2Lip". Với `max_shift=S`, mỗi clip được chấm ở độ lệch cố định tốt nhất trong ±S frame, nên một độ lệch đều nhỏ không bị phạt; train vẫn dùng `max_shift=0`.

Vòng lặp train ở `src/training/01_train.py`, tính chỉ số ở `src/evaluation/`. Checkpoint tải về nằm trong `weights/`; CLI train lưu checkpoint tại `experiments/<run_id>/best.pt` khi chọn thư mục output tương ứng.
