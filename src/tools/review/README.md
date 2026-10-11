# Review — mở file có số rồi Run

Thực hiện sau `src/data/04_build_manifest.py`. Dùng interpreter đã chọn cho pipeline. Cấu hình tên nhóm và tên của bạn trong [settings.py](../../data/preparation/settings.py).

| File | Tác dụng |
|---|---|
| [01_build_roi_preview.py](01_build_roi_preview.py) | Tùy chọn: dựng video vùng miệng có tiếng. Có thể bỏ qua và xem clip gốc. |
| [02_assign_reviewers.py](02_assign_reviewers.py) | Chia clips.csv cho REVIEWERS; mỗi clip một reviewer. Mặc định chưa có tập calibration chung. |
| [03_review_clips.py](03_review_clips.py) | Mở giao diện local, lưu vào reviews/exports/clips/<tên>/review_<tên>.csv. |
| [04_merge_reviews.py](04_merge_reviews.py) | Kiểm tra đủ quyết định, ghi reviewed_clips.csv; nếu thiếu/xung đột thì dừng và xuất danh sách cần xử lý. |
| [05_export_batch_optional.py](05_export_batch_optional.py) | Tùy chọn sau khi phân công: gom media của một reviewer để chuyển máy; không cần đợi gộp kết quả. |

File không có số là phần xử lý mà các bước này gọi. `build_review_manifest.py` chỉ dùng nếu cần ghép thêm điểm đo từ các công cụ quality. `mouth_roi.py` là hàm crop, không phải bước chạy độc lập. Công cụ quality/active-speaker là nhánh tùy chọn, không bắt buộc chạy trước review thủ công.

Trước bước 05 của pipeline, mở `data/manifests/<dataset_version>/reviewed_clips.csv`, bổ sung `speaker_id` nhất quán giữa mọi tập (và speaker_ids nếu liên quan nhiều người). Giao diện hiện chỉ review chất lượng, **chưa tự gán danh tính người nói**. Giữ source_video/canonical_source_id để kiểm tra leakage. Review pass chưa đồng nghĩa clip đã sẵn sàng train.

## Review theo người và gộp kết quả

Rubric v4: Keep giữ nguyên hành vi và tùy chọn tự chuyển. Bấm Reject (R) để mở
popup, chọn một lý do (hoặc phím 1–7) sẽ lưu và chuyển clip tiếp theo, kể cả khi
tắt tùy chọn tự chuyển của Keep. Hủy/Esc đóng popup mà không ghi nhãn. Không cần A/B.
Clip cuối sẽ hiện trạng thái đã lưu và ở lại màn hình.

Kết quả v3 cũ và các interval cũ vẫn được giữ, có thể tiếp tục cùng CSV với v4;
không cần review lại. Bước merge mặc định nhận cả v3/v4, xuất review_labels_v4.csv.
V4 không bắt buộc interval; nếu phân xử Reject, điền final_reason hợp lệ.
Các clip calibration vẫn cần đủ reviewer; bất đồng Keep/Reject/lý do vẫn cần phân xử.
Video gốc và ROI đồng bộ tua/phát/dừng/tốc độ. Đây là đồng bộ trình phát, không
sửa lệch audio/video có sẵn trong file. Khởi động lại chương trình review và refresh
trình duyệt sau khi cập nhật code.

Đặt video và assignment của mình vào
`data/manifests/<dataset_version>/reviews/exports/clips/<tên>/`. Không tạo thêm một folder
tên người bên trong folder đó. Có thể đặt video vào `media/` và ROI vào `roi/` bên trong
folder của mình. Từ root repo, mỗi người mở giao diện bằng tên mình:

```powershell
python src/tools/review/03_review_clips.py --reviewer nguyenvanlinh
```

Lâm Anh dùng `--reviewer nguyenlamanh`, Nhật dùng `--reviewer nguyenminhnhat`.
Không cần sửa settings.py. Giao diện tự lưu `review_<tên>.csv` trong folder đó; khi nghỉ,
dừng bằng Ctrl+C. Không xóa hay ghi đè CSV kết quả khi giải nén lại. Khi cả nhóm xong,
người tổng hợp gom các `review_<tên>.csv` về đúng chỗ rồi chạy
`python src/tools/review/04_merge_reviews.py`.

Review 1 của V1 đã gộp xong ngày 18/09/2026: đủ 2.252 quyết định, 1.143 Keep, 1.109 Reject
(`data/manifests/dataset_v1/reviews/merged/`). Khi đó các CSV review được đồng bộ qua Git
trên nhánh `codex/research-reset`. Từ commit `b27d891` (21/09/2026) cả `data/` bị Git ignore,
nên cách đó không còn dùng được; các `review_<tên>.csv` của review 1 chỉ còn trong lịch sử
Git trước commit đó. Đợt review sau cần thống nhất cách chia sẻ CSV (ví dụ Drive) trước khi bắt đầu.
