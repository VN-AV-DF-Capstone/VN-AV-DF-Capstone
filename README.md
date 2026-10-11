# VN-AV-DF-Capstone

Nghiên cứu phát hiện **lip-sync manipulation trong video một người nói tiếng Việt**, sử dụng audio và hình ảnh vùng miệng.

> Phương pháp X có cải thiện so với baseline Y trên người/nguồn độc lập, generator chưa thấy và video bị nén hay không?

Y là detector audio–visual; X dùng cùng detector, dữ liệu và ngân sách nhưng thêm yêu cầu nhất quán giữa các bản nén của cùng clip. AV-HuBERT là ứng viên cần kiểm chứng, chưa phải model đã chạy tốt trên tiếng Việt.

## Trạng thái

Repo đã chuyển sang nền xử lý dữ liệu cho hướng mới. Công cụ cắt, kiểm tra media, review, chia split và nén đã được tách khỏi pipeline AVSP-Net. `dataset_v1` đã qua review clip; từ 09/10/2026 được ghép với part1 thành split 11 người và có 946 cặp real/fake Wav2Lip chưa review bằng mắt. MuseTalk mới có các run xem chất lượng. Nguồn mở rộng đang tải trong `dataset_v2`. **Kết quả sơ bộ trên Wav2Lip (09–11/10/2026):** điểm khớp tiếng–môi từ AV-HuBERT, không cần train, đạt test AUC 0,875 và bắt được tiếng ghép lệch; detector phân loại đạt 1,000 nhưng chỉ bắt vết mờ của Wav2Lip. Chi tiết và giới hạn ở [báo cáo 11/10](docs/reports/2026-10-11_WAV2LIP_SYNC_RESULTS.md). Chưa có kết quả trên generator chưa thấy, chưa chạy X.

Xem [PROJECT.md](PROJECT.md) để phân biệt phần đã triển khai và phần dự kiến. Kết quả pseudo-fake/AVSP-Net nằm trong [archive](docs/archives/legacy_avsp/README.md).

## Bắt đầu

**Luồng chính: mở file có số rồi Run Python File trong IDE.** Chọn interpreter `vn_av_df`; đường dẫn dữ liệu do `src/data/preparation/settings.py` quyết định. File này đang đặt `DATASET_VERSION = 'dataset_v2'` (đã commit): URL nằm ở `data/sources/dataset_v2/videos.csv`, danh sách đã mở rộng ở `selected_videos.csv`, và bước 02 ghi vào `data/raw/dataset_v2/download_001/` (đến 26/09/2026: 966/1.954 video đã tải, 908 pending, 80 failed). Khi tải đủ và xử lý lỗi, chạy `src/data/03_cut_clips.py` → `04_build_manifest.py`, review bằng các file có số trong `src/tools/review/`, rồi chia split ở bước 05. Bước 01/02 dùng để thu thập hoặc tiếp tục tải; không chạy lại bước 01 với danh sách khác rồi tiếp tục vào `download_001` đã khóa nguồn. Kiểm tra giấy phép hiện là bước riêng tùy chọn.

Xem [thứ tự và đầu vào/đầu ra từng bước](src/data/README.md), [cấu trúc data hoàn chỉnh](data/README.md) và [tests theo nhóm](tests/README.md). File cấu hình chung là [settings.py](src/data/preparation/settings.py). Các lệnh CLI bên dưới chỉ là lựa chọn khi cần thao tác terminal.

Chọn interpreter Python của môi trường data đã chuẩn bị; phiên bản tham chiếu và giới hạn kiểm chứng nằm trong [environments/README.md](environments/README.md). Lệnh bên dưới chạy từ root repo; không tự tải dataset/model.

```powershell
python -m unittest discover -s tests -q
python -m src.data.preparation.build_splits --help
python -m src.data.preparation.compress --help
python -m src.data.preparation.download --help
python src/data/preparation/cut_clips.py --help
python src/data/preparation/build_manifest.py --help
python src/tools/review/clip_review.py --help
```

Đọc [hướng dẫn công cụ data](src/data/README.md) trước khi chạy với dữ liệu thật. `configs/templates/` chứa CSV mẫu chỉ có header, không phải dataset đã duyệt. File config cắt mẫu chưa điền số nguồn nên cố ý từ chối chạy.

## Cấu trúc đang có

| Đường dẫn | Nội dung |
|---|---|
| `src/data/` | File chạy 01–07 ở ngoài cùng; code hỗ trợ và settings.py nằm trong `preparation/`. |
| `src/data/preparation/quality/` | Đo mặt, gom người và active-speaker tùy chọn; cần kiểm chứng ngưỡng trên nguồn mới. |
| `src/tools/review/` | Preview miệng có tiếng, review, chia/gộp công việc. |
| `src/generators/` | Wav2Lip (run `wav2lip_multi_001` đã tạo 946 cặp) và MuseTalk (mới có run xem chất lượng). |
| `src/features/` | Trích đặc trưng AV-HuBERT Base (`01_extract.py`) và thêm Transformer hợp nhất (`02_fuse.py`). |
| `src/models/`, `src/training/` | Detector Y/X và model căn chỉnh tiếng–môi; đã train trên feature thật của run Wav2Lip. |
| `src/evaluation/` | Đánh giá 01–05 (AUC/EER, kiểm tra dấu hiệu, độ lệch tiếng–hình, điểm không cần train) và `check_shortcuts.py` chẩn đoán metadata. |
| `configs/`, `environments/` | Mẫu đầu vào và hướng dẫn môi trường. |
| `data/`, `cache/`, `weights/`, `experiments/` | Dữ liệu, feature, weights local và kết quả run; media/feature/weights không commit, `experiments/` chỉ commit CSV. |
| `vn-av-df-data-part1/` | 852 clip đã review của người 2, bước 07 đọc `review.csv` ở đây; bị ignore như `data/`. Xem [data/README.md](data/README.md#clip-của-người-2-part1). |
| `docs/` | Nghiên cứu hiện hành, planning, báo cáo và archive cũ. |

Các phiên bản dataset nằm song song theo loại dữ liệu: `data/sources/dataset_v2/`, `data/raw/dataset_v2/`, rồi `data/real/dataset_v2/` và `data/manifests/dataset_v2/` khi chạy các bước sau. `data/dataset/_V2` không thuộc cấu trúc mà pipeline hiện đọc. `dataset_v1` giữ clip real, quyết định review, split sơ bộ và split nhiều người `real_splits_multi_001.csv` (V1 + part1); tránh trộn hai phiên bản hoặc đổi tên thư mục V1. Generator, nén và feature có settings riêng hiện vẫn dùng V1. Xem [kiểm kê dọn workspace sau review 1](docs/reports/2026-09-23_WORKSPACE_CLEANUP_AUDIT.md) trước khi bỏ media/cache của các run thử.

Model/train/eval AVSP-Net cũ nằm trong backup và lịch sử Git. Bản mới có detector, model căn chỉnh tiếng–môi, training Y/X và alignment, các bước đánh giá; mở [01_smoke_test.py](src/models/01_smoke_test.py) để thử bằng tensor giả lập. Luồng V1 + part1 → Wav2Lip → nén → [01_extract.py](src/features/01_extract.py) → `02_fuse.py` → train → đánh giá đã chạy trên dữ liệu thật ngày 09–11/10/2026 ([bảng lệnh](src/README.md), [cách đặt weights](src/features/README.md)). Fake chưa review bằng mắt, val/test chỉ có 3/4 người, chưa có generator chưa thấy; đây chưa phải benchmark. Môi trường model tối thiểu ở [requirements-model.txt](environments/requirements-model.txt), bổ sung cho extractor ở [requirements-features.txt](environments/requirements-features.txt); chưa thử cài mới từ đầu. Xem [vai trò từng thư mục source](src/README.md).

## Dữ liệu và báo cáo

Nhóm nguồn phục vụ độ đa dạng; giấy phép được ghi riêng theo từng video. Không mặc định playlist VLR hoặc nội dung công khai có CC. Hồ sơ cần phân biệt nghiên cứu, cách lấy dữ liệu, tạo biến thể và phát hành media. Không phân phối lại video chỉ vì có file trong máy.

Mọi biến thể giữ nguồn và split của clip real. Người nói, tập nguồn và bản đăng lại liên quan phải cùng nhóm; cùng host có thể nối nhiều tập thành một nhóm lớn. Nén không làm clip real thành fake.

Tài liệu: [hướng nghiên cứu](docs/research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md), [protocol data](docs/research/DATA_PROTOCOL.md), [pilot](docs/planning/KE_HOACH_PILOT_LIP_SYNC_VI.md), [chỉ mục](docs/README.md).
