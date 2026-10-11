# Trích xuất audio và visual bằng AV-HuBERT Base

Mở **`01_extract.py`**, sửa các biến ở đầu file, rồi **Run Python File**.
Đường dẫn cấu hình tương đối được tính từ root repo, không phụ thuộc thư mục terminal.

## 1. Đặt các file ở đâu?

```text
weights/
  avhubert/
    base_vox_iter5.pt
  face_landmarks/
    shape_predictor_68_face_landmarks.dat
    20words_mean_face.npy
external/
  av_hubert/                  # Repo chính thức, không phải repo capstone
    avhubert/resnet.py
    avhubert/preparation/align_mouth.py
```

- Checkpoint: [Base pretrained LRS3 + VoxCeleb2 English](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt), không dùng bản finetuned lip-reading/ASR. File này đã có trên máy đang làm việc. Loader dùng `weights_only=True`, chỉ cho phép thêm kiểu metadata dictionary không thực thi code Fairseq; không nạp pickle không giới hạn.
- Landmark predictor: tải [file nén chính thức của dlib](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), giải nén thành `.dat`.
- Mean face: tải file raw từ [20words_mean_face.npy](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy), không lưu trang HTML thành `.npy`.
- Mã nguồn: `git clone --depth 1 https://github.com/facebookresearch/av_hubert.git external/av_hubert`. Không cần submodule Fairseq cho chế độ hai nhánh này. Upstream đã được clone trên máy đang làm việc; mỗi máy khác cần chuẩn bị riêng.

Weights, external code và feature cache không đưa lên Git. `run.json` lưu hash checkpoint, upstream code, assets và code preprocessing để nhận diện chính xác một lần trích xuất.

## 2. Môi trường

Dùng môi trường Python 3.10 có Torch >= 2.6 phù hợp CPU/CUDA, và các gói trong `environments/requirements-features.txt`. FFmpeg/FFprobe cần ở PATH. Có thể tạo môi trường riêng hoặc clone môi trường data đang hoạt động; không cần Fairseq/OmegaConf cho checkpoint Base này. Trên máy hiện tại, đã cài dlib CPU 20.0.1 vào `vn_av_df` bằng Conda theo yêu cầu người dùng. Có thể dùng `conda install -n vn_av_df -c conda-forge "dlib=20.0.1=cpu*" "libblas=*=*openblas" --freeze-installed` cho môi trường phù hợp; xem phương án dependency trước khi áp dụng ở máy khác. Các khoảng phiên bản trong requirements chưa phải lockfile của môi trường cài mới đã kiểm chứng.

Kiểm tra thiếu gì, không chạy inference:

```powershell
python src/features/01_extract.py --check
```

## 3. Chạy thử đúng một clip

Trong `01_extract.py`:

- Đặt `INPUT_VIDEO` thành đường dẫn **video gốc có audio và một người nói**, không phải ROI preview.
- Đặt `START_FRAME`, `NUM_FRAMES` theo đơn vị 25 fps. Ví dụ đoạn 0–2 giây: `0`, `50`; đoạn 1–3 giây: `25`, `50`. Cửa sổ tối đa 10 giây để giới hạn bộ nhớ.
- Chọn `OUTPUT` chưa tồn tại; mặc định `cache/features/dataset_v1/extract_001`.
- Để `DEVICE = 'cpu'` khi chưa xác minh CUDA. Bấm Run.

Kết quả là `000000_feature_path.pt` và `run.json`. Clip thử có nhãn `None`, split `unassigned`: không tự coi video chưa duyệt là real và không dùng cache thử làm dữ liệu train.

Mặc định `SAVE_PREVIEWS = True`: mỗi feature còn có `000000_feature_path_mouth.mp4` (miệng grayscale sau center-crop, 88×88 với Base, kèm audio cùng cửa sổ) và `000000_feature_path_mouth.png` (tối đa 8 frame mẫu trước chuẩn hóa, giữ đúng giá trị pixel). MP4 chỉ để xem/nghe, không đưa ngược vào model; PNG được lấy trực tiếp từ tensor đầu vào sau khi hoàn tác chuẩn hóa. Trong `run.json`, quality lưu chỉ số frame mẫu, số mặt phát hiện và box khuôn mặt được chọn từng frame. Có thể tắt bằng `SAVE_PREVIEWS = False` hoặc `--no-previews` khi đã kiểm tra và cần giảm output.

Ngày 18/09/2026, clip `5CNS_gaH16c_s0000015202_e0000020958` lỗi ở frame 47 do Dlib nhận nhầm bàn tay là mặt phụ. Preprocessing mới khởi tạo từ một mặt duy nhất, sau đó bám box có độ chồng lấn rõ nhất với box trước đó (IoU tối thiểu 0,3 và hơn ứng viên thứ hai ít nhất 0,15). Không chọn mặt lớn nhất và không nhảy sang mặt ở vị trí khác khi mất target. Nếu ngay đầu có nhiều mặt hoặc việc bám mặt không rõ ràng, vẫn dừng; đây chưa phải model xác định ai đang nói. Không loại clip chỉ vì lỗi extractor. Quy tắc này cần kiểm chứng khi mở rộng dữ liệu.

CLI tương đương:

```powershell
python src/features/01_extract.py --video "data/real/path/to/clip.mp4" --start-frame 0 --num-frames 50 --out cache/features/dataset_v1/smoke_001
```

## 4. Chạy các cặp nén cho training

`src/data/06_compress.py` tự ghi manifest này (`media_pairs.csv`, theo `configs/templates/media_pairs.csv`) cạnh các bản nén. Mặc định `INPUT_VIDEO = None` và `INPUT_MANIFEST` trỏ tới file đó; chỉ cần Run.

| Cột | Ý nghĩa |
|---|---|
| `sample_id` | ID duy nhất của clip real **hoặc** fake cùng cửa sổ; các cửa sổ khác nhau có ID khác nhau |
| `label` | 0 real, 1 lip-sync fake; không tự suy ra từ tên file |
| `split` | train/val/test đã khóa trước đó; extractor không chia lại |
| `media_path`, `paired_media_path` | Hai bản nén của **cùng clip mang cùng nhãn**, đường dẫn tương đối với CSV hoặc tuyệt đối |
| `view_id`, `paired_view_id` | Hai tên khác nhau, ví dụ original/crf23 |
| `start_frame`, `num_frames` | Cùng cửa sổ 25 fps cho cả hai bản; mốc 0 là thời điểm bắt đầu container |

**Không ghép video real với video fake thành một cặp consistency.** Manifest phải đến từ nguồn nén có provenance; extractor không thể chứng minh chỉ từ nội dung media rằng hai file là các bản nén của cùng clip. Nó cũng không thay thế bước chống leakage.

Mỗi view được xử lý riêng và lưu feature riêng. Kết quả đầy đủ có `pairs.csv`, các `.pt`, và `run.json`. `pairs.csv` dùng trực tiếp với `src/training/01_train.py --manifest ...`. Cả baseline Y và X dùng cùng CSV, chỉ khác consistency weight. Mẫu nào lỗi (ví dụ không bám được mặt) được ghi vào `failures.csv` kèm nhãn/split và bị loại khỏi `pairs.csv`; cuối run in số real/fake còn lại mỗi split. Kiểm tra tỷ lệ lỗi theo nhãn trước khi tin kết quả. Bị ngắt thì Run lại cùng `OUTPUT`: cache trùng khớp extractor, manifest và hash media được dùng lại. Run đã có `pairs.csv` không bị ghi đè; đổi code hoặc manifest thì cần `OUTPUT` mới.

## 5. Code xử lý những gì?

1. FFmpeg cắt hai luồng theo cùng cửa sổ, chuyển video sang 25 fps lossless tạm thời và audio PCM mono 16 kHz. Giữ độ lệch timestamp vốn có giữa hai stream; không tự chỉnh môi/tiếng cho khớp.
2. Dlib tìm và bám một mặt trên bản thu nhỏ cạnh dài 640 px (nhanh hơn khoảng 11 lần so với ảnh 1080p), đổi box về ảnh gốc, rồi lấy 68 landmark trên frame gốc. Dừng khi chọn mặt không rõ ràng hoặc mất target quá lâu; cho nội suy tối đa 3 frame liên tiếp và tổng không quá 10%. Đây là quy tắc preprocessing có ghi lại, cần kiểm tra tỷ lệ lỗi theo nhãn/generator.
3. Dùng `crop_patch` của upstream để căn chỉnh mặt và cắt miệng 96×96; grayscale rồi center-crop, chuẩn hóa theo cấu hình checkpoint (Base thường 88×88). Không dùng ROI YOLO phục vụ review. Không nén lại ROI bằng CRF 20 như bước xuất file của upstream.
4. Audio dùng log-filterbank 26 chiều, ghép mỗi 4 frame thành 104 chiều tại 25 Hz, rồi chuẩn hóa theo checkpoint. Chỉ cho lệch chiều dài tối đa một frame ở biên; không kéo giãn feature để che lỗi timeline.
5. Nạp **đúng weights hai nhánh trước fusion**: visual ResEncoder + projection và audio projection. Đóng băng, `eval`, không gradient. Với Base, mỗi đầu ra là `[T,768]`. Thiếu/sai key weights thì lỗi, không dùng random weights thay thế.

Đây là bộ trích xuất **pre-fusion**, chưa phải toàn bộ Transformer AV-HuBERT 103M. Base không có Transformer riêng trong mỗi nhánh (`sub_encoder_layers=0`), vì vậy nhánh audio ở đây chỉ là phép chiếu acoustic features; khả năng học quan hệ thời gian hiện đến từ GRU của detector. Transformer hợp nhất được thêm ở bước riêng `02_fuse.py` (mục 6); không diễn giải hai nhánh này như hai encoder Transformer hoàn chỉnh.

`avhubert.py` phụ trách weights; `media.py` xử lý media; `extraction.py` điều phối/lưu cache; chỉ cần Run `01_extract.py`.

Đối chiếu: [upstream model](https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert.py), [input transforms](https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert_dataset.py), [mouth preparation](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation). Khác biệt có chủ đích: dùng HOG một mặt, không CNN fallback; giới hạn nội suy; không re-encode ROI lossy. Đây là cấu hình của repo cần đánh giá, không tuyên bố tái lập bit-for-bit pipeline training upstream.

Đã nạp hai nhánh từ checkpoint thật (SHA-256 `dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc`) và chạy forward trên đầu vào tổng hợp: mỗi nhánh trả `[1,5,768]` hữu hạn; có 11,661,312 tham số được nạp trong hai nhánh. Upstream được đối chiếu tại commit `258fb50e155134eec2c4b49c2ae8de267075fd18`. Bộ test kiểm tra thêm media tổng hợp, checkpoint giả lập và ghép cache với loader training.

Ngày 16/09/2026: đã cài dlib CPU 20.0.1, nạp predictor `.dat` và chạy **end-to-end trên một clip thật** được reviewer đánh dấu `keep`: `j60KF89vXkg_s0000138818_e0000141598`, cửa sổ 0–2 giây. Video đầu vào 1920×1080, 50 fps, audio 44,1 kHz; sau xử lý có 50 frame, 32.000 mẫu audio 16 kHz, không nội suy landmark. Audio/visual features đều có shape `[50,768]`, giá trị hữu hạn. Thời gian toàn lệnh trên CPU khoảng 34,48 giây, gồm cả nạp model; không phải benchmark throughput.

Kết quả ở `cache/features/dataset_v1/real_smoke_001/`: `000000_feature_path.pt` và `run.json` lưu hash/config/quality. Cache thử giữ nhãn `None`, split `unassigned`; không tự đưa vào training. Kiểm tra kỹ thuật một clip chưa chứng minh chất lượng ROI trên toàn dataset hay accuracy detector.

## 6. Thêm Transformer hợp nhất (`02_fuse.py`)

Audio/visual trong cache của `01_extract.py` chính là đầu ra hai nhánh mà Transformer hợp nhất nhận vào. Vì vậy `02_fuse.py` chỉ đọc cache đã có, không đọc lại video và không chạy lại dlib.

- Làm như `extract_finetune` của upstream: ghép [audio, video], LayerNorm, `post_extract_proj`, Transformer 12 lớp (LayerNorm trước mỗi khối), LayerNorm cuối, lấy lớp cuối.
- Mỗi view có hai đặc trưng `[T,768]`: `audio` là đầu ra khi chỉ có tiếng (hình đặt 0), `visual` là đầu ra khi chỉ có hình (tiếng đặt 0). Tên file và `pairs.csv` giữ nguyên, nên các script train/đánh giá sẵn có đọc được.
- Transformer được viết lại bằng PyTorch thuần theo `TransformerEncoder` wav2vec2 của Fairseq, không cần cài Fairseq. Nạp strict: thiếu, thừa key hay sai shape đều lỗi. Chỉ bỏ qua hai nhánh và các head dùng cho pretraining (`mask_emb`, `label_embs_concat`, `final_proj`).
- Test so từng lớp với Conv1d weight-norm và `TransformerEncoderLayer` của PyTorch trên weights giả lập. Chưa so trực tiếp với Fairseq gốc.

Sau khi chạy, chương trình in độ tương đồng cosine giữa đặc trưng chỉ-hình ở frame t và chỉ-tiếng ở frame t+k (k từ −5 đến 5), trung bình theo view, riêng real và fake. Nếu Transformer được nạp đúng, hàng real phải cao nhất ở k=0. Nếu không, dừng lại trước khi train. Đây là kiểm tra hợp lý, không chứng minh AV-HuBERT (huấn luyện bằng tiếng Anh) hoạt động tốt với tiếng Việt.

```powershell
python src/features/02_fuse.py
```

Mặc định đọc `cache/features/dataset_v1/wav2lip_multi_001_crf23_32` và ghi `cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused`. Output có sẵn không bị ghi đè; nếu bị ngắt giữa chừng, xóa thư mục output chưa có `pairs.csv` rồi chạy lại.
