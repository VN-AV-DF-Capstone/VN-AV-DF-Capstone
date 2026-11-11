# Đề xuất detector v006 và v007

> **Đổi tên và cập nhật 11/11/2026 (MN):** v005 của Linh đã đóng (M3-R `trihead_v5`, [docs/v005_report.md](v005_report.md)), nên đề xuất "M1 + P1–P6" nay là **v006 · M6** và "ViSyl" nay là **v007 · M7**. MN chốt **M6 = nền M3-R + P4, P5, P6**: P1–P3 đã được ablation v004 / v005 của Linh trả lời (bảng §0b). Phần §2 dưới đây là thiết kế cũ trên khung M1, giữ để đối chiếu; `trihead_v3` vẫn trong `models.SPECS` nhưng không còn trong đội hình.
>
> **Code (11/11/2026):** M6 = `trihead_v5` (đội hình `settings.VERSIONS["v006"]`) + P4 / P6 trong `evaluate.py` (chạy cho mọi model); P5 chưa có encoder. M7 = `visyl` (+ 3 ablation), `features/syllables.py`. Chạy bằng `src/models/v006/`, `src/models/v007/`. Cách train theo v005 của Linh (nhãn mismatch đo được, dịch tiếng −2…+5, học thêm bản nén 480p / 360p, chọn checkpoint theo val AUC trung bình bản sạch và bản nén), không theo protocol v003 ghi trong §2, §3. Chưa train, chưa đo trên dữ liệu thật.

Ngày 30/10/2026 · viết cho MN và team VN-AV-DF-Capstone

> **Đổi tên 02/11/2026 (MN):** repo của Linh đã có v004 riêng (M3 = `trihead_v4`, môi–tiếng–đầu). Vì vậy đề xuất "M1 + P1–P6" (trước gọi là v004 · M1 (v004)) nay là **v005 · M5**, và "ViSyl" (trước gọi là v005 · M3) nay là **v006 · M6**. Tên file review (`review-v004`, `review-v005`) và Claude Doc "v004 · Đề xuất nâng cấp M1" giữ tên cũ.
>
> **Trạng thái.** Cả v006 và v007 đều là **đề xuất**: chưa có code, chưa có lần train nào, chưa có số đo. Mọi số của v003 trong file này lấy từ slide của Linh (`vn-av-df-slide`, số nhập tay trong `talk.js`, part 1, 1 seed), chưa đối chiếu được với output gốc (`history.json`, `predictions-*.npz`, `report.md` không có trong repo nào của team).
>
> **Cập nhật 30/10/2026 sau review v007:** v007 đã có bản sửa riêng, `plans/2026-10-30_v007_sau-review.md` (review ở `reviews/2026-10-30_review-v005.md`). §3 và §4 của file này là bản trước review, giữ để đối chiếu.
>
> Phần v006 chép lại và sắp xếp lại từ Claude Doc "v004 · Đề xuất nâng cấp M1" (28/10/2026, bản đã qua một vòng review độc lập, báo cáo review ở `reviews/2026-10-28_review-v004.md`). Phần v007 là đề xuất mới ngày 30/10/2026.
>
> **Mức tin:** ● số đo (slide hoặc dữ liệu của mình) · ✔ đã kiểm trong code (commit `2b3a70c` trên `main`) hoặc đọc trực tiếp từ paper gốc · ○ suy luận, ước lượng, chưa kiểm.

---

## 0. Tóm tắt một trang

**v006 = M6 (v006).** Giữ M1 (TriHead-v2) làm model đề xuất chính và nâng cấp bằng 6 bản vá có tên (P1–P6), mỗi bản vá đo được riêng so với M1-v003: nhánh sync thuần (P1), AV-HuBERT hình sang nhánh hình (P2), mỗi đầu ra một nhánh (P3), thay FiLM bằng chuẩn hoá điểm theo người (P4), đổi front-end giọng nếu kiểm zero-shot ủng hộ (P5), sửa thước đo định vị (P6). Không phụ thuộc ngôn ngữ, không đổi dữ liệu.

**v007 = M7 (v007) "ViSyl".** Một model **mới, khác họ với M1**, không thay M1. Ý chính: detector "hiểu tiếng Việt".
- Đơn vị phân tích là **âm tiết** (mốc thời gian từng từ của PhoWhisper; tiếng Việt viết cách theo âm tiết) cộng các khoảng lặng, thay vì từng frame 40 ms.
- Mỗi âm tiết mang **kỳ vọng khẩu hình suy ra từ chính tả tiếng Việt**: môi phải khép ở đầu âm tiết (b, m, p), ở cuối âm tiết (-m, -p) và ở cuối các vần -ung, -uc, -ông, -ôc, -ong, -oc (phụ âm cuối môi-ngạc mềm [ŋ͡m], [k͡p̚], đặc trưng riêng của tiếng Việt). Generator lip-sync học chủ yếu trên tiếng Anh, nơi -ng không bao giờ khép môi; giả thuyết là chúng vẽ sai đúng ở những chỗ này.
- Rẻ: không đọc crop miệng, ~0,75 M tham số, ước ~0,3–0,6 giờ train / lần trên part 1 (○).
- Cho được lời giải thích theo âm tiết (ví dụ "âm tiết 'không' lúc 2,31 s: kỳ vọng khép môi cuối âm tiết, môi vẫn mở").

| | v006 · M6 (v006) | v007 · M7 (v007) "ViSyl" |
|---|---|---|
| Loại | Nâng cấp M1 bằng 6 bản vá | Model mới, khác họ với M1 |
| Câu hỏi nghiên cứu | M1 có tốt hơn khi nối dây đúng, chuẩn hoá điểm theo người, front-end giọng tốt hơn? | Đưa kiến thức tiếng Việt (âm tiết, luật khép môi) vào detector có bắt tốt hơn ở chỗ M1 yếu? |
| Đơn vị | frame 40 ms | âm tiết + khoảng lặng |
| Ngôn ngữ | không phụ thuộc (AV-HuBERT tiếng Anh giữ nguyên) | có chủ đích: luật chính tả → khẩu hình; tuỳ chọn encoder AV-HuBERT tiếng Việt |
| Crop miệng (trace, NPR) | giữ | bỏ (dùng landmark môi + AV-HuBERT hình) |
| Tham số học | gần M1-v003 (1,94 M), chưa tính chính xác (○) | ~0,75 M (○) |
| Giờ train part 1 (T4) | ~4 giờ / lần (M1-v003 mất 4,09 giờ ●) | ~0,3–0,6 giờ / lần (○, M2 không crop miệng mất 0,25 giờ ●) |
| Cần thêm | output v003 của Linh; env fairseq riêng nếu làm P5 | cache âm tiết PhoWhisper (~1–2 giờ GPU cho part 1 + 2, ○); không env mới |
| Cổng chính | AI? M6 (v006) không kém M1-v003 (cận dưới CI ≥ −0,01) | AI? M7 không kém M1-v003 (cận dưới CI ≥ −0,02) và hơn ở ít nhất một lát khai báo trước (định vị hoặc generator chưa thấy) |
| Nấc 0 (không train) | chuẩn hoá điểm, thước đo định vị, zero-shot front-end giọng | pilot "môi có khép đúng luật tiếng Việt không" trên real và fake; độ chính xác mốc âm tiết; kiểm lối tắt vết ghép |

**Quan hệ giữa hai bản.** M1 vẫn là model đề xuất chính; M2 và các baseline giữ vai như v003. v007 thêm một đề xuất thứ hai để team có hai giả thuyết đối chứng nhau: v006 sửa cách M1 nối dây và hiệu chuẩn (không phụ thuộc ngôn ngữ), v007 đưa kiến thức tiếng Việt vào detector. Hai bản dùng chung dữ liệu, split, protocol, test khoá (part 2), chuẩn hoá điểm (P4) và thước đo định vị mới (P6), nên so trực tiếp được với nhau và với M1-v003.

**Vì sao v007 là model mới chứ không phải "v006 + thêm bản vá".** v006 chưa "tối ưu hết", nhưng những gì còn lại trong khuôn M1 (loss one-class, loss xếp hạng theo cặp, kênh Φ của AVH-Align trong nhánh sync, đầu hồi quy biên đoạn, pseudo-fake hình kiểu SAVe) đều đã bị hoãn vì chưa có bằng chứng trên dữ liệu mình. Gom chúng thành v007 sẽ lặp lại lỗi "quá nhiều thay đổi cùng lúc" mà vòng review v006 đã chỉ ra. Còn các giới hạn mà chính v006 ghi là không giải (AV-HuBERT pretrain tiếng Anh, tức rủi ro X3 của v003; định vị dựa trên frame; không giải thích được vì sao một clip bị nghi) nằm ngoài khuôn M1. v007 nhắm vào đúng các chỗ đó.


## 0b. Cập nhật 11/11/2026 theo kết quả v004 / v005 của Linh

Nguồn: [docs/v004_report.md](v004_report.md), [docs/v005_report.md](v005_report.md) (Kaggle, 3 part, 31 người test, 7.368 clip, seed 42; số "local" là thử nghiệm part 3 của Linh, 800–4.800 step). Mức tin: ● Linh đo, ○ suy luận của mình.

**Mốc so sánh mới.** M3-R (`trihead_v5`) AI? 0,973 (sạch) / 0,962 (480p) / 0,914 (360p), hơn BL-Ens-AV +0,039 / +0,050 / +0,065 ●. M1 không được train lại ở v004 và v005, nên mọi cổng "so với M1-v003" của file này đổi thành "so với M3-R v005, cùng protocol".

| Bản vá | Bằng chứng v004 / v005 | M6 |
|---|---|---|
| P1 · Sync thuần | Nhánh sync của M3 / M3-R chỉ còn cost volume [chiếu, cosine, Φ] → CNN 2D, không có đường `content`; M3-R đạt 0,973 ● | đã có trong nền M3-R |
| P2 · AV-HuBERT hình sang nhánh hình | Luồng appearance (AV-HuBERT hình → TCN) không giúp: bỏ đi 0,947 so với 0,951 (local, 800 step) ●; M3-R đã bỏ | bỏ |
| P3 · Mỗi đầu một nhánh (visual ← trace) | Trace một mình 0,924, sync một mình 0,954, cả hai 0,950 (local) ●. Mục tiêu của P3 (đầu visual không đọc lệch tiếng) M3-R đạt bằng luồng sync tương đối + sham làm mẫu âm: sham so với real AUC 0,42, tiếng trễ +2…+4 frame AUC 0,52–0,53 ● | bỏ |
| P4 · Chuẩn hoá điểm theo người | M3-R không có FiLM; chưa ai đo chuẩn hoá theo người. Ngưỡng cố định ở 360p báo nhầm 8,5 % real (mục tiêu 5 %) ●: chỗ P4 có thể giúp (○) | giữ, chạy ở evaluate trên run M3-R |
| P5 · Front-end giọng | Giọng chưa thấy là chỗ yếu nhất của M3-R: VoxCPM2 0,905, F5 inpainting 0,911 (M2 không có XLS-R lại hơn: 0,964, 0,933) ● | giữ, lý do mạnh hơn; chưa có encoder |
| P6 · Thước đo định vị | AP@0,5 = 0,023 cho mọi model ở v005; Linh ghi thước đo cần sửa trước khi đọc ● | giữ, đã có code |

**Các phát hiện khác ảnh hưởng cả M6 và M7.**
- **Video nén.** Mọi detector sập khi video bị nén lại nếu không học bản nén (M3 0,965 → 0,761 ở 360p, local ●). Từ v005 mọi model học thêm bản c480 / c360; M7 lấy bản nén ngẫu nhiên (p = 0,5, không theo cặp: cặp + nhất quán chỉ thêm +0,004 ở 480p, tốn gấp đôi giờ GPU ●).
- **Nhãn mismatch** đo được nhưng gần như do generator quyết định (Wav2Lip 100 %, LatentSync 0 %), và đầu mismatch học "có lip-sync không" hơn là "lệch bao nhiêu" (AUC trong cùng generator 0,45–0,82 ●).
- **Đầu visual dễ** vì độ nét miệng và dấu vết đường ống tạo fake (v004 §6.2 ●); M3-R qua được phép thử real xử lý lành tính (1–2 % báo nhầm, local ●).
- **Dưới 360p** đặc trưng AV-HuBERT hình mất tác dụng (AUC visual ~0,5 ở 288p, local ●); cổng độ phân giải (`settings.VISUAL_MIN_SHORT_SIDE`) áp cho mọi model.

**Thứ tự làm M6 từ đây.**
1. Nấc 0, không train: chép run `trihead_v5_seed42` của v005 (Output Kaggle của Linh) vào `runs/train_all_v006/`, chạy `src/models/v006/03_evaluate.py` → P4 và P6 cho M3-R. Đây là lần chấm test thứ hai của checkpoint này (ghi trong `test-log.json`): báo là post-hoc.
2. P5 nấc 0: zero-shot AntiDeepfake XLS-R-1B đóng băng trên audio val (cần env fairseq riêng và tải weight: chờ MN cho phép).
3. Nếu P5 đạt: thêm encoder giọng mới cho nền M3-R và train M6 theo protocol v005, so với M3-R bằng bootstrap theo người.

---

## 1. Bối cảnh chung: v003

### 1.1. Bài toán và dữ liệu

Sáu ô (`docs/v003_generation.md` §1 ✔):

| | Tiếng gốc | Tiếng thật của clip khác (cùng người) | Giọng AI nhân bản giọng người đó |
|---|---|---|---|
| **Hình thật (môi gốc)** | ① real | ② sham | ③ voice |
| **Môi do generator vẽ lại** | ④ source | ⑤ donor | ⑥ both |

| Ô | AI? | Phần bị giả | Tiếng khớp môi? |
|---|---|---|---|
| ① real | 0 | - | khớp |
| ② sham | **0** | - | **không** |
| ③ voice | 1 | tiếng | không |
| ④ source | 1 | môi | khớp |
| ⑤ donor | 1 | môi (câu mới) | khớp |
| ⑥ both | 1 | môi + một cụm tiếng AI (track B) | khớp |

- Ba câu hỏi cho mỗi mẫu: **visual fake?**, **audio fake?**, **AV mismatch?**; điểm tổng **AI? = max(visual, audio)**. Nhãn theo frame 0,04 s (`fake_intervals`, `visual_fake_intervals`, `audio_fake_intervals` trong manifest ✔).
- Mỗi ô (trừ ①) có bản **full** và **partial** (một đoạn 0,3–3 s, cách hai đầu ≥ 0,5 s). Chỗ nối hình: 50 % cắt thẳng, 50 % trộn 2–6 frame.
- Ô ③ có track A (đọc lại cả câu, co giãn ≤ ±15 % cho vừa clip), track B (đổi một cụm từ, ghép bằng đúng hàm ghép của ② sham), track C (F5 inpainting, **chỉ test**).
- Mỗi part 40 người (20 nam + 20 nữ), chia theo người: train 24, val 6, test 10. Part 1: 7.622 mẫu (val 1.052, test 2.336) ●; part 2: 7.772 mẫu ●.
- Generator train/val: Wav2Lip-GAN, MuseTalk 1.5, IP-LAP, VideoReTalking; **chỉ test**: LatentSync 1.5, Diff2Lip. Giọng train/val: VieNeu-TTS v3, F5-TTS tiếng Việt, OmniVoice, TTS → Seed-VC; **chỉ test**: VoxCPM2, viXTTS, F5 inpainting.
- Cache đặc trưng (lưới 25 Hz ✔): `av` (AV-HuBERT Large `self_large_vox_433h`, lớp cuối, chạy riêng tiếng `a` và riêng hình `v`, 1024 chiều), `ssl` (XLS-R 300M, 3 nhóm lớp 1–4 / 5–8 / 9–12, 50 Hz → 25 Hz, [T, 3, 1024]), `mouth` (crop miệng 96×96 độ phân giải gốc, `lips` 20 điểm môi đã trừ tâm và chia bề rộng miệng, `pose` yaw/pitch/roll từ solvePnP), landmark dlib 68 điểm thô (cache `landmarks`; bản float16 cũng nằm trong cache `avhubert`, khoá `landmarks` ✔), `lipinc`, `wave`.

### 1.2. M1 · TriHead-v2 (`trihead_v2`, 1,94 M tham số học ●)

Code: `src/vn_av_df/training/models.py`, lớp `TriHeadV2` ✔. Train bằng `src/models/v003/02_train.py`; mô tả ở `docs/v003_model.md` §5.

- **Nhánh sync** (`SyncTrunk`): chiếu `a`, `v` 1024 → 256 (Linear + LayerNorm); cost volume theo độ trễ δ ∈ [−15, 15] gồm hai kênh (tích vô hướng sau chiếu / 16 và cosine thô) → CNN 2D 3 lớp (giãn 1, 2, 4 trên trục thời gian) → điểm theo độ trễ → soft-argmin (độ lệch), đỉnh, entropy + pool mean/max theo trục độ trễ (67 chiều). **Ghép thêm đường `content` = Linear(2048 → 128) trên [a, v] thô.** → Linear → TCN (Conv1d giãn 1, 2, 4).
- **Nhánh trace** (`TraceTrunk`): crop miệng → 3 luồng [ảnh − 0,5; NPR đúng Tan et al. 2024 = x − upsample(x[::2, ::2]); residual thời gian x − (trước + sau)/2] → Conv3D 3 → 24 → 48 → 64 (stride không gian 2) → pool mean + max; ghép động học môi (y, vận tốc, gia tốc của 20 điểm = 60) + tư thế đầu và vận tốc (6) → TCN.
- **Nhánh audio** (`AudioTrunk`): XLS-R đóng băng (cache 3 nhóm lớp) → trọng số softmax học được trên nhóm → chiếu 128 → 2 lớp self-attention (4 head).
- **Tham chiếu người**: 50 frame đầu của 3 clip real cùng người (đã loại clip gốc của chính mẫu) đi qua chính các nhánh (không grad) → FiLM (khởi tạo 0) + ghép h − ref.
- **Đầu ra**: visual ← trace + sync; audio ← audio; mismatch ← sync + audio. AI? = max(visual, audio), không có đầu AI riêng.
- **Huấn luyện**: BCE theo frame + BCE theo video (gộp top-5 frame) cho từng đầu; pseudo-fake thay đoạn tiếng p = 0,25 (đoạn 0,3–3 s từ clip real khác cùng người, nhãn mismatch = 1, AI = 0); loss gây nhầm lẫn người nói (bị chặn, λ tăng 0 → 0,3) trên nhánh audio và trace; augment dịch tiếng −1…+2 frame; augment miệng; train trên đoạn 75 frame (3 s), chấm cả clip 4–7 s. AdamW lr 3e-4, weight decay 1e-4, batch 16, tối đa 30 epoch, patience 5.

### 1.3. M2 · AVH-Align+ (`avh_align_plus`, 1,92 M, trong đó 1,74 M là Φ ●)

Φ = `AlignMLP` của AVH-Align khởi tạo từ checkpoint công bố (học tiếp lr 1e-5). Cost volume Φ(v_t, a_t+δ), δ ∈ [−15, 15] + kênh vị trí δ/15 → CNN 2D → độ lệch, đỉnh, entropy; ghép vận tốc môi (20) + tư thế đầu và vận tốc (6) → TCN → hai đầu ai và mismatch ✔. Không có đặc trưng giọng giả, không crop miệng.

### 1.4. Kết quả v003 (slide, part 1, 1 seed, test 2.336 mẫu của 10 người chưa thấy, tác vụ "AI?")

| Model | AUC val → test | AUC sham so với real (0,5 = không nhầm) | Giờ train (T4) |
|---|---|---|---|
| M1 | 0,986 → 0,947 | 0,543 | 4,09 |
| M2 | 0,946 → 0,908 | 0,792 | 0,25 |
| BL-M/sup (AVH-Align có nhãn) | 0,879 → 0,843 | 0,527 | 1,12 |
| BL-V (LIPINC-V2) | 0,563 → 0,568 | | 0,47 |
| BL-A (XLS-R + AASIST) | 0,506 → 0,512 | | 3,34 |
| BL-M (AVH-Align chỉ real) | chỉ real → 0,416 | 0,981 | 0,08 |

- BL-Ens = 0,564. M1 − BL-M/sup = +0,104 (CI 95 % theo người [0,080; 0,125]) ●.
- BL-M/sup cũng không nhầm sham (0,527), nên tính chất "không nhầm sham" đến từ nhãn sham trong dữ liệu v003, không riêng kiến trúc M1.
- BL-A gần như chưa học được (việc mở #12 trong `docs/v003_model.md`), BL-V chỉ nhỉnh hơn ngẫu nhiên: so với BL-A, BL-V, BL-Ens hiện chưa có giá trị.
- M1 theo lát: generator đã thấy 0,995 / chưa thấy 0,942 (LatentSync 0,902); giọng đã thấy 0,990 / chưa thấy 0,889 (VoxCPM2 0,848, F5 inpainting 0,856); full 0,971 / partial 0,941; frame điểm cao nhất nằm trong đoạn giả 0,90 (partial); AP@0,5 ≈ 0,02 ●.

### 1.5. Điểm yếu của M1 và M2 (bài học dùng cho cả v006 và v007)

| # | Điểm yếu | Dẫn chứng | Nguyên nhân (code, giả thuyết) | Mức tin |
|---|---|---|---|---|
| W1 | Chưa biết nhánh trace đóng góp gì | Bỏ trace: visual 0,981 → 0,978; AI? 0,947 → 0,948 | Ablation `no_trace` bỏ cả ảnh, NPR, residual thời gian lẫn động học môi và tư thế đầu. Giả thuyết: đường `content` của nhánh sync đưa AV-HuBERT [a, v] thô vào đầu visual và đã đủ bắt dấu vết 4 generator đã thấy | ● số, ✔ code, ○ lý do |
| W2 | Ablation nhánh sync khó đọc | Bỏ sync: mismatch 0,989 → 0,690, AI? 0,947 → 0,922 | `no_sync` bỏ cả cost volume lẫn `content` | ● số, ✔ code |
| W3 | Tham chiếu người (FiLM) chưa cho thấy đóng góp | Bỏ ref: AI? 0,947 → 0,949 (nằm trong nhiễu 1 seed) | Giả thuyết: loss gây nhầm lẫn người nói làm tham chiếu mất thông tin người | ● số, ✔ code, ○ lý do |
| W4 | Điểm vận hành trôi | Ngưỡng FPR 1 % chọn trên val cho FPR 2,9 % ở test (M2 0,9 %, BL-M/sup 1,0 %) | AI? = max hai sigmoid chưa hiệu chuẩn; nhưng ngưỡng 1 % trên val chỉ do ~4 clip quyết định, 2,9 % có khoảng 1,5–4,3 % (mô phỏng) nên có thể chỉ là nhiễu | ● số, ○ lý do |
| W5 | Giọng lạ | Đã thấy 0,990, chưa thấy 0,889 | XLS-R-300M đóng băng, chỉ lớp 1–12, cache sẵn, chưa augment sóng âm; 4 hệ giọng khi train. Lối tắt cần kiểm: chỉ giọng AI bị co giãn (`stretch_ratio`): track A trong ±15 %, track B không giới hạn (sửa 30/10 sau review v007) | ● số, ✔ code, ○ lý do |
| W6 | Generator lạ | Đã thấy 0,995, chưa thấy 0,942; LatentSync 0,902 | 4 họ generator khi train; Scaling Laws: lỗi giảm khi thêm phương pháp giả vào train | ● số, ○ lý do |
| W7 | Chưa đo được định vị | AP@0,5 ≈ 0,02 | **Lỗi thước đo**: `proposals()` sinh đoạn theo phân vị của chính video, chấm đoạn bằng trung bình frame, AP tính trên mọi clip kể cả full; bộ chấm hoàn hảo trên dữ liệu tổng hợp chỉ được 0,009 (mọi clip) / 0,031 (partial) | ● số, ✔ code |
| W8 | M2 không có đặc trưng giọng | Sham 0,792; AI? 0,908 | M2 chỉ dùng Φ của AV-HuBERT; BL-M/sup (cùng Φ) chỉ đạt AUC audio 0,548. Đây là giới hạn phạm vi của M2 | ● số, ○ lý do |
| W9 | Bằng chứng mỏng | 1 seed, 1 part, 10 người test; val → test giảm ~0,04 | Mọi lựa chọn của v006 / v007 đã được thông tin bởi test part 1: so sánh mới trên đó là post-hoc | ● |

Hai điểm phụ: (1) lối tắt độ lệch tiếng–hình (trên dữ liệu trước v003, fake Wav2Lip lệch +2 frame còn real phần lớn +1; v003 đã sửa offset real và augment −1…+2, nhưng CNN 2D vẫn thấy vị trí đỉnh tuyệt đối; chưa kiểm trên v003 ○); (2) dừng sớm theo tổng val loss 3 đầu (việc mở #14).

---

## 2. v006 · M6 (v006): M1 + 6 bản vá

### 2.1. Ý chính

v006 giữ M1 làm model đề xuất. Những gì M1 đã chứng minh được giữ nguyên (hơn BL-M/sup +0,104 với CI > 0; giữ được cả ba câu hỏi cùng lúc: AI? 0,947, mismatch 0,989, sham 0,543). Mỗi thay đổi là một bản vá có tên (P1–P6), nhắm một bài học ở §1.5 và đo được riêng bằng ablation cộng dồn từ M1-v003. Trong code, M6 (v006) sẽ là `trihead_v3`, đăng ký cạnh `trihead_v2`; lớp `TriHeadV2` không bị sửa.

### 2.2. Bài học từ literature cho v006

Năm bài học (trang nguồn đã mở 28/10/2026; số là của paper, khác điều kiện dữ liệu mình):
1. Đặc trưng hình có "nghe" tiếng (AV-HuBERT V, BRAVEn V) tổng quát tốt nhất trong các đặc trưng SSL đã thử ([Boldisor et al. 2025](https://arxiv.org/abs/2511.17181): ngoài miền BRAVEn V 84,6 > AV-HuBERT V 80,0), nhưng phần lớn sức mạnh của AV-HuBERT nằm ở đặc trưng thô chứ không riêng ở sync ([AV-Lip-Sync+](https://arxiv.org/abs/2311.02733): chỉ đặc trưng 0,94 / 0,98; chỉ sync 0,95 / 0,95; cả hai 0,96 / 0,99).
2. Phát hiện giọng giả phụ thuộc mạnh vào hệ TTS ([VietPrism](https://arxiv.org/abs/2609.30005): cùng checkpoint AntiDeepfake bắt trọn VoxCPM2 nhưng EER 56 % với MiniMax; bản MMS-300M tệ nhất, EER trung bình 51,08 %); thêm dữ liệu đúng ngôn ngữ có bằng chứng rõ hơn đổi front-end ([Dao et al. 2026](https://arxiv.org/abs/2606.08669); [SEA-Spoof](https://arxiv.org/abs/2509.19865)).
3. Tổng quát sang generator lạ tăng chủ yếu khi thêm phương pháp giả vào train ([Scaling Laws](https://arxiv.org/abs/2510.16320): 1 − AUC ∝ N^−0,375); chọn checkpoint bằng val cùng miền với train là cách được khuyên ([DomainBed](https://arxiv.org/abs/2007.01434)).
4. Hiệu chuẩn từng kênh rồi lấy max chỉ hơn trung bình đơn giản một chút ([KLASSify](https://arxiv.org/abs/2508.07337): 92,78 so với 91,97), và v003 đã có cách này trong BL-Ens.
5. Tham chiếu người không cho thấy lợi rõ, kể cả trong paper gốc ([Referee](https://arxiv.org/abs/2510.27475): bản đủ 99,09, bỏ tham chiếu 99,33); chuẩn hoá điểm đơn điệu không đổi EER, chỉ dời điểm vận hành ([Zhou & Wang 2026](https://arxiv.org/abs/2606.21584)).

Không đưa vào v006: MLLM (fine-tune 7B không vừa T4), X-AVDT (~1 phút / clip), DF-MoE (424 M tham số), so nội dung ASR với đọc môi (chưa có model đọc môi tiếng Việt; xem §3.13 về ViCocktail).

### 2.3. Sáu bản vá

| Bản vá | Thay đổi so với M1-v003 | Nhắm | Căn cứ, rủi ro | Nấc |
|---|---|---|---|---|
| **P1 · Sync thuần** | Bỏ đường `content` (Linear trên [a, v] AV-HuBERT thô) khỏi nhánh S; giữ cost volume ±15 + CNN 2D | W2 | Ablation đọc được. Rủi ro (AV-Lip-Sync+): AI? có thể giảm nếu P2 không bù đủ | 1 |
| **P2 · AV-HuBERT hình sang nhánh V** | Đặc trưng AV-HuBERT hình (chỉ `v`) vào nhánh V cùng trace (ảnh, NPR, residual thời gian) + động học môi, tư thế đầu | W1 | Boldisor et al. P2 không tự chứng minh được trace: phải đo trace riêng trên generator chưa thấy và CRF 35 / 40 | 1 |
| **P3 · Mỗi đầu một nhánh** | visual ← V, audio ← A, mismatch ← S | W2 | Rẻ; mỗi ablation chỉ ảnh hưởng đầu của nó | 1 |
| **P4 · Tham chiếu thành chuẩn hoá điểm** | Bỏ FiLM. Điểm từng đầu z-score theo real val, rồi trừ trung bình điểm của 3 clip real tham chiếu cùng người (không chia độ lệch chuẩn của 3 giá trị). AI? = max(z_visual, z_audio) | W3, W4 | `auc_ref_norm` đã có trong `metrics.json` của v003 (hàm `ref_norm` chia cả độ lệch chuẩn ✔) → kiểm trên output v003 trước khi train. Zhou & Wang: không kỳ vọng tăng AUC trong từng người, chỉ ổn định điểm vận hành | 0 rồi 1 |
| **P5 · Front-end giọng** | Chỉ khi kiểm zero-shot trên val ủng hộ: AntiDeepfake XLS-R-1B (2B nếu đủ chỗ cache) đóng băng thay XLS-R-300M, kèm 1–2 bản augment codec cache cho cả real và fake. Không dùng MMS-300M. Song song: thêm dữ liệu giả tiếng Việt bên ngoài cho nhánh A | W5 | AntiDeepfake chưa được kiểm ở chế độ đóng băng; VietPrism (phụ thuộc hệ TTS); Dao et al. và SEA-Spoof (dữ liệu đúng ngôn ngữ) | 2 |
| **P6 · Thước đo định vị** | Hàm đánh giá **mới**, không sửa `proposals()`: AP trên partial + real, đoạn = khúc vượt ngưỡng cố định chọn trên val, thêm đoạn cả clip khi điểm video cao. Chỉ inference lại `best.pt` của M1-v003 (vì `predictions-*.npz` chỉ lưu điểm video ✔). Chỉ thêm đầu biên nếu M1 vẫn kém sau khi đo đúng | W7 | Mô phỏng của vòng review; mốc [LoCC](https://arxiv.org/abs/2606.22772) (AV-1M AP@0,5 41,2), UMMAFormer giám sát đủ AV-1M mAP@0,5 90,97 ([WMMT](https://arxiv.org/abs/2508.02179)) | 0 |

**Giữ nguyên (bản sắc của M1):** ba câu hỏi visual / audio / mismatch và AI? = max; ② sham và pseudo-fake thay đoạn tiếng p = 0,25 (mismatch = 1, AI = 0); cost volume ±15 + CNN 2D; loss gây nhầm lẫn người nói trên V và A; gộp top-5 frame; augment dịch tiếng −1…+2; augment miệng.

**Hoãn (chưa có bằng chứng trên dữ liệu mình):** loss one-class, loss xếp hạng theo cặp, kênh Φ của AVH-Align trong S, chỉnh độ lệch theo trung vị (chỉ khi kiểm thấy lối tắt, và khi đó dịch cả cost volume), đầu biên đoạn, val "giả chưa thấy", ô đối chứng hình, pseudo-fake hình kiểu SAVe, BRAVEn V.

### 2.4. Sơ đồ M6 (v006)

```
mouth crop 96×96 ─┐
lips, pose ───────┤ nhánh V: [ảnh, NPR, residual thời gian] → Conv3D        ─→ TCN ─→ đầu visual
v (AV-HuBERT) ────┘          ‖ AV-HuBERT v (P2) ‖ động học môi + tư thế đầu

a, v (AV-HuBERT) ── nhánh S: cost volume ±15 (chiếu học được + cosine) → CNN 2D
                    → độ lệch, đỉnh, entropy (P1: bỏ đường content)  ─→ TCN ─→ đầu mismatch

ssl (XLS-R) ─────── nhánh A: 3 nhóm lớp, trộn softmax → 2 lớp attention     ─→ đầu audio
                    (P5: AntiDeepfake đóng băng nếu nấc 0 ủng hộ)

Sau mô hình (P4): z = (điểm − μ_real val) / σ_real val, rồi trừ trung bình z của 3 clip real
tham chiếu cùng người. AI? = max(z_visual, z_audio). Định vị: thước đo P6.
```

Sau P1–P3 mỗi nhánh nuôi đúng một đầu, nên bỏ một nhánh chỉ ảnh hưởng đầu của nó. P5 chỉ đổi phần bên trong nhánh A, P6 chỉ đổi thước đo.

### 2.5. M2 trong v006

Giữ nguyên kiến trúc AVH-Align+ và vai đề xuất rẻ, chỉ dùng AV-HuBERT. Chỉ chạy lại theo protocol v006 (3 seed, test part 2) để so công bằng. Nếu muốn nâng M2 thì chỉ nhận bản vá không đổi kiến trúc (P4 và dừng sớm theo từng đầu). W8 ghi là giới hạn phạm vi của M2.

### 2.6. Kiểm tra cho P1: giữ sync, đổi hình

Sync thuần hợp lệ vẫn bắt được môi do generator vẽ (AVH-Align 85,2 AUC trên AV-1M; `peak_sync_001` không train đạt 0,875 trên Wav2Lip của nhóm ●), nên điểm cao của S trên ④ ⑤ không chứng minh S đang nhìn dấu vết hình. Phép thử đúng: lấy ~100 clip real của val, làm mờ / JPEG / thu nhỏ vùng miệng, trích lại AV-HuBERT (một job GPU nhỏ) và xem điểm mismatch của S có tăng không.

### 2.7. Lưu ý file bị hash

Mọi `src/vn_av_df/training/*.py` và `features/encoders.py` được hash vào provenance của run (`train.py:589-590` ✔). Thêm `trihead_v3`, hàm thước đo mới hay encoder AntiDeepfake đều đổi hash của run mới. Vì vậy chỉ thêm lớp và hàm mới, không sửa `TriHeadV2` hay `proposals()` tại chỗ, để run v003 vẫn đánh giá lại được.

### 2.8. Thứ tự làm (dừng ở nấc nào không đạt)

So sánh chính là **M6 (v006) − M1-v003 cùng protocol**: cùng dữ liệu, cùng split, cùng seed. Thay đổi kiến trúc và thay đổi dữ liệu không đi chung một bước.

1. **Nấc 0, không cần GPU** (cần output v003 của Linh: `predictions-val.npz`, `predictions-test.npz`, `history.json`, `metrics.json`): AI? = max(z_visual, z_audio) theo real val; đọc `auc_ref_norm` và thử chuẩn hoá chỉ trừ trung bình (P4); phân rã AUC tác vụ visual / audio theo generator, giọng, full / partial để điền cột M1-v003 của G2, G3; kiểm lối tắt co giãn (AUC tác vụ audio theo mức `|stretch_ratio − 1|`) và lối tắt độ lệch (độ trễ có cosine cao nhất từ cache AV-HuBERT, theo ô × generator).
2. **Nấc 0, chỉ inference**: P6 (inference lại `best.pt` của M1-v003); zero-shot AntiDeepfake XLS-R-1B / 2B trên audio val (cần env fairseq riêng).
3. **Nấc 1, train trên part 1**: M6 (v006) với P1–P4, cùng dữ liệu, cách dừng sớm và seed 42 như M1-v003, nên so được trên val mà không cần train lại M1. Ablation cộng dồn thêm 2 lần train: +P1 và +P1–P3.
4. **Nấc 2, chỉ khi nấc 0 ủng hộ**: P5.
5. **Bước kết**: train M1-v003, M6 (v006), M2 và các baseline trên part 1 + 2 cùng protocol, 3 seed, mở test part 2 một lần.

**Dữ liệu** (không sinh lại fake): part 1 + 2 với split mới là một phiên bản dữ liệu mới, không ghi đè `vn-av-df-data-part1`, ghi provenance, fake theo split của clip real. Validation giữ cùng generator và giọng với train như v003 (bản trước đề xuất giữ VideoReTalking + Seed-VC ngoài train để chọn model; đã bỏ theo DomainBed và vì Seed-VC là hệ chuyển giọng duy nhất khi train).

**Protocol**: test khoá là test part 2 (10 người mới, chưa ai xem). Test part 1 chỉ để so với số v003, ghi là post-hoc. 3 seed (42–44). CI bootstrap theo người kèm Δ của từng người (10 giá trị), vì bootstrap với 10 cụm thường cho CI hẹp hơn thực tế; có 3 seed thì lấy trung bình điểm qua seed trước khi bootstrap. Báo TPR ở FPR 1 % và 5 % cạnh AUC. Thêm bản test nén lại CRF 35 / 40 (không train lại). Lúc test có dùng 3 clip real tham chiếu của chính người đó (enrollment): ghi rõ trong báo cáo.

### 2.9. Cổng đặt trước (ngưỡng là đề xuất, team chốt trước khi mở test)

| Cổng | So sánh | Tác vụ · tập | Đạt khi | M1-v003 (part 1) |
|---|---|---|---|---|
| **G1 (chính)** | M6 (v006) − M1-v003 | AI? · toàn bộ test | cận dưới CI 95 % của Δ ≥ −0,01 | 0,947 |
| G2 | M6 (v006) − M1-v003 | visual · LatentSync, Diff2Lip | Δ ≥ +0,02 và cận dưới CI > 0 | chưa có (lấy AUC tác vụ visual ở nấc 0) |
| G3 | M6 (v006) − M1-v003 | audio · VoxCPM2, viXTTS, F5 inpainting | Δ ≥ +0,03 và cận dưới CI > 0 | chưa có (lấy AUC tác vụ audio ở nấc 0) |
| G4 | M6 (v006) | sham so với real | AUC ≤ 0,60 | 0,543 |
| G5 | M6 (v006) so với M1-v003 | FPR test ở ngưỡng FPR 1 % và 5 % chọn trên val, CI Clopper–Pearson; phân tán FPR theo người | phân tán giảm, FPR không tệ hơn | 2,9 % ở ngưỡng 1 % |
| G6 | M6 (v006) so với M1-v003 | định vị, thước đo P6 | đặt sau khi đo lại M1-v003 bằng thước đo mới | chưa đo đúng |
| G7 | M6 (v006), nhánh S | giữ sync, đổi hình (real val làm mờ / JPEG miệng) | điểm mismatch gần như không đổi | chưa đo |

**Ablation**: cộng dồn từ M1-v003 như ở nấc 1, mỗi bước ≥ 2 seed nếu đủ quota (AUC val của M1 dao động 0,969–0,986 giữa các epoch nên chênh 0,002 nằm trong nhiễu). Bỏ-từng-khối chỉ cho P1 và P4. Trace (W1) đo riêng trên generator chưa thấy và CRF 35 / 40.

### 2.10. Rủi ro, chi phí, quyết định

- Rủi ro lớn nhất là phần giọng: không front-end nào thắng mọi hệ TTS (VietPrism). P5 chỉ làm khi zero-shot trên val ủng hộ.
- P1 có thể làm giảm AI? (AV-Lip-Sync+), P2 phải bù đủ; ablation cộng dồn ở nấc 1 cho thấy ngay.
- AntiDeepfake: giấy phép CC BY-NC-SA 4.0 (dùng cho capstone và demo phi thương mại được, phải ghi nguồn); checkpoint cần fairseq 0.12.2 / Python 3.9 → worker env riêng, encoder mới trong `features/encoders.py`, sửa `tools/make_notebooks.py`.
- AV-HuBERT vẫn là bản pretrain tiếng Anh (X3, v006 không giải).
- Chi phí (○): nấc 0 không cần GPU (trừ zero-shot AntiDeepfake và inference lại M1-v003). Nấc 1: 3 lần train × ~4 giờ T4 ≈ 12 giờ (1 seed). Bước kết trên part 1 + 2: M1-v003, M6 (v006), M2 và 4 baseline × 3 seed + ablation ≈ 100–140 giờ GPU, tức 3–5 tuần quota Kaggle (30 giờ / tuần / tài khoản). Cache AntiDeepfake XLS-R-1B part 1 với 2 bản augment ≈ 17 GB, sát giới hạn Output 20 GB của Kaggle; XLS-R-2B vượt và không trích được trên RTX 4050 6 GB.

| Quyết định team | Đề xuất | Vì sao |
|---|---|---|
| M6 (v006) = M1 + P1–P6; M2 và baseline giữ vai như v003 | Đồng ý | Đúng vai đã chốt; mỗi bản vá đo được so với M1-v003 |
| Làm nấc 0 trước mọi lần train | Đồng ý | Nhiều giả thuyết kiểm được không cần GPU; cần output v003 của Linh |
| Bỏ FiLM, chuyển sang chuẩn hoá điểm (P4) | Đồng ý nếu nấc 0 xác nhận | Bỏ ref không đổi kết quả |
| Đổi front-end giọng (P5) | Chỉ khi zero-shot trên val ủng hộ | Không front-end nào thắng mọi hệ TTS |
| Khoá test part 2 | Đồng ý | Test part 1 đã được xem và đã định hướng v006 |
| Sửa BL-A trước khi so trên part 2 | Đồng ý | BL-A gần 0,5 |
| Ô đối chứng hình, pseudo-fake hình kiểu SAVe | Hoãn | Đổi bước sinh; theo CLAUDE.md pseudo-fake cần team đồng ý; riêng LipBlend chỉ 0,48 ở c40 |

### 2.11. Vòng review đã sửa gì (tóm tắt)

Đóng khung lại (M1 + bản vá, không "một model mới thay M1"); ghi đúng +0,104 của M1 so với BL-M/sup; AP@0,5 ≈ 0,02 là lỗi thước đo (P6 sửa thước đo trước, bỏ đầu định vị mới); thay G7 cũ (sẽ trượt oan) bằng phép thử giữ sync đổi hình; P5 chỉ sau zero-shot; bỏ val "giả chưa thấy"; hoãn loss xếp hạng (cặp ② / ③ khác cả lời lẫn co giãn); G1 dùng cận dưới CI, G2 / G3 dùng AUC tác vụ, G5 thêm CI và FPR 5 %; sửa trích dẫn và thêm 11 công trình; thêm cảnh báo file bị hash; chia 3 nấc, ablation cộng dồn.

---

## 3. v007 · M7 (v007) "ViSyl": detector neo theo âm tiết tiếng Việt

> **Bản trước review.** Bản hiện hành là `plans/2026-10-30_v007_sau-review.md`; review tìm 2 lỗi chặn ở mục này (phép thử định vị H-A và pilot khép môi 0c).

### 3.1. v006 để lại gì

| Giới hạn | v006 xử lý? | v007 nhắm? |
|---|---|---|
| X3: AV-HuBERT pretrain tiếng Anh, thanh điệu không thấy trên môi | Không (ghi rõ là không giải) | Có: luật khẩu hình tiếng Việt (§3.3) và, có điều kiện, encoder AV-HuBERT tinh chỉnh tiếng Việt (§3.8, nấc 2) |
| W6: generator lạ (LatentSync 0,902) | Gián tiếp (P2, đo trace) | Có, bằng một tín hiệu không học từ generator: môi phải khép ở các âm tiết mà luật tiếng Việt bắt khép |
| W7: định vị | Sửa thước đo (P6) | Có: đơn vị âm tiết cho biên đoạn trùng biên từ; đoạn ứng viên = dãy âm tiết |
| Giải thích kết quả cho người dùng (demo) | Không | Có: lý do theo âm tiết |
| Chi phí train | ~4 giờ / lần | ~0,3–0,6 giờ / lần (○) |
| W5: giọng lạ | P5 (có điều kiện) | Không nhắm riêng; nhánh tiếng như M1 (XLS-R đóng băng). Nếu P5 thành công thì M7 dùng lại được |

### 3.2. Hai giả thuyết của v007 (đều có thể bác bỏ)

- **H-A · Neo theo âm tiết.** Phân tích theo âm tiết (đơn vị mà một lần sửa lời thay thế: tên, số, "có" ↔ "không") cho biên đoạn chính xác hơn và đỡ loãng hơn với đoạn giả ngắn (0,3 s ≈ 1–2 âm tiết) so với phân tích theo frame. Bằng chứng gần nhất: [WAFL](https://arxiv.org/abs/2603.06220) (định vị neo theo từ, AP@0,5 LAV-DF 99,76, AV-1M 97,26; chéo bộ AP@0,95 44,89 so với 0,69 của AuViRe). Bác bỏ khi: M7 không hơn bản cửa sổ đều 0,2 s (`visyl_uniform`) trên định vị (G5).
- **H-B · Kỳ vọng khẩu hình tiếng Việt.** Chính tả tiếng Việt cho biết gần như chắc chắn âm tiết nào bắt môi khép (đầu b / m / p; cuối -m / -p; cuối -ng / -c sau u, ô, o). Người thật khép môi đúng các chỗ này; generator lip-sync, học chủ yếu trên tiếng Anh và tiếng Trung, có thể không vẽ khép ở các chỗ riêng của tiếng Việt (nhất là [ŋ͡m], [k͡p̚] như trong "không", "ông", "học", "cũng"). Bằng chứng gần nhất: [Agarwal et al. CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w39/Agarwal_Detecting_Deep-Fake_Videos_From_Phoneme-Viseme_Mismatches_CVPRW_2020_paper.html) (nhiều deepfake không khép môi ở M, B, P), [PIA](https://arxiv.org/abs/2510.14241) (luồng hình theo âm vị quan trọng nhất: bỏ đi AUC 98,06 → 65,57 trên DeepSpeak v2; **chỉ tiếng Anh**). Bác bỏ khi: pilot nấc 0 (§3.8) không thấy real khép môi rõ hơn fake ở các âm tiết đó, hoặc M7 không hơn bản bỏ kỳ vọng (`visyl_no_phon`) trên lát có âm tiết bắt khép (G4).

### 3.3. Cơ sở âm vị học: luật chính tả → kỳ vọng khẩu hình

Nguồn: [Vietnamese phonology (Wikipedia)](https://en.wikipedia.org/wiki/Vietnamese_phonology), đã mở 30/10/2026. Câu chính: "When the velar consonants /k, ŋ/ are after /u, o, ɔ/, they are articulated with a simultaneous bilabial closure"; Hà Nội: ong, oc → [ăwŋ͡m], [ăwk͡p̚], ung, uc → [ʊŋ͡m], [ʊk͡p̚]; Sài Gòn: "After rounded vowels /aw, u, o/, many speakers close their lips, i.e. they pronounce /k, ŋ/ as [k͡p, ŋ͡m]"; phụ âm tắc cuối /p, t, k/ không bật hơi ([p̚, t̚, k̚]).

| Vị trí | Chữ viết | Kỳ vọng khẩu hình | Thẻ |
|---|---|---|---|
| Âm đầu | b, m, p | hai môi khép rồi mở | `onset = bilabial` |
| Âm đầu | ph, v | môi dưới chạm răng trên, độ mở nhỏ | `onset = labiodental` |
| Âm đầu / âm đệm | qu; o trước a, ă, e (oa, oă, oe); u trước â, ê, y, ơ (uâ, uê, uy, uơ) | tròn môi ở đầu vần | `round_on = 1` |
| Âm chính | u, ô, o (gồm cả uô ở đầu vần) | tròn môi | `round_nuc = 1` |
| Âm cuối | m, p | **khép hai môi ở cuối âm tiết** | `coda = bilabial` |
| Âm cuối | ng, c sau u, ô, o (vần ung, uc, ông, ôc, ong, oc; không gồm uông, uôc, ương, ươc) | **khép môi ở cuối âm tiết** ([ŋ͡m], [k͡p̚]) | `coda = labial_velar` |
| Âm cuối | u, o (bán nguyên âm: au, ao, âu, êu, iu, ưu, eo…) | tròn môi ở cuối | `coda = round_glide` |
| Âm cuối | n, t, nh, ch, i, y; ng, c sau nguyên âm khác | không ràng buộc môi | `coda = other` |
| Thanh | không dấu, huyền, sắc, hỏi, ngã, nặng | không thấy trên môi; chỉ dùng phía tiếng | `tone ∈ 6 lớp` |
| Khác | số, chữ nước ngoài, ASR không tách được | không suy kỳ vọng | `type = unknown` |

- Luật viết tay ~40 dòng (tách âm đầu dài nhất, âm cuối, tách dấu thanh sau chuẩn hoá Unicode NFC), có test trên vài chục từ. Có thể đối chiếu với [viphoneme](https://pypi.org/project/viphoneme/) hoặc [underthesea IPA](https://underthesea.readthedocs.io/en/latest/_modules/underthesea/pipeline/ipa.html), nhưng không cần thêm thư viện.
- Đây là thẻ **phân lớp thô** (khép / tròn / không ràng buộc), không phải bản đồ viseme đầy đủ. Bản đồ âm vị → viseme khác nhau nhiều giữa các nghiên cứu và giữa người nói ([Bear & Harvey, "Phoneme-to-viseme mappings: the good, the bad, and the ugly"](https://ueaeprints.uea.ac.uk/id/eprint/64350/)); khép môi là lớp ổn định nhất.
- Thẻ được tính từ **lời ASR của chính tiếng trong mẫu** lúc train lẫn lúc test. Không dùng transcript, `edit` hay span lưu ở bước sinh (đó là rò nhãn).
- Biến thể phương ngữ: Sài Gòn ghi "many speakers", nên kỳ vọng [ŋ͡m] là xác suất chứ không tuyệt đối; model học mức độ từ real. Thẻ thanh không tạo phép kiểm "đúng / sai thanh" vì ASR suy thanh từ chính tiếng đó (vòng tròn); nó chỉ cho nhánh tiếng biết âm tiết đang mang thanh nào.

### 3.4. Công trình gần nhất và phần mới

| Khối | Gần nhất đã có | Phần của v007 (chưa tìm thấy ở nơi khác, tính đến 30/10/2026) |
|---|---|---|
| Âm vị ↔ khẩu hình | Agarwal et al. CVPRW 2020 (M, B, P, tiếng Anh); [NPVForensics](https://arxiv.org/abs/2306.06885) (2023: âm vị "không then chốt", tiếng Anh, pretrain real); PIA (ICCVW 2025: WhisperX + phonemizer, tiếng Anh) | luật khép môi riêng của tiếng Việt (phụ âm cuối môi-ngạc mềm), suy từ chính tả, không cần G2P học máy |
| Neo theo từ | WAFL (03/2026: token từ, LoRA r = 8 trên VideoMAE / wav2vec 2.0 đóng băng, 2,54 M tham số học) | âm tiết tiếng Việt + khoảng lặng, kèm ② sham và 3 đầu ra visual / audio / mismatch |
| Động học môi | [BioLip / LADDER](https://arxiv.org/abs/2604.16808) (landmark môi, 4 bậc động học; LipSyncTIMIT zero-shot 0,927, IP-LAP 0,82); [LipDA](https://arxiv.org/abs/2610.08417) (ICML 2026, môi ↔ tư thế đầu, code công bố) | độ mở môi đo đúng ở đầu / cuối âm tiết, có điều kiện theo thẻ kỳ vọng |
| Đa ngôn ngữ | [BabelFake](https://arxiv.org/abs/2610.06339) (10/2026, 5 ngôn ngữ châu Âu, **không có tiếng Việt**; SpeechForensics ổn định nhất giữa các ngôn ngữ, 77–84) | thiết kế riêng cho một ngôn ngữ thanh điệu, đơn âm tiết |
| Encoder AV tiếng Việt | [ViCocktail](https://arxiv.org/abs/2506.04635) (Interspeech 2025: AV-HuBERT Large tinh chỉnh AVSR trên 269 giờ tiếng Việt, 2.375 người) | dùng làm encoder cho phát hiện giả (nấc 2, có điều kiện giấy phép) |

Báo cáo nên nói "kết hợp có lý do, mỗi khối kiểm bằng ablation", không nói "kiến trúc mới hoàn toàn" (cùng giọng với `docs/v003_model.md` §5.0).

### 3.5. Kiến trúc M7 (v007)

#### 3.5.1. Tách đơn vị

1. Chạy **PhoWhisper-small** (`vinai/PhoWhisper-small`, BSD-3-Clause; đã có trong pipeline: `src/vn_av_df/generation/voice/asr.py`, `return_timestamps="word"` ✔) trên tiếng 16 kHz của **chính mẫu** → danh sách `{word, start, end}`. Tiếng Việt viết cách theo âm tiết, nên mỗi "word" là một âm tiết.
2. Đổi sang lưới 25 Hz: `s = round(start·25)`, `e = max(s + 1, round(end·25))`; cắt về [0, T); chồng lấn thì chia ở điểm giữa.
3. Frame không thuộc âm tiết nào thành **đơn vị khoảng lặng** (`type = gap`); khoảng lặng dài hơn 12 frame (0,48 s) chia thành khúc ≤ 12 frame. Mọi frame thuộc đúng một đơn vị, nên đoạn giả nằm trong khoảng lặng (④ partial rơi vào chỗ ngừng) vẫn được phủ.
4. Mỗi âm tiết gắn thẻ ở §3.3. Tối đa 64 đơn vị / clip (clip 7 s nói nhanh ~6 âm tiết/s ≈ 42 âm tiết + khoảng lặng).
5. Nếu ASR không nhận được gì: cả clip chia thành khúc 0,2 s `type = unknown` (đường dự phòng, báo riêng ở G8).

Bản ablation `visyl_uniform`: đơn vị = cửa sổ đều 5 frame (0,2 s, gần độ dài trung bình một âm tiết), thẻ lấy theo âm tiết chồng lấn nhiều nhất; mọi thứ khác giữ nguyên.

#### 3.5.2. Đặc trưng mỗi đơn vị (frame [s − 2, e + 2), bỏ frame không hợp lệ)

| Luồng | Từ cache | Cách gộp | Chiều |
|---|---|---|---|
| Hình nói | `v` AV-HuBERT (1024) | Linear 1024 → 128, attention pooling (query học được, softmax có mask) | 128 |
| Tiếng nói | `a` AV-HuBERT (1024) | như trên | 128 |
| Tiếng (giả giọng) | `ssl` XLS-R [T, 3, 1024] | trộn 3 nhóm lớp bằng softmax học được (như `AudioTrunk`), Linear → 128, attention pooling | 128 |
| Sync cục bộ | `a`, `v` | hồ sơ cosine theo độ trễ cos(a_t, v_t+δ), δ ∈ [−15, 15], **không tham số**, trung bình trên frame của đơn vị; thêm giá trị đỉnh, vị trí đỉnh / 15, entropy | 34 |
| Hình học môi | landmark dlib 68 điểm thô (khoá `landmarks` trong cache `avhubert` ✔, cùng file với `a`, `v`) | độ mở môi trong (trung bình khoảng cách dọc 61–67, 62–66, 63–65) chia bề rộng miệng (48–54); bề rộng miệng chia khoảng cách hai khoé mắt ngoài (36–45) (đo tròn môi); trong đơn vị: min, max, trung bình độ mở; độ mở ở 2 frame đầu và 2 frame cuối (+1 frame sau, chịu sai số mốc); tỉ lệ frame "khép" (độ mở < τ, τ = phân vị 5 % độ mở trên frame có tiếng của real train, tính một lần); bề rộng trung bình; vận tốc độ mở; vận tốc tư thế đầu | ~14 |
| Kỳ vọng | thẻ §3.3 | embedding 8 chiều cho mỗi thẻ (`onset`, `coda`, `round_on`, `round_nuc`, `tone`, `type`) | ~48 |
| Vị trí | | độ dài (s), vị trí tương đối trong clip | 2 |

Ghép (~480 chiều) → Linear → 128 (LayerNorm, GELU) + embedding thời gian hình sin theo tâm đơn vị.

#### 3.5.3. Mạng

- 2 lớp Transformer encoder trên chuỗi đơn vị (d = 128, 4 head, FFN 256, dropout 0,1, pre-norm, mask đệm). Mỗi đơn vị thấy ngữ cảnh cả câu: một âm tiết bất thường được so với các âm tiết khác của cùng người trong cùng clip.
- 3 đầu MLP (128 → 64 → 1) trên mỗi đơn vị: **visual**, **audio**, **mismatch**.
- Thiết kế **gộp** có chủ đích (khác P3 của v006): phép so "kỳ vọng từ tiếng ↔ quan sát trên môi" cần cả hai phía trong cùng một biểu diễn. Các đầu được tách bằng nhãn chứ không bằng dây nối: ② sham (lệch nhưng AI = 0) và ③ voice (môi thật, visual = 0) dạy đầu visual rằng "môi không khớp lời" chưa chắc là môi giả. Rủi ro nhầm sham được kiểm bằng G6.
- Tham số học ước ~0,75 M (○): 3 attention pooling ~0,39 M, Transformer ~0,26 M, chiếu đầu vào ~0,06 M, 3 đầu ~0,03 M.

#### 3.5.4. Đầu ra, gộp, định vị, giải thích

- Điểm mỗi đơn vị được **gán cho mọi frame của đơn vị đó**, nên M7 trả đúng giao diện hiện có của `Model` (logit theo frame [B, T] cho từng đầu). Mọi phần train / evaluate / report / BL-Ens dùng nguyên, không viết loss hay thước đo riêng.
- Gộp video: top-5 frame như M1 (vì điểm theo đơn vị nên ≈ đơn vị điểm cao nhất).
- AI? = max(z_visual, z_audio) với z theo real val, cộng chuẩn hoá theo 3 clip tham chiếu (đúng P4 của v006).
- Định vị: thước đo P6 của v006 áp nguyên; đoạn ứng viên tự nhiên là dãy đơn vị liên tiếp vượt ngưỡng, biên đoạn trùng biên âm tiết.
- Giải thích cho demo (Gradio tab 2): với mỗi đơn vị điểm cao, in lời ASR, thời gian, điểm 3 đầu, và bằng chứng đo tay khi có, ví dụ: "âm tiết 'không' 2,31–2,58 s · kỳ vọng khép môi cuối âm tiết [ŋ͡m] · độ mở nhỏ nhất 0,21 (real cùng người: 0,04)". Bằng chứng này là số đo landmark, không phải trọng số model; ghi rõ như vậy trên giao diện.

#### 3.5.5. Loss và huấn luyện

- Loss mặc định của `Model`: BCE theo frame (trên logit đã gán theo đơn vị) + BCE theo video cho từng đầu, `pos_weights` như v003. Vì logit frame được gán theo đơn vị, BCE theo frame tương đương BCE theo đơn vị có trọng số theo độ dài đơn vị.
- **Không dùng pseudo-fake thay đoạn tiếng** (`pseudo_fake = 0`, khác M1): thay đặc trưng tiếng trong không gian đặc trưng sẽ làm thẻ ASR (tính từ tiếng gốc) không còn khớp với tiếng mới. Ví dụ "lệch nhưng không AI" đã có thật trong dữ liệu: ② sham full và partial.
- Augment dịch tiếng −1…+2 frame giữ như v003; **mốc đơn vị dịch theo tiếng** (đơn vị neo vào tiếng vì lấy từ ASR), độ mở môi tính trên frame hình tương ứng.
- Train trên cả clip (`crop_frames = None`, clip ≤ 175 frame), không cắt 75 frame như M1, vì đơn vị cần trọn âm tiết và ngữ cảnh câu.
- Cân theo người (`speaker_balance`), AdamW lr 3e-4, weight decay 1e-4, batch 16, tối đa 30 epoch, patience 5, chọn checkpoint theo val loss: giống M1 để so công bằng. Không loss gây nhầm lẫn người nói (M7 không có nhánh trace; W3 cho thấy cơ chế người của M1 chưa chứng minh được đóng góp).

#### 3.5.6. Sơ đồ

```
tiếng của mẫu ── PhoWhisper ─→ âm tiết {lời, start, end} ─→ luật chính tả ─→ thẻ kỳ vọng
                                      │                                          │
                                      ▼                                          │
cache (25 Hz):                 chia frame thành đơn vị                           │
  v  (AV-HuBERT hình) ─┐       (âm tiết + khoảng lặng)                           │
  a  (AV-HuBERT tiếng) ┤                │                                         │
  ssl (XLS-R 3 nhóm)  ─┤──→ gộp theo đơn vị: attention pool v, a, ssl            │
  landmark 68 điểm    ─┤                 + hồ sơ cosine theo độ trễ ±15          │
  pose                ─┘                 + độ mở môi đầu / cuối, tròn môi        │
                                         + embedding thẻ  ←──────────────────────┘
                                                │
                                   2 lớp Transformer trên chuỗi đơn vị
                                                │
                          ┌─────────────────────┼─────────────────────┐
                       visual                 audio               mismatch     (mỗi đơn vị)
                          └──── gán về frame ───┴──── top-5 → điểm video ─────┘
                       AI? = max(z_visual, z_audio) · định vị = dãy đơn vị vượt ngưỡng (P6)
```

### 3.6. M7 khác M6 (v006) ở đâu

| Thành phần | M6 (v006) | M7 (v007) |
|---|---|---|
| Đơn vị | frame 40 ms, TCN (trường nhìn ~21–29 frame) | âm tiết + khoảng lặng, Transformer trên cả câu |
| Sync | cost volume ±15 có chiếu học được + CNN 2D + soft-argmin | hồ sơ cosine ±15 không tham số, gộp theo đơn vị; so khớp do Transformer học |
| Hình | crop miệng (ảnh, NPR, residual thời gian) + AV-HuBERT v + động học môi | AV-HuBERT v + độ mở môi đo đúng đầu / cuối âm tiết; không crop |
| Kiến thức ngôn ngữ | không | thẻ kỳ vọng khẩu hình từ chính tả tiếng Việt; thẻ thanh cho phía tiếng |
| Nối đầu ra | mỗi đầu một nhánh (P3) | gộp, tách bằng nhãn |
| Tham chiếu người | chuẩn hoá điểm (P4) | chuẩn hoá điểm (P4), dùng chung |
| Pseudo-fake | thay đoạn tiếng p = 0,25 | không (giữ ② sham thật) |
| Loss người nói | có (V, A) | không |
| Cắt khi train | 75 frame | cả clip |
| Giải thích | không | theo âm tiết |
| Cache cần | `av`, `ssl`, `mouth` (crop, môi, tư thế đầu) | `av` (kèm landmark thô), `ssl`, tư thế đầu (đọc từ file cache `mouth`, không đọc crop), **`syll` mới** |

### 3.7. Dữ liệu, cache mới, code

- **Không sinh lại fake, không đổi split.** Dùng part 1 (nấc 0, 1) và part 1 + 2 (bước kết) đúng như v006.
- **Cache mới `syll`** (một encoder mới, theo đúng mẫu npz + json, lưới 25 Hz, `feature_signature` gồm tên model ASR, phiên bản luật, tham số chia khoảng lặng): danh sách đơn vị (frame đầu, frame cuối, loại, lời, các thẻ). Chạy trong env chính (transformers), không cần env mới. Ước (○): PhoWhisper-small trên T4 ~0,2–0,5 s / clip 5 s → ~1–2 giờ GPU cho ~15,4 nghìn mẫu của part 1 + 2; trên CPU chậm hơn ~10 lần. Dung lượng không đáng kể (vài KB / mẫu).
- **Code** (chỉ thêm, theo §2.7): lớp `ViSyl` với 4 cấu hình trong `models.py` (bản chính `visyl` và 3 ablation `visyl_no_phon`, `visyl_uniform`, `visyl_uniform_no_phon`); nạp `syll` và landmark trong `data.py` (need mới `syll`); encoder `syll` trong `features/encoders.py` (hoặc stage riêng trong `prepare.py`); hàm luật chính tả + test trong `tests/`; notebook sinh lại bằng `tools/make_notebooks.py`. Tất cả đổi hash của run mới, không đụng run v003.
- **Thời gian suy luận thêm** ở demo: một lần PhoWhisper-small cho mỗi clip.

### 3.8. Thứ tự làm (dừng ở nấc nào không đạt)

**Nấc 0, không train (gần như không cần GPU; chỉ dùng train + val, không đụng test):**

- **0a · Cache âm tiết** cho train + val part 1 (~8,3 nghìn mẫu, ~0,5–1 giờ GPU ○).
- **0b · Độ chính xác mốc âm tiết.** Ô ③ track B có sẵn "đáp án" biên: cụm TTS chèn vào nằm đúng `partial_span` (đã co giãn vừa khít, `generate.py` ✔). So biên của các âm tiết ASR thuộc cụm đó với `partial_span`. Đạt khi trung vị sai số ≤ 2 frame (80 ms). Không đạt thì thêm căn chỉnh CTC ép buộc bằng [wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) (95 M, CC BY-NC 4.0, dùng phi thương mại được) trên lời của PhoWhisper.
- **0c · Pilot "môi có khép đúng luật không" (cổng đi / dừng của H-B).** Trên ① real và ④ source full (cùng tiếng, cùng lời với clip real gốc, chỉ khác môi) theo từng generator đã thấy:
  1. Real: độ mở môi ở cuối âm tiết `coda ∈ {bilabial, labial_velar}` so với âm tiết đối chứng (`coda = other`, cùng người). Đạt khi AUC phân biệt ≥ 0,70: landmark 25 Hz đủ thấy cử động khép.
  2. Fake: so **từng cặp âm tiết** giữa ④ và clip real gốc của nó (cùng lời, cùng thời điểm), độ mở cuối âm tiết fake − real. Đạt khi ≥ 2 / 4 generator có trung vị chênh > 0 với cận dưới CI bootstrap theo người > 0 ở âm tiết bắt khép, và chênh này lớn hơn ở âm tiết đối chứng.
  3. Làm tương tự cho âm đầu b / m / p, và tách riêng `labial_velar` để biết phần "riêng tiếng Việt" có đóng góp không.
  
  Không đạt (1): bỏ đặc trưng độ mở đo tay, chỉ giữ thẻ. Không đạt (2) ở mọi generator: H-B không có cơ sở, M7 chỉ còn H-A (hoặc team dừng v007). Thiết kế cặp ④ ↔ ① khử được khác biệt người, câu nói và ánh sáng.
- **0d · Lối tắt vết ghép theo từ** (§4): trên ② sham partial của val, so điểm tác vụ audio của M1-v003 giữa nhóm có hai biên đoạn trùng biên âm tiết (±1 frame) và nhóm cắt giữa âm tiết. Cần output v003 của Linh + cache 0a.
- **0e · Encoder AV-HuBERT tiếng Việt (chỉ khi được phép dùng, §3.11).** Trích `av` bằng [AV-HuBERT-CTC-Attention-VI](https://huggingface.co/nguyenvulebinh/AV-HuBERT-CTC-Attention-VI) cho val; so với encoder hiện tại bằng điểm sync không train kiểu SpeechForensics (AUC real so với ② sham full, real so với ⑤ donor). Đạt khi hơn ≥ +0,02 ở cả hai.

**Nấc 1, train trên part 1 (rẻ):** thiết kế giai thừa 2 × 2 để tách hai giả thuyết, mỗi cấu hình 2 seed (42, 43):

| | có thẻ kỳ vọng + độ mở đo tay | không |
|---|---|---|
| **đơn vị âm tiết** | `visyl` (M7) | `visyl_no_phon` |
| **cửa sổ đều 0,2 s** | `visyl_uniform` | `visyl_uniform_no_phon` |

8 lần train × ~0,3–0,6 giờ ≈ 2,5–5 giờ T4 (○). Chọn trên **val** (AI?, 3 tác vụ, định vị P6, sham). So với M1-v003 trên val (cùng dữ liệu, cùng seed 42).

**Nấc 2, có điều kiện:** nếu 0e đạt và giấy phép rõ: trích lại `av` part 1 bằng encoder tiếng Việt, train lại `visyl`. Nếu encoder tiếng Việt giúp M7, nó cũng nên được thử trên M6 (v006) để không nhầm "encoder tốt hơn" với "M7 tốt hơn".

**Bước kết:** chung với v006: part 1 + 2, 3 seed, test part 2 mở một lần; M7 và 3 ablation chạy cùng đợt với M1-v003, M6 (v006), M2 và baseline.

### 3.9. Cổng đặt trước cho M7 (team chốt trước khi mở test part 2)

M7 chỉ được giữ làm đề xuất trong báo cáo nếu **G1 đạt và ít nhất một trong G2, G3 đạt**. G4, G5 quyết định phần nào của M7 được nhận là đóng góp.

| Cổng | So sánh | Tác vụ · tập | Đạt khi |
|---|---|---|---|
| **G1 (chính)** | M7 − M1-v003 | AI? · toàn bộ test | cận dưới CI 95 % ≥ −0,02 (không kém đáng kể, với chi phí train ~1/8) |
| G2 | M7 − M1-v003 | định vị, thước đo P6 (AP trung bình IoU 0,5–0,95, partial + real) | cận dưới CI > 0 |
| G3 | M7 − M1-v003 | visual · LatentSync, Diff2Lip | Δ ≥ +0,02 và cận dưới CI > 0 |
| G4 (H-B) | M7 − `visyl_no_phon` | visual · mẫu có ≥ 1 âm tiết bắt khép trong đoạn giả | cận dưới CI > 0 |
| G5 (H-A) | M7 − `visyl_uniform` | định vị P6 | cận dưới CI > 0 |
| G6 | M7 | sham so với real | AUC ≤ 0,60 |
| G7 | M7 so với M1-v003 | FPR test ở ngưỡng 1 % / 5 % chọn trên val, CI Clopper–Pearson; phân tán theo người | không tệ hơn |
| G8 (báo cáo, không là cổng) | M7 | mẫu ASR nhận < 3 âm tiết | báo tỉ lệ và AUC riêng |

Có M6 (v006) thì báo thêm M7 − M6 (v006) như so sánh phụ. Mọi Δ kèm Δ của từng người (10 giá trị).

### 3.10. Rủi ro và cách giảm

| Rủi ro | Mức | Cách giảm / phát hiện |
|---|---|---|
| H-B sai: generator vẽ khép môi đủ tốt, hoặc landmark 25 Hz không thấy khép ngắn (phụ âm tắc cuối [p̚], [k͡p̚] chỉ 1–2 frame) | cao | Pilot 0c trước mọi lần train; tách âm mũi (dài hơn) với âm tắc |
| Mốc từ của Whisper lệch (PhoWhisper fine-tune, đầu căn chỉnh không chắc đúng) | trung bình | 0b có đáp án từ track B; dự phòng CTC |
| ASR sai hoặc bịa chữ ở đoạn ồn / im | trung bình | đơn vị `unknown` + khoảng lặng phủ mọi frame; G8 |
| M7 ít khối hơn M1 (không crop miệng) nên kém ở AI? | trung bình | G1; nếu chỉ thua ở visual đã thấy thì cân nhắc thêm lại nhánh trace (sau, có ablation) |
| Lối tắt vết ghép theo từ (§4) | chưa biết | 0d; nếu có thì team quyết thêm sham ghép theo âm tiết |
| Khoá vào tiếng Việt; câu chen tiếng Anh | thấp (có chủ đích) | thẻ `unknown`; ghi giới hạn phạm vi |
| Phương ngữ (Sài Gòn: "many speakers" khép môi ở [ŋ͡m]) | thấp | model học mức độ từ real; nếu có nhãn vùng thì báo riêng |
| Giấy phép encoder ViCocktail chưa ghi | chặn nấc 2 | hỏi tác giả trước; không có thì bỏ nấc 2 |
| Test part 1 đã xem; thiết kế M7 đã được thông tin bởi kết quả v003 | chắc chắn | chỉ test part 2 là bằng chứng sạch |

### 3.11. Chi phí và tài sản ngoài

- Chi phí (○): nấc 0 ~1 giờ GPU (cache âm tiết train + val) + vài giờ CPU phân tích; nấc 1 ~2,5–5 giờ T4; cache âm tiết phần còn lại (test part 1, part 2) ~1 giờ; bước kết: 4 cấu hình × 3 seed trên part 1 + 2 ≈ 12 lần × ~0,6–1,2 giờ ≈ 7–15 giờ. Tổng v007 ≈ 15–25 giờ GPU, nhỏ so với 100–140 giờ của bước kết v006.
- PhoWhisper-small: BSD-3-Clause ✔. wav2vec2-base-vietnamese-250h (chỉ khi cần CTC): CC BY-NC 4.0 ✔.
- **ViCocktail** ([paper](https://arxiv.org/abs/2506.04635), [repo](https://github.com/nguyenvulebinh/viCocktail), [model](https://huggingface.co/nguyenvulebinh/AV-HuBERT-CTC-Attention-VI), [dataset](https://huggingface.co/datasets/nguyenvulebinh/ViCocktail)): AV-HuBERT Large (24 lớp) khởi tạo từ encoder MuAViC tiếng Anh, tinh chỉnh AVSR (CTC + attention) trên 269 giờ tiếng Việt, 2.375 người; crop miệng 96×96 từ landmark dlib 48–67 (cùng cách với v003). Model card ghi 0,4 B tham số, nạp bằng lớp `AVHubertAVSR` của transformers (có thể cần bản transformers riêng của tác giả, chưa thử ○). Paper nói "publicly available" nhưng **không ghi giấy phép** ở repo, model card hay dataset card (kiểm 30/10/2026): phải hỏi tác giả trước khi dùng.
- **Bộ VLR** ([Zenodo 19705466](https://zenodo.org/records/19705466), 04/2026, CC BY 4.0): ~116 nghìn cặp clip–lời mức từ, 70,5 GB, YouTube tiếng Việt; nhóm tác giả có Trung Nguyen Quoc (ĐH FPT), có thể là thầy Nguyễn Quốc Trung. v007 không cần bộ này; nó là nguồn nếu sau này muốn tự tinh chỉnh encoder hình tiếng Việt.

### 3.12. Cân nhắc nhưng không đưa vào lõi v007

| Hướng | Bằng chứng | Vì sao chưa đưa vào |
|---|---|---|
| Tinh chỉnh nhẹ encoder (chỉ LayerNorm, LoRA) | [GenD](https://arxiv.org/abs/2508.06248) (WACV 2026): chỉ tune LayerNorm (0,03 % tham số) + loss căn chỉnh / đồng đều, trung bình 14 bộ ~91 AUROC so với probe đóng băng 76,0 (CLIP); full fine-tune overfit nhanh. WAFL: LoRA r = 8. [MoLEx](https://arxiv.org/abs/2509.09175): LoRA experts trên WavLM, In-the-Wild ~9,6 % EER | Phải chạy encoder trong vòng train (AV-HuBERT ở env fairseq riêng) → nhiều việc kỹ thuật; không phụ thuộc ngôn ngữ nên nên thử chung cho cả M1 và M7 ở một bước sau, không trộn vào v007 |
| Lấy mẫu batch theo cặp cùng nguồn | GenD, LNCLIP-DF nhấn mạnh cặp real–fake cùng nguồn | Rẻ, có thể thử sau cho cả M1 và M7 mà không đổi kiến trúc |
| Nhánh crop miệng (trace, NPR) | W1: chưa thấy đóng góp | Thêm lại chỉ khi M7 thua ở visual |
| Đầu hồi quy biên đoạn | WAFL cho thấy neo theo từ đã cho biên chính xác | Đơn vị âm tiết đã cho biên |
| Nhất quán khoảng cách–tiếng ([PADD](https://arxiv.org/abs/2610.18485), 10/2026) | AUC 0,73 trên DF24; thua SpeechForensics trên video tĩnh (0,65 so với 0,78) | Clip v003 chủ yếu chính diện, ít di chuyển |
| MLLM, X-AVDT, DF-MoE | | Quá nặng cho T4 (như v006) |

Về đánh giá: [DF26](https://arxiv.org/abs/2609.07369) (09/2026) cho thấy điểm gộp che lỗi theo từng generator (có detector về mức ngẫu nhiên ở một generator) → v007 cũng báo theo từng generator và giọng, như v003 đã làm.

### 3.13. Quyết định team cần cho v007

| Quyết định | Đề xuất | Vì sao |
|---|---|---|
| Nhận v007 (M7 "ViSyl") là đề xuất thứ hai, cạnh M6 (v006); không thay M1 | Đồng ý ở mức nấc 0; quyết lại sau pilot 0c | Đa dạng hướng tiếp cận; rẻ; nhắm đúng chỗ v006 không giải |
| Chạy pilot 0c trước mọi lần train M7 | Đồng ý | Không có hiệu ứng khép môi thì H-B không có cơ sở |
| Hỏi tác giả ViCocktail về giấy phép model | Đồng ý, làm song song | Chặn nấc 2 nếu không có |
| Thêm sham ghép theo âm tiết (biến thể dữ liệu mới) | Hoãn, chỉ khi 0d thấy lối tắt | Đổi bước sinh; theo CLAUDE.md cần team đồng ý |
| Phân việc | Chưa đề xuất | Team tự chia theo vai trò đã có |

---

## 4. Phát hiện khi đọc code: vết ghép theo từ (ảnh hưởng cả v003, v006, v007)

> Bản đầy đủ hơn (thêm biên trộn hình của ⑥, ③C nối 5 ms, bảng sửa cố định, co giãn không giới hạn) ở §3 của `plans/2026-10-30_v007_sau-review.md`.

Trong `src/vn_av_df/generation/generate.py` ✔:
- ② sham partial (dòng 236) và ④ / ⑤ partial (dòng 439) lấy đoạn bằng `random_span`: vị trí ngẫu nhiên đều trên lưới frame, **không theo biên từ**.
- ③ track B (dòng 286–301) và ⑥ (dòng 430–436) lấy đoạn từ `choose_edit(words, …)` trên mốc từ của PhoWhisper: **biên đoạn trùng biên từ**. Track C (chỉ test) cũng dùng đoạn này.
- Tiếng được ghép bằng cùng hàm `A.splice` (crossfade, khớp độ to, tiếng nền) cho ② và ③ B, đúng như thiết kế "vết ghép không thành dấu hiệu riêng của giọng AI". Nhưng **vị trí** của vết ghép so với biên từ vẫn khác nhau theo nhãn.

Hệ quả (○, chưa đo): trong các mẫu partial có ghép tiếng, nhãn "giọng AI" trùng với "ghép đúng biên từ", còn ② sham và ⑤ donor (không giọng AI) thường cắt giữa âm tiết. Model nào thấy được "cắt giữa âm tiết" (M1 qua XLS-R, M7 qua đơn vị âm tiết) có thể học "cắt giữa âm tiết ⇒ không phải giọng AI". Ngoài đời, người sửa lời thường cắt đúng biên từ, nên lối tắt này làm model dễ báo nhầm các đoạn ghép tiếng thật nhưng cắt khéo. Vì tiếng Việt nói ~5–6 âm tiết/s, một biên ngẫu nhiên vẫn hay rơi gần biên âm tiết, nên lối tắt có thể chỉ một phần.

Kiểm không cần GPU (0d ở §3.8, cũng nên đưa vào nấc 0 của v006): so điểm tác vụ audio của M1-v003 trên ② sham partial giữa nhóm biên trùng biên âm tiết và nhóm cắt giữa âm tiết. Điểm cao hơn rõ ở nhóm trùng biên thì model đang dùng lối tắt; khi đó team quyết có thêm biến thể sham ghép theo âm tiết hay không (đổi bước sinh).

---

## 5. Lịch làm chung đề xuất

| Bước | v006 | v007 | Cần |
|---|---|---|---|
| 1 · Không GPU | nấc 0: P4, phân rã AUC, lối tắt co giãn và độ lệch | 0b, 0c, 0d (sau khi có cache 0a) | output v003 của Linh; cache âm tiết train + val part 1 (~1 giờ GPU) |
| 2 · Chỉ inference | P6 trên `best.pt` M1-v003; zero-shot AntiDeepfake | 0e nếu được phép dùng ViCocktail | env fairseq cho AntiDeepfake |
| 3 · Train part 1 | 3 lần (+P1, +P1–P3, M6 (v006)) ~12 giờ | 8 lần (2 × 2 × 2 seed) ~2,5–5 giờ | quota Kaggle |
| 4 · Có điều kiện | P5 | encoder tiếng Việt | kết quả bước 1–2 |
| 5 · Bước kết | part 1 + 2, 3 seed, test part 2 một lần, đội hình: BL-V, BL-A (đã sửa), BL-M, BL-M/sup, BL-Ens, M1-v003, M6 (v006), M2, M7 (v007) + ablation | | part 2 sinh xong ●; tổng ~115–165 giờ GPU (○) |

Thứ tự ưu tiên nếu thiếu quota: v006 nấc 0 và v007 0c trước (đều gần như không tốn GPU), vì chúng quyết định có nên tốn GPU cho từng bản hay không.

---

## 6. Nguồn

Trang mở ngày 28/10/2026 (lượt v006 và review) và 30/10/2026 (lượt v007). Số liệu là của paper, điều kiện khác dữ liệu mình nên không so trực tiếp.

**Repo và tài liệu nội bộ**
- `docs/v003_model.md`, `docs/v003_generation.md`, `docs/v003_data.md`; `src/vn_av_df/training/models.py` (`TriHeadV2`, `AVHAlignPlus`), `data.py` (`replace_audio`, `fixed_references`), `evaluate.py` (`proposals`, `ref_norm`), `src/vn_av_df/generation/generate.py` (`random_span`, `run_voice`, `run_lipsync`), `src/vn_av_df/generation/voice/asr.py` (commit `2b3a70c`).
- Claude Doc [v004 · Đề xuất nâng cấp M1](https://claude.ai/artifact/3TyxqJuzgkkh27WmHdBEsk); `reviews/2026-10-28_review-v004.md`; slide [vn-av-df-slide](https://github.com/VN-AV-DF-Capstone/vn-av-df-slide) (`talk.js`).

**Dùng cho v006** (chi tiết ở Claude Doc §3)
- Boldisor et al. [arXiv 2511.17181](https://arxiv.org/abs/2511.17181) · AV-Lip-Sync+ [arXiv 2311.02733](https://arxiv.org/abs/2311.02733) · AVH-Align [arXiv 2412.00175](https://arxiv.org/abs/2412.00175) · HAVIC [arXiv 2603.23960](https://arxiv.org/abs/2603.23960) · LipDA [arXiv 2610.08417](https://arxiv.org/abs/2610.08417) · BioLip [arXiv 2604.16808](https://arxiv.org/abs/2604.16808) · SAVe [arXiv 2603.25140](https://arxiv.org/abs/2603.25140) · LoCC [arXiv 2606.22772](https://arxiv.org/abs/2606.22772) · WMMT [arXiv 2508.02179](https://arxiv.org/abs/2508.02179) · LNCLIP-DF / GenD (cùng một bài) [arXiv 2508.06248](https://arxiv.org/abs/2508.06248) · Scaling Laws [arXiv 2510.16320](https://arxiv.org/abs/2510.16320) · DomainBed [arXiv 2007.01434](https://arxiv.org/abs/2007.01434) · DF-MoE [arXiv 2608.23363](https://arxiv.org/abs/2608.23363) · Referee [arXiv 2510.27475](https://arxiv.org/abs/2510.27475) · KLASSify [arXiv 2508.07337](https://arxiv.org/abs/2508.07337) · Pindrop [arXiv 2508.08141](https://arxiv.org/abs/2508.08141) · Zhou & Wang [arXiv 2606.21584](https://arxiv.org/abs/2606.21584) · AV-Deepfake1M++ [arXiv 2507.20579](https://arxiv.org/abs/2507.20579) · MAVOS-DD [arXiv 2505.11109](https://arxiv.org/abs/2505.11109) · TalkingHeadBench [arXiv 2505.24866](https://arxiv.org/abs/2505.24866) · AntiDeepfake [arXiv 2506.21090](https://arxiv.org/abs/2506.21090) · XLSR-MamBo [arXiv 2601.02944](https://arxiv.org/abs/2601.02944) · Dao et al. [arXiv 2606.08669](https://arxiv.org/abs/2606.08669) · VietPrism [arXiv 2609.30005](https://arxiv.org/abs/2609.30005) · SEA-Spoof [arXiv 2509.19865](https://arxiv.org/abs/2509.19865) · Luong et al. [arXiv 2507.03468](https://arxiv.org/abs/2507.03468)

**Dùng cho v007** (mở 30/10/2026)
- WAFL, Word-Anchored Temporal Forgery Localization: [arXiv 2603.06220](https://arxiv.org/abs/2603.06220)
- Agarwal et al., Detecting Deep-Fake Videos From Phoneme-Viseme Mismatches, CVPRW 2020: [CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w39/Agarwal_Detecting_Deep-Fake_Videos_From_Phoneme-Viseme_Mismatches_CVPRW_2020_paper.html)
- NPVForensics: [arXiv 2306.06885](https://arxiv.org/abs/2306.06885)
- PIA, ICCVW 2025: [arXiv 2510.14241](https://arxiv.org/abs/2510.14241)
- BabelFake: [arXiv 2610.06339](https://arxiv.org/abs/2610.06339)
- DF26: [arXiv 2609.07369](https://arxiv.org/abs/2609.07369)
- PADD (khoảng cách–tiếng): [arXiv 2610.18485](https://arxiv.org/abs/2610.18485)
- GenD, WACV 2026: [arXiv 2508.06248](https://arxiv.org/abs/2508.06248)
- MoLEx: [arXiv 2509.09175](https://arxiv.org/abs/2509.09175)
- ViCocktail, Interspeech 2025: [arXiv 2506.04635](https://arxiv.org/abs/2506.04635) · [repo](https://github.com/nguyenvulebinh/viCocktail) · [model](https://huggingface.co/nguyenvulebinh/AV-HuBERT-CTC-Attention-VI) · [dataset](https://huggingface.co/datasets/nguyenvulebinh/ViCocktail)
- Vietnamese Lip Reading (VLR): [Zenodo 19705466](https://zenodo.org/records/19705466)
- PhoWhisper-small: [Hugging Face](https://huggingface.co/vinai/PhoWhisper-small) · wav2vec2-base-vietnamese-250h: [Hugging Face](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)
- Vietnamese phonology: [Wikipedia](https://en.wikipedia.org/wiki/Vietnamese_phonology)
- Bear & Harvey, Phoneme-to-viseme mappings: the good, the bad, and the ugly: [UEA](https://ueaeprints.uea.ac.uk/id/eprint/64350/)
- viphoneme: [PyPI](https://pypi.org/project/viphoneme/) · underthesea IPA: [docs](https://underthesea.readthedocs.io/en/latest/_modules/underthesea/pipeline/ipa.html)
