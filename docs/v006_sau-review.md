# v006 · M6 "ViSyl" sau review

> **Code (03/11/2026):** M6 = `visyl`, `visyl_no_phon`, `visyl_uniform_no_phon`, `visyl_shuffled_tags` (`training/models.py`), cache `syll` (`features/syllables.py`, bộ căn PhoWhisper DTW), pseudo-fake theo đơn vị (`data.replace_audio`), P4 / P6 (`evaluate.py`). Chưa có: hình học môi và kỳ vọng theo người (chờ 0c'), phép đảo thẻ lúc suy luận, cổng G1-G8 trong `report.py`. Cách train theo v004 của Linh (nhãn mismatch đo được, dịch tiếng −2…+5, chọn theo val AUC), không theo protocol v003 ghi trong file này (MN chốt 03/11). Chưa train, chưa đo trên dữ liệu thật; bản sao của file đề xuất trong project, giữ nguyên nội dung.

Ngày 30/10/2026 · viết cho MN và team VN-AV-DF-Capstone

> **Đổi tên 02/11/2026 (MN):** repo của Linh đã có v004 riêng (M3 = `trihead_v4`, môi–tiếng–đầu). Vì vậy đề xuất "M1 + P1–P6" (trước gọi là v004 · M1 (v004)) nay là **v005 · M5**, và "ViSyl" (trước gọi là v005 · M3) nay là **v006 · M6**. Tên file review (`review-v004`, `review-v005`) và Claude Doc "v004 · Đề xuất nâng cấp M1" giữ tên cũ.
>
> **Trạng thái.** Đề xuất: chưa có code, chưa train, chưa đo gì trên dữ liệu thật. File này **thay cho §3 (v006)** và mở rộng §4 của `plans/2026-10-30_v005-v006_de-xuat-detector.md` (bản trước review). Báo cáo review đầy đủ, có dẫn dòng code và URL cho từng phát hiện: `reviews/2026-10-30_review-v005.md`. Mọi số của v003 vẫn lấy từ slide của Linh (part 1, 1 seed), chưa đối chiếu được với output gốc.
>
> **Mức tin:** ● số đo (slide hoặc dữ liệu nhóm) · ✔ đã kiểm trong code (commit `2b3a70c` trên `main`) hoặc nguồn gốc · ✔ (review) reviewer đã mở nguồn, mình chưa mở lại · ○ suy luận, ước lượng, chưa kiểm.

---

## 0. Tóm tắt

**v006 vẫn là M6 "ViSyl"**: một model mới, khác họ với M1, là đề xuất thứ hai cạnh M5 (v005) chứ không thay M1. Ý chính giữ nguyên: phân tích theo âm tiết tiếng Việt và kiểm xem môi có khép đúng ở những chỗ mà tiếng Việt bắt khép hay không.

**Review kết luận: chưa nên train như bản 30/10, phải sửa nấc 0 trước.** Reviewer (một agent độc lập, vai senior reviewer / researcher) tìm 29 phát hiện: 2 chặn, 6 nặng, 12 vừa, 9 nhẹ. Hai lỗi chặn:
1. Phép thử định vị của H-A thắng sẵn do cách dựng dữ liệu: hơn nửa số đoạn giả partial có AI có biên chính là mốc từ của PhoWhisper, cùng nguồn với đơn vị của M6.
2. Pilot khép môi 0c không tách được "generator không khép môi" khỏi "dlib đo lệch trên miệng mờ, nhỏ".

Mình đã tự kiểm lại các claim có ảnh hưởng lớn nhất trong code và nguồn (§1.5), thấy đúng, và nhận cả 29 phát hiện. Ở ba chỗ (C1, N4, V10) mình sửa khác hoặc mạnh hơn đề xuất của reviewer, lý do ghi ở chỗ tương ứng.

**Thay đổi chính:**

| | Bản 30/10 (trước review) | Sau review |
|---|---|---|
| Giả thuyết định vị (H-A) | Đơn vị âm tiết cho biên chính xác hơn, kiểm trên mọi partial | Tách hai: **H-A1** là tính chất thiết kế (biên trùng biên sửa theo từ), chỉ báo cáo, không làm cổng; **H-A2** (gộp theo âm tiết giúp bắt đoạn giả ngắn) kiểm trên tầng đoạn ngẫu nhiên ②, ④, ⑤ |
| Mốc âm tiết | PhoWhisper DTW; 0b so với `partial_span` của ③B | Bộ căn chọn ở 0b' bằng 200–300 biên gán tay, so PhoWhisper DTW, CTC (wav2vec2 tiếng Việt), MFA |
| Pilot khép môi (H-B) | ④ so với ①, đo bằng dlib | 0c': thêm real bị làm suy biến giống generator, phép đo thứ hai, nhãn người, phân tầng cỡ miệng, kiểm nén dải, tiêu chí riêng cho `labial_velar` |
| Thẻ phụ âm | cố định theo giọng Bắc | có tính phương ngữ: lớp riêng cho un, ut, ôn, ôt và cho "v"; kỳ vọng khép theo từng người |
| Đặc trưng | có độ dài đơn vị; độ mở môi tính sẵn | bỏ độ dài tường minh; độ mở tính sau khi dịch tiếng, chuẩn hoá theo người |
| Đầu ra | 3 đầu đọc chung một biểu diễn | đầu audio có đường riêng chỉ đọc tiếng; đầu visual và mismatch đọc đường gộp không có XLS-R |
| Pseudo-fake | bỏ | giữ như M1 (p = 0,25), làm theo đơn vị, một nửa số lần cắt đúng biên âm tiết |
| Ablation | 2 × 2, 2 seed | 4 cấu hình (thêm thẻ xáo trộn) + đảo thẻ lúc suy luận, 5 seed, một tiêu chí chọn định trước |
| Cổng | G1 biên −0,02; M6 dùng P4 còn M1 không | cùng hậu xử lý, điểm trung bình 3 seed, nói rõ lực kiểm định; thêm cổng cho đầu visual trên ②, ③ và đầu audio trên ④, ⑤ |
| Code | "train / evaluate / report dùng nguyên" | danh sách 10 việc code thật; module `syll` riêng để không làm cũ cache XLS-R, mouth, LIPINC, wave |
| Claim tính mới | "độ mở môi có điều kiện theo kỳ vọng âm vị" | thu hẹp: luật phụ âm cuối môi-ngạc mềm của tiếng Việt suy từ chính tả, đo theo đơn vị âm tiết, có tính phương ngữ |

**Đường đi tiếp.** Nấc 0 (gần như chỉ CPU, cộng khoảng 8–12 giờ công gán nhãn tay) quyết định tất cả: 0b' trượt thì bỏ H-A; 0c' trượt thì bỏ phần đo môi của H-B; cả hai trượt thì M6 chỉ còn là một baseline rẻ, không claim gì về âm tiết hay khép môi (hoặc team dừng v006). Chi phí GPU của v006 tăng từ ~15–25 lên ~20–60 giờ (○), vẫn dưới bước kết ~100–140 giờ của v005.

---

## 1. Review tìm thấy gì

Reviewer đọc §3, §4 của bản 30/10 (đọc §0–§2, §5 làm bối cảnh), code ở `2b3a70c` và nguồn gốc trên web (30/10/2026). Không chạy gì trên media hay GPU vì container không có media; có một smoke check trên dữ liệu tổng hợp (không phải kết quả nghiên cứu).

### 1.1. Hai lỗi chặn

**C1 · Phép thử H-A thắng sẵn do cách dựng dữ liệu.**
- ③ track B lấy đoạn sửa từ mốc từ PhoWhisper trên tiếng gốc (`generate.py:255`, `:286-295`; span = [start của từ đầu, end của từ cuối], `voice/edit.py:100-105`). Track C (chỉ test) dùng đúng span đó. ⑥ dùng span này cho đoạn tiếng giả, và ở ⑥ partial cho cả **biên trộn hình** (`generate.py:435-447`). ②, ④, ⑤ partial dùng `random_span`, vị trí đều trên lưới frame ✔.
- Tỉ lệ (○, suy từ `plan.py` và `generators_per_clip` = 2 / 2 / 3): khoảng 56 % (train, val) và 60 % (test) đoạn giả partial có AI là đoạn theo biên từ.
- Smoke check của reviewer (3.000 clip tổng hợp, bộ chấm hoàn hảo, chỉ bị giới hạn bởi lưới đơn vị): trung bình P(IoU ≥ t), t = 0,5…0,95, là 1,000 với đơn vị âm tiết so với 0,907 với cửa sổ 0,2 s trên đoạn theo biên từ; trên đoạn ngẫu nhiên hai loại gần bằng nhau (0,905 so với 0,899).
- Hệ quả: G5 và G2 ở bản 30/10 gần như chắc đạt mà không nói gì về chất lượng model, và luật giữ M6 ("G1 và G2 hoặc G3") có thể thoả vì lý do sai.
- Mình xử lý mạnh hơn đề xuất của review: tách H-A thành H-A1 (tính chất thiết kế, không làm cổng) và H-A2 (giả thuyết thật, kiểm trên tầng đoạn ngẫu nhiên). Lý do: với đoạn sửa theo biên từ, bộ tách âm tiết nào đủ chính xác cũng cho biên đúng, nên đó là điều kiểm được bằng độ chính xác bộ căn (0b'), không cần train.

**C2 · Pilot 0c không tách được "không khép" khỏi "đo lệch".**
- Landmark là dlib 68 điểm, toạ độ nguyên pixel, chạy ở khung có cạnh dài ≤ 640 px, không làm mượt (`avhubert_worker.py:110-125`, `:163-167`) ✔ (review). Miệng rộng khoảng 20–30 px, khe môi lúc gần khép chỉ 0–3 px (○).
- Generator vẽ miệng hoặc mặt ở độ phân giải thấp rồi dán lại: VideoReTalking L-Net 96 × 96, Diff2Lip 128 × 128, MuseTalk và LatentSync 256 × 256 ✔ (review). BioLip: giảm độ phân giải làm AUC giảm 0,218; Agarwal et al.: độ chính xác đo khép môi giảm từ 90,46 % xuống 83,80 % sau khi resize ✔ (review).
- Với miệng mờ, bộ hồi quy hình dạng như của dlib dễ kéo điểm về hình trung bình ("nén dải"): độ mở nhỏ nhất không về 0, độ mở lớn nhất bị thu lại. Âm tiết bắt khép (ở đầu dưới của dải) lệch nhiều nhất, âm tiết đối chứng (ở giữa dải) lệch ít, nên tiêu chí "chênh lớn hơn ở âm tiết đối chứng" đúng là thứ một lỗi đo thuần tuý sẽ tạo ra, kể cả khi generator khép môi hoàn hảo (○).
- τ toàn cục (phân vị 5 % của real) cũng thiên vị: fake mờ hiếm khi xuống dưới τ, nên "tỉ lệ frame khép" bắt độ mờ chứ không bắt việc khép (○).

### 1.2. Sáu lỗi nặng

| ID | Vấn đề | Bằng chứng chính | Xử lý trong bản này |
|---|---|---|---|
| N1 | Mốc âm tiết dựa vào DTW trên attention chéo của Whisper; `alignment_heads` của PhoWhisper-small **giống hệt** whisper-small (không chọn lại sau tinh chỉnh); DTW hay kéo từ vào khoảng lặng; `asr.py` bỏ từ dài 0; đáp án 0b (`partial_span` của ③B) chính là mốc PhoWhisper trên tiếng gốc nên không độc lập | ✔ `generation_config.json`, `asr.py:53`, `generate.py:255-299`; CrisperWhisper; Rousso et al. 2024 (TIMIT, biên trong 50 ms: MFA 89,4 %, WhisperX 82,4 %) ✔ (review) | 0b' với đáp án gán tay, so 3 bộ căn, tiêu chí chặt hơn; giữ từ dài 0; ghim revision (§2.3, §4.1) |
| N2 | Tín hiệu H-B là phép đo nhất quán tiếng–hình nên cũng bị vi phạm ở ② sham và ③ voice (môi thật). ② partial ngắn có thể làm đầu visual bật; lỗi ở ③ bị đầu audio che vì AI? = max; G6 gộp full với partial, ngưỡng 0,60 lỏng so với M1 0,543 | ✔ nhãn `cells.py:29-60` | G6 tách full / partial / ≤ 0,6 s và so với M1; thêm G6b cho đầu visual; pseudo-fake theo đơn vị; quy tắc hiển thị trên demo. Không làm ablation "đầu visual không thấy tiếng" vì so tiếng với môi là chính cơ chế của H-B |
| N3 | Miền Nam và miền Trung đọc un, ut, ôn, ôt thành [ŋ͡m], [k͡p̚] (đang gắn `other`); Bắc Trung Bộ đọc nhiều vần môi-ngạc mềm mà không khép; "v" miền Nam thường là [j]; dữ liệu không có nhãn vùng | ✔ Wikipedia "Vietnamese phonology"; `docs/v003_data.md:945` ✔ (review) | thẻ có tính phương ngữ, kỳ vọng khép theo người, 0f ghi vùng, đối chiếu từ điển MFA (§2.2) |
| N4 | Đoạn sửa của ③B, ⑥, ③C lấy từ một bảng cố định (có ↔ không, số đếm, vài từ thay thế); track B co giãn **không giới hạn** (±15 % chỉ có ở track A); độ dài đơn vị đi thẳng vào model; ③C ở test dùng cùng bảng nên lối tắt làm điểm test đẹp lên chứ không lộ ra | ✔ `edit.py:10-50`, `audio.py:67-100`, `:255-264`, `generate.py:297-299`; arXiv 2603.11205 (model chỉ đọc transcript đạt WordF1 ~84 % trên AV-1M dev) ✔ (review) | probe ở 0d'; bỏ độ dài tường minh. Mình thêm một giới hạn mà review chưa nói: Transformer vẫn suy được độ dài từ vị trí các đơn vị kề nhau, nên chỉ sửa dữ liệu mới chặn tận gốc |
| N5 | 2 × 2 gộp thẻ kỳ vọng với hình học môi đo tay; bản cửa sổ đều đo môi ở mép cửa sổ và có độ dài hằng số, nên tương tác có sẵn; G5 lẫn ba yếu tố | ○ suy từ đặc tả | H-A2 so hai bản không thẻ; bản cửa sổ đều không dùng ASR; thêm thẻ xáo trộn và đảo thẻ (§4.2) |
| N6 | Với 10 người test (SE ≈ 0,0115 suy từ CI trên slide), G1 biên −0,02 chỉ đạt khoảng 41 % khi hai model ngang nhau; M6 dùng P4 còn M1-v003 dùng max thô; `report.py` tính cổng trên seed đầu | ✔ `report.py:131`, `:169`; `evaluate.py:129-141`; ○ tính lực | cùng hậu xử lý, trung bình 3 seed, ghi lực vào báo cáo, đề xuất biên −0,03 cho team chốt (§4.5) |

### 1.3. Lỗi vừa, lỗi nhẹ và cách xử lý

| ID | Vấn đề (một dòng) | Xử lý |
|---|---|---|
| V1 | "Train / evaluate / report dùng nguyên" sai: thiếu dịch đơn vị theo tiếng, nạp landmark, P4, P6, bootstrap AP, nhiều seed, lát G4; cache ASR của bước sinh khoá theo clip gốc, dùng lại sẽ rò lời gốc vào thẻ của ③B, ⑥ | §5 liệt kê việc code thật; cache khoá theo hash tiếng của chính mẫu |
| V2 | Thêm `syll` vào `features/encoders.py` đổi `sha(__file__)` trong chữ ký cache XLS-R, mouth, LIPINC, wave, nên mọi cache đó bị trích lại | module riêng `features/syllables.py` |
| V3 | 0d vòng tròn (ASR trên tiếng đã ghép), ±1 frame quá rộng với âm tiết 4–5 frame, lực thấp, đo M1 chứ không đo M6 | 0d' dùng ASR của tiếng gốc và tiếng donor, thêm probe, thêm tập phân tích ② theo biên từ (cần team đồng ý) |
| V4 | 2 seed không ước được nhiễu seed; chọn trên val theo 4 tiêu chí chưa định trước | 5 seed, một tiêu chí chính định trước |
| V5 | Tiền đề "generator học chủ yếu tiếng Anh (và tiếng Trung)" mâu thuẫn giữa §0 và §3.2 cũ; tiếng Trung chỉ được ghi cho LatentSync 1.5 (chỉ test); generator nhận tiếng, không đọc chính tả | §2.1 viết lại thành câu hỏi thực nghiệm, kèm bảng dữ liệu train của 6 generator và bằng chứng hai phía |
| V6 | PIA bỏ **ảnh** môi trong ablation và đã dùng độ mở miệng theo âm vị; số AV-1M của WAFL là trên val, AP@0,5 chéo bộ của WAFL thấp hơn DiMoDif và AuViRe; thiếu He & Wang (PRCV 2025) và 4 nguồn khác | sửa trích dẫn, thu hẹp claim tính mới (§2.10) |
| V7 | Bước kết part 1 + 2: code ghép part không đổi split; người test part 1 đã bị xem, có generator và giọng chỉ-test | §4.4 nêu hai cách, đề xuất một (quyết định chung với v005) |
| V8 | Bỏ pseudo-fake thêm một khác biệt với M1 | giữ pseudo-fake, làm theo đơn vị (§2.7) |
| V9 | `v` của AV-HuBERT là bản tinh chỉnh đọc môi (VSR) nên đã mã hoá khép môi; `visyl_no_phon` vẫn học được kỳ vọng ngầm; G4 không xác nhận cũng không bác được H-B | thêm phép đảo thẻ, so theo đơn vị, probe tuyến tính trên `v` |
| V10 | Đầu gộp: đầu audio thấy đặc trưng hình; ở ⑥ đoạn tiếng giả trùng đoạn môi giả, nên giải thích theo từng đầu không tin được | mạnh hơn đề xuất của review (review chỉ đề xuất báo cáo và cân nhắc chặn gradient): đầu audio có đường riêng chỉ đọc tiếng; đường gộp bỏ XLS-R, để đầu visual khó học "tiếng giống TTS ⇒ môi giả" từ ⑥ |
| V11 | §4 cũ thiếu: biên trộn hình của ⑥ cũng theo biên từ; ④ partial không ghép tiếng; ③C nối 5 ms, không khớp độ to; bảng từ; co giãn không giới hạn | §3 viết lại |
| V12 | Lát G4 gần bằng toàn bộ fake hình; G6, G7 không có biên rõ | G4 thành phép thử tương tác; G6, G7 có biên |
| L1 | "~8,3 nghìn mẫu" ở 0a sai: train + val part 1 = 5.286 | sửa |
| L2 | Bear & Harvey không đỡ câu "khép môi là lớp ổn định nhất" | bỏ câu đó |
| L3 | Luật chính tả: oong, ooc; tách qu, gi trước khi tìm vần; chữ số | §2.2 |
| L4 | Câu đoán danh tính tác giả bộ VLR; "crop cùng cách với v003" của ViCocktail chưa kiểm; 0e có thể đo sai mục tiêu | bỏ câu đoán; ghi ○; thêm probe real so với ④ vào 0e |
| L5 | Ước tốc độ PhoWhisper lạc quan (fp32, gọi từng clip); M6 đọc XLS-R như M1 nên ước giờ train theo M2 là lạc quan | đo 50 clip và 1 epoch trước khi lập lịch |
| L6 | arXiv 2604.16808 bản v4 tên "BioLip", không thấy "LADDER" | sửa |
| L7 | Tỉ lệ ASR thất bại có thể khác theo ô và thành dấu hiệu | báo theo ô (0d', G8) |
| L8 | Số đo landmark trên demo dễ được đọc như bằng chứng khi phép đo chưa được kiểm | chỉ hiện sau khi 0c' đạt (§2.6) |
| L9 | LipDA không ghi venue trên arXiv; BabelFake thấy không ngôn ngữ nào khó hơn một cách hệ thống | bỏ "ICML 2026"; BabelFake thành bằng chứng ngược nhẹ |

### 1.4. Những gì review xác nhận là đúng

- Landmark thô (khoá `landmarks` trong cache `avhubert`) và tư thế đầu (cache `mouth`, đọc được mà không đọc crop) có sẵn; chỉ số dlib (61–67, 62–66, 63–65; 48–54; 36–45) đúng.
- Ước ~0,75 M tham số của bản 30/10 (tính lại 0,746 M) và ~480 chiều đầu vào.
- Giao diện `Model`, gộp top-5, `crop_frames = None`, `pos_weights`, cân theo người có sẵn trong code.
- Các dòng code dẫn ở §4 cũ; giấy phép PhoWhisper (BSD-3), wav2vec2-base-vietnamese-250h (CC BY-NC 4.0), VLR (CC BY 4.0); ViCocktail không ghi giấy phép.
- Câu trích Wikipedia về giọng Hà Nội và Sài Gòn; phụ âm cuối môi-ngạc mềm là đặc trưng riêng của tiếng Việt (trang "Labial–velar consonant" chỉ liệt kê tiếng Việt có loại phụ âm cuối này).
- Phần nên giữ: nấc 0 trước khi tốn GPU; cặp ④ ↔ ① (cùng tiếng, cùng lời, cùng người, chỉ khác môi); thẻ lấy từ ASR của chính mẫu chứ không từ transcript lúc sinh; phần mới hẹp nhất (luật [ŋ͡m], [k͡p̚]) chưa thấy ai làm trong các nguồn tìm được.

### 1.5. Mình đã kiểm lại

Trước khi sửa, mình kiểm lại các claim có ảnh hưởng lớn nhất (30/10/2026):
- `generate.py:255`: cache ASR của bước sinh khoá bằng `sha(clip sạch gốc)` ✔. `:297-299`: cụm TTS của track B được co giãn tới đúng độ dài [p, q], không kẹp tỉ lệ ✔; `audio.py` `fit_whole` kẹp ±15 % và chỉ dùng cho track A ✔. `:334-341`: track C nối `crossfade_s=0.005`, `match_loudness=False` ✔ (mặc định của `splice` là 15 ms và có khớp độ to ✔). `:439-447`: ④ partial giữ nguyên tiếng gốc, ⑤ partial ghép tiếng donor vào `random_span`, `blend_compose` dùng cùng a, b với đoạn đó ✔.
- `cells.py:29-60`: ⑥ có `audio = part` cả khi là bản full ✔, tức ⑥ full cũng có một đoạn tiếng AI theo biên từ.
- `data.py`: `lag_audio` chỉ dịch `a`, `ssl`, `wave` ✔. `asr.py`: bỏ từ có `end <= start`, `chunk_length_s=30` ✔.
- `encoders.py`: 4 chữ ký cache chứa `"code": sha(__file__)` (dòng 82, 222, 372, 447) ✔. `dataset.py`: ghép part mà không đổi split ✔. `report.py`: cổng tính trên seed đầu ✔. `evaluate.py`: AI? = max của hai điểm video thô ✔.
- `vinai/PhoWhisper-small`, `generation_config.json`: `alignment_heads` = [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]], trùng danh sách của whisper-small ✔.
- Wikipedia "Vietnamese phonology": bảng phương ngữ Trung / Nam có un, ut → [uːŋ͡m], [uːk͡p̚]; câu "...but not with ôn, ôt as pronounced [oːŋ͡m], [oːk͡p̚]" (tức ở Sài Gòn ôn, ôt đọc môi-ngạc mềm, chỉ là không nhập với ông, ôc); Bắc Trung Bộ đọc các vần này "without a simultaneous bilabial closure"; "vợ" ở Sài Gòn là [jə˨˧] hoặc [və˨˧] ✔.
- Chưa kiểm lại: smoke check của reviewer, bảng dữ liệu train của generator, các số trong paper (đều ghi "✔ (review)").

---

## 2. Thiết kế M6 (v006) sau review

### 2.1. Câu hỏi và giả thuyết

**Câu hỏi nghiên cứu.** Đưa kiến thức tiếng Việt (âm tiết; luật khép môi suy từ chính tả, có tính phương ngữ) vào detector có giúp ở những chỗ M1 yếu không: đoạn giả ngắn, generator chưa thấy, và giải thích được kết quả.

- **H-A1 · tính chất thiết kế, không phải giả thuyết về model.** Khi đoạn bị sửa theo biên từ (kiểu sửa lời ngoài đời, và kiểu ③B, ③C, ⑥ trong dữ liệu), một lưới âm tiết chính xác cho biên đoạn trùng biên sửa. Điều này đúng do cấu trúc (C1), nên chỉ kiểm độ chính xác của bộ căn (0b') và báo kết quả định vị trên tầng theo biên từ riêng, ghi rõ "biên đoạn giả = mốc ASR". Không dùng để giữ M6.
- **H-A2 · giả thuyết.** Gộp đặc trưng theo âm tiết cho điểm ít loãng hơn gộp theo cửa sổ đều với đoạn giả ngắn (0,3–0,6 s, khoảng 1–3 âm tiết), kể cả khi biên đoạn không trùng biên âm tiết. Kiểm trên tầng đoạn ngẫu nhiên (②, ④, ⑤ partial) bằng cặp `visyl_no_phon` so với `visyl_uniform_no_phon` (G5). WAFL (định vị neo theo từ) là công trình gần nhất, nhưng số tốt của WAFL đến từ dữ liệu sửa theo từ (V6), nên chưa có bằng chứng trực tiếp cho H-A2.
- **H-B · giả thuyết.** Ở các âm tiết mà tiếng Việt bắt khép môi, nhất là phụ âm cuối môi-ngạc mềm [ŋ͡m], [k͡p̚] (không có trong tiếng Anh, cũng không có trong tiếng Trung phổ thông), môi do generator vẽ khép kém hơn môi thật của cùng câu. Đây là **câu hỏi thực nghiệm**: generator nhận tiếng chứ không đọc chính tả, nên câu hỏi đúng là đặc trưng âm thanh của [ŋ͡m], [k͡p̚] có kéo generator về hình miệng khép hay không. Âm môi chung với tiếng Anh (m, b, p) có thể đã được generator vẽ đúng (tiền đề của NPVForensics), nên phần riêng của tiếng Việt mới là chỗ H-B còn hy vọng. Kiểm bằng 0c' (trước khi train) và G4 (sau khi train).

Dữ liệu train của các generator trong v003 (✔ review, nguồn ở §10):

| Generator | Vai trong v003 | Dữ liệu train theo nguồn | Độ phân giải vẽ |
|---|---|---|---|
| Wav2Lip-GAN | train / val | LRS2 (BBC, tiếng Anh) | 96 × 96 (○, chưa mở lại nguồn) |
| MuseTalk 1.5 | train / val | paper: bộ talking-head công khai ~24 giờ; README: "HDTF and private dataset"; dữ liệu của bản 1.5 không được nêu | 256 × 256 |
| IP-LAP | train / val | LRS2, LRS3 (tiếng Anh) | chưa rõ |
| VideoReTalking | train / val | D-Net: VoxCeleb; L-Net, E-Net: LRS2 | L-Net 96 × 96 |
| LatentSync 1.5 | chỉ test | VoxCeleb2 + HDTF; model card 1.5: "we added Chinese data to the training" | 256 × 256 |
| Diff2Lip | chỉ test | VoxCeleb2 (nhiều ngôn ngữ, paper không chia theo ngôn ngữ) | 128 × 128 |

Không nguồn nào ghi tiếng Việt. Bằng chứng ủng hộ H-B: Agarwal et al. 2020 (nhiều deepfake không khép môi ở M, B, P, tiếng Anh); MuEx (tác giả nói model hiện tại "struggle with non-English languages"). Bằng chứng ngược: NPVForensics (deepfake tiên tiến căn được các vùng âm vị–khẩu hình then chốt); LipDA (generator mới khớp môi tốt và xoá dấu vết); BabelFake (5 ngôn ngữ châu Âu, không ngôn ngữ nào khó hơn một cách hệ thống).

### 2.2. Thẻ kỳ vọng khẩu hình, có tính phương ngữ

Nguồn: Wikipedia "Vietnamese phonology" ✔ (mở 30/10/2026). Trước khi dùng, đối chiếu đầu ra của luật với từ điển MFA tiếng Việt bản Hà Nội và TP.HCM (CC BY 4.0) trên toàn bộ từ vựng của dữ liệu và liệt kê chỗ khác nhau. vPhon (GPL-3.0) chỉ dùng để đối chiếu ngoài repo, không chép vào code.

| Vị trí | Chữ viết | Kỳ vọng khẩu hình | Thẻ |
|---|---|---|---|
| Âm đầu | b, m, p | hai môi khép rồi mở | `onset = bilabial` |
| Âm đầu | ph | môi dưới chạm răng trên | `onset = ph` |
| Âm đầu | v | Bắc: như ph; Nam: thường [j], không ràng buộc môi | `onset = v` (lớp riêng) |
| Âm đệm | qu; o trước a, ă, e; u trước â, ê, y, ơ | tròn môi ở đầu vần | `round_on = 1` |
| Âm chính | u, ô, o, uô | tròn môi | `round_nuc = 1` |
| Âm cuối | m, p | khép hai môi | `coda = bilabial` |
| Âm cuối | ng, c sau u, ô, o (ung, uc, ông, ôc, ong, oc) | khép môi [ŋ͡m], [k͡p̚]; Bắc Trung Bộ có thể không khép | `coda = labial_velar` |
| Âm cuối | n, t sau u, ô (un, ut, ôn, ôt) | Nam, Trung: khép [ŋ͡m], [k͡p̚]; Bắc: không | `coda = round_alveolar` (lớp riêng) |
| Âm cuối | u, o bán nguyên âm (au, ao, âu, êu, iu, ưu, eo…) | tròn môi ở cuối | `coda = round_glide` |
| Âm cuối | n, t, nh, ch, i, y còn lại; ng, c sau nguyên âm khác (gồm uông, uôc, ương, ươc) | không ràng buộc môi | `coda = other` |
| Ngoại lệ | oong, ooc; từ nước ngoài; viết tắt; chữ số không đọc được thành chữ | không suy kỳ vọng | `type = unknown` |
| Thanh | 6 thanh | không thấy trên môi | `tone`, chỉ vào đường tiếng |

**Parser** (L3): chuẩn hoá Unicode NFC; đổi chữ số thành chữ trước khi gắn thẻ (PhoWhisper có thể ra "5" thay vì "năm"); tách âm đầu dài nhất, trong đó "qu" và "gi" là âm đầu ("quốc" → vần "ôc", "giục" → "uc", "gióng" → "ong"); tách "oo" trước khi xét "ng, c sau o"; test cho từng trường hợp. Việc loại uông, uôc hợp với vPhon ("after back monophthongs") và với câu Wikipedia (/u, o, ɔ/ là nguyên âm đơn), nhưng chưa có nguồn nói thẳng: kiểm bằng từ điển MFA.

**Kỳ vọng theo người** (thay câu "model tự học mức độ từ real" của bản cũ). Từ 3 clip real tham chiếu của người đó (cùng nguồn với P4), tính tỉ lệ khép thật của từng lớp (`bilabial` đầu, `bilabial` cuối, `labial_velar`, `round_alveolar`) và τ khép riêng của người đó. Bốn tỉ lệ này vào model như đặc trưng của mỗi âm tiết thuộc lớp tương ứng. Nhờ vậy người Bắc không khép ở "un" và người Nam khép ở "un" đều được coi là bình thường. Chỉ dùng khi 0c' xác nhận phép đo.

**Nhãn vùng phương ngữ** (0f) chỉ dùng để phân tích và báo cáo theo nhóm, không làm đầu vào model, vì lúc dùng thật (demo) không có nhãn này.

### 2.3. Tách đơn vị

1. **Lời:** PhoWhisper-small (BSD-3) chạy trên tiếng 16 kHz của **chính mẫu**, ghim revision HF.
2. **Mốc:** bộ căn thắng ở 0b' trong ba ứng viên: PhoWhisper DTW (đã có); căn CTC ép buộc bằng wav2vec2-base-vietnamese-250h trên lời của PhoWhisper (chạy trong env chính; CC BY-NC 4.0); MFA tiếng Việt (cần env riêng, chỉ thử khi CTC trượt). Không bộ nào đạt thì bỏ H-A (§4.1).
3. Giữ từ dài 0 (gán 1 frame) thay vì bỏ như `asr.py:53`.
4. Đổi sang lưới 25 Hz như bản cũ; frame không thuộc âm tiết nào thành đơn vị `gap` (mỗi khúc ≤ 12 frame); ASR không ra gì thì cả clip thành khúc 0,2 s `type = unknown` (tỉ lệ báo theo ô và split ở G8).
5. **Cache `syll`:** module riêng `features/syllables.py` (V2). Khoá là hash của tiếng 16 kHz đã giải mã của chính mẫu, thư mục cache riêng; **không** dùng lại cache ASR của bước sinh (khoá theo clip gốc, sẽ rò lời gốc vào ③B và ⑥, V1). Mẫu có tiếng giống hệt nhau tự trùng khoá nên ASR chỉ chạy một lần. Chữ ký gồm: model ASR và revision, bộ căn và revision, phiên bản transformers, tham số pipeline (`chunk_length_s`, ngôn ngữ, `return_timestamps`), dtype, phiên bản luật thẻ, tham số chia khoảng lặng, hash code của module.
6. Khi train có augment dịch tiếng (−1…+2 frame), mốc đơn vị dịch cùng tiếng vì đơn vị lấy từ tiếng (V1). Mọi đặc trưng môi theo đơn vị tính **sau** khi dịch, trong dataset hoặc model, không tính sẵn trong cache.

Bản ablation `visyl_uniform_no_phon`: đơn vị là cửa sổ đều 5 frame phủ cả clip, **không dùng ASR**.

### 2.4. Đặc trưng mỗi đơn vị

Hai đường (V10): **đường gộp** cho đầu visual và mismatch, **đường tiếng** cho đầu audio.

| Luồng | Nguồn | Cách gộp | Chiều | Đường gộp | Đường tiếng |
|---|---|---|---|---|---|
| Hình nói | `v` AV-HuBERT (1024) | Linear 1024 → 128, attention pooling có mask | 128 | có | không |
| Tiếng nói | `a` AV-HuBERT (1024) | như trên (một pooling dùng chung cho hai đường) | 128 | có | có |
| Tiếng (giả giọng) | `ssl` XLS-R [T, 3, 1024] | trộn 3 nhóm lớp bằng softmax học được, Linear → 128, attention pooling | 128 | **không** | có |
| Sync cục bộ | `a`, `v` | hồ sơ cosine cos(a_t, v_t+δ), δ ∈ [−15, 15], không tham số, trung bình trên frame của đơn vị; thêm đỉnh, vị trí đỉnh / 15, entropy | 34 | có | không |
| Hình học môi (chỉ khi 0c' đạt) | `lips` (cache mouth, đã chia bề rộng miệng), landmark thô (cache avhubert), `pose` | độ mở môi trong / bề rộng: min, max, trung bình, giá trị quanh frame cuối âm tiết của bộ căn; bề rộng miệng / khoảng cách khoé mắt ngoài; tỉ lệ frame dưới τ của người đó; vận tốc độ mở; vận tốc tư thế đầu; chuẩn hoá z theo clip tham chiếu của người đó | ~12 | có | không |
| Kỳ vọng theo người (chỉ khi 0c' đạt) | 3 clip real tham chiếu | tỉ lệ khép của lớp tương ứng (§2.2) | 4 | có | không |
| Thẻ | §2.2 | embedding 8 chiều mỗi thẻ: `onset`, `coda`, `round_on`, `round_nuc`, `type`; `tone` chỉ ở đường tiếng | 40 / 16 | có (không có `tone`) | `tone`, `type` |
| Vị trí | | vị trí tương đối trong clip; **không có độ dài tường minh** (N4) | 1 | có | có |

Về độ dài: bỏ đặc trưng độ dài tường minh, nhưng Transformer vẫn suy được độ dài từ vị trí của các đơn vị kề nhau (embedding thời gian theo tâm đơn vị). Vì vậy lối tắt co giãn chỉ chặn tận gốc được ở dữ liệu (giới hạn tỉ lệ co giãn của track B); probe ở 0d' cho biết nó mạnh tới đâu.

### 2.5. Mạng

- **Đường gộp:** Linear (~347 → 128, LayerNorm, GELU) + embedding thời gian hình sin theo tâm đơn vị → 2 lớp Transformer encoder (d = 128, 4 head, FFN 256, dropout 0,1, pre-norm, mask đệm) → 2 đầu MLP (128 → 64 → 1): **visual**, **mismatch**.
- **Đường tiếng:** Linear (~273 → 128) + embedding thời gian → 1 lớp Transformer như trên → đầu **audio**.
- Đầu visual và mismatch vẫn thấy cả hai phía, vì phép so "kỳ vọng từ tiếng ↔ quan sát trên môi" là chính cơ chế của H-B; nhưng không thấy XLS-R, nên khó học "tiếng giống TTS ⇒ môi giả" từ ⑥. Rủi ro đầu visual bật nhầm trên ② và ③ không loại hẳn được bằng kiến trúc; nó được đo bằng G6b và hạn chế trên demo.
- Tham số học (○): 3 attention pooling ~0,39 M; đường gộp ~0,33 M (chiếu đầu vào ~0,05 M, 2 lớp Transformer ~0,27 M, 2 đầu ~0,02 M); đường tiếng ~0,18 M. Tổng **~0,9 M** (bản cũ ~0,75 M; M1-v003 1,94 M).

### 2.6. Đầu ra, gộp, định vị, giải thích

- Điểm mỗi đơn vị được gán cho mọi frame của đơn vị đó, nên M6 giữ giao diện `Model` (logit theo frame [B, T] cho từng đầu) và loss mặc định (BCE theo frame + BCE theo video, `pos_weights` như v003).
- Gộp video: top-5 frame như M1.
- AI? = max(z_visual, z_audio) với z theo real val và chuẩn hoá theo 3 clip tham chiếu (P4 của v005). Khi so với M1-v003, cả hai dùng **cùng hậu xử lý** (N6).
- Định vị: thước đo P6 của v005, báo theo **tầng**: tầng đoạn ngẫu nhiên (②, ④, ⑤ partial) là tầng chính; tầng theo biên từ (③B, ③C, ⑥) báo riêng với ghi chú "biên = mốc ASR"; thêm sai số biên (ms), AP@0,5 và lát đoạn ≤ 0,6 s.
- Giải thích trên demo (Gradio): chỉ bật sau khi 0c' xác nhận phép đo. Hiện điểm visual và mismatch cạnh nhau, vì một âm tiết lệch tiếng–hình có thể là ② sham chứ không phải môi giả. Không hiện số đo môi khi miệng nhỏ hơn một ngưỡng pixel (đặt theo 0c'). Ghi rõ đó là số đo landmark, không phải trọng số model.

### 2.7. Huấn luyện

- **Pseudo-fake theo đơn vị, p = 0,25 như M1 (V8).** Thay đoạn [s, e) dài 0,3–3 s của `a`, `ssl`, `wave` bằng đoạn cùng vị trí của một clip real khác cùng người (như `replace_audio`), và **thay luôn các đơn vị và thẻ** của đoạn đó bằng của clip nguồn (cắt ở hai mép). Một nửa số lần đặt s, e đúng biên âm tiết của clip nguồn. Nhãn: mismatch = 1, AI = 0. Cách này giữ M6 ngang M1 về cách train, dạy đầu visual rằng "kỳ vọng bị vi phạm" chưa chắc là môi giả, và đưa "ghép theo biên từ nhưng không có AI" vào train để chống lối tắt ở §3 mà không phải sinh lại dữ liệu.
- Augment dịch tiếng −1…+2 frame, mốc đơn vị dịch theo (§2.3).
- Train trên cả clip (`crop_frames = None`, clip ≤ 175 frame) vì đơn vị cần trọn âm tiết và ngữ cảnh câu.
- Cân theo người; AdamW lr 3e-4, weight decay 1e-4, batch 16, tối đa 30 epoch, patience 5; chọn checkpoint theo val loss. Giống M1 để so công bằng (dừng sớm theo tổng val loss 3 đầu là việc mở #14 của v003, áp chung cho mọi model).
- Không dùng loss gây nhầm lẫn người nói (M6 không có nhánh trace; W3 cho thấy cơ chế người của M1 chưa chứng minh được đóng góp).

### 2.8. Sơ đồ

```
tiếng của mẫu ─ PhoWhisper (lời) ─ bộ căn chọn ở 0b' ─→ âm tiết {lời, start, end} ─→ luật chính tả ─→ thẻ
                                                                │
                                                     chia frame thành đơn vị
                                                     (âm tiết + khoảng lặng)
cache 25 Hz ──────────────────────────────────────────────────┐ │
  v (AV-HuBERT hình), a (AV-HuBERT tiếng), ssl (XLS-R)         ├─→ attention pooling theo đơn vị
  lips, landmark thô, pose ──→ hình học môi theo đơn vị (sau khi dịch tiếng, chuẩn hoá theo người)
3 clip real tham chiếu ─────→ kỳ vọng khép theo người, τ theo người

đường gộp:  [v, a, hồ sơ cosine ±15, hình học môi, kỳ vọng theo người, thẻ] → 2 lớp Transformer → visual, mismatch
đường tiếng: [a, ssl, thanh, loại đơn vị]                                  → 1 lớp Transformer → audio

điểm đơn vị → gán về frame → top-5 → điểm video
AI? = max(z_visual, z_audio) (P4) · định vị = dãy đơn vị vượt ngưỡng (P6, báo theo tầng)
```

### 2.9. M6 sau review so với M5 (v005)

| Thành phần | M5 (v005) | M6 (v006) sau review |
|---|---|---|
| Đơn vị | frame 40 ms, TCN | âm tiết (bộ căn chọn ở 0b') + khoảng lặng, Transformer trên cả câu |
| Sync | cost volume ±15 có chiếu học được + CNN 2D | hồ sơ cosine ±15 không tham số, gộp theo đơn vị |
| Hình | crop miệng (ảnh, NPR, residual thời gian) + AV-HuBERT v + động học môi | AV-HuBERT v + hình học môi theo âm tiết (chỉ khi 0c' đạt); không crop |
| Kiến thức ngôn ngữ | không | thẻ khẩu hình từ chính tả, có tính phương ngữ; kỳ vọng khép theo người |
| Nối đầu ra | mỗi đầu một nhánh (P3) | audio có đường riêng; visual và mismatch chung đường gộp (không có XLS-R) |
| Tham chiếu người | chuẩn hoá điểm (P4) | P4 + kỳ vọng khép và τ theo người |
| Pseudo-fake | thay đoạn tiếng, p = 0,25 | thay đoạn tiếng + đơn vị + thẻ, p = 0,25, một nửa theo biên âm tiết |
| Cắt khi train | 75 frame | cả clip |
| Tham số học | gần 1,94 M (○) | ~0,9 M (○) |
| Giải thích | không | theo âm tiết, chỉ sau khi 0c' xác nhận phép đo |
| Cache cần | `av`, `ssl`, `mouth` (crop, môi, tư thế đầu) | `av` (kèm landmark thô), `ssl`, `mouth` (chỉ môi và tư thế đầu, không đọc crop), **`syll` mới** |

### 2.10. Phần mới, sau khi thu hẹp

| Khối | Đã có | Phần của v006 |
|---|---|---|
| Âm vị ↔ khẩu hình | Agarwal et al. 2020 (M, B, P; tiếng Anh; người sửa transcript rồi căn bằng P2FA); PIA (ICCVW 2025: luồng ảnh môi theo âm vị, và độ mở miệng MAR theo âm vị cho /m/, /b/, /p/ và nguyên âm tròn; tiếng Anh); NPVForensics (2023) | luật phụ âm cuối môi-ngạc mềm [ŋ͡m], [k͡p̚] của tiếng Việt suy từ chính tả, có tính phương ngữ, đo theo đơn vị âm tiết với kỳ vọng theo người |
| Neo theo từ hoặc âm vị | WAFL (03/2026: sửa theo từ, mốc từ Google Speech-to-Text); He & Wang, PRCV 2025 (nhất quán ngữ nghĩa mức từ giữa đọc môi và ASR, cộng căn chỉnh mức âm vị, chỉ train trên real) | âm tiết tiếng Việt + khoảng lặng, kèm ② sham và 3 đầu ra |
| Đa ngôn ngữ | BabelFake (5 ngôn ngữ châu Âu, không có tiếng Việt); ArEnAV (bộ deepfake nghe nhìn Ả Rập–Anh) | thiết kế cho một ngôn ngữ thanh điệu, đơn âm tiết |

Sửa trích dẫn so với bản cũ (V6, L6, L9): ablation 98,06 → 65,57 của PIA là bỏ **ảnh** môi theo âm vị, không phải bỏ thẻ; số AV-1M của WAFL là trên tập validation, và ở phép thử chéo bộ AP@0,5 của WAFL (45,22) thấp hơn DiMoDif (59,71) và AuViRe (48,54); arXiv 2604.16808 bản v4 tên "BioLip" (`docs/v003_model.md` ghi bản v5 đổi tên "LADDER", chưa kiểm); LipDA chưa có nguồn ghi venue; bỏ câu "khép môi là lớp ổn định nhất" (Bear & Harvey không nói vậy).

Hai nguồn reviewer tìm thấy nhưng không mở được ("Lip Forgery Video Detection via Multi-phoneme Selection" của USTC; một bài trên isj.vn về phát hiện deepfake nghe nhìn) có thể đụng phần tính mới: phải đọc trước khi chốt claim trong báo cáo. Báo cáo nên viết "kết hợp có lý do, mỗi khối kiểm bằng ablation", không viết "kiến trúc mới hoàn toàn".

---

## 3. Lối tắt trong dữ liệu v003 (thay §4 cũ; ảnh hưởng cả v003, v005, v006)

Cách mỗi ô chọn đoạn và ghép (`src/vn_av_df/generation/generate.py:236-237`, `:286-301`, `:323-357`, `:430-447`; `voice/edit.py:10-50`, `:88-115`; `voice/audio.py:67-100`, `:230-264`) ✔:

| Ô / biến thể | Cách chọn đoạn | Tiếng | Hình |
|---|---|---|---|
| ② sham partial | `random_span` (đều trên lưới frame) | ghép tiếng thật của clip khác bằng `A.splice` (crossfade 15 ms, khớp độ to, tiếng nền) | môi gốc |
| ③ track A (full) | cả câu | TTS đọc lại, co giãn trong ±15 % rồi đệm hoặc cắt | môi gốc |
| ③ track B (partial) | `choose_edit` trên mốc từ PhoWhisper của tiếng gốc, theo bảng sửa cố định | cụm TTS co giãn **không giới hạn** cho vừa [p, q], ghép bằng `A.splice` như ② | môi gốc |
| ③ track C (chỉ test) | cùng span với track B | F5 inpainting, nối **5 ms, không khớp độ to** | môi gốc |
| ④ source partial | `random_span` | **không ghép**, tiếng gốc | môi vẽ lại trong đoạn ngẫu nhiên |
| ⑤ donor partial | `random_span` | ghép tiếng donor vào đoạn | môi vẽ lại trong đoạn ngẫu nhiên |
| ⑥ both | span của track B | tiếng của track B (cả bản full lẫn partial đều có đoạn tiếng AI theo biên từ) | full: môi cả clip; partial: **biên trộn hình theo biên từ** |

**Hệ quả** (○, chưa đo):
1. "Ghép đúng biên từ" đi cùng nhãn "giọng AI" (③B, ③C, ⑥), và ở ⑥ partial đi cùng cả nhãn "môi giả"; "ghép giữa âm tiết" đi cùng ② (lệch, không AI) và ⑤ (môi giả, tiếng thật). Model nào thấy được vị trí vết ghép so với biên âm tiết (M1 qua XLS-R, M6 qua đơn vị âm tiết) có thể học lối tắt này. Ngoài đời người sửa lời thường cắt đúng biên từ, nên lối tắt làm model dễ báo nhầm các đoạn ghép tiếng thật nhưng cắt khéo.
2. Bảng sửa cố định (có ↔ không, số đếm, vài từ thay thế) tạo lối tắt từ vựng. Nhiều từ trong bảng mang thẻ khép môi ("không", "hôm", "năm", "tám", "trăm"; "một", "bốn" với người miền Nam), nên lối tắt này lẫn vào chính tín hiệu của H-B.
3. Co giãn không giới hạn ở track B làm tốc độ âm tiết trong đoạn lệch khỏi tốc độ nói của người đó.
4. Độ dài đoạn phân bố khác nhau theo ô (đều theo frame ở `random_span`, theo cụm từ ở `choose_edit`).
5. ③C nối 5 ms và không khớp độ to, khác train: lát "giọng chưa thấy" lẫn cả "kiểu vết ghép chưa thấy".
6. Vì ③C ở test dùng cùng bảng và cùng cách chọn đoạn, các lối tắt trên **làm điểm test cao lên** chứ không làm lộ ra; chúng còn cho mọi model neo theo âm tiết một lợi thế định vị có sẵn (C1).

**Kiểm** ở 0d' (chỉ val, §4.1). **Sửa tận gốc** chỉ có ở dữ liệu (bảng sửa rộng hơn, giới hạn co giãn, bảng riêng cho test): việc của team cho phiên bản dữ liệu sau, không làm trong v006.

---

## 4. Kế hoạch thí nghiệm

### 4.1. Nấc 0 (chỉ train + val part 1; không đụng test)

| Bước | Nội dung | Đạt khi | Không đạt thì |
|---|---|---|---|
| 0a | Cache âm tiết cho 5.286 mẫu train + val part 1 (train 4.234, val 1.052); đo tốc độ trên 50 clip trước | có cache và số đo tốc độ | |
| 0b' | Gán tay (Praat) 200–300 biên âm tiết trên 20–30 clip real thuộc train / val, nhiều người, cả nam và nữ, cả hai miền nếu có; so PhoWhisper DTW, CTC wav2vec2 tiếng Việt, MFA; phép thử biên ghép của ③B giữ làm kiểm tra phụ | một bộ căn có trung vị sai số ≤ 1 frame và P90 ≤ 2 frame ở **cuối âm tiết** (báo riêng cuối âm tiết đứng trước khoảng lặng) | bỏ H-A1, H-A2 và hình học môi đo ở cuối âm tiết; M6 chỉ còn là baseline rẻ, hoặc dừng |
| 0c' | Pilot khép môi có đối chứng (chi tiết dưới bảng) | `labial_velar` **tự** đạt trên chênh "fake − real đã suy biến" ở ≥ 2 / 4 generator train, cận dưới CI theo người > 0; phép đo được nhãn người xác nhận ở tầng miệng lớn (đề xuất: độ chính xác ≥ 0,8, kappa giữa hai người gán ≥ 0,6) | bỏ hình học môi đo tay và kỳ vọng theo người; H-B chỉ còn thẻ (kiểm bằng G4); bỏ claim "riêng tiếng Việt" |
| 0d' | Probe lối tắt và kiểm §3 (chi tiết dưới bảng) | biết lối tắt nào có mặt và đã quyết cách chặn trước khi train | |
| 0e | Encoder AV-HuBERT tiếng Việt (ViCocktail), **chỉ khi có giấy phép**: probe tuyến tính real so với ④ trên val, kèm điểm sync không train | probe hơn encoder hiện tại ≥ +0,02 | bỏ nấc 2 |
| 0f | Ghi vùng phương ngữ (Bắc / Trung / Nam, ước bằng tai) cho ~80 người của part 1 + 2 trong ledger | có nhãn | báo cáo không chia được theo vùng |

**0c' chi tiết** (C2; chỉ CPU, trừ bước chạy lại landmark trong env worker AV-HuBERT):
1. Cặp ④ ↔ ① full theo từng generator train (cùng tiếng, cùng lời, cùng người, chỉ khác môi), như bản cũ.
2. **Real bị làm suy biến:** với mỗi cặp, làm suy biến vùng mặt hoặc miệng của ① đúng như pipeline của generator đó (thu về 96, 128 hoặc 256 px, phóng lại, dán vào, encode lại CRF 18), chạy lại landmark bằng đúng cấu hình worker. Tiêu chí dùng chênh "fake − real đã suy biến", không dùng "fake − real".
3. **Phép đo thứ hai** không dùng landmark: profile cường độ dọc qua tâm miệng trên crop 96 × 96 có sẵn trong cache `mouth` (kiểu Agarwal et al.; chỉ cần numpy). MediaPipe Face Mesh là tuỳ chọn nếu team đồng ý cài thêm.
4. **Nhãn người:** hai người gán mù "khép / không khép" cho khoảng 200 frame cuối âm tiết lấy từ ① và ④; báo kappa và độ chính xác của từng phép đo theo ô và theo cỡ miệng.
5. **Phân tầng theo cỡ miệng** (bề rộng 48–54 tính bằng pixel ở khung 640) và theo generator; hiệu ứng phải còn ở nửa có miệng to.
6. **Kiểm nén dải:** đo cả độ mở lớn nhất ở nguyên âm mở (a, á…). Nếu fake cũng mở ít hơn ở đó thì là nén dải, không phải lỗi khép môi.
7. Bỏ các âm tiết sát khoảng lặng (môi thường khép khi nghỉ) và các âm tiết đứng trước âm đầu b, m, p (khép do đồng cấu âm); đo ở mốc cuối âm tiết của bộ căn chọn ở 0b'.
8. Tiêu chí riêng cho `bilabial`, `labial_velar`, `round_alveolar`, hiệu chỉnh Holm; τ theo người; báo theo nhóm vùng (0f).

**0d' chi tiết** (N4, V3, V11, L7; chỉ val):
1. Probe hồi quy logistic nhỏ dự đoán "đơn vị là tiếng giả", mỗi probe một nhóm đặc trưng: (a) chỉ thẻ và từ ASR quanh đơn vị; (b) chỉ độ dài đơn vị chia trung vị của clip và tốc độ âm tiết; (c) chỉ cấu trúc ASR (âm tiết mỗi giây, số khoảng lặng, số từ bị rơi). Định trước: AUC của probe > 0,6 nghĩa là có lối tắt.
2. Điểm tác vụ audio của M1-v003 chia theo loại sửa (đảo nghĩa, số, thay thế) và theo |stretch_ratio − 1| (có trong manifest).
3. FPR trên clip real của val có chứa từ trong bảng sửa so với clip không chứa.
4. Tỉ lệ đi đường dự phòng `unknown` theo ô và split.
5. Vị trí vết ghép so với biên âm tiết, với biên lấy từ ASR của **tiếng gốc và tiếng donor** (không phải tiếng đã ghép), và ngưỡng chặt hơn ±1 frame.
6. (Cần team đồng ý; chỉ CPU, cộng vài phút GPU suy luận) tập phân tích nhỏ "② ghép theo biên từ" chỉ cho val, dựng bằng đúng `A.splice` trên cache ASR của tiếng gốc; so điểm M1-v003 với ② ngẫu nhiên của cùng clip. Không dùng để train.

Nếu thấy lối tắt: chặn bằng cách bỏ đặc trưng tương ứng hoặc bằng pseudo-fake theo biên âm tiết (§2.7), nếu không chặn được thì ghi là giới hạn; sửa dữ liệu là quyết định của team.

### 4.2. Nấc 1: train trên part 1

| Cấu hình | Kiểm gì | Seed |
|---|---|---|
| `visyl` (M6 đầy đủ) | mô hình chính | 5 |
| `visyl_no_phon` (bỏ thẻ khẩu hình, hình học môi, kỳ vọng theo người) | so với `visyl`: H-B tổng thể | 5 |
| `visyl_uniform_no_phon` (cửa sổ đều 0,2 s phủ cả clip, không dùng ASR) | so với `visyl_no_phon`: H-A2 | 5 |
| `visyl_shuffled_tags` (thẻ xáo trộn trong cùng clip, giữ phân bố thẻ) | thẻ có mang thông tin của đúng âm tiết hay không | 5 |
| `visyl`, đảo `labial_velar` → `other` lúc suy luận | phần riêng của tiếng Việt; không train thêm | dùng 5 seed của `visyl` |
| (nếu còn quota) `visyl_tags_only`, `visyl_geom_only` | tách thẻ khỏi hình học môi | 3 |

- 20 lần train chính. Đo 1 epoch trước khi lập lịch: M6 đọc XLS-R như M1, nên ước theo M2 (0,25 giờ, không đọc XLS-R) là lạc quan. Ước 0,5–1,5 giờ T4 mỗi lần, tổng ~10–30 giờ (○).
- Tiêu chí chính định trước: AI? trên val với P4, trung bình 5 seed. Tiêu chí phụ theo thứ tự: định vị P6 trên tầng đoạn ngẫu nhiên, sham, 3 tác vụ. Mục đích của bước chọn: quyết cấu hình nào vào bước kết và claim nào còn đáng kiểm trên test.
- So với M1-v003 trên val chỉ để định hướng (M1-v003 chỉ có 1 seed).

### 4.3. Nấc 2 (có điều kiện)

0e đạt và có giấy phép: trích lại `av` part 1 bằng encoder tiếng Việt, train lại `visyl` (5 seed). Nếu encoder giúp M6 thì cũng thử trên M5 (v005), để không nhầm "encoder tốt hơn" với "M6 tốt hơn".

### 4.4. Bước kết (chung với v005)

- part 1 + 2, 3 seed, test part 2 mở một lần; `visyl`, `visyl_no_phon`, `visyl_uniform_no_phon` chạy cùng đợt với M1-v003, M5 (v005), M2 và baseline (`visyl_shuffled_tags` chỉ dùng ở nấc 1).
- **Người test của part 1** (V7; quyết định chung với v005). Code ghép các part mà không đổi split (`dataset.py:205`), nên phải chọn trước: (a) bỏ hẳn họ khỏi bước kết; hoặc (b) chuyển họ sang train nhưng loại mọi mẫu dùng generator hoặc giọng chỉ-test (LatentSync, Diff2Lip, VoxCPM2, viXTTS, F5 inpainting). Đề xuất (b): thêm 10 người cho train mà không làm hỏng lát "chưa thấy". Cách nào cũng là một phiên bản dữ liệu mới, cần code lọc và ghi provenance, không ghi đè `vn-av-df-data-part1`.

### 4.5. Cổng (định trước; team chốt trước khi mở test part 2)

Tất cả trên test part 2. Lấy trung bình điểm 3 seed cho từng mẫu rồi bootstrap theo người (10 người), kèm Δ của từng người. M1-v003 và M6 dùng cùng hậu xử lý (P4 là bản chính, báo thêm bản thô).

| Cổng | So sánh · tập | Đạt khi |
|---|---|---|
| **G1 (chính)** | M6 − M1-v003 · AI? · toàn bộ test | cận dưới CI 95 % ≥ −0,03 (đề xuất; xem lực bên dưới) |
| G2 (H-A2, định vị) | M6 − M1-v003 · P6 trên tầng đoạn ngẫu nhiên (②, ④, ⑤ partial + real) | cận dưới CI > 0 (bootstrap AP theo người) |
| G3 | M6 − M1-v003 · tác vụ visual · LatentSync, Diff2Lip | Δ ≥ +0,02 và cận dưới CI > 0 |
| G4 (H-B) | tương tác: (`visyl` − `visyl_no_phon`) ở partial có âm tiết `labial_velar` trong đoạn giả, trừ cùng hiệu ở partial không có, trong cùng dải độ dài đoạn; kèm phép đảo thẻ | cận dưới CI > 0; đảo thẻ làm điểm visual của các âm tiết đó giảm ở ④ (chiều định trước) và gần như không đổi ở ① |
| G5 (H-A2) | `visyl_no_phon` − `visyl_uniform_no_phon` · P6 trên tầng đoạn ngẫu nhiên, lát đoạn ≤ 0,6 s | cận dưới CI > 0 |
| G6 | M6 · AI? sham so với real, tách full, partial, partial ≤ 0,6 s | ≤ AUC của M1-v003 + 0,03 và ≤ 0,60 |
| G6b | M6 · đầu visual: ① so với ② partial; ① so với ③ | ≤ đầu visual của M1-v003 + 0,03 |
| G6c | M6 · đầu audio: ① so với ④, ⑤ (tiếng thật, môi giả) | AUC ≤ 0,60 |
| G7 | FPR ở ngưỡng 5 % chọn trên val | ≤ FPR của M1-v003 + 2 điểm %; báo kèm FPR của người cao nhất |
| G8 (chỉ báo cáo) | tầng theo biên từ (③B, ③C, ⑥); tỉ lệ dự phòng ASR theo ô; kết quả theo nhóm vùng | báo |

**Luật giữ.** M6 được giữ làm đề xuất nếu G1 đạt và (G2 hoặc G3) đạt. Claim H-A2 chỉ khi G5 đạt. Claim H-B chỉ khi 0c' và G4 đạt. Claim "riêng tiếng Việt" chỉ khi phần `labial_velar` tự đạt ở 0c' và ở phép đảo thẻ. Các biên 0,03 và 2 điểm % là đề xuất (của reviewer và của mình) để team chốt; điều quan trọng là có biên trước khi mở test.

**Lực kiểm định (○, phải ghi vào báo cáo).** CI của M1 − BL-M/sup trên slide là [0,080; 0,125], tức SE khoảng 0,0115 với 10 người. Nếu Δ của M6 − M1-v003 có độ rộng tương tự:

| Δ thật | P(G1 đạt), biên −0,02 | P(G1 đạt), biên −0,03 |
|---|---|---|
| −0,01 | 0,14 | 0,41 |
| 0 | 0,41 | 0,74 |
| +0,01 | 0,74 | 0,94 |

Với biên −0,02, G1 gần như là phép thử "M6 không thua về điểm". Các cổng "cận dưới CI > 0" cần hiệu ứng thật cỡ ≥ 0,03 AUC mới có lực ~0,8. Hai model giống nhau thường có Δ tương quan theo người nên SE thật có thể nhỏ hơn, nhưng chưa đo được.

---

## 5. Code cần viết (thay câu "dùng nguyên" của bản cũ)

Nguyên tắc: chỉ thêm, không sửa tại chỗ `TriHeadV2`, `proposals()` hay `features/encoders.py`. Mọi `training/*.py` và `encoders.py` bị hash vào provenance của run (`train.py:589-590`); sửa `encoders.py` còn làm cũ cache XLS-R, mouth, LIPINC, wave (V2).

1. `features/syllables.py`: ASR, bộ căn, luật thẻ, chia khoảng lặng; cache npz + json khoá theo hash tiếng của chính mẫu; chữ ký đầy đủ (§2.3). Đăng ký need `syll` trong `NEEDS` của `training/prepare.py` và `ENCODERS_FOR_NEED` của `data.py`.
2. Luật chính tả → thẻ, có test trong `tests/` (qu, gi, oo, chữ số, uông / uôc, các vần môi-ngạc mềm, un / ut / ôn / ôt).
3. `data.py`: need mới đọc `syll` và `landmarks` (khoá trong cache avhubert); dịch mốc đơn vị cùng `lag_audio`; pseudo-fake theo đơn vị.
4. `models.py`: lớp `ViSyl` và các cấu hình ở §4.2, đăng ký trong `SPECS` (`crop_frames = None`, `pseudo_fake = 0.25`).
5. Hình học môi theo đơn vị và kỳ vọng theo người, tính sau khi dịch tiếng; τ và chuẩn hoá theo clip tham chiếu.
6. P4 (đã có trong kế hoạch v005) và P6 theo tầng (đoạn ngẫu nhiên, đoạn theo biên từ, ≤ 0,6 s); sai số biên theo ms.
7. `report.py`: lấy trung bình điểm theo seed trước khi bootstrap (hiện chỉ dùng seed đầu); bootstrap theo người cho AP; cổng G1–G8 với biên định trước.
8. Lát đánh giá đọc `syll` (G4, G8) và phép đảo thẻ lúc suy luận.
9. Kiểm hồi quy: evaluate chỉ ghi hash checkpoint, không kiểm hash code, nên sau khi sửa `data.py` hay `evaluate.py` phải chạy lại M1-v003 trên val và thấy điểm giống hệt trước khi sửa.
10. Sinh lại notebook bằng `tools/make_notebooks.py` (không sửa notebook tay).

Ước công (○): khoảng 3–5 ngày người cho mục 1–8, chưa kể nấc 0.

---

## 6. Chi phí và giấy phép

| Hạng mục | Ước (○) |
|---|---|
| 0a: cache âm tiết train + val part 1 | 0,5–1,5 giờ GPU (đo 50 clip trước) |
| 0b': gán tay 200–300 biên, so 3 bộ căn | 3–5 giờ công + vài giờ CPU; MFA cần env riêng (chỉ khi CTC trượt) |
| 0c': real suy biến, chạy lại landmark, phép đo thứ hai, nhãn người 200 frame | vài giờ CPU + 1–2 giờ công |
| 0d': probe và phân tích | CPU; tập ② theo biên từ: CPU + vài phút GPU |
| 0f: nhãn vùng ~80 người | 1–2 giờ công |
| Nấc 1 (20 lần train) | ~10–30 giờ T4 |
| Cache phần còn lại (test part 1, part 2) | ~1–2 giờ GPU |
| Bước kết (3 cấu hình × 3 seed trên part 1 + 2) | ~9–27 giờ T4 |
| **Tổng v006** | **~20–60 giờ GPU + ~8–12 giờ công gán nhãn** (bước kết của v005: ~100–140 giờ GPU) |

Giấy phép:
- PhoWhisper-small: BSD-3-Clause ✔.
- wav2vec2-base-vietnamese-250h: CC BY-NC 4.0 ✔. Hợp với đồ án phi thương mại; nếu demo về sau có yếu tố thương mại thì không dùng được.
- MFA tiếng Việt (mô hình âm học và từ điển Hà Nội, TP.HCM): CC BY 4.0 ✔ (review).
- vPhon: GPL-3.0, chỉ đối chiếu ngoài repo.
- ViCocktail (model `nguyenvulebinh/AV-HuBERT-CTC-Attention-VI`): không ghi giấy phép ở repo, model card và dataset card ✔; phải hỏi tác giả trước khi dùng. Paper ghi crop 96 × 96 từ landmark dlib 48–67; v003 căn theo mean face rồi cắt 88 từ ROI 96, nên "cùng cách crop với v003" chưa kiểm (○).
- Bộ VLR (Zenodo 19705466): CC BY 4.0 ✔, nhưng nguồn là video YouTube; giấy phép của bộ dữ liệu không tự cho phép mọi cách dùng nội dung gốc (CLAUDE.md, Data Rules). v006 không cần bộ này.

---

## 7. Giới hạn còn lại sau khi sửa

1. **10 người test.** Hiệu ứng nhỏ hơn ~0,03 AUC sẽ không kết luận được, kể cả khi có thật.
2. **H-B có thể sai.** Generator có thể khép môi đủ tốt ở [ŋ͡m], [k͡p̚]; khi đó v006 chỉ còn H-A2.
3. **Phép đo ở 25 Hz.** Phụ âm tắc cuối [p̚], [k͡p̚] có thể chỉ kéo dài 1–2 frame, và mức khép thật của [ŋ͡m], [k͡p̚] (khép hẳn hay khép một phần) chưa được kiểm trong tài liệu ngữ âm. Đối chứng ở 0c' cho biết phép đo có đáng tin không, nhưng không làm nó nhạy hơn.
4. **Phương ngữ.** Nhãn vùng ước bằng tai; khác biệt ngay trong một vùng ("many speakers") chỉ được bù bằng kỳ vọng theo người, vốn dựa trên 3 clip tham chiếu.
5. **Lối tắt trong dữ liệu.** Lối tắt từ vựng, độ dài và vị trí vết ghép không loại hẳn được bằng kiến trúc; chỉ dữ liệu mới sửa tận gốc. Kết quả trên ③C ở test có thể bị thổi lên.
6. **Định vị trên tầng theo biên từ** là lợi thế cấu trúc, không phải bằng chứng model tốt hơn; vì vậy nó không làm cổng.
7. **Đầu visual nhìn cả tiếng.** Đó là cơ chế của H-B nên không tách được; bật nhầm trên ② và ③ chỉ được đo (G6b) và hạn chế trên demo.
8. **AV-HuBERT pretrain tiếng Anh** (rủi ro X3 của v003) vẫn còn nếu ViCocktail không có giấy phép.
9. **Số v003 chỉ từ slide.** Nấc 0 cần output của Linh (`predictions-*.npz`, `history.json`, `metrics.json`, `best.pt`).
10. **Test part 1 đã được xem.** Thiết kế này đã được thông tin bởi kết quả v003 và bởi chính vòng review này; chỉ test part 2 là bằng chứng sạch.

---

## 8. Ảnh hưởng tới v005 và v003

Các phát hiện sau cũng đúng cho v005 (và cho cách đọc kết quả v003); nên đưa vào nấc 0 của v005:
- **Lối tắt dữ liệu (§3):** thêm biên trộn hình của ⑥ theo biên từ, ③C nối 5 ms, bảng sửa cố định, co giãn không giới hạn.
- **Một chi tiết bị ghi sai** ở W5 của file tổng hợp và trong review v005: giới hạn co giãn ±15 % chỉ có ở track A; track B co giãn không giới hạn. Câu W5 trong file tổng hợp đã được sửa.
- **Encoder AntiDeepfake (P5)** không được thêm vào `features/encoders.py` (làm cũ cache XLS-R, mouth, LIPINC, wave); dùng module riêng như `syll`.
- **G1 của v005** (biên −0,01): với cùng giả định SE, khi M5 (v005) thật sự ngang M1-v003 thì xác suất đạt chỉ khoảng 0,14 (○). Team nên xem lại biên hoặc ghi rõ lực.
- **`report.py` tính cổng trên seed đầu:** bước kết 3 seed của cả hai bản cần sửa chỗ này.
- **Người test của part 1 ở bước kết** (§4.4): quyết định chung.

---

## 9. Quyết định team cần

| Quyết định | Đề xuất | Vì sao |
|---|---|---|
| Nhận v006 bản sau review ở mức nấc 0 | Đồng ý; quyết lại sau 0b', 0c' | Nấc 0 rẻ và quyết định được cả H-A lẫn H-B |
| Ai gán tay biên âm tiết (0b'), khép môi (0c'), vùng phương ngữ (0f) | Chia trong team, ~8–12 giờ công | Không có đáp án độc lập thì 0b', 0c' không có giá trị |
| Biên của G1 | −0,03 kèm câu về lực; hoặc giữ −0,02 và ghi rõ đó là phép thử "không thua" | Chỉ có 10 người test |
| Người test part 1 ở bước kết (chung với v005) | Chuyển sang train, loại mẫu của generator và giọng chỉ-test | Thêm dữ liệu mà giữ được lát "chưa thấy" |
| Tập phân tích "② theo biên từ" chỉ cho val | Đồng ý (chỉ phân tích, không train) | Đo trực tiếp được lối tắt §3 |
| Sửa dữ liệu (bảng sửa rộng hơn, giới hạn co giãn, bảng riêng cho test) | Cho phiên bản dữ liệu sau, không làm trong v006 | Đổi bước sinh, ảnh hưởng mọi model |
| Hỏi tác giả ViCocktail về giấy phép | Làm song song | Chặn nấc 2 nếu không có |
| Cài MFA (env riêng) | Chỉ khi CTC trượt 0b' | Không thêm env khi chưa cần |

---

## 10. Nguồn

**Nội bộ.** `plans/2026-10-30_v005-v006_de-xuat-detector.md` (bản trước review); `reviews/2026-10-30_review-v005.md` (review, có dòng code và URL cho từng phát hiện, kèm smoke check); `reviews/2026-10-28_review-v004.md`. Repo `VN-AV-DF-Capstone` commit `2b3a70c`: `src/vn_av_df/generation/generate.py`, `generation/cells.py`, `generation/plan.py`, `generation/voice/asr.py`, `voice/edit.py`, `voice/audio.py`; `src/vn_av_df/training/models.py`, `data.py`, `train.py`, `evaluate.py`, `report.py`, `prepare.py`; `src/vn_av_df/features/encoders.py`, `avhubert.py`, `avhubert_worker.py`; `src/vn_av_df/dataset.py`; `settings.py`; `docs/v003_data.md`, `docs/v003_model.md`, `docs/v003_generation.md`.

**Web** (mở 30/10/2026 bởi mình hoặc reviewer)
- Vietnamese phonology: [Wikipedia](https://en.wikipedia.org/wiki/Vietnamese_phonology) · Labial–velar consonant: [Wikipedia](https://en.wikipedia.org/wiki/Labial%E2%80%93velar_consonant)
- PhoWhisper-small: [Hugging Face](https://huggingface.co/vinai/PhoWhisper-small), [generation_config.json](https://huggingface.co/vinai/PhoWhisper-small/blob/main/generation_config.json) · whisper-small: [generation_config.json](https://huggingface.co/openai/whisper-small/blob/main/generation_config.json)
- wav2vec2-base-vietnamese-250h: [Hugging Face](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)
- MFA tiếng Việt: [mô hình](https://huggingface.co/MontrealCorpusTools/vietnamese_mfa), [từ điển](https://mfa-models.readthedocs.io/en/latest/dictionary/Vietnamese/Vietnamese%20MFA%20dictionary%20v3_0_0.html) · vPhon: [GitHub](https://github.com/kirbyj/vPhon)
- CrisperWhisper: [arXiv 2408.16589](https://arxiv.org/abs/2408.16589) · Rousso et al., Interspeech 2024: [arXiv 2406.19363](https://arxiv.org/abs/2406.19363)
- "Can LLMs Help Localize Fake Words in Partially Fake Speech?": [arXiv 2603.11205](https://arxiv.org/abs/2603.11205)
- WAFL: [arXiv 2603.06220](https://arxiv.org/abs/2603.06220) · PIA, ICCVW 2025: [arXiv 2510.14241](https://arxiv.org/abs/2510.14241)
- Agarwal et al., CVPRW 2020: [CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w39/Agarwal_Detecting_Deep-Fake_Videos_From_Phoneme-Viseme_Mismatches_CVPRW_2020_paper.html) · NPVForensics: [arXiv 2306.06885](https://arxiv.org/abs/2306.06885)
- He & Wang, PRCV 2025, "Towards Highly Generalized Lip-Sync Deepfake Detection via Detailed Audio-Visual Inconsistency Analysis": [Springer](https://link.springer.com/chapter/10.1007/978-981-95-5740-0_36)
- BioLip: [arXiv 2604.16808](https://arxiv.org/abs/2604.16808) · LipDA ("Ariadne's Thread of LipSync"): [arXiv 2610.08417](https://arxiv.org/abs/2610.08417)
- BabelFake: [arXiv 2610.06339](https://arxiv.org/abs/2610.06339) · DF26: [arXiv 2609.07369](https://arxiv.org/abs/2609.07369) · MuEx: [arXiv 2510.06612](https://arxiv.org/abs/2510.06612) · ArEnAV: [arXiv 2505.22581](https://arxiv.org/abs/2505.22581)
- ViCocktail: [arXiv 2506.04635](https://arxiv.org/abs/2506.04635), [repo](https://github.com/nguyenvulebinh/viCocktail), [model](https://huggingface.co/nguyenvulebinh/AV-HuBERT-CTC-Attention-VI), [dataset](https://huggingface.co/datasets/nguyenvulebinh/ViCocktail) · VLR: [Zenodo 19705466](https://zenodo.org/records/19705466)
- Generator: Wav2Lip [arXiv 2008.10010](https://arxiv.org/abs/2008.10010); MuseTalk [arXiv 2410.10122](https://arxiv.org/abs/2410.10122), [GitHub](https://github.com/TMElyralab/MuseTalk); IP-LAP [arXiv 2305.08293](https://arxiv.org/abs/2305.08293); VideoReTalking [arXiv 2211.14758](https://arxiv.org/abs/2211.14758); LatentSync [arXiv 2412.09262](https://arxiv.org/abs/2412.09262), [model 1.5](https://huggingface.co/ByteDance/LatentSync-1.5); Diff2Lip [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Mukhopadhyay_Diff2Lip_Audio_Conditioned_Diffusion_Models_for_Lip-Synchronization_WACV_2024_paper.pdf)
- Bear & Harvey 2017: [UEA](https://ueaeprints.uea.ac.uk/id/eprint/64350/) (chỉ đọc tóm tắt)
- Chưa mở được (reviewer): "Lip Forgery Video Detection via Multi-phoneme Selection" (USTC); một bài trên isj.vn (`journal_STIS/article/view/1248`)
