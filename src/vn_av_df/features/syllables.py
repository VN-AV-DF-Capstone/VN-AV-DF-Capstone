"""v006 · M6 (ViSyl): đơn vị âm tiết của tiếng mẫu + thẻ kỳ vọng khẩu hình suy từ chính tả.

Thiết kế: docs/v006_sau-review.md §2.2-§2.3 (đề xuất, chưa train).

- Lời và mốc: PhoWhisper-small (generation/voice/asr.py) chạy trên tiếng 16 kHz của **chính mẫu**,
  mốc từng từ bằng DTW của Whisper. ASR được cache theo hash tiếng đã giải mã (thư mục
  `<cache>/syll/asr/`), **không** dùng cache ASR của bước sinh (khoá theo clip gốc: rò lời gốc
  vào ③B, ⑥). Mẫu có tiếng giống hệt nhau dùng chung một lần ASR.
- Đơn vị: mỗi từ (tiếng Việt viết rời từng âm tiết) là một âm tiết [s, e) trên lưới 25 Hz, từ dài
  0 vẫn giữ 1 frame; khoảng giữa các từ là đơn vị `gap` (mỗi khúc ≤ 12 frame); ASR không ra gì thì
  cả clip thành khúc 0,2 s `type = unknown`.
- Cache `syll` [T, 7] float32 theo frame: cột 0 = frame mở đơn vị, cột 1-6 = chỉ số thẻ (0 = đệm).
  Dạng theo frame để trượt tiếng (data.lag_audio), thay đoạn pseudo-fake và cắt đi cùng tiếng.

Bộ căn chọn ở bước 0b' (PhoWhisper DTW / CTC wav2vec2 tiếng Việt / MFA) chưa làm: đây là PhoWhisper
DTW. Luật thẻ chưa đối chiếu với từ điển MFA Hà Nội / TP.HCM (bước trước khi dùng, §2.2).
"""

import math
import unicodedata
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import fingerprint, read_json, save_npz, sha, write_json

RULES_VERSION = "visyl-tags-v1"
MODEL = "vinai/PhoWhisper-small"
GAP_MAX, FALLBACK = 12, 5  # frame: khúc lặng dài nhất; khúc dự phòng 0,2 s
COLUMNS = ("boundary", "onset", "coda", "round_on", "round_nuc", "type", "tone")
VOCAB = {
    "onset": ("pad", "none", "bilabial", "ph", "v", "other"),
    "coda": ("pad", "none", "bilabial", "labial_velar", "round_alveolar", "round_glide", "other"),
    "round_on": ("pad", "no", "yes"),
    "round_nuc": ("pad", "no", "yes"),
    "type": ("pad", "syllable", "gap", "unknown"),
    "tone": ("pad", "ngang", "huyen", "sac", "hoi", "nga", "nang", "none"),
}
TONES = {"̀": "huyen", "́": "sac", "̉": "hoi", "̃": "nga", "̣": "nang"}
LETTERS = set("aăâbcdđeêghiklmnoôơpqrstuưvxy")
VOWELS = set("aăâeêioôơuưy")
ONSETS = sorted(
    "ngh ng nh ch gh gi kh ph qu th tr b c d đ g h k l m n p r s t v x".split(),
    key=len,
    reverse=True,
)
CODAS = ("ng", "nh", "ch", "c", "m", "n", "p", "t", "i", "y", "o", "u")
EMPTY = {"onset": "none", "coda": "none", "round_on": "no", "round_nuc": "no", "tone": "none"}


def split_tone(word):
    """'Quốc' → ('quôc', 'sac'): chữ thường NFC đã bỏ dấu thanh, và thanh."""
    tone, kept = "ngang", []
    for ch in unicodedata.normalize("NFD", word.lower()):
        if ch in TONES:
            tone = TONES[ch]
        else:
            kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept)), tone


def tag(word):
    """Thẻ kỳ vọng khẩu hình của một âm tiết viết (bảng §2.2). Ngoại lệ (oong / ooc, chữ ngoại,
    chữ số, không có nguyên âm) → type = unknown, không suy kỳ vọng."""
    base, tone = split_tone(word.strip(".,!?;:…\"'()[]-"))
    unknown = {**EMPTY, "type": "unknown"}
    # shortcut: chữ số không đổi thành chữ (một mốc thời gian cho nhiều âm tiết) → unknown;
    # đổi khi 0d' cho thấy tỉ lệ unknown đáng kể.
    if not base or not set(base) <= LETTERS:
        return unknown
    onset = next((o for o in ONSETS if base.startswith(o)), "")
    rhyme = base[len(onset) :]
    if onset == "gi" and rhyme[:1] not in VOWELS:
        rhyme = "i" + rhyme  # "gì", "gìn": i là âm chính
    round_on = onset == "qu"
    if not round_on and len(rhyme) > 1:
        if rhyme[0] == "o" and rhyme[1] in "aăe" or rhyme[0] == "u" and rhyme[1] in "âêyơ":
            round_on, rhyme = True, rhyme[1:]
    if not rhyme or rhyme.startswith("oo") or not set(rhyme) & VOWELS:
        return unknown
    coda = next((c for c in CODAS if rhyme.endswith(c) and len(rhyme) > len(c)), "")
    nucleus = rhyme[: len(rhyme) - len(coda)]
    if not set(nucleus) <= VOWELS:
        return unknown
    if coda in ("m", "p"):
        place = "bilabial"
    elif coda in ("ng", "c") and nucleus in ("u", "ô", "o"):
        place = "labial_velar"  # [ŋ͡m], [k͡p̚] sau nguyên âm tròn đơn (không gồm uô, ươ)
    elif coda in ("n", "t") and nucleus in ("u", "ô"):
        place = "round_alveolar"  # Nam, Trung khép môi; Bắc không
    elif coda in ("u", "o"):
        place = "round_glide"
    else:
        place = "other" if coda else "none"
    first = {"b": "bilabial", "m": "bilabial", "p": "bilabial", "ph": "ph", "v": "v", "": "none"}
    return {
        "onset": first.get(onset, "other"),
        "coda": place,
        "round_on": "yes" if round_on else "no",
        "round_nuc": "yes" if nucleus in ("u", "ô", "o", "uô", "ua") else "no",  # ua = uô mở
        "type": "syllable",
        "tone": tone,
    }


def encode(tags):
    return [VOCAB[k].index(tags[k]) for k in COLUMNS[1:]]


def units(words, frames, gap_max=GAP_MAX, fallback=FALLBACK):
    """[{"word", "start", "end"}] (giây) → [frames, 7]: cột 0 mở đơn vị, cột 1-6 thẻ."""
    out = np.zeros((frames, len(COLUMNS)), np.float32)
    gap = encode({**EMPTY, "type": "gap"})

    def put(s, e, codes, size):
        for a in range(s, e, size):
            out[a, 0] = 1
            out[a : min(e, a + size), 1:] = codes

    cursor, found = 0, False
    for w in words:
        s = max(cursor, int(math.floor(w["start"] * 25)))
        if s >= frames:
            break
        e = min(frames, max(s + 1, int(round(w["end"] * 25))))
        put(cursor, s, gap, gap_max)
        put(s, e, encode(tag(w["word"])), e - s)
        cursor, found = e, True
    if not found:
        put(0, frames, encode({**EMPTY, "type": "unknown"}), fallback)
    else:
        put(cursor, frames, gap, gap_max)
    return out


class SyllableEncoder:
    """Cache `syll` cho prepare (cùng giao diện các encoder trong encoders.py)."""

    kind = "syll"

    def __init__(self, cfg, transcriber=None):
        from importlib.metadata import PackageNotFoundError, version

        self.cfg = dict(cfg)
        self.transcriber = transcriber
        try:
            transformers = version("transformers")
        except PackageNotFoundError:
            transformers = None
        # shortcut: chưa ghim revision HF của PhoWhisper (Transcriber không nhận revision);
        # ghim khi chốt bộ căn ở 0b'.
        self.signature = fingerprint(
            {
                "code": sha(__file__),
                "asr": self.cfg.get("model", MODEL),
                "aligner": "phowhisper-dtw",
                "transformers": transformers,
                "pipeline": {"chunk_length_s": 30, "language": "vi", "timestamps": "word"},
                "rules": RULES_VERSION,
                "gap_max": GAP_MAX,
                "fallback": FALLBACK,
            }
        )

    def start(self):
        if self.transcriber is None:
            from vn_av_df.generation.voice.asr import Transcriber

            self.transcriber = Transcriber(
                self.cfg.get("model", MODEL), self.cfg.get("device", "cpu")
            )

    def extract(self, request, output):
        from vn_av_df.features.encoders import audio_16k, fresh, source_fingerprint

        video = Path(request["video"])
        source = source_fingerprint(video)
        if fresh(output, self.signature, source):
            return read_json(Path(output).with_suffix(".json"))
        pcm = audio_16k(video)
        frames = max(1, int(round(len(pcm) / 640)))
        import hashlib

        key = hashlib.sha256(pcm.tobytes()).hexdigest()
        asr = Path(output).parent / "asr" / f"{key}.json"
        if asr.is_file() and read_json(asr).get("signature") == self.signature:
            words = read_json(asr)["words"]
        else:
            self.start()
            words = self.transcriber(pcm)
            write_json(asr, {"signature": self.signature, "words": words})
        syll = units(words, frames)
        save_npz(output, syll=syll, valid=np.ones(frames, bool), times_s=np.arange(frames) / 25)
        types = syll[:, COLUMNS.index("type")].astype(int)
        meta = {
            "format": "syll-v1",
            "step_s": 0.04,
            "duration_s": frames / 25,
            "feature_signature": self.signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
            "audio_sha256": key,
            "words": [w["word"] for w in words],
            "units": int(syll[:, 0].sum()),
            "unknown_ratio": float(np.mean(types == VOCAB["type"].index("unknown"))),
        }
        write_json(Path(output).with_suffix(".json"), meta)
        return meta
