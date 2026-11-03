"""v005 (M5 trihead_v3, P4, P6) và v006 (M6 ViSyl, thẻ âm tiết): kiểm thành phần mới."""

import random

import numpy as np
import pytest

from vn_av_df.features.syllables import COLUMNS, VOCAB, tag, units

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "word, onset, coda, round_on, round_nuc",
    [
        ("ông", "none", "labial_velar", "no", "yes"),
        ("học", "other", "labial_velar", "no", "yes"),
        ("chúng", "other", "labial_velar", "no", "yes"),
        ("Quốc", "other", "labial_velar", "yes", "yes"),  # qu là âm đầu, vần ôc
        ("giục", "other", "labial_velar", "no", "yes"),  # gi là âm đầu, vần uc
        ("gióng", "other", "labial_velar", "no", "yes"),
        ("uống", "none", "other", "no", "yes"),  # uô: không khép môi-ngạc mềm
        ("bún", "bilabial", "round_alveolar", "no", "yes"),
        ("một", "bilabial", "round_alveolar", "no", "yes"),
        ("cam", "other", "bilabial", "no", "no"),
        ("đẹp", "other", "bilabial", "no", "no"),
        ("sao", "other", "round_glide", "no", "no"),
        ("toán", "other", "other", "yes", "no"),  # âm đệm o
        ("phải", "ph", "other", "no", "no"),
        ("vui", "v", "other", "no", "yes"),
        ("gì", "other", "none", "no", "no"),
        ("của", "other", "none", "no", "yes"),
    ],
)
def test_orthography_tags(word, onset, coda, round_on, round_nuc):
    t = tag(word)
    assert (t["onset"], t["coda"], t["round_on"], t["round_nuc"], t["type"]) == (
        onset,
        coda,
        round_on,
        round_nuc,
        "syllable",
    )


def test_exceptions_and_tones():
    for word in ("xoong", "5", "COVID-19", "wifi"):
        assert tag(word)["type"] == "unknown"
    tones = [tag(w)["tone"] for w in ("ma", "mà", "má", "mả", "mã", "mạ")]
    assert tones == ["ngang", "huyen", "sac", "hoi", "nga", "nang"]


def test_units_keep_zero_length_words_and_chunk_gaps():
    words = [
        {"word": "xin", "start": 0.1, "end": 0.3},
        {"word": "chào", "start": 0.3, "end": 0.3},  # dài 0 → giữ 1 frame
        {"word": "ông", "start": 1.2, "end": 1.5},
    ]
    syll = units(words, 50)
    starts = np.flatnonzero(syll[:, 0]).tolist()
    assert starts == [0, 2, 8, 9, 21, 30, 38]  # gap 9-30 chia khúc ≤ 12 frame
    kind = syll[:, COLUMNS.index("type")].astype(int)
    assert kind[8] == VOCAB["type"].index("syllable") and kind[9] == VOCAB["type"].index("gap")
    coda = syll[30, COLUMNS.index("coda")]
    assert VOCAB["coda"][int(coda)] == "labial_velar"
    empty = units([], 12)
    assert np.flatnonzero(empty[:, 0]).tolist() == [0, 5, 10]
    assert set(empty[:, COLUMNS.index("type")]) == {VOCAB["type"].index("unknown")}


def test_pseudo_fake_replaces_units_and_cuts_edges():
    from vn_av_df.training.data import replace_audio

    n = 150
    item = {"a": np.zeros((n, 4), np.float32), "syll": units([], n)}
    item["syll"][:, 0] = 0
    other = {"a": np.ones((n, 4), np.float32), "syll": np.zeros((n, 7), np.float32)}
    other["syll"][::7, 0] = 1
    other["syll"][:, 1:] = 2
    snapped = 0
    for seed in range(40):
        it = {k: v.copy() for k, v in item.items()}
        targets = {"mismatch": np.zeros(n, np.float32)}
        start, end = replace_audio(it, other, targets, random.Random(seed))
        assert it["syll"][start, 0] == 1 and (end == n or it["syll"][end, 0] == 1)
        assert (it["syll"][start:end, 1:] == 2).all() and (it["a"][start:end] == 1).all()
        snapped += start % 7 == 0 and end % 7 == 0
    assert 10 <= snapped < 40  # khoảng một nửa theo biên âm tiết


def test_visyl_units_follow_audio_lag():
    from vn_av_df.training.data import lag_audio

    syll = units([{"word": "ông", "start": 0.4, "end": 0.6}], 25)
    item = lag_audio({"a": np.zeros((25, 2), np.float32), "syll": syll.copy()}, 2)
    assert np.flatnonzero(item["syll"][:, 0]).tolist() == [s + 2 for s in [0, 10, 15]]


def test_trihead_v3_routes_one_branch_per_head():
    from vn_av_df.training import models as zoo
    from vn_av_df.training.benchmark import synthetic_batch

    torch.manual_seed(0)
    model = zoo.build("trihead_v3", dim=16, hidden=8)
    assert model.visual_from == ("trace",) and model.mismatch_from == ("sync",)
    assert not hasattr(model.sync, "content") and model.references == 0
    model.eval()
    batch = synthetic_batch(model, 2, 40, 16, "cpu")
    base = model(batch)
    batch["ssl"] = torch.randn_like(batch["ssl"])  # đổi tiếng XLS-R: chỉ đầu audio đổi
    changed = model(batch)
    assert torch.equal(base["visual"], changed["visual"])
    assert torch.equal(base["mismatch"], changed["mismatch"])
    assert not torch.equal(base["audio"], changed["audio"])
    assert zoo.build("trihead_v3_p1", dim=16, hidden=8).references == 3  # còn FiLM


def test_visyl_scores_are_constant_within_units():
    from vn_av_df.training import models as zoo
    from vn_av_df.training.benchmark import synthetic_batch

    torch.manual_seed(0)
    model = zoo.build("visyl", dim=16, hidden=16).eval()
    batch = synthetic_batch(model, 2, 30, 16, "cpu")
    with torch.no_grad():
        out = model(batch)["visual"][0]
    for s in range(0, 30, 5):
        assert torch.allclose(out[s : s + 5], out[s].expand(5))
    assert zoo.build("visyl_uniform_no_phon", dim=16, hidden=16).needs == {"av", "ssl"}


def test_p6_perfect_scorer_and_p4_reference_centering():
    from vn_av_df.training import evaluate as ev

    class Fake:
        heads = {"visual": "visual", "audio": "audio"}

    rows, preds = [], {}
    for split in ("validation", "test"):
        for i in range(8):
            spk = f"{split}-s{i % 2}"
            for cell, kind in (("real", "full"), ("source", "partial"), ("voice", "partial")):
                sid = f"{split}-{cell}-{i}"
                fake = cell != "real"
                span = [[1.0, 1.4 if cell == "source" else 2.0]] if fake else []
                frames = np.zeros(100)
                if fake:
                    frames[int(span[0][0] * 25) : int(span[0][1] * 25)] = 0.9
                rows.append(
                    {
                        "sample_id": sid,
                        "split": split,
                        "speaker_id": spk,
                        "source_clip_id": f"{split}-clip{i}",
                        "cell": cell,
                        "kind": kind,
                        "label": int(fake),
                        "synthetic_visual": cell == "source",
                        "synthetic_audio": cell == "voice",
                        "fake_intervals": span,
                        "visual_fake_intervals": span if cell == "source" else [],
                        "audio_fake_intervals": span if cell == "voice" else [],
                    }
                )
                video = {"visual": 0.8 if cell == "source" else 0.1}
                video["audio"] = 0.8 if cell == "voice" else 0.1
                preds[sid] = {
                    "video": video,
                    "frames": {"visual": frames, "audio": frames},
                }
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    out = ev.p6_localization(Fake(), val, preds, test, preds, "ai")
    assert out["tiers"]["random"]["ap@0.5"] == pytest.approx(1.0)
    assert out["tiers"]["word"]["ap@0.75"] == pytest.approx(1.0)
    assert out["tiers"]["random_short"]["segments"] == 8  # đoạn 0,4 s
    assert out["tiers"]["random"]["boundary_error_ms"] == pytest.approx(0.0)
    scores = {t: {sid: p["video"][t] for sid, p in preds.items()} for t in ("visual", "audio")}
    stats = {t: (0.0, 1.0) for t in scores}
    p4 = ev.p4_scores(test, scores, stats, 1)
    # tham chiếu = clip real đầu của mỗi người: bị loại cùng mọi mẫu sinh từ nó; real còn lại → 0
    assert "test-real-0" not in p4 and "test-source-0" not in p4
    assert p4["test-real-2"] == pytest.approx(0.0)
    assert p4["test-source-2"] > 0


def test_version_bootstrap_uses_v003_protocol():
    import settings

    try:
        settings.bootstrap("v006")
        cfg = settings.config()
        assert cfg["models"][0] == "visyl" and cfg["seeds"] == [42, 43, 44, 45, 46]
        assert cfg["runs"].endswith("train_part1_v006")
        assert cfg["training"]["audio_lag_aug"] == [-1, 2]
        assert cfg["training"]["monitor"] == "loss" and cfg["training"]["scheduler"] is None
        assert cfg["mismatch_labels"]["mode"] == "construction"
        assert len(cfg["encoders"]["xlsr"]["layer_groups"]) == 3
    finally:
        settings.VERSION = None
    assert settings.config()["models"] == settings.MODELS + settings.ABLATIONS
