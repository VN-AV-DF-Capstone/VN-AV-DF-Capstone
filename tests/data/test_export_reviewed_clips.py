"""Safety checks for the preview-to-master handoff."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.preparation.export_reviewed_clips import (
    export_reviewed,
    source_map,
    validate_reviews,
)


class ReviewedExportTest(unittest.TestCase):
    def test_review_requires_exact_coverage_and_explicit_decisions(self):
        candidates = [{"clip_id": "a"}, {"clip_id": "b"}]
        valid = [{"clip_id": "a", "decision": "Keep"},
                 {"clip_id": "b", "decision": "reject"}]
        self.assertEqual(validate_reviews(candidates, valid),
                         {"a": "keep", "b": "reject"})
        with self.assertRaisesRegex(ValueError, "chưa phủ đủ"):
            validate_reviews(candidates, valid[:1])
        with self.assertRaisesRegex(ValueError, "rỗng/trùng"):
            validate_reviews(candidates, [valid[0], valid[0]])
        with self.assertRaisesRegex(ValueError, "keep/reject"):
            validate_reviews(candidates, [valid[0], {"clip_id": "b", "decision": ""}])

    def test_source_path_cannot_escape_source_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "raw"
            source_dir.mkdir()
            (source_dir / "good.mp4").write_bytes(b"source")
            mapped = source_map([{"filename": "good.mp4"}], source_dir)
            self.assertEqual(mapped["good"], source_dir / "good.mp4")
            with self.assertRaisesRegex(ValueError, "vượt source_dir"):
                source_map([{"filename": "../escape.mp4"}], source_dir)

    def test_changed_source_is_rejected_before_any_master_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "preview_batch"
            batch.mkdir()
            sources = root / "raw"
            sources.mkdir()
            (sources / "video.mp4").write_bytes(b"x")
            clip_id = "video_s0000001000_e0000004000"
            (batch / "preview_candidates.csv").write_text(
                f"clip_id,source_video,start_time,end_time\n{clip_id},video,1,4\n",
                encoding="utf-8",
            )
            (batch / "input_inventory.csv").write_text(
                "filename,size_bytes\nvideo.mp4,2\n", encoding="utf-8",
            )
            (batch / "run_summary.json").write_text(
                json.dumps({"output_mode": "preview", "accepted_clips": 1}),
                encoding="utf-8",
            )
            (batch / "config.json").write_text(json.dumps({
                "tier": "podcast", "dataset_dir": str(sources),
                "input_csv": "unused.csv", "output_root": str(root / "preview"),
                "run_id": "preview", "expected_input_count": 1,
                "preview_height": 480,
            }), encoding="utf-8")
            decisions = root / "decisions.csv"
            decisions.write_text(f"clip_id,decision\n{clip_id},keep\n", encoding="utf-8")
            output = root / "masters"
            with self.assertRaisesRegex(ValueError, "Kích thước"):
                export_reviewed(batch, decisions, sources, output, "master_001")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
