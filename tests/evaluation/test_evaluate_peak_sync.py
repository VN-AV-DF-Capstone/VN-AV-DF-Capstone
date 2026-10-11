"""Training-free sync score on synthetic features with known offsets, not research results."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from src.features.sync import best_offset_misalignment

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'src/evaluation/05_evaluate_peak_sync.py'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PeakSyncTest(unittest.TestCase):
    def test_padding_is_ignored(self):
        torch.manual_seed(0)
        audio, visual = torch.randn(2, 12, 6), torch.randn(2, 12, 6)
        padded = audio.clone()
        padded[1, 8:] = 100
        scores = best_offset_misalignment(padded, visual, torch.tensor([12, 8]), max_offset=3)
        short = best_offset_misalignment(audio[1:, :8], visual[1:, :8], torch.tensor([8]), max_offset=3)
        torch.testing.assert_close(scores[1], short[0])

    def test_reports_known_offsets_as_in_sync(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, clips = [], []
            # Real clips are in sync at a constant offset; fake lips are unrelated to the audio.
            for i, (split, label, offset) in enumerate([('val', 0, 0), ('val', 0, 2), ('val', 1, 0), ('val', 1, 0),
                                                        ('test', 0, -2), ('test', 0, 1), ('test', 1, 0),
                                                        ('test', 1, 0)]):
                base = torch.randn(12, 6)
                visual = torch.randn(12, 6) if label else base.roll(-offset, dims=0)
                kind = 'fake' if label else 'real'
                for view in ('crf23', 'crf32'):
                    torch.save(dict(sample_id=f'c{i}__run__{kind}', label=label, split=split, view_id=view,
                                    extractor_id='synthetic_fixture', timeline_id='25fps:0:12',
                                    audio=base, visual=visual), root / f'{i}_{view}.pt')
                rows.append(dict(sample_id=f'c{i}__run__{kind}', label=label, split=split,
                                 feature_path=f'{i}_crf23.pt', paired_feature_path=f'{i}_crf32.pt'))
                clips.append(dict(clip_id=f'c{i}', speaker_id=f'p_{split}', source_video=f'v{i}', start_time='0'))
            write_csv(root / 'pairs.csv', rows)
            write_csv(root / 'real.csv', clips)
            out = root / 'peak_sync'
            command = [sys.executable, str(SCRIPT), '--manifest', str(root / 'pairs.csv'),
                       '--real-manifest', str(root / 'real.csv'), '--out', str(out), '--max-offset', '3']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out / 'metrics.csv').open(newline='') as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual([(m['split'], m['view']) for m in metrics],
                             [(s, v) for s in ('val', 'test') for v in ('both', 'crf23', 'crf32')])
            self.assertTrue(all(float(m['auc']) == 1.0 for m in metrics))
            with (out / 'mismatch_summary.csv').open(newline='') as handle:
                summary = list(csv.DictReader(handle))
            self.assertTrue(all(float(s['auc_vs_genuine_real']) == 1.0 for s in summary))
            with (out / 'mismatch_scores.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 8)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)  # Never overwrites.


if __name__ == '__main__':
    unittest.main()
