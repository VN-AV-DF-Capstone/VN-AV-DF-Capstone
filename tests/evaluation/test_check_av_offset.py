"""Audio-video offset diagnostic on synthetic features with known shifts, not research results."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'src/evaluation/04_check_av_offset.py'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CheckAvOffsetCliTest(unittest.TestCase):
    def test_recovers_known_offsets(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, clips = [], []
            for i, (label, offset) in enumerate([(0, 0), (1, 3), (0, -4)]):
                base = torch.randn(50, 8)
                kind = 'fake' if label else 'real'
                for view in ('crf23', 'crf32'):
                    # Video frame t carries the content of audio frame t+offset.
                    torch.save(dict(audio=base, visual=base.roll(-offset, dims=0)), root / f'{i}_{view}.pt')
                rows.append(dict(sample_id=f'c{i}__run__{kind}', label=label, split='test',
                                 feature_path=f'{i}_crf23.pt', paired_feature_path=f'{i}_crf32.pt'))
                clips.append(dict(clip_id=f'c{i}', speaker_id='p1', source_video=f'v{i}'))
            write_csv(root / 'pairs.csv', rows)
            write_csv(root / 'real.csv', clips)
            out = root / 'offsets.csv'
            command = [sys.executable, str(SCRIPT), '--manifest', str(root / 'pairs.csv'),
                       '--real-manifest', str(root / 'real.csv'), '--out', str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with out.open(newline='') as handle:
                self.assertEqual([int(r['offset']) for r in csv.DictReader(handle)], [0, 3, -4])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)  # Never overwrites.


if __name__ == '__main__':
    unittest.main()
