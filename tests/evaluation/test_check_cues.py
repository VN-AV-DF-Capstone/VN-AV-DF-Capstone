"""Cue check: mismatch partner rules and the CLI on synthetic features, not research results."""
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'src/evaluation/02_check_cues.py'
spec = importlib.util.spec_from_file_location('check_cues', SCRIPT)
check_cues = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_cues)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MismatchPartnersTest(unittest.TestCase):
    def test_same_speaker_without_time_overlap(self):
        clips = {'a': dict(speaker_id='p1', source_video='v1', start_time='0'),
                 'b': dict(speaker_id='p1', source_video='v1', start_time='1'),
                 'c': dict(speaker_id='p1', source_video='v2', start_time='0'),
                 'd': dict(speaker_id='p2', source_video='v3', start_time='0')}
        rows = [dict(sample_id=f'{c}__run__real', label='0') for c in clips]
        rows.append(dict(sample_id='a__run__fake', label='1'))
        partners = check_cues.mismatch_partners(rows, clips, seconds=2, seed=7)
        self.assertEqual(partners['a__run__real'], 'c__run__real')  # b overlaps a in time.
        self.assertEqual(partners['b__run__real'], 'c__run__real')
        self.assertIn(partners['c__run__real'], {'a__run__real', 'b__run__real'})
        self.assertNotIn('d__run__real', partners)  # Only speaker p2 clip.
        self.assertNotIn('a__run__fake', partners)


class CheckCuesCliTest(unittest.TestCase):
    def test_ablations_and_mismatch_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, clips = [], []
            for i in range(12):
                split, label = ('train', 'val', 'test')[i // 4], i % 2
                for view in ('crf23', 'crf32'):
                    torch.save(dict(sample_id=f's{i}', label=label, split=split, view_id=view,
                                    extractor_id='synthetic_fixture', timeline_id='25fps:0:5',
                                    audio=torch.randn(5, 4) + label, visual=torch.randn(5, 6)),
                               root / f'{i}_{view}.pt')
                rows.append(dict(sample_id=f's{i}', label=label, split=split,
                                 feature_path=f'{i}_crf23.pt', paired_feature_path=f'{i}_crf32.pt'))
                clips.append(dict(clip_id=f's{i}', speaker_id=f'p_{split}', source_video=f'v{i}',
                                  start_time='0'))
            manifest, real_manifest = root / 'pairs.csv', root / 'real.csv'
            write_csv(manifest, rows)
            write_csv(real_manifest, clips)
            run = root / 'y_001'
            train = subprocess.run([sys.executable, str(ROOT / 'src/training/01_train.py'), '--manifest',
                                    str(manifest), '--out', str(run), '--epochs', '2', '--hidden-dim', '8'],
                                   capture_output=True, text=True)
            self.assertEqual(train.returncode, 0, train.stderr)
            command = [sys.executable, str(SCRIPT), '--run', str(run), '--real-manifest', str(real_manifest)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = run / 'cue_check'
            with (output / 'metrics.csv').open(newline='') as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual([(m['model'], m['split']) for m in metrics],
                             [(m, s) for m in ('audio+visual', 'visual_only', 'audio_only')
                              for s in ('val', 'test')])
            with (output / 'mismatch_scores.csv').open(newline='') as handle:
                mismatch = list(csv.DictReader(handle))
            # Two real samples per val/test split, each paired with the other, in both views.
            self.assertEqual(len(mismatch), 8)
            self.assertTrue(all(m['sample_id'] != m['audio_from'] for m in mismatch))
            with (output / 'mismatch_summary.csv').open(newline='') as handle:
                self.assertEqual([s['split'] for s in csv.DictReader(handle)], ['val', 'test'])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
