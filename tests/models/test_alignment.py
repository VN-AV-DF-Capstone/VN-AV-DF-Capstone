"""Alignment scorer: scoring rules, real-only training and evaluation on synthetic features."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from src.features.sync import crop_to_best_offset
from src.models.alignment import AlignmentModel

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AlignmentModelTest(unittest.TestCase):
    def test_synchronized_audio_scores_lower_than_shifted_audio(self):
        torch.manual_seed(0)
        model = AlignmentModel(6, 6, hidden_dim=8, max_offset=3)
        with torch.no_grad():
            model.audio_projection[-1].bias.zero_()
        model.visual_projection.load_state_dict(model.audio_projection.state_dict())
        features, lengths = torch.randn(3, 12, 6), torch.tensor([12, 12, 12])
        aligned = model(features, features, lengths)  # Identical heads: frame t matches itself best.
        shifted = model(features.roll(2, dims=1), features, lengths)
        self.assertTrue(torch.all(aligned < shifted))

    def test_shift_tolerance_forgives_a_constant_offset(self):
        torch.manual_seed(0)
        model = AlignmentModel(6, 6, hidden_dim=8, max_offset=3)
        with torch.no_grad():
            model.audio_projection[-1].bias.zero_()
        model.visual_projection.load_state_dict(model.audio_projection.state_dict())
        features, lengths = torch.randn(3, 12, 6), torch.tensor([12, 12, 12])
        late = features.roll(2, dims=1)  # Audio two frames late: lips at t match audio at t+2.
        strict, tolerant = model(late, features, lengths), model(late, features, lengths, max_shift=2)
        self.assertTrue(torch.all(tolerant < strict))
        torch.testing.assert_close(model(late, features, lengths, max_shift=0), strict)
        with self.assertRaises(ValueError):
            model(late, features, lengths, max_shift=4)

    def test_crop_to_best_offset_lines_up_audio_and_lips(self):
        torch.manual_seed(0)
        base = torch.randn(20, 5)
        for offset in (2, -3):
            visual = base.roll(-offset, dims=0)  # Video frame t carries audio frame t+offset.
            views = crop_to_best_offset([(base, visual), (base.clone(), visual.clone())], max_offset=4)
            for audio, cropped in views:
                self.assertEqual(len(audio), 20 - abs(offset))
                torch.testing.assert_close(cropped, audio)

    def test_padding_is_ignored_and_stays_finite(self):
        torch.manual_seed(0)
        model = AlignmentModel(4, 6, hidden_dim=8, max_offset=3)
        audio, visual = torch.randn(2, 12, 4), torch.randn(2, 12, 6)
        lengths = torch.tensor([12, 3])  # Frames 6..11 of the second clip have no valid candidate.
        padded = audio.clone()
        padded[1, 3:] = 100
        score = model(padded, visual, lengths)
        short = model(audio[1:, :3], visual[1:, :3], torch.tensor([3]))
        torch.testing.assert_close(score[1].detach(), short[0].detach())
        score.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


class AlignmentCliTest(unittest.TestCase):
    def test_trains_on_real_only_and_evaluates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, clips, fakes = [], [], []
            for i in range(12):
                split, label = ('train', 'val', 'test')[i // 4], i % 2
                for view in ('crf23', 'crf32'):
                    cache = (root / f'{i}_{view}.pt',
                             dict(sample_id=f's{i}', label=label, split=split, view_id=view,
                                  extractor_id='synthetic_fixture', timeline_id='25fps:0:12',
                                  audio=torch.randn(12, 4), visual=torch.randn(12, 6)))
                    if label:
                        fakes.append(cache)
                    else:
                        torch.save(cache[1], cache[0])
                rows.append(dict(sample_id=f's{i}', label=label, split=split,
                                 feature_path=f'{i}_crf23.pt', paired_feature_path=f'{i}_crf32.pt'))
                clips.append(dict(clip_id=f's{i}', speaker_id=f'p_{split}', source_video=f'v{i}',
                                  start_time='0'))
            manifest, real_manifest = root / 'pairs.csv', root / 'real.csv'
            write_csv(manifest, rows)
            write_csv(real_manifest, clips)
            run = root / 'align_001'
            # Fake caches do not exist yet, so training cannot read them.
            train = subprocess.run([sys.executable, str(ROOT / 'src/training/02_train_alignment.py'),
                                    '--manifest', str(manifest), '--out', str(run), '--epochs', '2',
                                    '--hidden-dim', '8', '--max-offset', '3'], capture_output=True, text=True)
            self.assertEqual(train.returncode, 0, train.stderr)
            checkpoint = torch.load(run / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['real_train_clips'], 2)
            for path, cache in fakes:
                torch.save(cache, path)
            command = [sys.executable, str(ROOT / 'src/evaluation/03_evaluate_alignment.py'), '--run', str(run),
                       '--real-manifest', str(real_manifest)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = run / 'evaluation'
            with (output / 'metrics.csv').open(newline='') as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual([(m['split'], m['view']) for m in metrics],
                             [(s, v) for s in ('val', 'test') for v in ('both', 'crf23', 'crf32')])
            with (output / 'mismatch_scores.csv').open(newline='') as handle:
                mismatch = list(csv.DictReader(handle))
            # Two real samples per val/test split, each paired with the other, in both views.
            self.assertEqual(len(mismatch), 8)
            self.assertTrue(all(m['sample_id'] != m['audio_from'] for m in mismatch))
            with (output / 'mismatch_summary.csv').open(newline='') as handle:
                self.assertEqual([s['split'] for s in csv.DictReader(handle)], ['val', 'test'])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            result = subprocess.run(command + ['--max-shift', '2'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((run / 'evaluation_shift2' / 'metrics.csv').is_file())
            resynced = root / 'align_002'
            train = subprocess.run([sys.executable, str(ROOT / 'src/training/02_train_alignment.py'),
                                    '--manifest', str(manifest), '--out', str(resynced), '--epochs', '1',
                                    '--hidden-dim', '8', '--max-offset', '3', '--resync', '2'],
                                   capture_output=True, text=True)
            self.assertEqual(train.returncode, 0, train.stderr)
            result = subprocess.run([sys.executable, str(ROOT / 'src/evaluation/03_evaluate_alignment.py'),
                                     '--run', str(resynced), '--real-manifest', str(real_manifest)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('resync 2', result.stdout)
            with (resynced / 'evaluation' / 'mismatch_scores.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 8)


if __name__ == '__main__':
    unittest.main()
