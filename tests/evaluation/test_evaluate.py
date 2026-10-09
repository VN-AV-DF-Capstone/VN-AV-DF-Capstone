"""Metric definitions and the evaluation CLI on synthetic features, not research results."""
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from src.evaluation.metrics import equal_error_rate, roc_auc, threshold_report

ROOT = Path(__file__).resolve().parents[2]


class MetricsTest(unittest.TestCase):
    def test_auc_eer_and_threshold_rates(self):
        labels, scores = [0, 0, 0, 1, 1, 1], [.1, .2, .7, .6, .8, .9]
        self.assertAlmostEqual(roc_auc(labels, scores), 8 / 9)
        self.assertEqual(roc_auc([0, 1], [.5, .5]), .5)
        eer, threshold = equal_error_rate(labels, scores)
        self.assertAlmostEqual(eer, 1 / 3)
        self.assertEqual(threshold, .7)
        report = threshold_report(labels, scores, threshold)
        self.assertAlmostEqual(report['real_false_alarm'], 1 / 3)
        self.assertAlmostEqual(report['fake_miss'], 1 / 3)
        self.assertAlmostEqual(threshold_report(labels, scores, .6)['accuracy'], 5 / 6)
        with self.assertRaises(ValueError):
            roc_auc([1, 1], [.2, .3])


class EvaluateCliTest(unittest.TestCase):
    def test_threshold_from_val_and_metrics_per_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i in range(12):
                split, label = ('train', 'val', 'test')[i // 4], i % 2
                for view in ('crf23', 'crf32'):
                    torch.save(dict(sample_id=f's{i}', label=label, split=split, view_id=view,
                                    extractor_id='synthetic_fixture', timeline_id='25fps:0:5',
                                    audio=torch.randn(5, 4) + label, visual=torch.randn(5, 6)),
                               root / f'{i}_{view}.pt')
                rows.append(dict(sample_id=f's{i}', label=label, split=split,
                                 feature_path=f'{i}_crf23.pt', paired_feature_path=f'{i}_crf32.pt'))
            manifest = root / 'pairs.csv'
            with manifest.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            run = root / 'y_001'
            train = subprocess.run([sys.executable, str(ROOT / 'src/training/01_train.py'), '--manifest',
                                    str(manifest), '--out', str(run), '--epochs', '2', '--hidden-dim', '8'],
                                   capture_output=True, text=True)
            self.assertEqual(train.returncode, 0, train.stderr)
            command = [sys.executable, str(ROOT / 'src/evaluation/01_evaluate.py'), '--run', str(run)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (run / 'evaluation/metrics.csv').open(newline='') as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual([(m['split'], m['view']) for m in metrics],
                             [(s, v) for s in ('val', 'test') for v in ('both', 'crf23', 'crf32')])
            self.assertEqual(len({m['threshold'] for m in metrics}), 1)
            self.assertEqual({(m['real'], m['fake']) for m in metrics if m['view'] != 'both'}, {('2', '2')})
            with (run / 'evaluation/predictions.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 16)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            manifest.write_text(manifest.read_text() + '\n')
            shutil.rmtree(run / 'evaluation')
            self.assertIn('differs', subprocess.run(command, capture_output=True, text=True).stderr)


if __name__ == '__main__':
    unittest.main()
