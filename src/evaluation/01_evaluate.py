"""Score val/test with a trained detector: AUC, EER and accuracy per split and compression view.

The threshold is the validation EER point and is applied unchanged to test. Val rows at that
threshold are optimistic. Never pick runs, epochs or thresholds from test numbers.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader
from src.evaluation.metrics import equal_error_rate, roc_auc, threshold_report
from src.evaluation.predict import predict_batch
from src.models.detector import AudioVisualDetector
from src.training.dataset import PairedFeatureDataset, collate_pairs


def score(model, dataset):
    """One row per sample and compression view."""
    rows = []
    for batch in DataLoader(dataset, batch_size=32, collate_fn=collate_pairs):
        for view, prefix in enumerate(('', 'paired_')):
            scores = predict_batch(model, batch[prefix + 'audio'], batch[prefix + 'visual'],
                                   batch['lengths'])['fake_score']
            rows += [dict(sample_id=sample_id, label=int(label), view_id=views[view], fake_score=float(value))
                     for sample_id, label, views, value in
                     zip(batch['sample_ids'], batch['labels'], batch['view_ids'], scores)]
    return rows


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Experiment directory containing best.pt')
    parser.add_argument('--manifest', help='Feature pairs.csv; defaults to the one used for training')
    args = parser.parse_args()
    run = Path(args.run)
    output = run / 'evaluation'
    if output.exists():
        parser.error(f'Already evaluated; keep it and train a new run instead: {output}')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    manifest = Path(args.manifest or checkpoint['args']['manifest'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint['manifest_sha256']:
        parser.error(f'{manifest} differs from the manifest used for training')
    model = AudioVisualDetector(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    predictions = {split: score(model, PairedFeatureDataset(manifest, split)) for split in ('val', 'test')}
    _, threshold = equal_error_rate([r['label'] for r in predictions['val']],
                                    [r['fake_score'] for r in predictions['val']])
    metrics = []
    for split, rows in predictions.items():
        views = {'both': rows} | {view: [r for r in rows if r['view_id'] == view]
                                  for view in sorted({r['view_id'] for r in rows})}
        for view, subset in views.items():
            labels, scores = [r['label'] for r in subset], [r['fake_score'] for r in subset]
            metrics.append(dict(split=split, view=view, real=labels.count(0), fake=labels.count(1),
                                auc=roc_auc(labels, scores), eer=equal_error_rate(labels, scores)[0],
                                threshold=threshold, **threshold_report(labels, scores, threshold)))
    output.mkdir()
    write_csv(output / 'predictions.csv', [dict(split=split, **row) for split, rows in predictions.items()
                                           for row in rows])
    write_csv(output / 'metrics.csv', metrics)
    print(f'Checkpoint epoch {checkpoint["epoch"]}; threshold {threshold:.4f} from val EER')
    print(f'{"split":5} {"view":6} {"real":>5} {"fake":>5} {"AUC":>6} {"EER":>6} {"acc":>6} '
          f'{"bal_acc":>7} {"real_FA":>7} {"fake_miss":>9}')
    for m in metrics:
        print(f'{m["split"]:5} {m["view"]:6} {m["real"]:5d} {m["fake"]:5d} {m["auc"]:6.3f} {m["eer"]:6.3f} '
              f'{m["accuracy"]:6.3f} {m["balanced_accuracy"]:7.3f} {m["real_false_alarm"]:7.3f} '
              f'{m["fake_miss"]:9.3f}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
