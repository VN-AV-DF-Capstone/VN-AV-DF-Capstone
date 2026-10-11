"""Check which cue a trained run relies on, using its own cached features.

1. Retrain with the same settings but one modality zeroed (visual only, audio only).
2. Score the run's model on REAL video + audio of ANOTHER real clip of the same speaker and
   split with no time overlap: out of sync, but without generator artifacts.
Each model's threshold is its val EER point, applied unchanged to test. Diagnostics only.
"""
import argparse
import copy
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader
from src.data.preparation.manifest_io import read_rows, unique_rows
from src.evaluation.metrics import equal_error_rate, roc_auc, threshold_report
from src.evaluation.mismatch import REAL_MANIFEST, mismatch_partners
from src.evaluation.predict import predict_batch
from src.models.detector import AudioVisualDetector
from src.training.dataset import PairedFeatureDataset, collate_pairs
from src.training.loop import run_epoch


def drop(batch, modality):
    """Zero one modality ('audio' or 'visual') in both compression views; None keeps both."""
    if modality:
        for prefix in ('', 'paired_'):
            batch[prefix + modality] = torch.zeros_like(batch[prefix + modality])
    return batch


def scores(model, dataset, modality=None):
    """Labels and fake scores, one per sample and compression view."""
    labels, values = [], []
    for batch in DataLoader(dataset, batch_size=32, collate_fn=collate_pairs):
        batch = drop(batch, modality)
        for prefix in ('', 'paired_'):
            values += predict_batch(model, batch[prefix + 'audio'], batch[prefix + 'visual'],
                                    batch['lengths'])['fake_score'].tolist()
            labels += batch['labels'].int().tolist()
    return labels, values


def retrain(config, settings, data, modality):
    """Same settings as 01_train with one modality zeroed; keeps the best val epoch."""
    torch.manual_seed(settings['seed'])
    model = AudioVisualDetector(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'])
    collate = lambda samples: drop(collate_pairs(samples), modality)
    train = DataLoader(data['train'], batch_size=settings['batch_size'], shuffle=True,
                       generator=torch.Generator().manual_seed(settings['seed']), collate_fn=collate)
    val = DataLoader(data['val'], batch_size=settings['batch_size'], collate_fn=collate)
    best, state = float('inf'), None
    for _ in range(settings['epochs']):
        run_epoch(model, train, optimizer, settings['consistency_weight'])
        loss = run_epoch(model, val)['supervised']
        if loss < best:
            best, state = loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model


def mismatch_scores(model, dataset, partners):
    index = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    rows = []
    for sample_id, partner_id in partners.items():
        own, other = dataset[index[sample_id]], dataset[index[partner_id]]
        for prefix, view in zip(('', 'paired_'), own['view_ids']):
            frames = min(len(own[prefix + 'visual']), len(other[prefix + 'audio']))
            score = predict_batch(model, other[prefix + 'audio'][None, :frames],
                                  own[prefix + 'visual'][None, :frames], torch.tensor([frames]))
            rows.append(dict(sample_id=sample_id, audio_from=partner_id, view_id=view,
                             fake_score=score['fake_score'].item()))
    return rows


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Experiment directory containing best.pt')
    parser.add_argument('--real-manifest', type=Path, default=REAL_MANIFEST,
                        help='Real clips with speaker_id, source_video and start_time')
    parser.add_argument('--seed', type=int, default=7, help='Choice of mismatch partners')
    args = parser.parse_args()
    run = Path(args.run)
    output = run / 'cue_check'
    if output.exists():
        parser.error(f'Already checked; keep it: {output}')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    manifest = Path(checkpoint['args']['manifest'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint['manifest_sha256']:
        parser.error(f'{manifest} differs from the manifest used for training')
    clips = unique_rows(read_rows(args.real_manifest))
    data = {split: PairedFeatureDataset(manifest, split) for split in ('train', 'val', 'test')}
    model = AudioVisualDetector(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])

    metrics, genuine = [], {}
    for name, modality in (('audio+visual', None), ('visual_only', 'audio'), ('audio_only', 'visual')):
        if modality:
            print(f'Retraining {name} ({checkpoint["args"]["epochs"]} epochs)...', flush=True)
        current = retrain(checkpoint['model_config'], checkpoint['args'], data, modality) if modality else model
        results = {split: scores(current, data[split], modality) for split in ('val', 'test')}
        threshold = equal_error_rate(*results['val'])[1]
        if not modality:
            genuine = {split: (threshold, [s for label, s in zip(*r) if label == 0])
                       for split, r in results.items()}
        for split, (labels, values) in results.items():
            metrics.append(dict(model=name, split=split, auc=roc_auc(labels, values),
                                eer=equal_error_rate(labels, values)[0], threshold=threshold,
                                **threshold_report(labels, values, threshold)))

    seconds = len(data['val'][0]['audio']) / 25
    mismatch, summary = [], []
    for split in ('val', 'test'):
        threshold, real_scores = genuine[split]
        rows = mismatch_scores(model, data[split],
                               mismatch_partners(data[split].rows, clips, seconds, args.seed))
        if not rows:
            raise ValueError(f'No same-speaker, non-overlapping real partners in {split}')
        values = [row['fake_score'] for row in rows]
        mismatch += [dict(split=split, **row) for row in rows]
        summary.append(dict(split=split, mismatched=len(values),
                            flagged_fake=sum(v >= threshold for v in values) / len(values),
                            genuine_real_flagged=sum(v >= threshold for v in real_scores) / len(real_scores),
                            auc_vs_genuine_real=roc_auc([0] * len(real_scores) + [1] * len(values),
                                                        real_scores + values)))

    output.mkdir()
    write_csv(output / 'metrics.csv', metrics)
    write_csv(output / 'mismatch_summary.csv', summary)
    write_csv(output / 'mismatch_scores.csv', mismatch)
    print(f'{"model":12} {"split":5} {"AUC":>6} {"EER":>6} {"acc":>6} {"real_FA":>7} {"fake_miss":>9}')
    for m in metrics:
        print(f'{m["model"]:12} {m["split"]:5} {m["auc"]:6.3f} {m["eer"]:6.3f} {m["accuracy"]:6.3f} '
              f'{m["real_false_alarm"]:7.3f} {m["fake_miss"]:9.3f}')
    print('\nReal video + other real audio (same speaker), scored by the run model:')
    print(f'{"split":5} {"n":>5} {"flagged_fake":>12} {"genuine_real_flagged":>20} {"AUC_vs_real":>11}')
    for s in summary:
        print(f'{s["split"]:5} {s["mismatched"]:5d} {s["flagged_fake"]:12.3f} '
              f'{s["genuine_real_flagged"]:20.3f} {s["auc_vs_genuine_real"]:11.3f}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
