"""Score val/test with an alignment run from src/training/02_train_alignment.py.

fake_score is the misalignment: higher means audio and lips agree less. The threshold is the val
EER point (real vs fake, both compression views) and is applied unchanged to test. A second check
scores REAL video with audio of another REAL clip of the same speaker and split without time
overlap: it shows whether the model reacts to audio that does not belong to the lips, with no
generator involved. Never pick runs, epochs or thresholds from test numbers.
--max-shift S scores each clip at its best constant audio-video shift within S frames, so a
real video whose recording is slightly out of sync is not called fake for that alone.
"""
import argparse
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
from src.models.alignment import AlignmentModel
from src.training.dataset import PairedFeatureDataset, collate_pairs


@torch.inference_mode()
def score(model, dataset, max_shift):
    """One row per sample and compression view."""
    rows = []
    for batch in DataLoader(dataset, batch_size=32, collate_fn=collate_pairs):
        for view, prefix in enumerate(('', 'paired_')):
            values = model(batch[prefix + 'audio'], batch[prefix + 'visual'], batch['lengths'], max_shift)
            rows += [dict(sample_id=sample_id, label=int(label), view_id=views[view], fake_score=float(value))
                     for sample_id, label, views, value in
                     zip(batch['sample_ids'], batch['labels'], batch['view_ids'], values)]
    return rows


@torch.inference_mode()
def mismatch_scores(model, dataset, partners, max_shift):
    index = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    rows = []
    for sample_id, partner_id in partners.items():
        own, other = dataset[index[sample_id]], dataset[index[partner_id]]
        for prefix, view in zip(('', 'paired_'), own['view_ids']):
            frames = min(len(own[prefix + 'visual']), len(other[prefix + 'audio']))
            value = model(other[prefix + 'audio'][None, :frames], own[prefix + 'visual'][None, :frames],
                          torch.tensor([frames]), max_shift)
            rows.append(dict(sample_id=sample_id, audio_from=partner_id, view_id=view, fake_score=value.item()))
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
    parser.add_argument('--max-shift', type=int, default=0, help='Tolerated constant offset in frames')
    args = parser.parse_args()
    run = Path(args.run)
    output = run / ('evaluation' if args.max_shift == 0 else f'evaluation_shift{args.max_shift}')
    if output.exists():
        parser.error(f'Already evaluated; keep it and train a new run instead: {output}')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    manifest = Path(checkpoint['args']['manifest'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint['manifest_sha256']:
        parser.error(f'{manifest} differs from the manifest used for training')
    model = AlignmentModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    clips = unique_rows(read_rows(args.real_manifest))
    data = {split: PairedFeatureDataset(manifest, split) for split in ('val', 'test')}
    predictions = {split: score(model, dataset, args.max_shift) for split, dataset in data.items()}
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

    seconds = len(data['val'][0]['audio']) / 25
    mismatch, summary = [], []
    for split, dataset in data.items():
        rows = mismatch_scores(model, dataset, mismatch_partners(dataset.rows, clips, seconds, args.seed),
                               args.max_shift)
        if not rows:
            raise ValueError(f'No same-speaker, non-overlapping real partners in {split}')
        values = [row['fake_score'] for row in rows]
        real_scores = [r['fake_score'] for r in predictions[split] if r['label'] == 0]
        mismatch += [dict(split=split, **row) for row in rows]
        summary.append(dict(split=split, mismatched=len(values),
                            flagged_fake=sum(v >= threshold for v in values) / len(values),
                            genuine_real_flagged=sum(v >= threshold for v in real_scores) / len(real_scores),
                            auc_vs_genuine_real=roc_auc([0] * len(real_scores) + [1] * len(values),
                                                        real_scores + values)))

    output.mkdir()
    write_csv(output / 'predictions.csv', [dict(split=split, **row) for split, rows in predictions.items()
                                           for row in rows])
    write_csv(output / 'metrics.csv', metrics)
    write_csv(output / 'mismatch_summary.csv', summary)
    write_csv(output / 'mismatch_scores.csv', mismatch)
    print(f'Checkpoint epoch {checkpoint["epoch"]}; max shift {args.max_shift}; '
          f'threshold {threshold:.4f} from val EER')
    print('Real vs Wav2Lip fake:')
    print(f'{"split":5} {"view":6} {"real":>5} {"fake":>5} {"AUC":>6} {"EER":>6} {"acc":>6} '
          f'{"bal_acc":>7} {"real_FA":>7} {"fake_miss":>9}')
    for m in metrics:
        print(f'{m["split"]:5} {m["view"]:6} {m["real"]:5d} {m["fake"]:5d} {m["auc"]:6.3f} {m["eer"]:6.3f} '
              f'{m["accuracy"]:6.3f} {m["balanced_accuracy"]:7.3f} {m["real_false_alarm"]:7.3f} '
              f'{m["fake_miss"]:9.3f}')
    print('\nReal video + other real audio (same speaker):')
    print(f'{"split":5} {"n":>5} {"flagged_fake":>12} {"genuine_real_flagged":>20} {"AUC_vs_real":>11}')
    for s in summary:
        print(f'{s["split"]:5} {s["mismatched"]:5d} {s["flagged_fake"]:12.3f} '
              f'{s["genuine_real_flagged"]:20.3f} {s["auc_vs_genuine_real"]:11.3f}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
