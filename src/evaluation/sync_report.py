"""Val/test report shared by audio-lip sync scorers.

score_fn(audio [B,T,Da], visual [B,T,Dv], lengths [B]) returns one score per clip; higher means
audio and lips agree less. The threshold is the val EER point (real vs fake, both compression
views) and is applied unchanged to test. A second check scores REAL video with audio of another
REAL clip of the same speaker and split without time overlap: it shows whether the score reacts
to audio that does not belong to the lips, with no generator involved.
"""
import csv

import torch
from torch.utils.data import DataLoader
from src.data.preparation.manifest_io import read_rows, unique_rows
from src.evaluation.metrics import equal_error_rate, roc_auc, threshold_report
from src.evaluation.mismatch import mismatch_partners
from src.features.sync import crop_to_best_offset
from src.training.dataset import PairedFeatureDataset, collate_pairs


@torch.inference_mode()
def score(score_fn, dataset):
    """One row per sample and compression view."""
    rows = []
    for batch in DataLoader(dataset, batch_size=32, collate_fn=collate_pairs):
        for view, prefix in enumerate(('', 'paired_')):
            values = score_fn(batch[prefix + 'audio'], batch[prefix + 'visual'], batch['lengths'])
            rows += [dict(sample_id=sample_id, label=int(label), view_id=views[view], fake_score=float(value))
                     for sample_id, label, views, value in
                     zip(batch['sample_ids'], batch['labels'], batch['view_ids'], values)]
    return rows


@torch.inference_mode()
def mismatch_scores(score_fn, dataset, partners, resync):
    """dataset must not be resynced: the cropping is estimated on the mismatched pair itself."""
    index = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    rows = []
    for sample_id, partner_id in partners.items():
        own, other = dataset[index[sample_id]], dataset[index[partner_id]]
        frames = min(len(own['visual']), len(other['audio']))
        views = [(other[prefix + 'audio'][:frames], own[prefix + 'visual'][:frames])
                 for prefix in ('', 'paired_')]
        if resync:
            views = crop_to_best_offset(views, resync)
        for (audio, visual), view in zip(views, own['view_ids']):
            value = score_fn(audio[None], visual[None], torch.tensor([len(audio)]))
            rows.append(dict(sample_id=sample_id, audio_from=partner_id, view_id=view, fake_score=value.item()))
    return rows


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(score_fn, manifest, real_manifest, output, seed, resync=0):
    """Score val/test of manifest, write the CSVs into the new directory output and print the tables.

    resync R > 0 crops every clip and every mismatched pair to its best offset within ±R frames.
    """
    clips = unique_rows(read_rows(real_manifest))
    raw = {split: PairedFeatureDataset(manifest, split) for split in ('val', 'test')}
    data = {split: PairedFeatureDataset(manifest, split, resync) for split in raw}
    predictions = {split: score(score_fn, dataset) for split, dataset in data.items()}
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

    seconds = len(raw['val'][0]['audio']) / 25
    mismatch, summary = [], []
    for split, dataset in raw.items():
        partners = mismatch_partners(dataset.rows, clips, seconds, seed)
        rows = mismatch_scores(score_fn, dataset, partners, resync)
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

    output.mkdir(parents=True)
    write_csv(output / 'predictions.csv', [dict(split=split, **row) for split, rows in predictions.items()
                                           for row in rows])
    write_csv(output / 'metrics.csv', metrics)
    write_csv(output / 'mismatch_summary.csv', summary)
    write_csv(output / 'mismatch_scores.csv', mismatch)
    print(f'Threshold {threshold:.4f} from val EER')
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
