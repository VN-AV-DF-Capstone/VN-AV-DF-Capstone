"""Estimate each clip's audio-video offset from fused AV-HuBERT features, without a trained model.

A clip's offset is the k (frames at 25 fps) at which video-only frame t is most similar to
audio-only frame t+k: the 02_fuse.py profile, averaged over both compression views. The table
gives the median per source video, so videos whose real clips are out of sync stand out.
A data diagnostic: a sync rule built on it must be fixed for ALL splits and drop real and fake
alike; never select clips by detector scores or by test results.
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.data.preparation.manifest_io import read_rows, unique_rows
from src.evaluation.mismatch import REAL_MANIFEST
from src.features.sync import offset_profile

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused/pairs.csv'
MAX_OFFSET = 15


def summary(items):
    """Count, median offset, share within one frame of that median, median peak cosine."""
    if not items:
        return f'{0:4d} {"":>6} {"":>5} {"":>5}'
    median = statistics.median(row['offset'] for row in items)
    near = sum(abs(row['offset'] - median) <= 1 for row in items) / len(items)
    peak = statistics.median(row['peak_cosine'] for row in items)
    return f'{len(items):4d} {median:+6.1f} {near:5.2f} {peak:5.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST, help='Fused feature pairs.csv from 02_fuse.py')
    parser.add_argument('--real-manifest', type=Path, default=REAL_MANIFEST,
                        help='Real clips with speaker_id and source_video')
    parser.add_argument('--out', type=Path, required=True, help='New CSV, one row per clip')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'Already exists; keep it: {args.out}')
    clips = unique_rows(read_rows(args.real_manifest))
    rows = []
    for row in read_rows(args.manifest):
        caches = [torch.load(args.manifest.parent / row[key], map_location='cpu', weights_only=True)
                  for key in ('feature_path', 'paired_feature_path')]
        profile = sum(offset_profile(c['audio'].float(), c['visual'].float(), MAX_OFFSET) for c in caches) / 2
        clip = clips[row['sample_id'].rsplit('__', 2)[0]]  # <clip_id>__<run_id>__<real|fake>
        rows.append(dict(sample_id=row['sample_id'], label=row['label'], split=row['split'],
                         speaker_id=clip['speaker_id'], source_video=clip['source_video'],
                         offset=int(profile.argmax()) - MAX_OFFSET, peak_cosine=float(profile.max())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(lambda: {'0': [], '1': []})
    for row in rows:
        groups[(row['split'], row['speaker_id'], row['source_video'])][row['label']].append(row)
    order = {'train': 0, 'val': 1, 'test': 2}
    print('Offset k: video frame t matches audio frame t+k (40 ms per frame). near = share within 1 frame '
          'of the median; peak = median best cosine.')
    print(f'{"split":5} {"speaker":12} {"source_video":14} | {"real":>4} {"offset":>6} {"near":>5} {"peak":>5} '
          f'| {"fake":>4} {"offset":>6} {"near":>5} {"peak":>5}')
    for (split, speaker, video), labels in sorted(groups.items(), key=lambda item: (order[item[0][0]], item[0])):
        print(f'{split:5} {speaker:12} {video:14} | {summary(labels["0"])} | {summary(labels["1"])}')
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
