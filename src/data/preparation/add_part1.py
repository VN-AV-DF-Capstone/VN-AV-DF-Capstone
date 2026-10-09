"""Add Person 2's reviewed part1 clips as new speakers and publish a NEW split version.

V1's locked split is only read. All V1 clips (one speaker) go to train, so val/test hold
only speakers never seen in training. Part1 is split by connected speaker/video/channel
groups; a channel shared by two speakers keeps both in one split.
"""
import argparse
import hashlib
import json
from pathlib import Path

from src.data.preparation.build_splits import STRICT_PROTOCOL, split_real_rows, verify_splits
from src.data.preparation.manifest_io import read_rows, write_rows

SPLIT_FIELDS = ('split', 'group_id', 'split_protocol')


def part1_rows(review_csv):
    """Keep decisions only, mapped to the main manifest columns with absolute media paths."""
    review_csv = Path(review_csv).resolve()
    rows = []
    for row in read_rows(review_csv):
        if row['decision'] != 'keep':
            continue
        if row.get('sync_status') != 'reviewed_match':
            raise ValueError(f'Kept part1 clip without reviewed sync: {row["clip_id"]}')
        path = review_csv.parent / row['file_path']
        if not path.is_file():
            raise FileNotFoundError(path)
        channel = row.get('channel', '').strip()
        rows.append(dict(clip_id=row['clip_id'], source_video=row['source_id'].removeprefix('yt_'),
                         start_time=row['source_start_s'], end_time=row['source_end_s'],
                         duration=row['duration_s'], face_ratio=row['face_ratio'],
                         file_path=str(path), url=row['url'], channel=channel,
                         shared_group=f'channel:{channel}' if channel else '',
                         decision='keep', speaker_id=row['speaker_id'], sha256=row['sha256'],
                         label='0', origin='part1'))
    if not rows:
        raise ValueError(f'No keep clips in {review_csv}')
    return rows


def merge_splits(v1_rows, part1, ratios, seed):
    v1 = [dict({k: v for k, v in row.items() if k not in SPLIT_FIELDS}, origin='v1') for row in v1_rows]
    rows = (split_real_rows(v1, (1.0, 0.0, 0.0), seed, STRICT_PROTOCOL)
            + split_real_rows(part1, ratios, seed, STRICT_PROTOCOL))
    rows.sort(key=lambda row: row['clip_id'])
    verify_splits(rows)  # Also catches a speaker or video shared by V1 and part1.
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', required=True, help="V1's locked real_splits.csv")
    parser.add_argument('--part1', required=True, help="Person 2's review.csv")
    parser.add_argument('--out', required=True)
    parser.add_argument('--ratios', default='0.4,0.3,0.3', help='Part1 groups only')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    ratios = tuple(map(float, args.ratios.split(',')))
    rows = merge_splits(read_rows(args.v1), part1_rows(args.part1), ratios, args.seed)
    write_rows(args.out, rows)
    summary = {split: dict(clips=sum(r['split'] == split for r in rows),
                           speakers=sorted({r['speaker_id'] for r in rows if r['split'] == split}))
               for split in ('train', 'val', 'test')}
    provenance = dict(v1_sha256=sha256(args.v1), part1_sha256=sha256(args.part1), ratios=ratios,
                      seed=args.seed, protocol=STRICT_PROTOCOL, v1_split='all train', splits=summary)
    Path(args.out).with_suffix('.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    for split, info in summary.items():
        print(f'{split}: {info["clips"]} clips, speakers {", ".join(info["speakers"])}')
    print(f'Output: {args.out}')


if __name__ == '__main__':
    main()
