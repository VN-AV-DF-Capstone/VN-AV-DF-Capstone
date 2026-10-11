"""Pair each real clip with audio of another real clip: out of sync, but no generator involved."""
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
# Speaker, source video and start time of each real clip (written by src/data/07_add_part1.py).
REAL_MANIFEST = ROOT / 'data/manifests/dataset_v1/real_splits_multi_001.csv'


def mismatch_partners(rows, clips, seconds, seed):
    """Real sample_id -> another real sample of the same speaker (rows are one split) whose
    window does not overlap in time. Samples without such a partner are left out."""
    def window(row):
        clip = clips[row['sample_id'].rsplit('__', 2)[0]]  # <clip_id>__<run_id>__real
        return clip['speaker_id'], clip['source_video'], float(clip['start_time'])
    real = sorted((row for row in rows if row['label'] == '0'), key=lambda row: row['sample_id'])
    rng, partners = random.Random(seed), {}
    for row in real:
        speaker, video, start = window(row)
        options = [other for other in real if other is not row and window(other)[0] == speaker
                   and (window(other)[1] != video or abs(window(other)[2] - start) >= seconds)]
        if options:
            partners[row['sample_id']] = rng.choice(options)['sample_id']
    return partners
