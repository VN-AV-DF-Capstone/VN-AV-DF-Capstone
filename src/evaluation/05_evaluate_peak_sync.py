"""Score val/test with a training-free audio-lip score on fused AV-HuBERT features (02_fuse.py).

fake_score is minus the best cosine between video-only frame t and audio-only frame t+k over
|k| <= --max-offset frames (src/features/sync.py), so a recording's constant offset is not
penalized. Nothing is trained. Same report as 03_evaluate_alignment.py: the val EER threshold is
applied unchanged to test, and real video is also scored with another real clip's audio.
A lower cosine on fakes may partly come from generator blur, not only from lips out of sync.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.mismatch import REAL_MANIFEST
from src.evaluation.sync_report import evaluate
from src.features.sync import best_offset_misalignment

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused/pairs.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST,
                        help='Fused feature pairs.csv from 02_fuse.py')
    parser.add_argument('--real-manifest', type=Path, default=REAL_MANIFEST,
                        help='Real clips with speaker_id, source_video and start_time')
    parser.add_argument('--out', type=Path, required=True, help='New directory for the report')
    parser.add_argument('--max-offset', type=int, default=15, help='Searched audio-video offset in frames')
    parser.add_argument('--seed', type=int, default=7, help='Choice of mismatch partners')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'Already exists; keep it: {args.out}')
    if args.max_offset < 0:
        parser.error('--max-offset must be 0 or more')
    print(f'Training-free score; max offset {args.max_offset}')
    evaluate(lambda audio, visual, lengths: best_offset_misalignment(audio, visual, lengths, args.max_offset),
             args.manifest, args.real_manifest, args.out, args.seed)


if __name__ == '__main__':
    main()
