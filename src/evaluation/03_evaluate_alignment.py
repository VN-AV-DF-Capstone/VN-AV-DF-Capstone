"""Score val/test with an alignment run from src/training/02_train_alignment.py.

fake_score is the misalignment: higher means audio and lips agree less. The threshold is the val
EER point (real vs fake, both compression views) and is applied unchanged to test. A second check
scores REAL video with audio of another REAL clip of the same speaker and split without time
overlap: it shows whether the model reacts to audio that does not belong to the lips, with no
generator involved (src/evaluation/sync_report.py). Never pick runs, epochs or thresholds from
test numbers.
--max-shift S scores each clip at its best constant audio-video shift within S frames, so a
real video whose recording is slightly out of sync is not called fake for that alone.
A run trained with --resync R is scored with the same cropping, on every clip and every
mismatched pair.
"""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.evaluation.mismatch import REAL_MANIFEST
from src.evaluation.sync_report import evaluate
from src.models.alignment import AlignmentModel


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
    resync = checkpoint['args'].get('resync', 0)  # Same cropping as in training.
    print(f'Checkpoint epoch {checkpoint["epoch"]}; resync {resync}; max shift {args.max_shift}')
    evaluate(lambda audio, visual, lengths: model(audio, visual, lengths, args.max_shift),
             manifest, args.real_manifest, output, args.seed, resync)


if __name__ == '__main__':
    main()
