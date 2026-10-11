"""Add AV-HuBERT's fused Transformer to a finished pre-fusion cache from 01_extract.py.

The cached audio/visual are exactly the branch outputs the fused model consumes, so no media is
decoded again. Each view gets audio-only and video-only final-layer features (the other modality
zeroed, as upstream extract_finetune), saved with the same file names and pairs.csv. Run this
file after editing SETTINGS below, or use --help for CLI overrides.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.features.sync import offset_profile

CHECKPOINT = ROOT / 'weights/avhubert/base_vox_iter5.pt'
SOURCE = ROOT / 'cache/features/dataset_v1/wav2lip_multi_001_crf23_32'
OUTPUT = ROOT / 'cache/features/dataset_v1/wav2lip_multi_001_crf23_32_fused'
DEVICE = 'cpu'
MAX_OFFSET = 5  # Frames either side in the printed audio-video similarity check.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--source', type=Path, default=SOURCE, help='Finished 01_extract.py output')
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=DEVICE)
    args = parser.parse_args()
    for key in ('checkpoint', 'source', 'out'):
        if not getattr(args, key).is_absolute():
            setattr(args, key, ROOT / getattr(args, key))
    source_manifest = args.source / 'pairs.csv'
    if not source_manifest.is_file():
        parser.error(f'Missing {source_manifest}; finish 01_extract.py first')
    if args.out.exists():
        parser.error(f'Output already exists; keep it, or delete it if it has no pairs.csv: {args.out}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this Python environment; select cpu')
    from src.features.avhubert import load_fused, sha256
    with source_manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    fused, backbone = load_fused(args.checkpoint, args.device)
    provenance = dict(backbone=backbone, source_pairs_sha256=sha256(source_manifest), device=args.device,
                      implementation=sha256(ROOT / 'src/features/avhubert.py'),
                      torch=importlib.metadata.version('torch'))
    args.out.mkdir(parents=True)
    source_id, extractor_id = None, None
    profiles = {'0': torch.zeros(2 * MAX_OFFSET + 1), '1': torch.zeros(2 * MAX_OFFSET + 1)}
    counts = {'0': 0, '1': 0}
    for index, row in enumerate(rows):
        if index % 100 == 0:
            print(f'[{index + 1}/{len(rows)}] {row["sample_id"]}', flush=True)
        for key in ('feature_path', 'paired_feature_path'):
            cache = torch.load(args.source / row[key], map_location='cpu', weights_only=True)
            if source_id is None:
                source_id = cache['extractor_id']
                identity = json.dumps(dict(provenance, source_extractor_id=source_id), sort_keys=True)
                extractor_id = hashlib.sha256(identity.encode()).hexdigest()
            if cache['extractor_id'] != source_id or cache['sample_id'] != row['sample_id']:
                raise ValueError(f'{row[key]}: another extractor or sample than the rest of the cache')
            audio, visual = fused(cache['audio'][None].float().to(args.device),
                                  cache['visual'][None].float().to(args.device))
            audio, visual = audio[0].cpu(), visual[0].cpu()
            if not torch.isfinite(audio).all() or not torch.isfinite(visual).all():
                raise ValueError(f'{row[key]}: fused features are not finite')
            torch.save(dict(cache, extractor_id=extractor_id, source_extractor_id=source_id,
                            audio=audio, visual=visual), args.out / row[key])
            profiles[row['label']] += offset_profile(audio, visual, MAX_OFFSET)
            counts[row['label']] += 1
    profiles = {label: (profile / max(counts[label], 1)).tolist() for label, profile in profiles.items()}
    report = dict(schema='avhubert_fused_run_v1', extractor_id=extractor_id, provenance=provenance,
                  source=str(args.source), source_extractor_id=source_id,
                  offset_profile=dict(offsets=list(range(-MAX_OFFSET, MAX_OFFSET + 1)),
                                      real=profiles['0'], fake=profiles['1']))
    (args.out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    shutil.copyfile(source_manifest, args.out / 'pairs.csv')  # Written last: marks a finished run.
    print('\nCosine of video-only frame t vs audio-only frame t+k (time-centred), mean over views:')
    print('k     ' + ''.join(f'{k:>7d}' for k in range(-MAX_OFFSET, MAX_OFFSET + 1)))
    for label, name in (('0', 'real'), ('1', 'fake')):
        print(f'{name:6}' + ''.join(f'{value:7.3f}' for value in profiles[label]))
    print('A correctly loaded Transformer should make the real row peak at k=0.')
    print(f'Completed: {args.out}')


if __name__ == '__main__':
    main()
