"""Train the audio-lip alignment scorer (src/models/alignment.py) on REAL clips only.

Fake rows of the manifest are never loaded: training uses real train clips and the kept epoch
has the lowest misalignment on real val clips, both compression views. Intended for fused
AV-HuBERT features from src/features/02_fuse.py.
--resync R first crops every clip to its best audio-video offset within R frames, estimated from
the raw features (src/features/sync.py). Recordings differ by a few frames of constant offset, so
without it the model can learn one typical offset as the sign of a real clip.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader
from src.models.alignment import AlignmentModel
from src.training.dataset import PairedFeatureDataset, collate_pairs


def real_only(manifest, split, resync):
    dataset = PairedFeatureDataset(manifest, split, resync)
    dataset.rows = [row for row in dataset.rows if row['label'] == '0']
    if not dataset.rows:
        raise ValueError(f'No real samples for {split}')
    return dataset


def run_epoch(model, loader, optimizer=None):
    """Mean misalignment over both compression views; trains when an optimizer is given."""
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for batch in loader:
        with torch.set_grad_enabled(optimizer is not None):
            loss = (model(batch['audio'], batch['visual'], batch['lengths']).mean()
                    + model(batch['paired_audio'], batch['paired_visual'], batch['lengths']).mean()) / 2
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += loss.item() * len(batch['lengths'])
        count += len(batch['lengths'])
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True, help='New experiment directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--max-offset', type=int, default=10, help='Candidate audio frames each side')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--resync', type=int, default=0, help='Best-offset search range in frames (0 = off)')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim, args.max_offset, args.lr) <= 0:
        parser.error('Positive epochs/batch/hidden/max-offset/lr required')
    if args.resync < 0:
        parser.error('--resync must be 0 or more')
    torch.manual_seed(args.seed)
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    train, val = real_only(args.manifest, 'train', args.resync), real_only(args.manifest, 'val', args.resync)
    signature = None
    for dataset in (train, val):
        for item in dataset:
            current = (item['extractor_id'], item['audio'].shape[1], item['visual'].shape[1])
            if signature is not None and signature != current:
                raise ValueError('All caches must share extractor and feature dimensions')
            signature = current
    config = dict(audio_dim=signature[1], visual_dim=signature[2], hidden_dim=args.hidden_dim,
                  max_offset=args.max_offset)
    model = AlignmentModel(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=collate_pairs)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate_pairs)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    print(f'{len(train)} real train clips, {len(val)} real val clips. Misalignment without any '
          f'audio-lip timing is about {math.log(2 * args.max_offset + 1):.2f}; lower is better.', flush=True)
    best = float('inf')
    with (output / 'epochs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_misalignment', 'val_real_misalignment'])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            row = dict(epoch=epoch, train_misalignment=run_epoch(model, train_loader, optimizer),
                       val_real_misalignment=run_epoch(model, val_loader))
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)
            if row['val_real_misalignment'] < best:
                best = row['val_real_misalignment']
                torch.save(dict(model=model.state_dict(), model_config=config, epoch=epoch,
                                args=vars(args), extractor_id=signature[0], val_real_misalignment=best,
                                manifest_sha256=manifest_hash, real_train_clips=len(train)),
                           output / 'best.pt')


if __name__ == '__main__':
    main()
