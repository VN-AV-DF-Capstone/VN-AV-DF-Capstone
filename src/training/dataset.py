"""Read paired cached features; never construct new train/test splits here."""
import csv
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from src.features.sync import crop_to_best_offset


class PairedFeatureDataset(Dataset):
    """CSV: sample_id,label,split,feature_path,paired_feature_path.

    Each .pt is a plain dictionary: sample_id, label, split, view_id,
    extractor_id, timeline_id, audio [T,Da], visual [T,Dv].
    sample_id identifies the authentic OR manipulated clip and exact window.
    timeline_id identifies timestamps/preprocessing; extractor_id identifies the
    checkpoint hash, extraction mode and preprocessing version, not just a name.
    Paths are relative to the CSV. Compression pairs differ only in view_id.
    resync R > 0 crops each clip to its best audio-video offset within ±R frames.
    """
    def __init__(self, manifest, split, resync=0):
        if split not in {'train', 'val', 'test'}:
            raise ValueError('Use train, val or test')
        self.root = Path(manifest).resolve().parent
        with open(manifest, encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        seen = set()
        for row in rows:
            if not row['sample_id'] or row['sample_id'] in seen:
                raise ValueError('sample_id must occur once in the manifest')
            seen.add(row['sample_id'])
            if row['label'] not in {'0', '1'} or row['split'] not in {'train', 'val', 'test'}:
                raise ValueError('Invalid label or split')
        self.rows = [row for row in rows if row['split'] == split]
        if not self.rows:
            raise ValueError(f'No samples for {split}')
        self.resync = resync

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        views = [torch.load(self.root / row[key], map_location='cpu', weights_only=True)
                 for key in ('feature_path', 'paired_feature_path')]
        for view in views:
            if (view['sample_id'] != row['sample_id'] or view['label'] != int(row['label'])
                    or view['split'] != row['split']):
                raise ValueError('Cache identity/label/split differs from manifest')
            a, v = view['audio'], view['visual']
            if (a.ndim != 2 or v.ndim != 2 or a.shape[0] != v.shape[0]
                    or min(a.shape) < 1 or min(v.shape) < 1
                    or not a.is_floating_point() or not v.is_floating_point()
                    or not torch.isfinite(a).all() or not torch.isfinite(v).all()):
                raise ValueError('Cache requires finite aligned floating point features')
        first, second = views
        for key in ('extractor_id', 'timeline_id'):
            if not first[key] or first[key] != second[key]:
                raise ValueError(f'Compression views must share {key}')
        if not first['view_id'] or not second['view_id'] or first['view_id'] == second['view_id']:
            raise ValueError('Expected two distinct compression views')
        if any(first[key].shape != second[key].shape for key in ('audio', 'visual')):
            raise ValueError('Compression views must share feature shapes')
        pairs = [(view['audio'].float(), view['visual'].float()) for view in views]
        if self.resync:
            pairs = crop_to_best_offset(pairs, self.resync)
        (audio, visual), (paired_audio, paired_visual) = pairs
        return dict(audio=audio, visual=visual, paired_audio=paired_audio, paired_visual=paired_visual,
                    label=int(row['label']), sample_id=row['sample_id'],
                    extractor_id=first['extractor_id'], view_ids=(first['view_id'], second['view_id']))


def collate_pairs(samples):
    if len({sample['extractor_id'] for sample in samples}) != 1:
        raise ValueError('Cannot mix extractors in a batch')
    result = {key: pad_sequence([sample[key] for sample in samples], batch_first=True)
              for key in ('audio', 'visual', 'paired_audio', 'paired_visual')}
    result['lengths'] = torch.tensor([len(sample['audio']) for sample in samples])
    result['labels'] = torch.tensor([sample['label'] for sample in samples], dtype=torch.float32)
    result['sample_ids'] = [sample['sample_id'] for sample in samples]
    result['paired_sample_ids'] = list(result['sample_ids'])
    result['view_ids'] = [sample['view_ids'] for sample in samples]
    return result
