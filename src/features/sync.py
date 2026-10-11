"""Audio-video timing seen through AV-HuBERT audio-only and video-only features."""
import torch
from torch.nn import functional as F


def offset_profile(audio, visual, max_offset):
    """Cosine between time-centred video-only frame t and audio-only frame t+k, mean over t, per k."""
    a = F.normalize(audio - audio.mean(0), dim=1)
    v = F.normalize(visual - visual.mean(0), dim=1)
    similarity = v @ a.T
    return torch.stack([similarity.diagonal(k).mean() for k in range(-max_offset, max_offset + 1)])


def crop_to_best_offset(views, max_offset):
    """Crop the (audio, visual) views of one clip so video frame t meets its best-matching audio frame.

    One offset within ±max_offset is estimated from all views together, so they stay equally long.
    """
    offset = int(sum(offset_profile(a, v, max_offset) for a, v in views).argmax()) - max_offset
    start, end = max(offset, 0), len(views[0][0]) + min(offset, 0)
    return [(a[start:end], v[start - offset:end - offset]) for a, v in views]
