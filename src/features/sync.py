"""Audio-video timing seen through AV-HuBERT audio-only and video-only features."""
import torch
from torch.nn import functional as F


def offset_profile(audio, visual, max_offset):
    """Cosine between time-centred video-only frame t and audio-only frame t+k, mean over t, per k."""
    a = F.normalize(audio - audio.mean(0), dim=1)
    v = F.normalize(visual - visual.mean(0), dim=1)
    similarity = v @ a.T
    return torch.stack([similarity.diagonal(k).mean() for k in range(-max_offset, max_offset + 1)])
