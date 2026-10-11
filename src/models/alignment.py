"""Audio-lip alignment scorer learned from REAL clips only, after AVH-Align (Smeu et al., CVPR 2025).

Same idea as AVH-Align (frozen AV-HuBERT audio-only and video-only features, train on real
videos only, score how well audio and lips agree); the head and loss here are simpler, so this
is not a reproduction of that paper.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _projection(input_dim, hidden_dim):
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, hidden_dim))


class AlignmentModel(nn.Module):
    """Inputs: audio [B,T,Da], visual [B,T,Dv], valid lengths [B]. Output: misalignment [B].

    For video frame t, audio frames t-K..t+K of the same clip are the candidates and frame t is
    the synchronized one. The output is the mean over frames of -log p(synchronized frame): about
    log(2K+1) when audio and lips share no timing, near 0 when they match. Training minimizes it
    on real clips; a high value on a test clip means audio and lips disagree.
    With max_shift S, the synchronized frame may instead be t+s for one constant s in [-S, S] per
    clip and the best s is kept: a constant offset of up to S frames is then not penalized.
    """
    temperature = 0.1

    def __init__(self, audio_dim, visual_dim, hidden_dim=256, max_offset=10):
        super().__init__()
        self.audio_projection = _projection(audio_dim, hidden_dim)
        self.visual_projection = _projection(visual_dim, hidden_dim)
        self.max_offset = max_offset

    def forward(self, audio, visual, lengths, max_shift=0):
        if audio.ndim != 3 or visual.ndim != 3 or audio.shape[:2] != visual.shape[:2]:
            raise ValueError('Audio and visual must share batch size and aligned time steps')
        if (lengths.shape != (audio.shape[0],) or lengths.is_floating_point()
                or torch.any(lengths < 1) or torch.any(lengths > audio.shape[1])):
            raise ValueError('Lengths must be integer valid-frame counts, one per sample')
        if not 0 <= max_shift <= self.max_offset:
            raise ValueError('max_shift must lie between 0 and max_offset')
        a = F.normalize(self.audio_projection(audio), dim=-1)
        v = F.normalize(self.visual_projection(visual), dim=-1)
        frames, device = audio.shape[1], audio.device
        offsets = torch.arange(-self.max_offset, self.max_offset + 1, device=device)
        candidates = torch.arange(frames, device=device)[:, None] + offsets  # [T, 2K+1]
        logits = torch.einsum('btd,btkd->btk', v, a[:, candidates.clamp(0, frames - 1)]) / self.temperature
        lengths = lengths.to(device)
        valid = (candidates >= 0) & (candidates < lengths[:, None, None])
        valid[..., self.max_offset] = True  # Keeps padded frames finite; they are dropped below.
        loss = -logits.masked_fill(~valid, float('-inf')).log_softmax(-1)
        counted = torch.arange(frames, device=device)[None] < lengths[:, None]
        scores = []
        for shift in range(self.max_offset - max_shift, self.max_offset + max_shift + 1):
            used = counted & valid[..., shift]  # Frames whose shifted audio frame exists.
            scores.append(torch.where(used, loss[..., shift], 0.0).sum(1) / used.sum(1).clamp(min=1))
        return torch.stack(scores).min(0).values
