"""Frozen pre-fusion branches of an already loaded official AVHubertModel.

Not the fused Transformer output. The loader below supports base_vox_iter5.pt.
Audio must be checkpoint-compatible stacked acoustic features [B,F,T]; video
must be normalized grayscale mouth crops [B,1,T,H,W] on the SAME timeline.
Reference: facebookresearch/av_hubert, avhubert/hubert.py, SubModel.
load_fused adds the fused Transformer on top of those cached branch outputs.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class FrozenAVHubertBranches(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.audio_branch = backbone.feature_extractor_audio
        self.visual_branch = backbone.feature_extractor_video
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, audio, video):
        if (audio.ndim != 3 or video.ndim != 5 or video.shape[1] != 1
                or audio.shape[0] != video.shape[0] or audio.shape[2] != video.shape[2]):
            raise ValueError('Expected aligned audio [B,F,T] and video [B,1,T,H,W]')
        # Frozen extraction must stay deterministic even if a parent calls train().
        self.eval()
        a = self.audio_branch(audio).transpose(1, 2).contiguous()
        v = self.visual_branch(video).transpose(1, 2).contiguous()
        if a.shape[:2] != v.shape[:2] or a.shape[1] != audio.shape[2]:
            raise ValueError('Backbone changed the shared timeline')
        return a, v


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_source(path, name):
    """Import an explicit upstream file without importing its Fairseq package."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing upstream source: {path}. See src/features/README.md')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ProjectionBranch(nn.Module):
    """Equivalent to upstream SubModel for sub_encoder_layers=0 only."""
    def __init__(self, input_dim, output_dim, resnet=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        if self.resnet is not None:
            inputs = self.resnet(inputs)
        return self.proj(inputs.transpose(1, 2)).transpose(1, 2)


class _DictionaryMetadata:
    """Inert holder for the checkpoint's unused Fairseq training dictionary.

    Only tensor weights and cfg are consumed below. No Fairseq dictionary code
    is executed, and no model layer/weight is substituted by this metadata type.
    """
    pass


def _read_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Place base_vox_iter5.pt at: {checkpoint.resolve()}')
    if tuple(int(part) for part in torch.__version__.split('.')[:2]) < (2, 6):
        raise RuntimeError('Checkpoint loader requires PyTorch >= 2.6')
    with torch.serialization.safe_globals([
            (_DictionaryMetadata, 'fairseq.data.dictionary.Dictionary')]):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if state.get('cfg') is None:
        raise ValueError('Expected official pretrained checkpoint with cfg/model; finetuned files are unsupported')
    return state


def load_checkpoint(checkpoint, upstream, device='cpu'):
    """Load the two exact pre-fusion branches, not the fused Transformer.

    The official Base file stores plain cfg and an unused Fairseq dictionary.
    Read with weights_only and narrowly allow that metadata class, without
    importing Fairseq. No unrestricted pickle or fallback to random weights.
    The upstream ResEncoder is reused; Fairseq itself is not needed for these
    branches because Base has no per-modality Transformer layers.
    """
    checkpoint, upstream = Path(checkpoint), Path(upstream)
    state = _read_checkpoint(checkpoint)
    model_cfg, task_cfg = state['cfg']['model'], state['cfg']['task']
    if (model_cfg.get('_name') != 'av_hubert'
            or model_cfg.get('sub_encoder_layers', 0) != 0
            or model_cfg.get('audio_feat_dim') != 104
            or model_cfg.get('encoder_embed_dim', 768) != 768):
        raise ValueError('This loader supports AV-HuBERT Base pre-fusion branches with 104 audio inputs only')
    from src.features.media import MediaConfig
    config = MediaConfig.from_task(task_cfg)
    resnet_path = upstream / 'avhubert/resnet.py'
    resnet = load_source(resnet_path, '_avhubert_resnet').ResEncoder(
        relu_type=model_cfg.get('resnet_relu_type', 'prelu'), weights=None)
    audio = _ProjectionBranch(104, 768)
    visual = _ProjectionBranch(resnet.backend_out, 768, resnet)
    for name, branch in [('audio', audio), ('video', visual)]:
        prefix = f'feature_extractor_{name}.'
        branch_state = {key[len(prefix):]: value for key, value in state['model'].items()
                        if key.startswith(prefix)}
        # Strict loading rejects incomplete or incompatible branches.
        branch.load_state_dict(branch_state, strict=True)
    adapter = FrozenAVHubertBranches(SimpleNamespace(
        feature_extractor_audio=audio, feature_extractor_video=visual)).to(device)
    provenance = dict(checkpoint_sha256=sha256(checkpoint),
                      resnet_sha256=sha256(resnet_path), mode='base_pre_fusion_v1',
                      loaded_parameters=sum(p.numel() for p in adapter.parameters()))
    return adapter, config, provenance


class _PositionalConv(nn.Module):
    """Fairseq wav2vec2 pos_conv: grouped Conv1d with weight norm over dim 2, SamePad, GELU."""
    def __init__(self, dim, kernel, groups):
        super().__init__()
        self.weight_g = nn.Parameter(torch.empty(1, 1, kernel))
        self.weight_v = nn.Parameter(torch.empty(dim, dim // groups, kernel))
        self.bias = nn.Parameter(torch.empty(dim))
        self.kernel, self.groups = kernel, groups

    def forward(self, x):
        norm = torch.linalg.vector_norm(self.weight_v, dim=(0, 1), keepdim=True)
        weight = self.weight_v * (self.weight_g / norm)
        x = F.conv1d(x, weight, self.bias, padding=self.kernel // 2, groups=self.groups)
        return F.gelu(x[:, :, :-1] if self.kernel % 2 == 0 else x)  # SamePad drops the extra frame.


class _EncoderLayer(nn.Module):
    """Fairseq TransformerSentenceEncoderLayer with layer_norm_first=True, without dropout."""
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.self_attn = nn.ModuleDict({name: nn.Linear(dim, dim)
                                        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj')})
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, frames, dim = x.shape
        h = self.self_attn_layer_norm(x)
        q, k, v = (self.self_attn[name](h).view(batch, frames, self.heads, -1).transpose(1, 2)
                   for name in ('q_proj', 'k_proj', 'v_proj'))
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, frames, dim)
        x = x + self.self_attn['out_proj'](h)
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class FusedAVHubert(nn.Module):
    """AVHubertModel.extract_finetune after the branches, in eval mode, output_layer=None.

    Input: cached branch outputs [B,T,D] (01_extract.py). concat [audio, video] -> layer_norm
    -> post_extract_proj -> wav2vec2 TransformerEncoder (pre-LN) -> encoder.layer_norm.
    No padding mask: every clip in a batch must have the same length.
    """
    def __init__(self, dim, layers, ffn_dim, heads, conv_pos, conv_pos_groups):
        super().__init__()
        self.layer_norm = nn.LayerNorm(2 * dim)
        self.post_extract_proj = nn.Linear(2 * dim, dim)
        self.encoder = nn.Module()
        self.encoder.pos_conv = nn.ModuleList([_PositionalConv(dim, conv_pos, conv_pos_groups)])
        self.encoder.layers = nn.ModuleList(_EncoderLayer(dim, ffn_dim, heads) for _ in range(layers))
        self.encoder.layer_norm = nn.LayerNorm(dim)
        self.requires_grad_(False)
        self.eval()

    def _encode(self, audio, visual):
        x = self.post_extract_proj(self.layer_norm(torch.cat([audio, visual], dim=-1)))
        x = x + self.encoder.pos_conv[0](x.transpose(1, 2)).transpose(1, 2)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.encoder.layer_norm(x)

    @torch.no_grad()
    def forward(self, audio, visual):
        """Audio-only and video-only final-layer features; the missing modality is zeros, as upstream."""
        if audio.ndim != 3 or audio.shape != visual.shape:
            raise ValueError('Expected cached audio and visual branch outputs [B,T,D] of equal shape')
        self.eval()
        return self._encode(audio, torch.zeros_like(visual)), self._encode(torch.zeros_like(audio), visual)


# Masked-prediction heads, not used for features.
_PRETRAINING_ONLY = {'mask_emb', 'label_embs_concat', 'final_proj.weight', 'final_proj.bias'}


def load_fused(checkpoint, device='cpu'):
    """Load the fused Transformer of an official pretrained AV-HuBERT checkpoint, strictly.

    Every weight outside the two branches and the pretraining heads must match this module.
    """
    checkpoint = Path(checkpoint)
    state = _read_checkpoint(checkpoint)
    cfg = state['cfg']['model']
    # Defaults are those of upstream AVHubertConfig.
    if (cfg.get('_name') != 'av_hubert' or cfg.get('modality_fuse', 'concat') != 'concat'
            or not cfg.get('layer_norm_first', False) or cfg.get('activation_fn', 'gelu') != 'gelu'):
        raise ValueError('Fused loader supports AV-HuBERT with concat fusion, layer_norm_first and GELU only')
    fused = FusedAVHubert(dim=cfg.get('encoder_embed_dim', 768), layers=cfg.get('encoder_layers', 12),
                          ffn_dim=cfg.get('encoder_ffn_embed_dim', 3072),
                          heads=cfg.get('encoder_attention_heads', 12), conv_pos=cfg.get('conv_pos', 128),
                          conv_pos_groups=cfg.get('conv_pos_groups', 16))
    weights = {key: value for key, value in state['model'].items()
               if not key.startswith('feature_extractor_') and key not in _PRETRAINING_ONLY}
    fused.load_state_dict(weights, strict=True)
    provenance = dict(checkpoint_sha256=sha256(checkpoint), mode='base_fused_v1',
                      loaded_parameters=sum(p.numel() for p in fused.parameters()))
    return fused.to(device), provenance
