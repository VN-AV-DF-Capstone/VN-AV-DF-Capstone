"""Fused AV-HuBERT Transformer: layer math against PyTorch references, strict loading and the
cache conversion CLI, all with synthetic weights (not the real checkpoint)."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

from src.features.avhubert import FusedAVHubert, _EncoderLayer, _PositionalConv, load_fused
from src.training.dataset import PairedFeatureDataset

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = dict(dim=8, layers=2, ffn_dim=16, heads=2, conv_pos=4, conv_pos_groups=2)
# Fields named as in upstream AVHubertConfig.
CFG = dict(model=dict(_name='av_hubert', layer_norm_first=True, encoder_embed_dim=8, encoder_layers=2,
                      encoder_ffn_embed_dim=16, encoder_attention_heads=2, conv_pos=4, conv_pos_groups=2),
           task={})


def synthetic_state():
    """Random fused weights plus branch and pretraining-head keys the fused loader must skip."""
    torch.manual_seed(0)
    state = {key: torch.randn(value.shape) * .5 for key, value in FusedAVHubert(**SETTINGS).state_dict().items()}
    return state | {'mask_emb': torch.randn(8), 'label_embs_concat': torch.randn(5, 4),
                    'final_proj.weight': torch.randn(4, 8), 'final_proj.bias': torch.randn(4),
                    'feature_extractor_audio.proj.weight': torch.randn(8, 104)}


class FusedLayersTest(unittest.TestCase):
    def test_positional_conv_matches_weight_norm_conv(self):
        torch.manual_seed(0)
        reference = nn.utils.weight_norm(nn.Conv1d(8, 8, kernel_size=4, padding=2, groups=2), dim=2)
        with torch.no_grad():
            reference.weight_g.mul_(torch.rand_like(reference.weight_g) + .5)
        ours = _PositionalConv(8, 4, 2)
        ours.load_state_dict(dict(weight_g=reference.weight_g, weight_v=reference.weight_v, bias=reference.bias))
        x = torch.randn(3, 8, 7)
        # Even kernel: the conv returns one extra frame, which fairseq's SamePad drops at the end.
        torch.testing.assert_close(ours(x), nn.functional.gelu(reference(x)[:, :, :-1]))

    def test_encoder_layer_matches_pytorch_pre_norm_layer(self):
        torch.manual_seed(0)
        reference = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, activation='gelu', batch_first=True,
                                               norm_first=True).eval()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(0, .5)
        attention = reference.self_attn
        weights, biases = attention.in_proj_weight.chunk(3), attention.in_proj_bias.chunk(3)
        state = {f'self_attn.{name}_proj.{kind}': value for name, weight, bias in zip('qkv', weights, biases)
                 for kind, value in (('weight', weight), ('bias', bias))}
        state |= {'self_attn.out_proj.weight': attention.out_proj.weight,
                  'self_attn.out_proj.bias': attention.out_proj.bias}
        for ours, theirs in (('self_attn_layer_norm', 'norm1'), ('final_layer_norm', 'norm2'),
                             ('fc1', 'linear1'), ('fc2', 'linear2')):
            for kind in ('weight', 'bias'):
                state[f'{ours}.{kind}'] = getattr(getattr(reference, theirs), kind)
        layer = _EncoderLayer(8, 16, 2)
        layer.load_state_dict(state)
        x = torch.randn(3, 7, 8)
        # With gradients enabled PyTorch takes its plain (not fused fast-path) implementation.
        torch.testing.assert_close(layer(x).detach(), reference(x).detach(), rtol=1e-5, atol=1e-5)


class LoadFusedTest(unittest.TestCase):
    def test_strict_loading_and_zeroed_modality(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pt'
            torch.save(dict(cfg=CFG, model=synthetic_state()), path)
            fused, provenance = load_fused(path)
            self.assertEqual(provenance['mode'], 'base_fused_v1')
            self.assertFalse(any(parameter.requires_grad for parameter in fused.parameters()))
            audio, visual = torch.randn(1, 6, 8), torch.randn(1, 6, 8)
            audio_only, video_only = fused(audio, visual)
            self.assertEqual((audio_only.shape, video_only.shape), ((1, 6, 8), (1, 6, 8)))
            other_audio_only, other_video_only = fused(audio, torch.randn(1, 6, 8))
            torch.testing.assert_close(audio_only, other_audio_only)  # Video input is zeroed.
            self.assertFalse(torch.allclose(video_only, other_video_only))

            state = synthetic_state()
            del state['encoder.layers.1.fc2.bias']
            torch.save(dict(cfg=CFG, model=state), path)
            with self.assertRaises(RuntimeError):  # Missing weights are never replaced by random ones.
                load_fused(path)
            cfg = copy.deepcopy(CFG)
            cfg['model']['layer_norm_first'] = False
            torch.save(dict(cfg=cfg, model=synthetic_state()), path)
            with self.assertRaises(ValueError):
                load_fused(path)


class FuseCliTest(unittest.TestCase):
    def test_converts_cache_and_keeps_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, source, out = root / 'model.pt', root / 'source', root / 'fused'
            torch.save(dict(cfg=CFG, model=synthetic_state()), checkpoint)
            source.mkdir()
            rows = []
            for i, (label, split) in enumerate([(0, 'train'), (1, 'train'), (0, 'val')]):
                for key, view in (('feature_path', 'crf23'), ('paired_feature_path', 'crf32')):
                    torch.save(dict(sample_id=f's{i}', label=label, split=split, view_id=view,
                                    extractor_id='pre_fusion', timeline_id='25fps:0:12', source_sha256='x',
                                    quality='{}', audio=torch.randn(12, 8), visual=torch.randn(12, 8)),
                               source / f'{i}_{key}.pt')
                rows.append(dict(sample_id=f's{i}', label=label, split=split, feature_path=f'{i}_feature_path.pt',
                                 paired_feature_path=f'{i}_paired_feature_path.pt'))
            with (source / 'pairs.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            command = [sys.executable, str(ROOT / 'src/features/02_fuse.py'), '--checkpoint', str(checkpoint),
                       '--source', str(source), '--out', str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((out / 'pairs.csv').read_bytes(), (source / 'pairs.csv').read_bytes())
            fused, _ = load_fused(checkpoint)
            before = torch.load(source / '0_feature_path.pt', weights_only=True)
            after = torch.load(out / '0_feature_path.pt', weights_only=True)
            expected = fused(before['audio'][None], before['visual'][None])
            torch.testing.assert_close(after['audio'], expected[0][0])
            torch.testing.assert_close(after['visual'], expected[1][0])
            self.assertEqual(after['source_extractor_id'], 'pre_fusion')
            self.assertNotEqual(after['extractor_id'], 'pre_fusion')
            for key in ('sample_id', 'label', 'split', 'view_id', 'timeline_id', 'source_sha256'):
                self.assertEqual(after[key], before[key])
            train = PairedFeatureDataset(out / 'pairs.csv', 'train')
            self.assertEqual((len(train), train[1]['audio'].shape), (2, (12, 8)))
            profile = json.loads((out / 'run.json').read_text())['offset_profile']
            self.assertEqual(len(profile['real']), len(profile['offsets']))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)  # Never overwrites.


if __name__ == '__main__':
    unittest.main()
