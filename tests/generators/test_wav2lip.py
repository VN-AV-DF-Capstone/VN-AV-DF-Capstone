"""Protocol and media plumbing checks; fixture output is NOT a real lip-sync fake."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from src.data.preparation.manifest_io import read_rows, write_rows
from src.generators.preparation.wav2lip import (check_media, finalize_existing, generate,
                                               prepare, select_plan, validate_plan)
from src.generators.preparation.worker import load_generator


class TinyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, audio, face):
        return face[:, :3] * self.scale + audio.mean() * 0


def real(cid, source, split='train', **extra):
    return dict(clip_id=cid, source_video=source, speaker_id='host', split=split,
                group_id=source, split_protocol='source_disjoint_single_speaker',
                decision='keep', label='0', duration='2.4', file_path=cid + '.mp4', **extra)


def timed(cid, speaker, source, split, start):
    return dict(clip_id=cid, source_video=source, speaker_id=speaker, split=split, group_id=speaker,
                split_protocol='speaker_source_disjoint', decision='keep', label='0', duration='5',
                start_time=str(start), end_time=str(start + 5), file_path=cid + '.mp4')


def write_sources(root, rows):
    from src.features.media import run_media_command
    for i, row in enumerate(rows):
        run_media_command(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                           'testsrc2=s=96x96:r=25:d=2.4', '-f', 'lavfi', '-i',
                           f'sine=frequency={440+i*200}:sample_rate=16000:duration=2.4',
                           '-c:v', 'libx264', '-c:a', 'aac', '-shortest', root/row['file_path']])


def copy_frames(config, work, log_path):
    from src.features.media import run_media_command
    log_path.write_text('Fixture copies frames; NOT Wav2Lip inference.', encoding='utf-8')
    run_media_command(['ffmpeg', '-v', 'error', '-i', work/'face.avi', '-i', work/'audio.wav',
                       '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'ffv1',
                       '-c:a', 'pcm_s16le', work/'generated.mkv'])


class Wav2LipTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_finalize_complete_pairs_after_interruption(self):
        from src.features.media import run_media_command
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = SimpleNamespace(MANIFEST=root/'real.csv', PLAN_DIR=root/'plan',
                                       MEDIA_DIR=root/'generated', MAX_PER_SPLIT=2, NUM_FRAMES=50,
                                       RUN_ID='run1', DEVICE='cpu', UPSTREAM=root/'upstream',
                                       CHECKPOINT=root/'fixture.pth')
            rows = [real('a', 'v1'), real('b', 'v2')]
            for i, row in enumerate(rows):
                source = root/row['file_path']
                run_media_command(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                                   'testsrc2=s=96x96:r=25:d=2.4', '-f', 'lavfi', '-i',
                                   f'sine=frequency={440+i*200}:sample_rate=16000:duration=2.4',
                                   '-c:v', 'libx264', '-c:a', 'aac', '-shortest', source])
            write_rows(settings.MANIFEST, rows)
            prepare(settings)
            plan = read_rows(settings.PLAN_DIR/'plan.csv')
            settings.MEDIA_DIR.mkdir()
            first = plan[0]
            for suffix in ('real', 'fake'):
                name = first['clip_id'].removesuffix('__fake') + f'__{suffix}.mp4'
                run_media_command(['ffmpeg', '-v', 'error', '-i', root/'a.mp4', '-t', '2',
                                   '-c:v', 'libx264', '-c:a', 'aac', settings.MEDIA_DIR/name])
            finalize_existing(settings)
            candidates = read_rows(settings.PLAN_DIR/'candidates.csv')
            self.assertEqual(len(candidates), 2)
            summary = json.loads((settings.PLAN_DIR/'summary.json').read_text())
            self.assertEqual(summary['successful_pairs'], 1)
            self.assertEqual(summary['unattempted'], 1)
            self.assertTrue(summary['partial'])
            with self.assertRaises(FileExistsError):
                finalize_existing(settings)

    def test_generator_accepts_torchscript_and_weights_only_state_dict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pth'
            model = TinyGenerator().eval()
            torch.jit.trace(model, (torch.zeros(1, 1, 80, 16), torch.zeros(1, 6, 96, 96))).save(str(path))
            loaded, kind = load_generator(path, TinyGenerator, 'cpu')
            self.assertEqual(kind, 'torchscript')
            self.assertFalse(loaded.training)
            torch.save({'state_dict': {'module.scale': model.scale.detach()}}, path)
            loaded, kind = load_generator(path, TinyGenerator, 'cpu')
            self.assertEqual(kind, 'state_dict')
            self.assertFalse(loaded.training)
            torch.save({'wrong_key': {}}, path)
            with self.assertRaisesRegex(ValueError, 'state_dict'):
                load_generator(path, TinyGenerator, 'cpu')
            torch.save({'state_dict': {'scale': torch.tensor(float('nan'))}}, path)
            with self.assertRaisesRegex(ValueError, 'forward check'):
                load_generator(path, TinyGenerator, 'cpu')

    def test_single_request_can_use_donor_outside_selected_requests(self):
        rows = [real('a', 'v1'), real('b', 'v2')]
        plan = select_plan(rows, 1, 50, 'one_clip')
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['audio_source_clip_id'], 'b')
        validate_plan(plan, rows)

    def test_failed_generation_is_reported_without_publishing_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = SimpleNamespace(MANIFEST=root/'real.csv', PLAN_DIR=root/'plan',
                                       MEDIA_DIR=root/'generated', MAX_PER_SPLIT=2, NUM_FRAMES=50,
                                       RUN_ID='run1', DEVICE='cpu', UPSTREAM=root/'upstream',
                                       CHECKPOINT=root/'fixture.pth')
            sfd = settings.UPSTREAM/'face_detection/detection/sfd/s3fd.pth'
            sfd.parent.mkdir(parents=True)
            sfd.write_bytes(b'fixture')
            settings.CHECKPOINT.write_bytes(b'fixture')
            rows = [real('a', 'v1'), real('b', 'v2')]
            for row in rows:
                (root/row['file_path']).write_bytes(b'fixture')
            write_rows(settings.MANIFEST, rows)
            prepare(settings)
            with patch('src.generators.preparation.wav2lip.check_setup'), \
                    patch('src.generators.preparation.wav2lip.generate_one', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'Some clips failed'):
                    generate(settings)
            self.assertFalse((settings.PLAN_DIR/'candidates.csv').exists())
            self.assertEqual(len(read_rows(settings.PLAN_DIR/'failures.csv')), 2)
            summary = json.loads((settings.PLAN_DIR/'summary.json').read_text())
            self.assertEqual(summary['status'], 'incomplete')
            self.assertEqual(summary['successful_pairs'], 0)

    def test_val_and_test_get_same_split_donors_from_non_overlapping_clips(self):
        rows = [timed('t1', 'a', 'va', 'train', 0), timed('t2', 'a', 'va', 'train', 4),
                timed('t3', 'a', 'va', 'train', 8), timed('t4', 'a', 'vb', 'train', 0),
                timed('v1', 'b', 'vc', 'val', 0), timed('v2', 'b', 'vc', 'val', 6),
                timed('x1', 'c', 'vd', 'test', 0), timed('x2', 'c', 'vd', 'test', 3)]
        plan = select_plan(rows, None, 50, 'run1')
        validate_plan(plan, rows)
        donors = {item['source_clip']: item['audio_source_clip_id'] for item in plan}
        self.assertEqual([item['split'] for item in plan][:2], ['val', 'val'])  # Evaluation first.
        self.assertEqual(donors['v1'], 'v2')
        self.assertNotIn('x1', donors)  # Its only same-speaker test clip overlaps in time.
        self.assertEqual(donors['t1'], 't3')  # Same video preferred; t2 overlaps t1.
        self.assertEqual(donors['t2'], 't4')  # t1 and t3 overlap t2, so another video.
        for donor in ('t2', 'v1'):
            with self.subTest(donor=donor), self.assertRaises(ValueError):
                validate_plan([dict(plan[-1], source_clip='t1', audio_source_clip_id=donor)], rows)
        self.assertEqual(len(select_plan(rows, 1, 50, 'run1')), 2)  # At most one per split.

    def test_plan_rotates_sources_deterministically(self):
        rows = [real('a', 'v1'), real('b', 'v1'), real('c', 'v2'),
                real('d', 'v3', 'val'), real('e', 'v4', 'test')]
        plan = select_plan(rows, 2, 50, 'run1')
        self.assertEqual([r['source_clip'] for r in plan], ['a', 'c'])
        self.assertEqual(plan, select_plan(list(reversed(rows)), 2, 50, 'run1'))
        validate_plan(plan, rows)
        self.assertEqual(plan[0]['audio_source_clip_id'], 'c')
        with self.assertRaises(ValueError):
            select_plan(rows, 2, 50, '../escape')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_rerun_retries_failures_and_keeps_complete_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = SimpleNamespace(MANIFEST=root/'real.csv', PLAN_DIR=root/'plan',
                                       MEDIA_DIR=root/'generated', MAX_PER_SPLIT=2, NUM_FRAMES=50,
                                       RUN_ID='run1', DEVICE='cpu', UPSTREAM=root/'upstream',
                                       CHECKPOINT=root/'fixture.pth')
            sfd = settings.UPSTREAM/'face_detection/detection/sfd/s3fd.pth'
            sfd.parent.mkdir(parents=True)
            sfd.write_bytes(b'fixture-not-weights')
            settings.CHECKPOINT.write_bytes(b'fixture-not-weights')
            rows = [real('a', 'v1'), real('b', 'v2')]
            write_sources(root, rows)
            write_rows(settings.MANIFEST, rows)
            prepare(settings)
            calls = []

            def fail_second(config, work, log_path):
                calls.append(log_path.name)
                if len(calls) == 2:
                    raise RuntimeError('fixture failure')
                copy_frames(config, work, log_path)

            with patch('src.generators.preparation.wav2lip.check_setup'), \
                    patch('src.generators.preparation.wav2lip.run_worker', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'run again'):
                    generate(settings)
                self.assertEqual(len(read_rows(settings.PLAN_DIR/'candidates.csv')), 2)
                generate(settings)
                self.assertEqual(len(calls), 3)  # Only the failed pair was generated again.
                with self.assertRaisesRegex(FileExistsError, 'complete'):
                    generate(settings)
            self.assertEqual(len(read_rows(settings.PLAN_DIR/'candidates.csv')), 4)
            self.assertFalse((settings.PLAN_DIR/'failures.csv').exists())
            self.assertEqual(json.loads((settings.PLAN_DIR/'summary.json').read_text())['status'],
                             'awaiting_visual_review')

    def test_cross_split_audio_and_bad_windows_are_rejected(self):
        rows = [real('a', 'v1'), real('b', 'v2'), real('c', 'v3', 'test')]
        plan = select_plan(rows, 2, 50, 'run1')
        for changed in ({'audio_source_clip_id': 'c'}, {'audio_source_clip_id': 'a'},
                        {'start_frame': -1}, {'num_frames': 500}, {'start_frame': 25}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_plan([dict(plan[0], **changed)], rows)
        with self.assertRaises(ValueError):
            select_plan([real('a', 'v1'), real('b', 'v1')], 2, 50, 'run1')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_generation_contract_with_stub_model_and_real_media_commands(self):
        from src.features.media import run_media_command
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = SimpleNamespace(MANIFEST=root/'real.csv', PLAN_DIR=root/'plan',
                                       MEDIA_DIR=root/'generated', MAX_PER_SPLIT=2, NUM_FRAMES=50,
                                       RUN_ID='run1', DEVICE='cpu', UPSTREAM=root/'upstream',
                                       CHECKPOINT=root/'fixture.pth')
            sfd = settings.UPSTREAM/'face_detection/detection/sfd/s3fd.pth'
            sfd.parent.mkdir(parents=True)
            sfd.write_bytes(b'fixture-not-weights')
            settings.CHECKPOINT.write_bytes(b'fixture-not-weights')
            rows = [real('a', 'v1'), real('b', 'v2')]
            for i, row in enumerate(rows):
                source = root/row['file_path']
                run_media_command(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                                   'testsrc2=s=96x96:r=25:d=2.4', '-f', 'lavfi', '-i',
                                   f'sine=frequency={440+i*200}:sample_rate=16000:duration=2.4',
                                   '-c:v', 'libx264', '-c:a', 'aac', '-shortest', source])
            write_rows(settings.MANIFEST, rows)
            prepare(settings)

            def stub_model(config, work, log_path):
                log_path.write_text('Fixture copies frames; NOT Wav2Lip inference.', encoding='utf-8')
                run_media_command(['ffmpeg', '-v', 'error', '-i', work/'face.avi', '-i', work/'audio.wav',
                                   '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'ffv1',
                                   '-c:a', 'pcm_s16le', work/'generated.mkv'])

            with patch('src.generators.preparation.wav2lip.check_setup'), \
                    patch('src.generators.preparation.wav2lip.run_worker', side_effect=stub_model):
                generate(settings)
                with self.assertRaises(FileExistsError):
                    generate(settings)
            candidates = read_rows(settings.PLAN_DIR/'candidates.csv')
            self.assertEqual(len(candidates), 4)
            self.assertEqual({r['label'] for r in candidates}, {'0', '1'})
            for row in candidates:
                check_media(row['file_path'], 50)
                self.assertEqual(row['split'], 'train')
                self.assertEqual(row['review_status'], 'pending')
            self.assertEqual(candidates[1]['audio_source_clip_id'], 'b')
            self.assertEqual(candidates[0]['audio_source_clip_id'], 'a')
            summary = json.loads((settings.PLAN_DIR/'summary.json').read_text())
            self.assertEqual(summary['successful_pairs'], 2)
            with settings.MANIFEST.open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                generate(settings)


if __name__ == '__main__':
    unittest.main()
