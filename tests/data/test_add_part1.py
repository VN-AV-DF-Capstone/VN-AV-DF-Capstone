"""Part1 import: V1 stays in train, part1 speakers are split without leakage."""
import csv
from pathlib import Path
import tempfile
import unittest

from src.data.preparation.add_part1 import merge_splits, part1_rows
from src.data.preparation.build_splits import verify_splits


def review(cid, speaker, source, channel, decision='keep', start='0', end='6'):
    return dict(clip_id=cid, source_id='yt_' + source, url='https://example.invalid/' + source,
                speaker_id=speaker, channel=channel, source_start_s=start, source_end_s=end,
                duration_s='6', face_ratio='1.0', file_path=f'clips/{cid}.mp4', sha256='x',
                decision=decision, sync_status='reviewed_match' if decision == 'keep' else 'unverified')


def v1(cid, source, split):
    return dict(clip_id=cid, source_video=source, start_time='0', end_time='5', duration='5',
                file_path=f'E:/v1/{cid}.mp4', decision='keep', speaker_id='spk_001', split=split,
                group_id='old', label='0', split_protocol='source_disjoint_single_speaker')


class AddPart1Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_review(self, rows):
        (self.root / 'clips').mkdir(exist_ok=True)
        for row in rows:
            (self.root / row['file_path']).write_bytes(b'fixture')
        path = self.root / 'review.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_keep_only_with_absolute_paths_and_channel_link(self):
        path = self.write_review([review('c1', 'sp1', 'A', 'Chan'), review('c2', 'sp2', 'B', 'Chan'),
                                  review('c3', 'sp3', 'C', 'Other', decision='reject')])
        rows = part1_rows(path)
        self.assertEqual([r['clip_id'] for r in rows], ['c1', 'c2'])
        self.assertEqual(rows[0]['source_video'], 'A')
        self.assertEqual(rows[0]['shared_group'], rows[1]['shared_group'])
        self.assertTrue(Path(rows[0]['file_path']).is_absolute())
        (self.root / 'clips/c1.mp4').unlink()
        with self.assertRaises(FileNotFoundError):
            part1_rows(path)

    def test_v1_goes_to_train_and_channel_speakers_share_a_split(self):
        part1 = [review(f'{s}_{i}', s, s.upper(), 'Shared' if s in ('p', 'q') else s, start=str(i * 6),
                        end=str(i * 6 + 6)) for s in 'pqrstu' for i in range(4)]
        rows = merge_splits([v1('v_a', 'va', 'test'), v1('v_b', 'vb', 'val')],
                            part1_rows(self.write_review(part1)), (0.4, 0.3, 0.3), 42)
        verify_splits(rows)
        split = {r['clip_id']: r['split'] for r in rows}
        self.assertEqual({split['v_a'], split['v_b']}, {'train'})
        self.assertEqual(split['p_0'], split['q_0'])
        self.assertEqual({r['split'] for r in rows}, {'train', 'val', 'test'})
        self.assertEqual({r['split_protocol'] for r in rows}, {'speaker_source_disjoint'})
        self.assertEqual({r['origin'] for r in rows if r['clip_id'].startswith('v_')}, {'v1'})

    def test_video_shared_by_v1_and_part1_is_leakage(self):
        part1 = [review(f'{s}_0', s, s.upper(), s) for s in 'pqr'] + [review('dup', 'z', 'va', 'z')]
        with self.assertRaisesRegex(ValueError, 'Leakage'):
            merge_splits([v1('v_a', 'va', 'test')], part1_rows(self.write_review(part1)), (0, .5, .5), 42)


if __name__ == '__main__':
    unittest.main()
