"""STEP 07, once BEFORE generation: add Person 2's reviewed clips as unseen speakers.

Reads V1's locked real_splits.csv, never rewrites it. V1 goes to train; part1 speakers
are split into train/val/test. The generator settings read OUTPUT.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation._run import run
from src.data.preparation.settings import ROOT

MANIFESTS = ROOT / 'data/manifests/dataset_v1'  # Fixed: independent of DATASET_VERSION.
PART1_REVIEW = ROOT / 'vn-av-df-data-part1/vn-av-df-data-part1/review.csv'  # Edit if the folder moves.
PART1_RATIOS = '0.4,0.3,0.3'  # Part1 groups only; V1 already adds ~1.1k train clips.
OUTPUT = MANIFESTS / 'real_splits_multi_001.csv'

if __name__ == '__main__':
    run('src.data.preparation.add_part1', ['--v1', MANIFESTS / 'real_splits.csv', '--part1', PART1_REVIEW,
                                           '--ratios', PART1_RATIOS, '--out', OUTPUT])
