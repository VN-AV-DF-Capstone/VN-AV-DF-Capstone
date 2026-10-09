"""Edit before Run; use a new RUN_ID when the plan or generator changes. 02 resumes the same RUN_ID."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
DATASET = 'dataset_v1'
RUN_ID = 'wav2lip_multi_001'  # V1 + part1 speakers, all three splits.
MANIFEST = ROOT / 'data/manifests' / DATASET / 'real_splits_multi_001.csv'  # Written by src/data/07_add_part1.py.
PLAN_DIR = ROOT / 'data/manifests' / DATASET / 'generators' / RUN_ID
MEDIA_DIR = ROOT / 'data/generated' / DATASET / RUN_ID
UPSTREAM = ROOT / 'external/Wav2Lip'
CHECKPOINT = ROOT / 'weights/wav2lip/wav2lipGAN.pth'
GENERATOR_PYTHON = Path(sys.executable)  # Can point to a separate environment.
MAX_PER_SPLIT = 600  # Source clips per split; None = all. Val/test currently have fewer than 600.
NUM_FRAMES = 50  # Two seconds at the normalized 25 fps, matching feature extraction.
DEVICE = 'cuda'
