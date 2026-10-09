"""STEP 06: AFTER Wav2Lip generation: compress every real/fake master at two CRFs.

Also writes media_pairs.csv (two CRF views per master) for src/features/01_extract.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation._run import run
from src.generators.preparation import settings as generator

CRFS = '23,32'  # Two training/consistency views. Keep a stronger level for the final held-out test.
MASTERS = generator.PLAN_DIR / 'candidates.csv'  # Real controls and fakes from the generator run.
COMPRESSION_RUN = f"{generator.RUN_ID}_crf{CRFS.replace(',', '_')}"

if __name__ == '__main__':
    run('src.data.preparation.compress', ['--input_csv', MASTERS,
                             '--out_dir', generator.ROOT / 'data/compressed' / generator.DATASET / COMPRESSION_RUN,
                             '--crfs', CRFS])
