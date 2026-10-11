"""STEP 03: Run after a fully successful download batch. Settings below control cutting."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation.settings import ROOT, RAW, REAL, CUT_RUN, SOURCE_GROUP

NUM_WORKERS = 1
USE_GPU = False
PREVIEW_HEIGHT = 0  # Set to 480 to write review previews into .tmp instead of REAL.


def main():
    if '--help' in sys.argv:
        print(__doc__)
        print('Edit NUM_WORKERS / USE_GPU / PREVIEW_HEIGHT here and dataset paths in preparation/settings.py.')
        return
    from src.data.preparation.cut_clips import CutConfig, run_batch
    from src.data.preparation.manifest_io import read_rows
    os.chdir(ROOT)
    input_csv = RAW / 'download_results.csv'
    rows = read_rows(input_csv)
    if any(row.get('status') != 'downloaded' for row in rows):
        raise ValueError('Download batch has failed rows. Resolve them before cutting; none will be silently skipped.')
    output_root = ROOT / '.tmp' / 'clip_previews' if PREVIEW_HEIGHT else REAL
    config = CutConfig(tier=SOURCE_GROUP, dataset_dir=str(RAW), input_csv=str(input_csv),
                       output_root=str(output_root), run_id=CUT_RUN, expected_input_count=len(rows),
                       num_workers=NUM_WORKERS, use_hwaccel_decode=USE_GPU, use_nvenc=USE_GPU,
                       preview_height=PREVIEW_HEIGHT)
    run_batch(config)


if __name__ == '__main__':
    main()
