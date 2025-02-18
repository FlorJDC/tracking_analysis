from pathlib import Path

TCSPC_TIME_OFFSET = 15.8
LASER_PERIOD = 50
K = 4
STEP_NM = 1
LIFETIME_WIN_BEG = 0
LIFETIME_WIN_END = 5
PSF_DIR_BASE = Path('testdata') / 'psf'
DATA_DIR_BASE = Path('testdata') / 'clock'