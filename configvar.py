from pathlib import Path

TCSPC_TIME_OFFSET_NS = 15.8
LASER_PERIOD_NS = 50
NUM_PULSES = 4
STEP_NM = 1
LIFETIME_WIN_BEG_NS = 0
LIFETIME_WIN_END_NS = 5
PSF_DIR_BASE = Path('testdata') / 'psf'
DATA_DIR_BASE = Path('testdata') / 'clock'