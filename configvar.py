from pathlib import Path

TCSPC_TIME_OFFSET_NS = 15.8
LASER_PERIOD_NS = 50
NUM_PULSES = 4
STEP_NM = 1
LIFETIME_WIN_BEG_NS = 0
LIFETIME_WIN_END_NS = 5
DIR_BASE = Path('testdata')
PSF_DIR_BASE = Path('testdata') / 'psf'
DATA_DIR_BASE = Path('testdata') / 'measurement'
LOCS_FILE_SUFFIX = "_locs_"
SIGMA_TOL_OUTLIERS = 3.0
COLOR_LIST = ['blue', 'orange', 'gray', 'yellow']