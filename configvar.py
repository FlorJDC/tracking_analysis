import numpy as np
from pathlib import Path

TCSPC_TIME_OFFSET_NS = 0 
#antes: 
#TCSPC_TIME_OFFSET_NS = 15.8
LASER_PERIOD_NS = 50
NUM_PULSES = 4
STEP_NM = 1
LIFETIME_WIN_BEG_NS = 0
LIFETIME_WIN_END_NS = 5
DIR_BASE = Path('C:\Data')
PSF_DIR_BASE = Path('C:\Data') / 'psf'
DATA_DIR_BASE = Path('C:\Data') / 'measurement'
LOCS_FILE_SUFFIX = "_locs_"
SIGMA_TOL_OUTLIERS = 3.0
COLOR_LIST = ['blue', 'orange', 'gray', 'yellow']
PULSES_POS_NS = np.array([2.2, 15.0, 27.3, 40.3])  # [ns] 
#antes:
#PULSES_POS_NS = np.array([0.98, 13.8, 26.12, 39.1])  # [ns] 