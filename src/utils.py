"""Project paths and schema constants.

The single source of truth for where things live and what columns are called.
Anything else in src/ imports from here.
"""
from pathlib import Path

# --- Project structure ---
ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
RAW_DIR = DATA / 'raw'
TRAIN_DIR = RAW_DIR / 'train'
TEST_DIR = RAW_DIR / 'test'
CACHE_DIR = ROOT / 'cache'
SUB_DIR = ROOT / 'submissions'
EXPERIMENTS_DIR = ROOT / 'experiments'
NOTEBOOKS_DIR = ROOT / 'notebooks'
CONFIGS_DIR = ROOT / 'configs'

# --- Schema constants (from notebook 01 schema discovery) ---
COL_MD = 'MD'
COL_X, COL_Y, COL_Z = 'X', 'Y', 'Z'
COL_GR = 'GR'
COL_TVT_INPUT = 'TVT_input'  # masked in eval zone
COL_TVT = 'TVT'              # ground truth (train only)
COL_FORMATION = 'Geology'    # in typewell
COL_WELLNAME = 'wellname'

# Train-only formation surface columns (auxiliary supervision; not yet used)
SURFACE_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']


def ensure_dirs():
    """Create the writable directories (cache, submissions, experiments)."""
    for d in (CACHE_DIR, SUB_DIR, EXPERIMENTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
