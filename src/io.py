"""Cache loaders for raw data and intermediate pickles.

These wrap the file-reading and caching logic that was scattered across notebooks
02–04. The cache files live in `cache/` and are regeneratable from `data/raw/`.
"""
import numpy as np
import pandas as pd

from .utils import CACHE_DIR, TRAIN_DIR, TEST_DIR

TRAIN_HW_CACHE  = CACHE_DIR / 'train_wells.pkl'
TRAIN_TW_CACHE  = CACHE_DIR / 'train_typewells.pkl'
TRAIN_FEAT_CACHE = CACHE_DIR / 'train_features.pkl'
TEST_FEAT_CACHE  = CACHE_DIR / 'test_features.pkl'


def load_train_horizontal(force_reload: bool = False) -> pd.DataFrame:
    """All train horizontal-well CSVs concatenated, with `well` and `row_idx` columns."""
    if TRAIN_HW_CACHE.exists() and not force_reload:
        df = pd.read_pickle(TRAIN_HW_CACHE)
        print(f'Loaded HW cache: {df.shape}')
        return df
    print('Reading train horizontal-well CSVs...')
    frames = []
    for f in sorted(TRAIN_DIR.glob('*__horizontal_well.csv')):
        well = f.name.replace('__horizontal_well.csv', '')
        d = pd.read_csv(f)
        d['well'] = well
        d['row_idx'] = np.arange(len(d), dtype=np.int32)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(TRAIN_HW_CACHE)
    print(f'Cached: {df.shape} -> {TRAIN_HW_CACHE.name}')
    return df


def load_train_typewells(force_reload: bool = False) -> dict:
    """Dict mapping well_id -> typewell DataFrame."""
    if TRAIN_TW_CACHE.exists() and not force_reload:
        d = pd.read_pickle(TRAIN_TW_CACHE)
        print(f'Loaded TW cache: {len(d)} typewells')
        return d
    print('Reading train typewell CSVs...')
    out = {}
    for f in sorted(TRAIN_DIR.glob('*__typewell.csv')):
        well = f.name.replace('__typewell.csv', '')
        out[well] = pd.read_csv(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, TRAIN_TW_CACHE)
    print(f'Cached {len(out)} typewells -> {TRAIN_TW_CACHE.name}')
    return out


def load_test_horizontal() -> dict:
    """Dict mapping well_id -> test horizontal DataFrame (with `well`, `row_idx`)."""
    out = {}
    for f in sorted(TEST_DIR.glob('*__horizontal_well.csv')):
        well = f.name.replace('__horizontal_well.csv', '')
        d = pd.read_csv(f)
        d['well'] = well
        d['row_idx'] = np.arange(len(d), dtype=np.int32)
        out[well] = d
    return out


def load_test_typewells() -> dict:
    out = {}
    for f in sorted(TEST_DIR.glob('*__typewell.csv')):
        well = f.name.replace('__typewell.csv', '')
        out[well] = pd.read_csv(f)
    return out
