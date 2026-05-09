"""Feature engineering.

Two main functions:
  - `predict_gr_match_v2(g, tw)` — runs the typewell GR-template correlation,
    returns (per-row TVT prediction, per-row cosine similarity). Originally from
    notebook 03, extended in 04 to return similarity for use as a feature.
  - `featurize_well(g, tw, well_id)` — produces one row per eval row of one well
    with the engineered features defined in `FEATURE_COLS`.

Plus batch helpers `featurize_all_train()` and `featurize_all_test()` that read
from `data/raw/`, run the featurizer over every well, and cache to disk.
"""
import numpy as np
import pandas as pd

from .utils import COL_MD, COL_X, COL_Y, COL_Z, COL_GR, COL_TVT_INPUT, COL_TVT
from .io import (
    TRAIN_FEAT_CACHE, TEST_FEAT_CACHE,
    load_train_horizontal, load_train_typewells,
    load_test_horizontal, load_test_typewells,
)


# Canonical feature list. Configs reference these names.
FEATURE_COLS = [
    'md', 'x', 'y', 'z', 'gr',
    'gr_local_mean_50', 'gr_local_std_50',
    'anchor_tvt', 'anchor_z', 'anchor_md', 'anchor_gr',
    'smooth_anchor_tvt_50', 'slope_tvt_md_50',
    'd_md_from_anchor', 'd_z_from_anchor', 'd_gr_from_anchor',
    'gr_match_tvt', 'gr_match_sim', 'gr_match_delta_anchor',
]


def predict_gr_match_v2(g: pd.DataFrame, tw: pd.DataFrame,
                        window: int = 51, search_radius: float = 50.0,
                        smooth: int = 11):
    """For each eval row, find the typewell TVT whose GR window best matches.

    Returns:
        pred: array of length len(g). For eval rows: median-smoothed best-match TVT.
              For known rows: original TVT_input value (NaN propagated).
        sim:  array of length len(g). Per-row cosine similarity (0 for fallback rows).
    """
    n = len(g)
    pred = g[COL_TVT_INPUT].values.astype(float).copy()
    sim  = np.zeros(n, dtype=float)
    eval_mask = np.isnan(pred)
    if not eval_mask.any():
        return pred, sim
    known_idx = np.where(~eval_mask)[0]
    if len(known_idx) == 0:
        return pred, sim
    anchor = float(pred[known_idx[-1]])
    eval_idx = np.where(eval_mask)[0]

    # Lateral GR: interpolate NaN, z-score per well
    lat_gr = pd.Series(g[COL_GR].values).interpolate(limit_direction='both').values.astype(float)
    if not np.isfinite(lat_gr).all():
        pred[eval_mask] = anchor
        return pred, sim
    lat_gr_z = (lat_gr - lat_gr.mean()) / (lat_gr.std() + 1e-9)

    # Typewell GR -> resample to 1.0 TVT grid, z-score
    if tw is None or COL_GR not in tw or COL_TVT not in tw:
        pred[eval_mask] = anchor
        return pred, sim
    tw_clean = (tw.dropna(subset=[COL_TVT, COL_GR])
                  .drop_duplicates(subset=[COL_TVT])
                  .sort_values(COL_TVT))
    if len(tw_clean) < window + 2:
        pred[eval_mask] = anchor
        return pred, sim
    tvt_grid = np.arange(np.floor(tw_clean[COL_TVT].min()),
                         np.ceil(tw_clean[COL_TVT].max()) + 1, 1.0)
    if len(tvt_grid) < window + 2:
        pred[eval_mask] = anchor
        return pred, sim
    gr_grid = np.interp(tvt_grid, tw_clean[COL_TVT].values, tw_clean[COL_GR].values)
    gr_grid_z = (gr_grid - gr_grid.mean()) / (gr_grid.std() + 1e-9)

    half = window // 2
    n_tw = len(gr_grid_z)
    TW_W = np.lib.stride_tricks.sliding_window_view(gr_grid_z, window).copy()
    norms = np.linalg.norm(TW_W, axis=1, keepdims=True); norms[norms < 1e-9] = 1.0
    TW_W /= norms
    tw_centers = tvt_grid[half : n_tw - half]

    cand_mask = np.abs(tw_centers - anchor) <= search_radius
    if not cand_mask.any():
        pred[eval_mask] = anchor
        return pred, sim

    n_lat = len(lat_gr_z)
    LAT_W_all = np.lib.stride_tricks.sliding_window_view(lat_gr_z, window)
    in_bounds = (eval_idx >= half) & (eval_idx <= n_lat - half - 1)
    pred[eval_idx[~in_bounds]] = anchor
    eval_full = eval_idx[in_bounds]
    if len(eval_full) == 0:
        return pred, sim

    LAT_W = LAT_W_all[eval_full - half].copy()
    lat_norms = np.linalg.norm(LAT_W, axis=1, keepdims=True); lat_norms[lat_norms < 1e-9] = 1.0
    LAT_W /= lat_norms

    S = LAT_W @ TW_W.T
    S_masked = S.copy(); S_masked[:, ~cand_mask] = -np.inf
    best = np.argmax(S_masked, axis=1)
    raw = tw_centers[best]
    raw_sim = S[np.arange(len(eval_full)), best]

    if smooth and smooth > 1:
        raw = pd.Series(raw).rolling(smooth, center=True, min_periods=1).median().values

    pred[eval_full] = raw
    sim[eval_full]  = raw_sim
    return pred, sim


def featurize_well(g: pd.DataFrame, tw: pd.DataFrame, well_id: str = None) -> pd.DataFrame:
    """Compute features for the eval rows of one well.

    Returns a DataFrame with columns ['well', 'row_idx', *FEATURE_COLS] and
    'target_tvt' if the well has labels (i.e., the COL_TVT column is present).
    """
    g = g.reset_index(drop=True)
    n = len(g)
    eval_mask = g[COL_TVT_INPUT].isna().values
    if not eval_mask.any():
        return pd.DataFrame(columns=['well', 'row_idx', *FEATURE_COLS])
    known_idx = np.where(~eval_mask)[0]
    if len(known_idx) == 0:
        return pd.DataFrame(columns=['well', 'row_idx', *FEATURE_COLS])

    anchor_idx = known_idx[-1]
    anchor_tvt = float(g[COL_TVT_INPUT].iloc[anchor_idx])
    anchor_z   = float(g[COL_Z].iloc[anchor_idx])
    anchor_md  = float(g[COL_MD].iloc[anchor_idx])
    gr_filled  = pd.Series(g[COL_GR].values).interpolate(limit_direction='both').fillna(0.0).values
    anchor_gr  = float(gr_filled[anchor_idx])

    K = 50
    last_K = known_idx[-K:] if len(known_idx) >= K else known_idx
    smooth_anchor = float(g[COL_TVT_INPUT].iloc[last_K].mean())
    if len(last_K) >= 2:
        slope, _ = np.polyfit(g[COL_MD].values[last_K], g[COL_TVT_INPUT].values[last_K], 1)
    else:
        slope = 0.0

    gr_series = pd.Series(gr_filled)
    gr_local_mean = gr_series.rolling(50, center=True, min_periods=1).mean().values
    gr_local_std  = gr_series.rolling(50, center=True, min_periods=1).std().fillna(0.0).values

    gr_match_pred, gr_match_sim = predict_gr_match_v2(g, tw)

    feats_full = pd.DataFrame({
        'md':                   g[COL_MD].values,
        'x':                    g[COL_X].values,
        'y':                    g[COL_Y].values,
        'z':                    g[COL_Z].values,
        'gr':                   gr_filled,
        'gr_local_mean_50':     gr_local_mean,
        'gr_local_std_50':      gr_local_std,
        'anchor_tvt':           anchor_tvt,
        'anchor_z':             anchor_z,
        'anchor_md':            anchor_md,
        'anchor_gr':            anchor_gr,
        'smooth_anchor_tvt_50': smooth_anchor,
        'slope_tvt_md_50':      float(slope),
        'd_md_from_anchor':     g[COL_MD].values - anchor_md,
        'd_z_from_anchor':      g[COL_Z].values  - anchor_z,
        'd_gr_from_anchor':     gr_filled        - anchor_gr,
        'gr_match_tvt':         gr_match_pred,
        'gr_match_sim':         gr_match_sim,
        'gr_match_delta_anchor': gr_match_pred - anchor_tvt,
    })
    feats_full['well']    = well_id if well_id is not None else (g['well'].iloc[0] if 'well' in g.columns else None)
    feats_full['row_idx'] = np.arange(n, dtype=np.int32)

    out = feats_full.loc[eval_mask].reset_index(drop=True)
    if COL_TVT in g.columns:
        out['target_tvt'] = g[COL_TVT].values[eval_mask]

    cols = ['well', 'row_idx', *FEATURE_COLS]
    if 'target_tvt' in out.columns:
        cols.append('target_tvt')
    return out[cols]


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
    return df


def featurize_all_train(force_reload: bool = False) -> pd.DataFrame:
    """Featurize every train well. Cached as `cache/train_features.pkl`."""
    if TRAIN_FEAT_CACHE.exists() and not force_reload:
        df = pd.read_pickle(TRAIN_FEAT_CACHE)
        print(f'Loaded train-feature cache: {df.shape}')
        return df
    print('Featurizing train wells (this takes ~30-90s)...')
    train_hw = load_train_horizontal()
    train_tw = load_train_typewells()
    rows = []
    for i, (well, g) in enumerate(train_hw.groupby('well', sort=False)):
        if i and i % 100 == 0:
            print(f'  ... {i} wells', flush=True)
        rows.append(featurize_well(g, train_tw.get(well), well))
    out = _downcast_floats(pd.concat(rows, ignore_index=True))
    out.to_pickle(TRAIN_FEAT_CACHE)
    print(f'Cached: {out.shape} -> {TRAIN_FEAT_CACHE.name}')
    return out


def featurize_all_test(force_reload: bool = False) -> pd.DataFrame:
    """Featurize every test well. Cached as `cache/test_features.pkl`."""
    if TEST_FEAT_CACHE.exists() and not force_reload:
        df = pd.read_pickle(TEST_FEAT_CACHE)
        print(f'Loaded test-feature cache: {df.shape}')
        return df
    print('Featurizing test wells...')
    test_hw = load_test_horizontal()
    test_tw = load_test_typewells()
    rows = []
    for well, g in test_hw.items():
        rows.append(featurize_well(g, test_tw.get(well), well))
    out = _downcast_floats(pd.concat(rows, ignore_index=True))
    out.to_pickle(TEST_FEAT_CACHE)
    print(f'Cached: {out.shape} -> {TEST_FEAT_CACHE.name}')
    return out
