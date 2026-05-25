"""Feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_well_zscore(df: pd.DataFrame, col: str) -> np.ndarray:
    """Z-score `col` within each well_id group."""
    g = df.groupby("well_id")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std")).to_numpy()


def _gr_interpolate(gr: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN runs in GR. Edge NaNs filled with nearest known."""
    gr = gr.astype(float).copy()
    n = len(gr)
    if np.isnan(gr).all():
        return gr  # leave as all-NaN; caller decides
    idx = np.arange(n)
    known = ~np.isnan(gr)
    gr[~known] = np.interp(idx[~known], idx[known], gr[known])
    return gr


def build_features_for_well(well: pd.DataFrame, anchor_idx: int, k_roll: int = 50) -> pd.DataFrame:
    """Build features for every row of one well, anchored at `anchor_idx`.

    Returns a DataFrame with the same length as `well`, columns:
        dmd, dz, anchor_tvt, anchor_z, gr_roll_mean_k, gr_roll_std_k, row_idx, MD
    Plus `target_tvt` (true TVT, NaN where unknown) and `anchor_tvt_value` for
    residual computation downstream.
    """
    md = well["MD"].to_numpy(dtype=float)
    z = well["Z"].to_numpy(dtype=float)
    gr = _gr_interpolate(well["GR"].to_numpy(dtype=float))
    tvt = well["TVT"].to_numpy(dtype=float)  # NaN at inference; known at training
    tvt_input = well["TVT_input"].to_numpy(dtype=float)

    anchor_md = md[anchor_idx]
    anchor_z = z[anchor_idx]
    anchor_tvt = tvt_input[anchor_idx]  # always known by construction
    if np.isnan(anchor_tvt):
        raise ValueError(f"anchor_idx={anchor_idx} points at a row with NaN TVT_input.")

    # Causal rolling stats: window ending at current row, length k_roll.
    s = pd.Series(gr)
    gr_mean = s.rolling(k_roll, min_periods=1).mean().to_numpy()
    gr_std = s.rolling(k_roll, min_periods=2).std().to_numpy()  # NaN at row 0

    return pd.DataFrame(
        {
            "row_idx": np.arange(len(well)),
            "MD": md,
            "dmd": md - anchor_md,
            "dz": z - anchor_z,
            "anchor_tvt": anchor_tvt,
            "anchor_z": anchor_z,
            "gr_roll_mean_k": gr_mean,
            "gr_roll_std_k": gr_std,
            "target_tvt": tvt,
            "tvt_input": tvt_input,
            "anchor_tvt_value": anchor_tvt,
        }
    )


def pick_training_anchor(well: pd.DataFrame, frac: float = 0.4) -> int:
    """Pick a synthetic anchor inside the known segment for training.

    Anchor is at frac * known_segment_length. Rows after the anchor (still inside
    the known segment) become the "eval-like" training rows.
    """
    known = ~np.isnan(well["TVT_input"].to_numpy(dtype=float))
    if not known.any():
        raise ValueError("Well has no known TVT_input rows.")
    known_idx = np.flatnonzero(known)
    last_known = known_idx[-1]
    # Anchor at frac * (last_known + 1) — i.e. inside the known segment.
    return int(np.clip(round(frac * (last_known + 1)), 1, last_known - 1))


def pick_inference_anchor(well: pd.DataFrame) -> int:
    """Inference anchor = last known TVT_input row (the natural anchor)."""
    known = ~np.isnan(well["TVT_input"].to_numpy(dtype=float))
    if not known.any():
        raise ValueError("Well has no known TVT_input rows.")
    return int(np.flatnonzero(known)[-1])
