"""Phase 2 — constant-prediction baselines.

All baselines take a single well's dataframe (with MD, TVT, TVT_input) and
return a full-length TVT prediction array. Scoring happens downstream via
evaluate.masked_rmse on the TVT_input.isna() rows.

Convention: `TVT_input` is the known/visible TVT (NaN on eval rows).
The "anchor" is the last non-NaN TVT_input row in the well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _anchor_idx(tvt_input: pd.Series) -> int:
    """Index of the last known TVT_input row. Raises if a well has zero known rows."""
    known = tvt_input.notna()
    if not known.any():
        raise ValueError("Well has no known TVT_input rows; cannot anchor.")
    return int(known[known].index[-1])


def predict_carry_forward(well: pd.DataFrame) -> np.ndarray:
    """Carry the last known TVT_input forward over the eval tail.

    Known rows return TVT_input verbatim; eval rows return the anchor value.
    """
    tvt_in = well["TVT_input"].to_numpy(dtype=float)
    anchor = _anchor_idx(well["TVT_input"])
    out = tvt_in.copy()
    out[np.isnan(out)] = tvt_in[anchor]
    return out


def predict_smooth_anchor(well: pd.DataFrame, k: int = 50) -> np.ndarray:
    """Predict the mean of the last K known TVT_input values for all eval rows.

    More robust than carry-forward when the last known row is noisy.
    """
    tvt_in = well["TVT_input"].to_numpy(dtype=float)
    known_mask = ~np.isnan(tvt_in)
    if not known_mask.any():
        raise ValueError("Well has no known TVT_input rows; cannot anchor.")
    known_vals = tvt_in[known_mask]
    anchor = float(np.mean(known_vals[-k:]))  # last K (or fewer if known < K)
    out = tvt_in.copy()
    out[np.isnan(out)] = anchor
    return out


def predict_linear_extrap(well: pd.DataFrame, k: int = 20) -> np.ndarray:
    """Fit a line to the last K known (MD, TVT_input) pairs and extrapolate.

    Prior-attempt number was RMSE ~107 — much worse than carry-forward. Including
    for completeness and to confirm we reproduce that gap.
    """
    tvt_in = well["TVT_input"].to_numpy(dtype=float)
    md = well["MD"].to_numpy(dtype=float)
    known_mask = ~np.isnan(tvt_in)
    if not known_mask.any():
        raise ValueError("Well has no known TVT_input rows; cannot anchor.")
    md_known = md[known_mask][-k:]
    tvt_known = tvt_in[known_mask][-k:]
    if len(md_known) < 2:
        # Degenerate; fall back to carry-forward on that well.
        anchor = float(tvt_known[-1])
        out = tvt_in.copy()
        out[np.isnan(out)] = anchor
        return out
    slope, intercept = np.polyfit(md_known, tvt_known, deg=1)
    out = tvt_in.copy()
    eval_idx = np.isnan(out)
    out[eval_idx] = slope * md[eval_idx] + intercept
    return out


def predict_causal_affine(well: pd.DataFrame, lam: float = 1.0) -> np.ndarray:
    """Causal per-well affine extrapolation, damped toward carry-forward by `lam`.

    Fit a line TVT_input ~ MD over the ENTIRE known zone (least squares), then
    extrapolate from the carry-forward anchor with the slope scaled by `lam`:

        pred[eval] = anchor_tvt + lam * slope * (MD[eval] - MD[anchor])

    Properties:
      - lam == 0.0  -> identical to predict_carry_forward (slope term vanishes).
      - lam == 1.0  -> full causal fitted slope, pivoted at the anchor.
      - The fit uses ONLY known rows; eval rows are never seen. No leak.

    Full known zone (not last-K) is deliberate: Phase 3 finding #1 showed a
    last-K local slope amplifies error (~107 RMSE). A whole-known-zone slope is
    the stable version; `lam` then controls how much of it to trust.

    Degenerate wells (<2 known rows) fall back to carry-forward on that well.
    """
    tvt_in = well["TVT_input"].to_numpy(dtype=float)
    md = well["MD"].to_numpy(dtype=float)
    anchor = _anchor_idx(well["TVT_input"])

    out = tvt_in.copy()
    eval_idx = np.isnan(out)
    anchor_tvt = tvt_in[anchor]
    md_anchor = md[anchor]

    known_mask = ~np.isnan(tvt_in)
    if int(known_mask.sum()) < 2:
        out[eval_idx] = anchor_tvt
        return out

    md_known = md[known_mask]
    tvt_known = tvt_in[known_mask]
    slope, _intercept = np.polyfit(md_known, tvt_known, deg=1)

    out[eval_idx] = anchor_tvt + lam * slope * (md[eval_idx] - md_anchor)
    return out
