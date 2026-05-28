"""Phase 3 features: typewell-derived features for LGBM v2.

Extends v1's 4 features (dmd, dz, gr_roll_mean_k, gr_roll_std_k) with 4 new
per-well scalars (broadcast to every eval row of the well):

    tw_slope_at_anchor       — local typewell dGR/dTVT around the anchor (50 ft window)
    gr_delta_eval_anchor     — eval-zone mean lateral GR minus anchor-local mean GR
    calib_a                  — known-zone slope of (lateral GR vs typewell GR at TVT)
    matcher_sim              — max Pearson r of gr_trend_match (informational)

Design rationale (see notebooks/03_typewell_explore.ipynb):
- Per-row TVT prediction from GR matching fails — eval-zone TVT window too narrow
  for the typewell GR character to be uniquely identifying.
- Affine TVT trend HAS recoverable signal (ceiling ~9 RMSE below CF on the 30-well
  sample), but no single matcher recovers it cleanly.
- The 30-well LGBM sandbox showed feature-importance signal for these 4 features
  even when the model didn't beat CF (sample too small; v1 also lost there).
- Decision: ship these as a feature bag for the full 770-well OOF, let LGBM
  decide. Result is honest either way.

All features are leakage-safe by construction: each new feature is derived from
(a) the typewell (always available, never masked) and/or (b) lateral GR (always
available, never masked). None touch TVT or TVT_input outside the known segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii_wellbore.features import _gr_interpolate, build_features_for_well


def _typewell_grid(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Resample typewell to a 1.0 ft ascending TVT grid.

    Drops NaN-GR rows, sorts by TVT ascending, interpolates to integer ft TVTs.
    Returns (tvt_grid, gr_grid). Both empty arrays if typewell is unusable.
    """
    tt = typewell["TVT"].to_numpy(dtype=float)
    tg = typewell["GR"].to_numpy(dtype=float)
    ok = np.isfinite(tt) & np.isfinite(tg)
    if ok.sum() < 10:
        return np.array([]), np.array([])
    tt, tg = tt[ok], tg[ok]
    order = np.argsort(tt)
    tt, tg = tt[order], tg[order]
    if tt[-1] - tt[0] < 10:
        return np.array([]), np.array([])
    grid = np.arange(np.ceil(tt[0]), np.floor(tt[-1]) + 1.0, 1.0)
    gr_grid = np.interp(grid, tt, tg)
    return grid, gr_grid


def _typewell_slope_at_anchor(
    tvt_grid: np.ndarray, gr_grid: np.ndarray, anchor_tvt: float, window_ft: float = 25.0
) -> float:
    """Local dGR/dTVT around anchor_tvt via linear fit on a +/-window_ft band.

    Returns 0.0 if window is too narrow or anchor is outside typewell coverage.
    """
    if len(tvt_grid) == 0:
        return 0.0
    m = (tvt_grid >= anchor_tvt - window_ft) & (tvt_grid <= anchor_tvt + window_ft)
    if m.sum() < 10:
        return 0.0
    slope, _ = np.polyfit(tvt_grid[m], gr_grid[m], 1)
    return float(slope)


def _known_zone_calibration(
    lat_gr_full: np.ndarray,
    tvt_input: np.ndarray,
    tvt_grid: np.ndarray,
    gr_grid: np.ndarray,
) -> tuple[float, float]:
    """Fit lat_GR ~ a * typewell_GR(TVT_input) + b on the known zone.

    Returns (a, b). Defaults to (1.0, 0.0) if there's no fittable overlap.
    """
    if len(tvt_grid) == 0:
        return 1.0, 0.0
    known = ~np.isnan(tvt_input)
    if known.sum() < 50:
        return 1.0, 0.0
    known_tvt = tvt_input[known]
    in_cov = (known_tvt >= tvt_grid[0]) & (known_tvt <= tvt_grid[-1])
    if in_cov.sum() < 50:
        return 1.0, 0.0
    tw_at = np.interp(known_tvt[in_cov], tvt_grid, gr_grid)
    lat_at = lat_gr_full[known][in_cov]
    if np.std(tw_at) < 1e-6:
        return 1.0, 0.0
    a, b = np.polyfit(tw_at, lat_at, 1)
    return float(a), float(b)


def _eval_zone_gr_delta(
    lat_gr_full: np.ndarray,
    anchor_idx: int,
    eval_mask: np.ndarray,
    anchor_window: int = 50,
) -> float:
    """Mean lateral GR in eval zone minus mean lateral GR in the last
    `anchor_window` rows of the known zone (anchor-local mean).

    Returns 0.0 if either side has no data.
    """
    if not eval_mask.any():
        return 0.0
    eval_mean = float(np.nanmean(lat_gr_full[eval_mask]))
    lo = max(0, anchor_idx - anchor_window)
    anchor_local = lat_gr_full[lo : anchor_idx + 1]
    if len(anchor_local) == 0 or np.all(np.isnan(anchor_local)):
        return 0.0
    return eval_mean - float(np.nanmean(anchor_local))


def _gr_trend_match_sim(
    lat_gr_full: np.ndarray,
    eval_mask: np.ndarray,
    anchor_idx: int,
    anchor_tvt: float,
    tvt_grid: np.ndarray,
    gr_grid: np.ndarray,
    r_offset: float = 50.0,
    r_slope: float = 50.0,
    n_offset: int = 51,
    n_slope: int = 51,
) -> float:
    """Run gr_trend_match (affine offset+slope by max Pearson r on eval zone)
    and return only the max sim. The matcher's predicted TVT is NOT used as a
    feature (notebook 03 showed it's unreliable). The sim score itself, however,
    carries weak per-well information about how well the eval-zone lateral GR
    aligns with ANY affine trajectory through the typewell.

    Returns NaN if eval zone is too small or typewell is unusable.
    """
    if len(tvt_grid) == 0:
        return float("nan")
    eval_idx = np.flatnonzero(eval_mask)
    if len(eval_idx) < 50:
        return float("nan")
    lat_eval = lat_gr_full[eval_idx]
    if not np.all(np.isfinite(lat_eval)):
        return float("nan")

    x = (eval_idx - anchor_idx).astype(float)
    offsets = np.linspace(-r_offset, r_offset, n_offset)
    slope_max = r_slope / max(len(eval_idx), 1)
    slopes = np.linspace(-slope_max, slope_max, n_slope)

    tw_lo, tw_hi = tvt_grid[0], tvt_grid[-1]
    best_sim = -np.inf

    for off in offsets:
        base = anchor_tvt + off
        for slp in slopes:
            tvt_pred = base + slp * x
            in_cov = (tvt_pred >= tw_lo) & (tvt_pred <= tw_hi)
            if in_cov.sum() < 50:
                continue
            tw_at = np.interp(tvt_pred[in_cov], tvt_grid, gr_grid)
            lw = lat_eval[in_cov]
            a = lw - lw.mean()
            b = tw_at - tw_at.mean()
            an = np.sqrt((a * a).sum())
            bn = np.sqrt((b * b).sum())
            if an == 0 or bn == 0:
                continue
            sim = float((a * b).sum() / (an * bn))
            if sim > best_sim:
                best_sim = sim
    return best_sim if np.isfinite(best_sim) else float("nan")


def compute_well_constants_v2(
    well: pd.DataFrame,
    typewell: pd.DataFrame | None,
    compute_matcher_sim: bool = True,
) -> dict[str, float]:
    """Compute the 4 v2 features as per-well constants using the REAL eval mask.

    Critical for leakage-safety AND train/inference distribution match:
    - The 4 v2 features are intrinsic well properties (typewell shape near anchor,
      known-zone calibration, eval-zone GR stats). They must NOT depend on which
      training anchor we picked.
    - "anchor_tvt" here is always the inference anchor (last known TVT_input row),
      not a synthetic anchor. This way the well-constant features are identical
      whether the row was sampled with frac=0.80 in training or with the inference
      anchor at the real eval boundary.

    Phase 2 lesson #1: any feature constant within a group whose distribution
    shifts between training and inference will sabotage tree models. Computing
    these features once per well from fixed reference points eliminates that risk.
    """
    out = {
        "tw_slope_at_anchor": 0.0,
        "gr_delta_eval_anchor": 0.0,
        "calib_a": 1.0,
        "matcher_sim": float("nan"),
    }
    if typewell is None or len(typewell) == 0:
        return out

    tvt_grid, gr_grid = _typewell_grid(typewell)
    if len(tvt_grid) == 0:
        return out

    tvt_input = well["TVT_input"].to_numpy(dtype=float)
    known = ~np.isnan(tvt_input)
    if not known.any():
        return out
    # Inference anchor: last known TVT_input row. Always.
    inference_anchor_idx = int(np.flatnonzero(known)[-1])
    inference_anchor_tvt = float(tvt_input[inference_anchor_idx])

    lat_gr_full = _gr_interpolate(well["GR"].to_numpy(dtype=float))
    real_eval_mask = np.isnan(tvt_input)

    out["tw_slope_at_anchor"] = _typewell_slope_at_anchor(tvt_grid, gr_grid, inference_anchor_tvt)
    out["gr_delta_eval_anchor"] = _eval_zone_gr_delta(
        lat_gr_full, inference_anchor_idx, real_eval_mask
    )
    out["calib_a"], _ = _known_zone_calibration(lat_gr_full, tvt_input, tvt_grid, gr_grid)
    if compute_matcher_sim:
        out["matcher_sim"] = _gr_trend_match_sim(
            lat_gr_full,
            real_eval_mask,
            inference_anchor_idx,
            inference_anchor_tvt,
            tvt_grid,
            gr_grid,
        )
    return out


def build_features_for_well_v2(
    well: pd.DataFrame,
    anchor_idx: int,
    typewell: pd.DataFrame | None = None,
    k_roll: int = 50,
    compute_matcher_sim: bool = True,
    well_constants: dict[str, float] | None = None,
) -> pd.DataFrame:
    """v1 features + 4 new typewell-derived features broadcast per row.

    The 4 new features are well-level constants computed from a FIXED reference
    point (the inference anchor) regardless of the `anchor_idx` used for v1
    features. This keeps the new features identically distributed between
    training (synthetic anchors) and inference (last-known anchor).

    `well_constants` can be pre-computed via `compute_well_constants_v2()` and
    reused across the multiple training-anchor calls per well (4x speedup on
    matcher_sim).
    """
    base = build_features_for_well(well, anchor_idx, k_roll=k_roll)

    if well_constants is None:
        well_constants = compute_well_constants_v2(
            well, typewell, compute_matcher_sim=compute_matcher_sim
        )

    base["tw_slope_at_anchor"] = well_constants["tw_slope_at_anchor"]
    base["gr_delta_eval_anchor"] = well_constants["gr_delta_eval_anchor"]
    base["calib_a"] = well_constants["calib_a"]
    base["matcher_sim"] = well_constants["matcher_sim"]
    return base
