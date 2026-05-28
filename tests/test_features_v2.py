"""Tests for features_v2 (Phase 3 typewell-derived features)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import rogii_wellbore.features_v2

# ---------- _typewell_grid ----------


def test_typewell_grid_resamples_to_1ft():
    tw = pd.DataFrame({"TVT": np.arange(1000.0, 1500.5, 0.5), "GR": np.linspace(50, 150, 1001)})
    grid, gr = rogii_wellbore.features_v2._typewell_grid(tw)
    assert grid[0] == 1000.0
    assert grid[-1] == 1500.0
    assert len(grid) == 501
    # Step is uniform 1.0
    assert np.allclose(np.diff(grid), 1.0)
    # Resampling is monotonic same as input
    assert np.all(np.diff(gr) > 0)


def test_typewell_grid_drops_nan_gr():
    # Wide-enough span that even after dropping NaNs we exceed the 10-ft floor
    n = 100
    tvt = np.arange(1000.0, 1000.0 + 0.5 * n, 0.5)
    gr = np.linspace(50, 150, n).tolist()
    gr[5] = np.nan
    gr[20] = np.nan
    gr[50] = np.nan
    tw = pd.DataFrame({"TVT": tvt, "GR": gr})
    grid, gr_out = rogii_wellbore.features_v2._typewell_grid(tw)
    assert grid.size > 0
    assert np.all(np.isfinite(gr_out))
    assert grid[0] >= 1000.0 and grid[-1] <= 1049.5


def test_typewell_grid_unusable_returns_empty():
    grid, gr = rogii_wellbore.features_v2._typewell_grid(
        pd.DataFrame({"TVT": [1.0, 2.0], "GR": [np.nan, np.nan]})
    )
    assert grid.size == 0 and gr.size == 0


# ---------- _typewell_slope_at_anchor ----------


def test_slope_at_anchor_linear_typewell_recovers_slope():
    """GR = 2 * TVT + 100 — slope should be 2.0."""
    tvt = np.arange(1000.0, 1500.0, 1.0)
    gr = 2.0 * tvt + 100.0
    s = rogii_wellbore.features_v2._typewell_slope_at_anchor(
        tvt, gr, anchor_tvt=1250.0, window_ft=25.0
    )
    assert abs(s - 2.0) < 1e-6


def test_slope_at_anchor_flat_typewell_is_zero():
    tvt = np.arange(1000.0, 1500.0, 1.0)
    gr = np.full_like(tvt, 80.0)
    s = rogii_wellbore.features_v2._typewell_slope_at_anchor(
        tvt, gr, anchor_tvt=1250.0, window_ft=25.0
    )
    assert abs(s) < 1e-9


def test_slope_at_anchor_outside_coverage_returns_zero():
    tvt = np.arange(1000.0, 1500.0, 1.0)
    gr = 2.0 * tvt + 100.0
    s = rogii_wellbore.features_v2._typewell_slope_at_anchor(
        tvt, gr, anchor_tvt=2000.0, window_ft=25.0
    )
    assert s == 0.0


# ---------- _known_zone_calibration ----------


def test_calibration_recovers_a_b_when_lateral_is_linear_in_typewell():
    """lateral_GR = 0.5 * typewell_GR(TVT_input) + 10 — should recover a=0.5, b=10."""
    tvt = np.arange(1000.0, 1500.0, 1.0)
    gr = 2.0 * tvt + 100.0  # typewell: GR linear in TVT

    # Lateral: known segment with TVT_input set, then eval (NaN)
    n_known = 400
    tvt_input = np.full(800, np.nan)
    tvt_input[:n_known] = np.linspace(1050.0, 1450.0, n_known)
    typewell_gr_at_known = 2.0 * tvt_input[:n_known] + 100.0
    lat_gr = np.full(800, 90.0)
    lat_gr[:n_known] = 0.5 * typewell_gr_at_known + 10.0

    a, b = rogii_wellbore.features_v2._known_zone_calibration(lat_gr, tvt_input, tvt, gr)
    assert abs(a - 0.5) < 1e-6
    assert abs(b - 10.0) < 1e-3


def test_calibration_defaults_when_no_overlap():
    tvt = np.arange(1000.0, 1100.0, 1.0)
    gr = np.linspace(50, 150, 100)
    tvt_input = np.full(200, np.nan)
    tvt_input[:100] = np.linspace(5000, 5100, 100)  # totally outside typewell
    lat_gr = np.full(200, 80.0)
    a, b = rogii_wellbore.features_v2._known_zone_calibration(lat_gr, tvt_input, tvt, gr)
    assert a == 1.0 and b == 0.0


# ---------- _eval_zone_gr_delta ----------


def test_eval_zone_gr_delta_positive_when_eval_gr_higher():
    lat_gr = np.full(200, 80.0)
    lat_gr[100:] = 95.0  # eval zone has higher GR
    eval_mask = np.zeros(200, dtype=bool)
    eval_mask[100:] = True
    delta = rogii_wellbore.features_v2._eval_zone_gr_delta(
        lat_gr, anchor_idx=99, eval_mask=eval_mask, anchor_window=20
    )
    assert abs(delta - 15.0) < 1e-6


def test_eval_zone_gr_delta_zero_when_flat():
    lat_gr = np.full(200, 80.0)
    eval_mask = np.zeros(200, dtype=bool)
    eval_mask[100:] = True
    delta = rogii_wellbore.features_v2._eval_zone_gr_delta(
        lat_gr, anchor_idx=99, eval_mask=eval_mask
    )
    assert abs(delta) < 1e-9


# ---------- _gr_trend_match_sim ----------


def test_gr_trend_match_sim_finds_high_correlation_when_lateral_matches_typewell():
    """Construct: typewell has a sinusoid GR. Lateral eval zone is a slice of
    that typewell at TVT = anchor_tvt + 0 + 0*x (no drift) — i.e. lateral GR
    is just typewell GR at anchor_tvt repeated. With a flat slice the matcher
    can't distinguish positions, so use a tilted slice instead.

    Lateral eval traces typewell along TVT = anchor + 0.01 * x_row, so the
    matcher should recover slope ~0.01 and find HIGH sim.
    """
    tvt = np.arange(1000.0, 1200.0, 1.0)
    gr = 80 + 20 * np.sin(0.5 * tvt)  # ~10 ft sinusoid period -> distinctive

    n = 600
    anchor_idx = 100
    anchor_tvt = 1100.0
    slope = 0.01  # ft per row -> 5 ft drift over 500 eval rows
    # Eval zone is rows 100..599 (500 rows). True TVT[i] = anchor + slope * (i - anchor_idx).
    true_tvt = anchor_tvt + slope * (np.arange(n) - anchor_idx)
    lat_gr = np.interp(true_tvt, tvt, gr)

    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[anchor_idx + 1 :] = True

    sim = rogii_wellbore.features_v2._gr_trend_match_sim(
        lat_gr,
        eval_mask,
        anchor_idx,
        anchor_tvt,
        tvt,
        gr,
        r_offset=20.0,
        r_slope=20.0,
        n_offset=41,
        n_slope=41,
    )
    # Should be near 1.0; allow some slack because grid is discrete.
    assert sim > 0.85, f"expected high sim, got {sim}"


def test_gr_trend_match_sim_handles_empty_eval():
    tvt = np.arange(1000.0, 1100.0, 1.0)
    gr = np.linspace(50, 150, 100)
    lat_gr = np.full(200, 80.0)
    eval_mask = np.zeros(200, dtype=bool)
    sim = rogii_wellbore.features_v2._gr_trend_match_sim(lat_gr, eval_mask, 100, 1050.0, tvt, gr)
    assert np.isnan(sim)


# ---------- build_features_for_well_v2 ----------


def _make_synthetic_well(n=300, anchor_known_end=100):
    md = np.arange(n, dtype=float) + 5000.0
    z = np.linspace(2000.0, 2050.0, n)
    gr = 80 + 5 * np.sin(0.1 * np.arange(n))
    tvt_input = np.full(n, np.nan)
    tvt_input[:anchor_known_end] = np.linspace(1000.0, 1099.0, anchor_known_end)
    true_tvt = np.linspace(1000.0, 1300.0, n)  # eval zone has real values too
    return pd.DataFrame({"MD": md, "Z": z, "GR": gr, "TVT": true_tvt, "TVT_input": tvt_input})


def _make_synthetic_typewell():
    tvt = np.arange(900.0, 1500.0, 1.0)
    gr = 80 + 20 * np.sin(0.05 * tvt)
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def test_build_features_v2_has_all_columns():
    well = _make_synthetic_well()
    tw = _make_synthetic_typewell()
    anchor_idx = 80
    feats = rogii_wellbore.features_v2.build_features_for_well_v2(
        well, anchor_idx, typewell=tw, compute_matcher_sim=False
    )
    # v1 columns preserved
    for c in (
        "row_idx",
        "MD",
        "dmd",
        "dz",
        "anchor_tvt",
        "anchor_z",
        "gr_roll_mean_k",
        "gr_roll_std_k",
    ):
        assert c in feats.columns, f"missing v1 col {c}"
    # new columns present
    for c in ("tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a", "matcher_sim"):
        assert c in feats.columns, f"missing v2 col {c}"
    assert len(feats) == len(well)


def test_build_features_v2_no_typewell_uses_defaults():
    well = _make_synthetic_well()
    feats = rogii_wellbore.features_v2.build_features_for_well_v2(
        well, anchor_idx=80, typewell=None
    )
    assert (feats["tw_slope_at_anchor"] == 0.0).all()
    assert (feats["gr_delta_eval_anchor"] == 0.0).all()
    assert (feats["calib_a"] == 1.0).all()
    assert feats["matcher_sim"].isna().all()


def test_build_features_v2_new_features_are_constant_per_well():
    """Each of the 4 new features is a per-well scalar broadcast to all rows."""
    well = _make_synthetic_well()
    tw = _make_synthetic_typewell()
    feats = rogii_wellbore.features_v2.build_features_for_well_v2(
        well, anchor_idx=80, typewell=tw, compute_matcher_sim=False
    )
    for c in ("tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a"):
        assert feats[c].nunique(dropna=False) == 1, f"{c} should be constant per well"


def test_build_features_v2_doesnt_touch_true_tvt():
    """Sanity: changing true TVT outside the known zone must not affect features.
    Tests leakage-safety by construction."""
    well1 = _make_synthetic_well()
    well2 = well1.copy()
    well2.loc[150:, "TVT"] = -9999.0  # corrupt true TVT in eval zone
    tw = _make_synthetic_typewell()

    f1 = rogii_wellbore.features_v2.build_features_for_well_v2(
        well1, 80, typewell=tw, compute_matcher_sim=False
    )
    f2 = rogii_wellbore.features_v2.build_features_for_well_v2(
        well2, 80, typewell=tw, compute_matcher_sim=False
    )
    # All v2 feature values must match — they should not depend on TVT in eval rows.
    for c in ("tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a"):
        assert (f1[c] == f2[c]).all(), f"{c} leaks from true TVT in eval zone"


def test_build_features_v2_invariant_to_training_anchor():
    """REGRESSION TEST for the smoke-test bug: changing the anchor_idx (as we
    do across synthetic training anchors) must NOT change the v2 features.
    If this fails, training and inference will see different feature
    distributions and the model will overfit catastrophically (train RMSE 0.2,
    OOF RMSE 19 — see notebook 03 smoke run)."""
    well = _make_synthetic_well()
    tw = _make_synthetic_typewell()
    # Two different "training" anchors inside the known segment
    f_a = rogii_wellbore.features_v2.build_features_for_well_v2(
        well, anchor_idx=60, typewell=tw, compute_matcher_sim=False
    )
    f_b = rogii_wellbore.features_v2.build_features_for_well_v2(
        well, anchor_idx=90, typewell=tw, compute_matcher_sim=False
    )
    for c in ("tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a"):
        assert f_a[c].iloc[0] == f_b[c].iloc[0], (
            f"{c} changed with anchor_idx (a={f_a[c].iloc[0]}, b={f_b[c].iloc[0]}) — "
            f"train/inference distribution mismatch will sabotage LGBM"
        )


def test_compute_well_constants_v2_returns_correct_shape():
    """compute_well_constants_v2 returns a dict with the 4 expected scalar keys."""
    from rogii_wellbore.features_v2 import compute_well_constants_v2

    well = _make_synthetic_well()
    tw = _make_synthetic_typewell()
    c = compute_well_constants_v2(well, tw, compute_matcher_sim=False)
    assert set(c.keys()) == {"tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a", "matcher_sim"}
    for k, v in c.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"
