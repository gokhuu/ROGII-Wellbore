"""LGBM v2: v1 + 4 typewell-derived features.

Self-contained — does NOT depend on models/lgbm.py to keep blast radius small.
Imports only from features.py, features_v2.py, and lightgbm.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

import rogii_wellbore.features_v2
from rogii_wellbore.features import pick_training_anchor

# v1's 4 features + 4 new typewell-derived features.
FEATURES_V2: list[str] = [
    # v1
    "dmd",
    "dz",
    "gr_roll_mean_k",
    "gr_roll_std_k",
    # v2 additions
    "tw_slope_at_anchor",
    "gr_delta_eval_anchor",
    "calib_a",
    "matcher_sim",
]


def default_params_v2() -> dict[str, Any]:
    """LGBM params matching v1 defaults — change is feature set only."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }


def assemble_training_matrix_v2(
    wells: dict[str, pd.DataFrame],
    typewells: dict[str, pd.DataFrame],
    well_ids: list[str],
    anchor_fracs: tuple[float, ...] = (0.95, 0.90, 0.85, 0.80),
    compute_matcher_sim: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build training (X, y, well_id_per_row) for the given wells.

    For each well, picks `len(anchor_fracs)` synthetic anchors inside the known
    segment. For each anchor, computes v2 features for all rows AFTER the anchor
    that are still inside the known segment (so we have ground truth TVT).
    Target = TVT - anchor_tvt (residual).
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    well_id_rows: list[str] = []
    for wid in well_ids:
        well = wells[wid]
        tw = typewells.get(wid)
        # Compute the 4 v2 features ONCE per well (anchor-independent) and
        # reuse across all training-anchor passes. This is correct (the v2
        # features must NOT depend on which synthetic anchor we picked, see
        # compute_well_constants_v2 docstring) AND ~4x faster on matcher_sim.
        well_consts = rogii_wellbore.features_v2.compute_well_constants_v2(
            well, tw, compute_matcher_sim=compute_matcher_sim
        )
        tvt_input = well["TVT_input"].to_numpy(dtype=float)
        true_tvt = well["TVT"].to_numpy(dtype=float)
        known = ~np.isnan(tvt_input)
        if not known.any():
            continue
        last_known = int(np.flatnonzero(known)[-1])
        for frac in anchor_fracs:
            anchor_idx = pick_training_anchor(well, frac=frac)
            anchor_tvt = float(tvt_input[anchor_idx])
            feats = rogii_wellbore.features_v2.build_features_for_well_v2(
                well,
                anchor_idx,
                typewell=tw,
                compute_matcher_sim=compute_matcher_sim,
                well_constants=well_consts,
            )
            # Use rows AFTER the synthetic anchor, still inside the known segment.
            row_idx = feats["row_idx"].to_numpy()
            in_window = (row_idx > anchor_idx) & (row_idx <= last_known)
            if not in_window.any():
                continue
            sub = feats.loc[in_window, FEATURES_V2].to_numpy()
            y = true_tvt[row_idx[in_window]] - anchor_tvt
            ok = np.isfinite(y)
            if not ok.any():
                continue
            Xs.append(sub[ok])
            ys.append(y[ok])
            well_id_rows.extend([wid] * int(ok.sum()))
    if not Xs:
        return (
            np.empty((0, len(FEATURES_V2))),
            np.empty(0),
            [],
        )
    return np.vstack(Xs), np.concatenate(ys), well_id_rows


def train_lgbm_v2(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train one LGBM model with early stopping on the val set."""
    p = params if params is not None else default_params_v2()
    dtr = lgb.Dataset(X_tr, y_tr, feature_name=FEATURES_V2)
    dva = lgb.Dataset(X_va, y_va, feature_name=FEATURES_V2, reference=dtr)
    model = lgb.train(
        p,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dtr, dva],
        valid_names=["tr", "va"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )
    meta = {
        "best_iteration": int(model.best_iteration or 0),
        "best_score_val": float(model.best_score["va"]["rmse"]),
    }
    return model, meta


def predict_lgbm_v2(
    model: lgb.Booster,
    well: pd.DataFrame,
    typewell: pd.DataFrame | None,
    anchor_idx: int,
    compute_matcher_sim: bool = True,
) -> np.ndarray:
    """Predict residual-from-anchor for every row of `well` using `model`.

    Returns array len(well). To get TVT, caller adds back anchor_tvt.
    """
    feats = rogii_wellbore.features_v2.build_features_for_well_v2(
        well, anchor_idx, typewell=typewell, compute_matcher_sim=compute_matcher_sim
    )
    X = feats[FEATURES_V2].to_numpy()
    return model.predict(X, num_iteration=model.best_iteration)
