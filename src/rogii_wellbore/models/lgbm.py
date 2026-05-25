"""Phase 2 — first LightGBM model. Residual-from-anchor target with multi-anchor training."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import (
    build_features_for_well,
    pick_inference_anchor,
    pick_training_anchor,  # noqa: F401  # re-exported for diagnostics
)

FEATURES = ["dmd", "dz", "gr_roll_mean_k", "gr_roll_std_k"]


def default_params() -> dict:
    return {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": -1,
    }


def _assemble_training_matrix(
    wells: dict[str, pd.DataFrame],
    well_ids: list[str],
    anchor_fracs: tuple[float, ...] = (0.95, 0.90, 0.85, 0.80),
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (X, y) using multiple anchors per well, each placed in the known segment.

    For each (well, anchor_frac), the anchor lands inside the known segment.
    Training rows are the post-anchor rows whose target_tvt is known.
    Anchors close to end-of-known mimic the inference setup; earlier anchors add diversity.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for wid in well_ids:
        w = wells[wid]
        known = ~np.isnan(w["TVT_input"].to_numpy(dtype=float))
        if known.sum() < 10:
            continue
        last_known = int(np.flatnonzero(known)[-1])
        for frac in anchor_fracs:
            anchor_idx = int(np.clip(round(frac * (last_known + 1)), 1, last_known - 1))
            feats = build_features_for_well(w, anchor_idx)
            keep = (feats["row_idx"] > anchor_idx) & feats["target_tvt"].notna()
            feats = feats.loc[keep]
            if feats.empty:
                continue
            residual = (feats["target_tvt"] - feats["anchor_tvt_value"]).to_numpy()
            xs.append(feats[FEATURES])
            ys.append(residual)
    if not xs:
        raise RuntimeError("No training rows assembled.")
    x = pd.concat(xs, ignore_index=True)
    y = np.concatenate(ys)
    # Shuffle so LightGBM bagging works across all anchors.
    perm = rng.permutation(len(y))
    return x.iloc[perm].reset_index(drop=True), y[perm]


def train_lgbm(
    wells: dict[str, pd.DataFrame],
    train_well_ids: list[str],
    es_well_ids: list[str],
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    anchor_fracs: tuple[float, ...] = (0.95, 0.90, 0.85, 0.80),
    seed: int = 42,
) -> lgb.Booster:
    """Train LGBM with internal early stopping on a held-out subset of train wells."""
    params = params or default_params()
    x_train, y_train = _assemble_training_matrix(
        wells, train_well_ids, anchor_fracs=anchor_fracs, seed=seed
    )
    x_es, y_es = _assemble_training_matrix(
        wells, es_well_ids, anchor_fracs=anchor_fracs, seed=seed + 1
    )
    dtrain = lgb.Dataset(x_train, y_train)
    des = lgb.Dataset(x_es, y_es, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, des],
        valid_names=["train", "es"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )


def predict_lgbm(model: lgb.Booster, well: pd.DataFrame) -> np.ndarray:
    """Full-length TVT prediction using the inference anchor (last known row).

    Known rows return TVT_input; eval rows return anchor_tvt + predicted residual.
    """
    anchor_idx = pick_inference_anchor(well)
    feats = build_features_for_well(well, anchor_idx)
    residual_pred = model.predict(feats[FEATURES])
    anchor_tvt = float(feats["anchor_tvt_value"].iloc[0])
    pred_tvt = anchor_tvt + residual_pred
    tvt_in = well["TVT_input"].to_numpy(dtype=float)
    out = tvt_in.copy()
    eval_idx = np.isnan(out)
    out[eval_idx] = pred_tvt[eval_idx]
    return out
