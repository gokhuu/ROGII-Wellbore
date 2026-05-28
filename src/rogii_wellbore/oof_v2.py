"""OOF runner for LGBM v2.

Self-contained: depends only on data.py, cv.py, evaluate.py, features.py
(via features_v2), and models/lgbm_v2.py. Does NOT modify or import the
existing oof.py — keeps blast radius small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rogii_wellbore.cv import grouped_well_splits
from rogii_wellbore.evaluate import eval_mask
from rogii_wellbore.features import pick_inference_anchor
from rogii_wellbore.models.lgbm_v2 import (
    FEATURES_V2,
    assemble_training_matrix_v2,
    default_params_v2,
    predict_lgbm_v2,
    train_lgbm_v2,
)


@dataclass
class OOFResultV2:
    pooled_rmse: float
    per_fold_rmse: list[float]
    n_eval_total: int
    oof_preds: dict[str, np.ndarray]  # well_id -> predicted TVT array, NaN where not eval


def _val_matrix_for_fold(
    wells: dict[str, pd.DataFrame],
    typewells: dict[str, pd.DataFrame],
    val_wells: list[str],
    compute_matcher_sim: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]:
    """Build a held-out matrix from the SYNTHETIC-ANCHOR view of val wells
    (mirrors training-matrix construction), used as the LGBM early-stopping
    val set. Returns (X_va, y_va, per_well_inference_info)."""
    from rogii_wellbore.models.lgbm_v2 import assemble_training_matrix_v2  # local to be explicit

    X_va, y_va, _ = assemble_training_matrix_v2(
        wells, typewells, val_wells, compute_matcher_sim=compute_matcher_sim
    )
    return X_va, y_va, []


def run_oof_lgbm_v2(
    wells: dict[str, pd.DataFrame],
    typewells: dict[str, pd.DataFrame],
    n_splits: int = 5,
    seed: int = 42,
    params: dict | None = None,
    compute_matcher_sim: bool = True,
) -> tuple[OOFResultV2, list[dict]]:
    """5-fold GroupKFold OOF on `wells`. For each fold:

      1. Build training matrix from train wells (synthetic anchors).
      2. Build val matrix from val wells (synthetic anchors) for early stopping.
      3. Train LGBM.
      4. For each val well, predict residuals on the REAL eval zone using the
         inference anchor (last known TVT_input row), add anchor back, score.

    Pooled RMSE is over all eval rows of all val wells, mirroring the competition
    metric.
    """
    well_ids = sorted(wells.keys())
    p = params if params is not None else default_params_v2()

    fold_meta: list[dict] = []
    per_fold_rmse: list[float] = []
    oof_preds: dict[str, np.ndarray] = {}
    sse_total = 0.0
    n_total = 0

    feature_importances = np.zeros(len(FEATURES_V2))

    for fold_idx, (tr_idx, va_idx) in enumerate(grouped_well_splits(well_ids, n_splits=n_splits)):
        # grouped_well_splits returns INDEX arrays into well_ids, not the ids themselves.
        # Be tolerant: accept either ints (indices) or strings (already well_ids).
        def _resolve(arr):
            arr = list(arr)
            if not arr:
                return []
            if isinstance(arr[0], int | np.integer):
                return [well_ids[i] for i in arr]
            return [str(x) for x in arr]

        tr_wids = _resolve(tr_idx)
        va_wids = _resolve(va_idx)

        print(f"\n--- fold {fold_idx} ---  train={len(tr_wids)} val={len(va_wids)}")

        X_tr, y_tr, _ = assemble_training_matrix_v2(
            wells, typewells, tr_wids, compute_matcher_sim=compute_matcher_sim
        )
        X_va, y_va, _ = _val_matrix_for_fold(
            wells, typewells, va_wids, compute_matcher_sim=compute_matcher_sim
        )
        print(f"  train matrix: {X_tr.shape}, val matrix: {X_va.shape}")

        model, meta = train_lgbm_v2(X_tr, y_tr, X_va, y_va, params=p)
        print(f"  best_iter={meta['best_iteration']}  val_rmse={meta['best_score_val']:.4f}")

        feature_importances += model.feature_importance(importance_type="gain")

        fold_sse = 0.0
        fold_n = 0
        for wid in va_wids:
            well = wells[wid]
            tw = typewells.get(wid)
            anchor_idx = pick_inference_anchor(well)
            tvt_input = well["TVT_input"].to_numpy(dtype=float)
            anchor_tvt = float(tvt_input[anchor_idx])

            pred_resid = predict_lgbm_v2(
                model, well, tw, anchor_idx, compute_matcher_sim=compute_matcher_sim
            )
            pred_tvt = anchor_tvt + pred_resid
            mask = eval_mask(tvt_input)

            true_tvt = well["TVT"].to_numpy(dtype=float)
            true_in_mask = true_tvt[mask]
            pred_in_mask = pred_tvt[mask]
            ok = np.isfinite(true_in_mask)
            if not ok.any():
                continue
            err = pred_in_mask[ok] - true_in_mask[ok]
            fold_sse += float((err * err).sum())
            fold_n += int(ok.sum())

            arr = np.full(len(well), np.nan)
            arr[mask] = pred_tvt[mask]
            oof_preds[wid] = arr

        fold_rmse = float(np.sqrt(fold_sse / max(fold_n, 1)))
        per_fold_rmse.append(fold_rmse)
        sse_total += fold_sse
        n_total += fold_n
        meta["fold_rmse"] = fold_rmse
        meta["fold_n"] = fold_n
        fold_meta.append(meta)
        print(f"  fold OOF RMSE on real eval zones: {fold_rmse:.4f}  ({fold_n:,} rows)")

    pooled = float(np.sqrt(sse_total / max(n_total, 1)))
    result = OOFResultV2(
        pooled_rmse=pooled,
        per_fold_rmse=per_fold_rmse,
        n_eval_total=n_total,
        oof_preds=oof_preds,
    )

    # Attach summed feature importance to last meta for convenience
    if fold_meta:
        fold_meta[-1]["feature_importance_summed"] = {
            name: float(g)
            for name, g in zip(FEATURES_V2, feature_importances)  # noqa: B905
        }

    return result, fold_meta
