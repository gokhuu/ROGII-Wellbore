"""Phase 2 — run constant baselines on train wells, pooled OOF RMSE, logged to MLflow.

Runs each baseline under two CV schemes:
  1. Well-grouped (GroupKFold by well_id) — the floor.
  2. Pad-grouped (GroupKFold by KMeans cluster of wellhead X/Y) — more pessimistic.
"""

from __future__ import annotations

import mlflow
import pandas as pd

from rogii_wellbore.config import MLFLOW_TRACKING_URI
from rogii_wellbore.data import list_wells, load_horizontal
from rogii_wellbore.models.constant import (
    predict_carry_forward,
    predict_linear_extrap,
    predict_smooth_anchor,
)
from rogii_wellbore.oof import run_oof_constant
from rogii_wellbore.pads import assign_pads

BASELINES = {
    "carry_forward": (predict_carry_forward, {}),
    "smooth_anchor_k50": (lambda w: predict_smooth_anchor(w, k=50), {"k": 50}),
    "smooth_anchor_k20": (lambda w: predict_smooth_anchor(w, k=20), {"k": 20}),
    "linear_extrap_k20": (lambda w: predict_linear_extrap(w, k=20), {"k": 20}),
}


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase2")

    well_ids = list_wells("train")
    print(f"Loading {len(well_ids)} train wells from parquet…")
    df = load_horizontal("train", well_ids=well_ids, source="parquet")
    wells = {wid: g.reset_index(drop=True) for wid, g in df.groupby("well_id", sort=True)}
    print(f"Loaded {len(wells)} wells, {sum(len(w) for w in wells.values()):,} rows total.\n")

    # --- Well-grouped CV (the floor) ---
    print("--- Well-grouped CV (GroupKFold by well_id) ---")
    for name, (fn, params) in BASELINES.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("baseline", name)
            mlflow.log_param("cv", "GroupKFold_by_well_id")
            mlflow.log_param("n_splits", 5)
            for k, v in params.items():
                mlflow.log_param(k, v)
            result = run_oof_constant(wells, predict_fn=fn, n_splits=5)
            mlflow.log_metric("oof_pooled_rmse", result.pooled_rmse)
            for i, r in enumerate(result.per_fold_rmse):
                mlflow.log_metric(f"oof_fold{i}_rmse", r)
            mlflow.log_metric("n_eval_rows", result.n_eval_total)
            print(
                f"{name:25s}  pooled={result.pooled_rmse:.4f}  "
                f"per_fold=[{', '.join(f'{r:.2f}' for r in result.per_fold_rmse)}]"
            )

    # --- Pad-grouped CV (more pessimistic) ---
    print("\n--- Pad-grouped CV (GroupKFold by KMeans of wellhead X/Y) ---")
    well_to_pad = assign_pads(wells, n_pads=20)
    pad_counts = pd.Series(well_to_pad).value_counts()
    print(
        f"Pad sizes (wells/pad): min={pad_counts.min()}, "
        f"median={int(pad_counts.median())}, max={pad_counts.max()}"
    )

    for name, (fn, params) in BASELINES.items():
        with mlflow.start_run(run_name=f"{name}__pad_cv"):
            mlflow.log_param("baseline", name)
            mlflow.log_param("cv", "GroupKFold_by_pad")
            mlflow.log_param("n_splits", 5)
            mlflow.log_param("n_pads", 20)
            for k, v in params.items():
                mlflow.log_param(k, v)
            result = run_oof_constant(wells, predict_fn=fn, n_splits=5, well_to_group=well_to_pad)
            mlflow.log_metric("oof_pooled_rmse", result.pooled_rmse)
            for i, r in enumerate(result.per_fold_rmse):
                mlflow.log_metric(f"oof_fold{i}_rmse", r)
            mlflow.log_metric("n_eval_rows", result.n_eval_total)
            print(
                f"{name:25s}  pooled={result.pooled_rmse:.4f}  "
                f"per_fold=[{', '.join(f'{r:.2f}' for r in result.per_fold_rmse)}]"
            )


if __name__ == "__main__":
    main()
