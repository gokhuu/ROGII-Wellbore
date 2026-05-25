"""Phase 2 — first LightGBM (residual-from-anchor), logged to MLflow as phase2."""

from __future__ import annotations

import mlflow

from rogii_wellbore.config import MLFLOW_TRACKING_URI, RANDOM_SEED
from rogii_wellbore.data import list_wells, load_horizontal
from rogii_wellbore.models.lgbm import FEATURES, default_params
from rogii_wellbore.oof import run_oof_lgbm


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase2")

    well_ids = list_wells("train")
    print(f"Loading {len(well_ids)} train wells from parquet…")
    df = load_horizontal("train", well_ids=well_ids, source="parquet")
    wells = {wid: g.reset_index(drop=True) for wid, g in df.groupby("well_id", sort=True)}
    print(f"Loaded {len(wells)} wells, {sum(len(w) for w in wells.values()):,} rows.\n")

    params = default_params()
    with mlflow.start_run(run_name="lgbm_v1"):
        mlflow.log_param("model", "lgbm_v1")
        mlflow.log_param("cv", "GroupKFold_by_well_id")
        mlflow.log_param("n_splits", 5)
        mlflow.log_param("frac_anchor", 0.4)
        mlflow.log_param("features", ",".join(FEATURES))
        for k, v in params.items():
            mlflow.log_param(f"lgbm__{k}", v)

        result, meta = run_oof_lgbm(wells, n_splits=5, seed=RANDOM_SEED)

        mlflow.log_metric("oof_pooled_rmse", result.pooled_rmse)
        for i, r in enumerate(result.per_fold_rmse):
            mlflow.log_metric(f"oof_fold{i}_rmse", r)
            mlflow.log_metric(f"fold{i}_best_iter", meta[i]["best_iteration"])
        mlflow.log_metric("n_eval_rows", result.n_eval_total)

        print(f"\nLGBM v1 pooled OOF RMSE: {result.pooled_rmse:.4f}")
        print(f"Per-fold RMSE: {[f'{r:.4f}' for r in result.per_fold_rmse]}")
        print(f"Best iters:    {[m['best_iteration'] for m in meta]}")
        delta = 15.9099 - result.pooled_rmse
        print("\nCarry-forward baseline: 15.9099")
        print(f"Delta vs carry-forward: {delta:+.4f} ({'better' if delta > 0 else 'worse'})")


if __name__ == "__main__":
    main()
