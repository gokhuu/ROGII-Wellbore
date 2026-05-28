"""Phase 3 LGBM v2 full OOF.

Runs 5-fold GroupKFold by well_id on all 770 train wells. Logs to MLflow under
the `phase3` experiment. Mirrors `scripts/run_lgbm.py` (the v1 run script) so
results are directly comparable.

Usage:
    python scripts/run_lgbm_v2.py
    python scripts/run_lgbm_v2.py --no-matcher-sim   # skip the expensive matcher_sim feature
"""

from __future__ import annotations

import argparse
import time

import mlflow
import numpy as np

from rogii_wellbore.config import MLFLOW_TRACKING_URI, RANDOM_SEED
from rogii_wellbore.data import list_wells, load_horizontal, load_typewell
from rogii_wellbore.evaluate import eval_mask
from rogii_wellbore.models.lgbm_v2 import FEATURES_V2, default_params_v2
from rogii_wellbore.oof_v2 import run_oof_lgbm_v2

# Phase 2 reference numbers (logged for delta calculations).
CF_OOF_RMSE = 15.9099
LGBM_V1_OOF_RMSE = 15.4199


def _carry_forward_rmse(wells: dict) -> float:
    sse, n = 0.0, 0
    for w in wells.values():
        tvt_input = w["TVT_input"].to_numpy(dtype=float)
        true_tvt = w["TVT"].to_numpy(dtype=float)
        known = ~np.isnan(tvt_input)
        if not known.any():
            continue
        anchor_tvt = float(tvt_input[np.flatnonzero(known)[-1]])
        m = eval_mask(tvt_input)
        true_eval = true_tvt[m]
        ok = np.isfinite(true_eval)
        if not ok.any():
            continue
        err = anchor_tvt - true_eval[ok]
        sse += float((err * err).sum())
        n += int(ok.sum())
    return float(np.sqrt(sse / max(n, 1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-matcher-sim",
        action="store_true",
        help="Skip the expensive gr_trend_match sim feature.",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="lgbm_v2")
    args = parser.parse_args()

    compute_matcher_sim = not args.no_matcher_sim

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase3")

    well_ids = list_wells("train")
    print(f"Loading {len(well_ids)} train wells from parquet...")
    t0 = time.time()
    df_h = load_horizontal("train", well_ids=well_ids, source="parquet")
    df_t = load_typewell("train", well_ids=well_ids, source="parquet")
    wells = {wid: g.reset_index(drop=True) for wid, g in df_h.groupby("well_id", sort=True)}
    typewells = {wid: g.reset_index(drop=True) for wid, g in df_t.groupby("well_id", sort=True)}
    print(
        f"Loaded {len(wells)} wells, {sum(len(w) for w in wells.values()):,} rows "
        f"in {time.time() - t0:.1f}s\n"
    )

    cf_rmse_actual = _carry_forward_rmse(wells)
    print(
        f"Carry-forward pooled RMSE (recomputed): {cf_rmse_actual:.4f}  "
        f"(Phase 2 reference: {CF_OOF_RMSE:.4f})\n"
    )

    params = default_params_v2()
    run_name = args.run_name + ("__no_matcher_sim" if args.no_matcher_sim else "")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", "lgbm_v2")
        mlflow.log_param("cv", "GroupKFold_by_well_id")
        mlflow.log_param("n_splits", args.n_splits)
        mlflow.log_param("anchor_fracs", "0.95,0.90,0.85,0.80")
        mlflow.log_param("features", ",".join(FEATURES_V2))
        mlflow.log_param("compute_matcher_sim", compute_matcher_sim)
        for k, v in params.items():
            mlflow.log_param(f"lgbm__{k}", v)

        t0 = time.time()
        result, meta = run_oof_lgbm_v2(
            wells,
            typewells,
            n_splits=args.n_splits,
            seed=RANDOM_SEED,
            compute_matcher_sim=compute_matcher_sim,
        )
        elapsed = time.time() - t0
        print(f"\nOOF complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)\n")

        mlflow.log_metric("oof_pooled_rmse", result.pooled_rmse)
        mlflow.log_metric("n_eval_rows", result.n_eval_total)
        mlflow.log_metric("oof_runtime_sec", elapsed)
        for i, r in enumerate(result.per_fold_rmse):
            mlflow.log_metric(f"oof_fold{i}_rmse", r)
            mlflow.log_metric(f"fold{i}_best_iter", meta[i]["best_iteration"])

        delta_cf = cf_rmse_actual - result.pooled_rmse
        delta_v1 = LGBM_V1_OOF_RMSE - result.pooled_rmse
        mlflow.log_metric("delta_vs_cf", delta_cf)
        mlflow.log_metric("delta_vs_v1", delta_v1)

        fi = meta[-1].get("feature_importance_summed", {})
        if fi:
            for name, gain in fi.items():
                mlflow.log_metric(f"feat_gain__{name}", float(gain))

        print(f"=== LGBM v2 pooled OOF RMSE: {result.pooled_rmse:.4f} ===")
        print(f"Per-fold RMSE: {[round(r, 4) for r in result.per_fold_rmse]}")
        print(f"Best iters:    {[m['best_iteration'] for m in meta]}")
        print(f"\nCarry-forward (recomputed): {cf_rmse_actual:.4f}")
        print(f"LGBM v1 (Phase 2 reference): {LGBM_V1_OOF_RMSE:.4f}")
        print(f"\nDelta vs CF:   {delta_cf:+.4f}  ({'BETTER' if delta_cf > 0 else 'WORSE'})")
        print(f"Delta vs v1:   {delta_v1:+.4f}  ({'BETTER' if delta_v1 > 0 else 'WORSE'})")
        if fi:
            print("\nFeature importance (gain, summed across folds):")
            for name, gain in sorted(fi.items(), key=lambda x: -x[1]):
                print(f"  {name:25s}  {gain:>14,.0f}")


if __name__ == "__main__":
    main()
