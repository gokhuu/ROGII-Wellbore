"""Phase 4 de-risking — causal per-well affine extrapolation, lambda sweep.

Tests whether ANY causally-fittable drift slope beats carry-forward, and how
much it must be damped before Phase 3 finding #1 (slope amplification) bites.

Mirrors scripts/run_baselines.py exactly:
  - source="parquet", n_splits=5, dict via groupby("well_id").reset_index
  - run_oof_constant(wells, predict_fn=..., n_splits=5)
Differences:
  - Excludes the 3 LB test wells from the wells dict (clean 770-well OOF).
  - Recomputes carry-forward on the SAME 770 wells so the bar is apples-to-apples
    (not the 773-well 15.9099 reference).
  - Sweeps lambda; lam=0.0 must reproduce carry-forward to the float (asserted).
  - Logs to a new `phase4_affine` experiment; does not touch phase2/phase3 runs.

Usage:
    python scripts/run_affine.py
"""

from __future__ import annotations

import mlflow

from rogii_wellbore.config import MLFLOW_TRACKING_URI
from rogii_wellbore.data import list_wells, load_horizontal
from rogii_wellbore.models.constant import predict_carry_forward, predict_causal_affine
from rogii_wellbore.oof import run_oof_constant

# The 3 wells that appear in BOTH train and test dirs (cell 33). Their train-side
# copies carry the true eval-tail TVT = the LB answer key. Exclude from OOF.
TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}

LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase4_affine")

    all_ids = list_wells("train")
    well_ids = [w for w in all_ids if w not in TEST_WELL_IDS]
    n_excluded = len(all_ids) - len(well_ids)
    print(
        f"list_wells('train') -> {len(all_ids)} wells; "
        f"excluding {n_excluded} LB test wells -> {len(well_ids)} for OOF."
    )

    df = load_horizontal("train", well_ids=well_ids, source="parquet")
    wells = {wid: g.reset_index(drop=True) for wid, g in df.groupby("well_id", sort=True)}
    print(f"Loaded {len(wells)} wells, {sum(len(w) for w in wells.values()):,} rows total.\n")

    # Bar: carry-forward on the SAME 770 wells, same folds.
    cf = run_oof_constant(wells, predict_fn=predict_carry_forward, n_splits=5)
    print(
        f"{'carry_forward (770w)':28s}  pooled={cf.pooled_rmse:.4f}  "
        f"per_fold=[{', '.join(f'{r:.2f}' for r in cf.per_fold_rmse)}]"
    )
    with mlflow.start_run(run_name="carry_forward_770w"):
        mlflow.log_param("baseline", "carry_forward")
        mlflow.log_param("n_wells", len(wells))
        mlflow.log_param("cv", "GroupKFold_by_well_id")
        mlflow.log_metric("oof_pooled_rmse", cf.pooled_rmse)
        for i, r in enumerate(cf.per_fold_rmse):
            mlflow.log_metric(f"oof_fold{i}_rmse", r)
        mlflow.log_metric("n_eval_rows", cf.n_eval_total)

    print()
    results = {}
    for lam in LAMBDAS:
        fn = lambda w, _lam=lam: predict_causal_affine(w, lam=_lam)  # noqa: E731
        res = run_oof_constant(wells, predict_fn=fn, n_splits=5)
        results[lam] = res
        delta = cf.pooled_rmse - res.pooled_rmse
        flag = "BETTER" if delta > 0 else "worse "
        print(
            f"affine lam={lam:<4}  pooled={res.pooled_rmse:.4f}  "
            f"delta_vs_CF={delta:+.4f} {flag}  "
            f"per_fold=[{', '.join(f'{r:.2f}' for r in res.per_fold_rmse)}]"
        )
        with mlflow.start_run(run_name=f"affine_lam{lam}"):
            mlflow.log_param("baseline", "causal_affine")
            mlflow.log_param("lambda", lam)
            mlflow.log_param("fit_window", "full_known_zone")
            mlflow.log_param("n_wells", len(wells))
            mlflow.log_param("cv", "GroupKFold_by_well_id")
            mlflow.log_metric("oof_pooled_rmse", res.pooled_rmse)
            mlflow.log_metric("delta_vs_cf", delta)
            for i, r in enumerate(res.per_fold_rmse):
                mlflow.log_metric(f"oof_fold{i}_rmse", r)
            mlflow.log_metric("n_eval_rows", res.n_eval_total)

    # Canary: lam=0 must reproduce carry-forward to the float.
    d0 = abs(results[0.0].pooled_rmse - cf.pooled_rmse)
    print(
        f"\nCanary |affine(lam=0) - CF| = {d0:.2e}  "
        f"({'OK' if d0 < 1e-9 else 'FAIL — investigate before trusting any row above'})"
    )

    best_lam = min(results, key=lambda L: results[L].pooled_rmse)
    best = results[best_lam]
    print(
        f"\nBest lambda: {best_lam}  pooled={best.pooled_rmse:.4f}  "
        f"vs CF {cf.pooled_rmse:.4f}  (delta {cf.pooled_rmse - best.pooled_rmse:+.4f})"
    )
    print(
        "\nRead: if best delta_vs_CF is within fold noise (~4.2 was 773-well; check "
        "770-well fold spread above), drift is NOT cleanly causally recoverable by a "
        "global MD-slope, and the sequence-model thesis needs the typewell to do the work "
        "MD-slope can't. If best lam is interior (not 0, not 1) and clears noise, there's "
        "a real damped drift signal to build on."
    )


if __name__ == "__main__":
    main()
