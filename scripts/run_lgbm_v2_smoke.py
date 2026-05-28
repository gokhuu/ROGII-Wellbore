"""Phase 3 LGBM v2 smoke test.

Runs OOF on a small subset (50 wells by default) WITHOUT MLflow logging, prints
a quick comparison vs CF. Use this to verify the pipeline runs end-to-end before
launching the full 770-well run.

Usage:
    python scripts/run_lgbm_v2_smoke.py
    python scripts/run_lgbm_v2_smoke.py --n_wells 100 --no-matcher-sim
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rogii_wellbore.config import RANDOM_SEED
from rogii_wellbore.data import list_wells, load_horizontal, load_typewell
from rogii_wellbore.evaluate import eval_mask
from rogii_wellbore.oof_v2 import run_oof_lgbm_v2


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
    parser.add_argument("--n_wells", type=int, default=50)
    parser.add_argument(
        "--no-matcher-sim",
        action="store_true",
        help="Skip the gr_trend_match sim feature (faster).",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    compute_matcher_sim = not args.no_matcher_sim
    all_ids = list_wells("train")
    # Deterministic subset
    rng = np.random.default_rng(0)
    subset = sorted(rng.choice(np.asarray(all_ids), size=args.n_wells, replace=False).tolist())
    print(f"Smoke test on {len(subset)} wells. compute_matcher_sim={compute_matcher_sim}")

    t0 = time.time()
    df_h = load_horizontal("train", well_ids=subset, source="parquet")
    df_t = load_typewell("train", well_ids=subset, source="parquet")
    print(f"  loaded horizontal+typewell in {time.time() - t0:.1f}s")

    wells = {wid: g.reset_index(drop=True) for wid, g in df_h.groupby("well_id", sort=True)}
    typewells = {wid: g.reset_index(drop=True) for wid, g in df_t.groupby("well_id", sort=True)}

    cf_rmse = _carry_forward_rmse(wells)
    print(f"\nCarry-forward pooled RMSE on this subset: {cf_rmse:.4f}")

    t0 = time.time()
    result, meta = run_oof_lgbm_v2(
        wells,
        typewells,
        n_splits=args.n_splits,
        seed=RANDOM_SEED,
        compute_matcher_sim=compute_matcher_sim,
    )
    print(f"\nOOF complete in {time.time() - t0:.1f}s")
    print(f"  LGBM v2 pooled OOF RMSE: {result.pooled_rmse:.4f}")
    print(f"  Per-fold RMSE: {[round(r, 4) for r in result.per_fold_rmse]}")
    print(f"  Delta vs CF: {cf_rmse - result.pooled_rmse:+.4f}")

    fi = meta[-1].get("feature_importance_summed", {})
    if fi:
        print("\nFeature importance (summed across folds, gain):")
        for name, gain in sorted(fi.items(), key=lambda x: -x[1]):
            print(f"  {name:25s}  {gain:>14,.0f}")


if __name__ == "__main__":
    main()
