"""Diagnose why LGBM v1 is worse than carry-forward.

Train one fold, print per-well stats: anchor, prediction range, residual range.
Compare predictions to carry-forward on the same wells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii_wellbore.config import RANDOM_SEED
from rogii_wellbore.cv import grouped_well_splits
from rogii_wellbore.data import list_wells, load_horizontal
from rogii_wellbore.evaluate import eval_mask, masked_rmse
from rogii_wellbore.features import (
    build_features_for_well,
    pick_inference_anchor,
    pick_training_anchor,
)
from rogii_wellbore.models.constant import predict_carry_forward
from rogii_wellbore.models.lgbm import FEATURES, predict_lgbm, train_lgbm


def main() -> None:
    well_ids = list_wells("train")
    df = load_horizontal("train", well_ids=well_ids, source="parquet")
    wells = {wid: g.reset_index(drop=True) for wid, g in df.groupby("well_id", sort=True)}

    row_well_ids = np.concatenate([np.full(len(wells[w]), w) for w in sorted(wells)])
    splits = list(grouped_well_splits(row_well_ids, n_splits=5))
    train_idx, val_idx = splits[0]
    train_wells = sorted(set(row_well_ids[train_idx]))
    val_wells = sorted(set(row_well_ids[val_idx]))

    rng = np.random.default_rng(RANDOM_SEED)
    n_es = max(1, int(0.1 * len(train_wells)))
    es_wells = rng.choice(train_wells, size=n_es, replace=False).tolist()
    train_only = [w for w in train_wells if w not in set(es_wells)]

    print(f"Training fold 0: {len(train_only)} train, {len(es_wells)} es, {len(val_wells)} val")
    model = train_lgbm(wells, train_only, es_wells)
    print(f"Best iter: {model.best_iteration}\n")

    # ------ Distribution check: dmd and residual at train vs inference ------
    def collect_train(wid: str) -> tuple[np.ndarray, np.ndarray]:
        w = wells[wid]
        a = pick_training_anchor(w, frac=0.4)
        f = build_features_for_well(w, a)
        keep = (f["row_idx"] > a) & f["target_tvt"].notna()
        f = f.loc[keep]
        return f["dmd"].to_numpy(), (f["target_tvt"] - f["anchor_tvt_value"]).to_numpy()

    def collect_inference(wid: str) -> tuple[np.ndarray, np.ndarray]:
        w = wells[wid]
        a = pick_inference_anchor(w)
        f = build_features_for_well(w, a)
        m = np.isnan(w["TVT_input"].to_numpy(dtype=float))
        return f.loc[m, "dmd"].to_numpy(), (w["TVT"].to_numpy()[m] - f["anchor_tvt_value"].iloc[0])

    sample_wells = val_wells[:50]
    tr_dmd = np.concatenate([collect_train(w)[0] for w in sample_wells])
    tr_res = np.concatenate([collect_train(w)[1] for w in sample_wells])
    inf_dmd = np.concatenate([collect_inference(w)[0] for w in sample_wells])
    inf_res = np.concatenate([collect_inference(w)[1] for w in sample_wells])
    print("Distribution comparison (50 val wells, training-anchor vs inference-anchor):")
    print(
        f"  dmd     train: min={tr_dmd.min():.0f} median={np.median(tr_dmd):.0f} max={tr_dmd.max():.0f}"
    )
    print(
        f"  dmd     inf:   min={inf_dmd.min():.0f} median={np.median(inf_dmd):.0f} max={inf_dmd.max():.0f}"
    )
    print(
        f"  residual train: mean={tr_res.mean():+.2f} std={tr_res.std():.2f} range=[{tr_res.min():+.1f},{tr_res.max():+.1f}]"
    )
    print(
        f"  residual inf:   mean={inf_res.mean():+.2f} std={inf_res.std():.2f} range=[{inf_res.min():+.1f},{inf_res.max():+.1f}]"
    )

    # ------ Per-well prediction trace, 5 val wells ------
    print("\nPer-well predictions (5 val wells):")
    print(
        f"  {'well_id':12s}  {'n_eval':>6s}  {'anchor':>7s}  {'true_mean':>9s}  {'cf_rmse':>8s}  {'lgbm_rmse':>9s}  {'lgbm_pred_mean':>14s}  {'lgbm_res_std':>12s}"
    )
    for wid in val_wells[:5]:
        w = wells[wid]
        m = eval_mask(w["TVT_input"].to_numpy(dtype=float))
        tvt_true = w["TVT"].to_numpy(dtype=float)
        cf_pred = predict_carry_forward(w)
        lg_pred = predict_lgbm(model, w)
        anchor = float(w["TVT_input"].dropna().iloc[-1])
        cf_rmse = masked_rmse(tvt_true, cf_pred, m)
        lg_rmse = masked_rmse(tvt_true, lg_pred, m)
        print(
            f"  {wid:12s}  {m.sum():>6d}  {anchor:>7.2f}  {tvt_true[m].mean():>9.2f}  "
            f"{cf_rmse:>8.3f}  {lg_rmse:>9.3f}  {lg_pred[m].mean():>14.2f}  {(lg_pred[m] - anchor).std():>12.2f}"
        )

    # ------ Feature importance ------
    print("\nFeature importance (gain):")
    imp = pd.DataFrame(
        {
            "feature": FEATURES,
            "gain": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
