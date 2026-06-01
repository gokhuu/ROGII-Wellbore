"""Probe: how much of the per-well anchor OFFSET is CAUSALLY predictable?

Established so far (770-well OOF):
  carry-forward pooled RMSE      = 15.92
  oracle per-well offset floor   =  9.03   (uses TRUE mean eval error; look-ahead)
  67.9% of CF error is per-well constant.

This probe asks: with NO look-ahead, can we predict that per-well offset from
typewell + known-zone features? Under the SAME 770-well GroupKFold, fit a model
on train-fold wells' offsets, predict held-out wells' offsets, apply as
anchor+offset, score on the REAL eval zone via masked_rmse.

Two models, both deliberately simple (the target is one number per well; high
capacity would just overfit 770 points):
  - ridge (standardized features)
  - small LGBM (depth-capped)

Reports pooled RMSE vs CF (15.92) and vs oracle floor (9.03). The gap from CF
tells you how much is REACHABLE; the gap to 9.03 tells you how much offset
remains unpredicted.

Leakage notes:
  - Target (true mean eval error) is computed per well, used ONLY as a fit target
    for OTHER wells via GroupKFold. A held-out well's own target never trains it.
  - Features come from compute_well_constants_v2: typewell shape + known-zone
    calibration + eval-zone GR stats. GR is never masked, so eval-zone GR is legit
    input for a TVT prediction. No feature reads TVT/TVT_input outside known zone.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rogii_wellbore.cv import grouped_well_splits
from rogii_wellbore.data import list_wells, load_horizontal, load_typewell
from rogii_wellbore.evaluate import eval_mask, masked_rmse
from rogii_wellbore.features_v2 import compute_well_constants_v2

TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}
FEATS = ["tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a", "matcher_sim"]

CF_REF = 15.9240
ORACLE_OFFSET_FLOOR = 9.0270


def build(wells, typewells):
    """Per-well: feature row, true offset (mean eval error), and the arrays
    needed to score (true tvt, anchor tvt, eval mask)."""
    rows, offsets, score_pack, wids = [], [], [], []
    for wid in sorted(wells.keys()):
        w = wells[wid]
        ti = w["TVT_input"].to_numpy(float)
        tv = w["TVT"].to_numpy(float)
        known = ~np.isnan(ti)
        ev = np.isnan(ti)
        if known.sum() < 2 or ev.sum() < 1:
            continue
        anchor_tvt = float(ti[np.flatnonzero(known)[-1]])
        true_eval = tv[ev]
        ok = np.isfinite(true_eval)
        if not ok.any():
            continue
        offset = float((true_eval[ok] - anchor_tvt).mean())  # target: signed correction
        c = compute_well_constants_v2(w, typewells.get(wid), compute_matcher_sim=True)
        feat = [c[f] for f in FEATS]
        if not all(np.isfinite(feat[:3])):  # matcher_sim may be NaN; handle below
            pass
        rows.append(feat)
        offsets.append(offset)
        score_pack.append((tv, ti, anchor_tvt))
        wids.append(wid)
    X = np.array(rows, float)
    # matcher_sim NaNs -> column median (computed on train folds only, below)
    return X, np.array(offsets), score_pack, wids


def score_predictions(score_pack, pred_offsets):
    yt, yp, m = [], [], []
    for (tv, ti, anchor), off in zip(score_pack, pred_offsets, strict=False):
        pred = ti.copy()
        ev = np.isnan(ti)
        pred[ev] = anchor + off
        yt.append(tv)
        yp.append(pred)
        m.append(eval_mask(ti))
    return masked_rmse(np.concatenate(yt), np.concatenate(yp), np.concatenate(m))


def run_model(make_model, X, y, score_pack, wids, n_splits=5):
    pred_off = np.zeros(len(y))
    for tr, va in grouped_well_splits(np.array(wids), n_splits=n_splits):
        Xtr, Xva = X[tr].copy(), X[va].copy()
        # impute matcher_sim (col 3) with TRAIN median only
        col = 3
        med = np.nanmedian(Xtr[:, col])
        for M in (Xtr, Xva):
            M[np.isnan(M[:, col]), col] = med
        model = make_model()
        model.fit(Xtr, y[tr])
        pred_off[va] = model.predict(Xva)
    rmse = score_predictions(score_pack, pred_off)
    # also report how well we predicted the offset itself (R^2-ish)
    ss_res = float(((pred_off - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r2, pred_off


def main():
    ids = [w for w in list_wells("train") if w not in TEST_WELL_IDS]
    print(f"Loading {len(ids)} wells (+typewells) from parquet...")
    dh = load_horizontal("train", well_ids=ids, source="parquet")
    dt = load_typewell("train", well_ids=ids, source="parquet")
    wells = {w: g.reset_index(drop=True) for w, g in dh.groupby("well_id", sort=True)}
    typewells = {w: g.reset_index(drop=True) for w, g in dt.groupby("well_id", sort=True)}

    X, y, score_pack, wids = build(wells, typewells)
    print(f"Built {len(y)} wells with features+offset.\n")
    print(
        f"Offset target: mean={y.mean():+.3f}  std={y.std():.3f}  "
        f"mean|.|={np.mean(np.abs(y)):.3f}\n"
    )

    print(f"{'baseline':28s} pooled_RMSE   offset_R2")
    print(f"{'carry-forward (ref)':28s} {CF_REF:9.4f}      0.000")
    print(f"{'oracle offset floor':28s} {ORACLE_OFFSET_FLOOR:9.4f}      1.000\n")

    rmse_r, r2_r, _ = run_model(
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X, y, score_pack, wids
    )
    print(
        f"{'ridge (alpha=1)':28s} {rmse_r:9.4f}   {r2_r:8.4f}   delta_vs_CF={CF_REF - rmse_r:+.4f}"
    )

    try:
        import lightgbm as lgb

        def mk_lgbm():
            return lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=7,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                verbose=-1,
            )

        rmse_l, r2_l, _ = run_model(mk_lgbm, X, y, score_pack, wids)
        print(
            f"{'small LGBM':28s} {rmse_l:9.4f}   {r2_l:8.4f}   delta_vs_CF={CF_REF - rmse_l:+.4f}"
        )
    except Exception as e:
        print(f"small LGBM skipped: {e}")

    print("\nRead:")
    print("  offset_R2 > 0 and pooled < 15.92  -> offset IS causally predictable;")
    print("    the gap CF->achieved is real RMSE you can submit. Phase 4 = this, refined.")
    print("  offset_R2 ~ 0 / pooled ~ 15.92    -> offset real but NOT predictable from")
    print("    these features. Need better features, or CF is near the stopping point.")
    print("  pooled approaching 9.03           -> near the offset ceiling; little left.")


if __name__ == "__main__":
    main()
