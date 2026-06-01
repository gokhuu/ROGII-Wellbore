"""Probe: typewell coverage of the eval band + GR cross-correlation alignment as
an offset predictor.

Two parts, one run:

PART A - COVERAGE (the gating question):
  For each well, does the eval-zone TVT band fall inside the typewell's TVT range?
  If a large fraction fall outside, neither this feature NOR a CNN can align against
  a typewell that doesn't cover the eval depth. Reports the distribution before any
  modeling, so we interpret Part B knowing how many wells even had a chance.

PART B - ALIGNMENT FEATURE:
  Physical idea: the per-well offset is a heel mis-tie - the lateral sits some feet
  above/below where its GR pattern matches the typewell GR-vs-TVT profile. Measure
  that shift directly:
    - Take known-zone rows. Each has true TVT_input and lateral GR.
    - Resample lateral GR onto the typewell's 0.5-ft TVT grid over the known band.
    - Z-score both GR signals within-well (GR not comparable across wells).
    - Cross-correlate; the lag (in TVT feet) maximizing correlation = alignment shift.
  Computed over two windows (last 100 ft, last 200 ft of known zone) - the offset is
  an anchor property so near-anchor is more relevant, but shorter = noisier. Report
  which predicts the offset better.

  Then run the SAME 770-well GroupKFold ridge as diag_offset_predictable, but with
  the alignment-lag feature(s) ADDED to the four v2 features. If offset_R2 jumps off
  zero, there's learnable alignment signal -> green light for a CNN over raw sequences.
  If flat, an explicit cross-correlation found nothing, and a CNN over the same
  sequences is very unlikely to do better.

Read-only. No MLflow. No files written.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rogii_wellbore.cv import grouped_well_splits
from rogii_wellbore.data import list_wells, load_horizontal, load_typewell
from rogii_wellbore.evaluate import eval_mask, masked_rmse
from rogii_wellbore.features import _gr_interpolate
from rogii_wellbore.features_v2 import _typewell_grid, compute_well_constants_v2

TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}
V2_FEATS = ["tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a", "matcher_sim"]
CF_REF = 15.9240
ORACLE_OFFSET_FLOOR = 9.0270
MAX_LAG_FT = 40.0  # search +/- 40 ft of TVT shift
GRID_STEP = 0.5  # typewell native step


def _zscore(a):
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else a - a.mean()


def alignment_lag(well, tvt_grid, gr_grid, window_ft):
    """Cross-correlation lag (TVT ft) aligning known-zone lateral GR to typewell GR.

    Returns (lag_ft, max_corr, valid_flag). Positive lag = lateral GR pattern sits
    at HIGHER typewell TVT than its recorded TVT_input (i.e. anchor reads shallow).
    """
    if len(tvt_grid) == 0:
        return np.nan, np.nan, False
    ti = well["TVT_input"].to_numpy(float)
    gr = _gr_interpolate(well["GR"].to_numpy(float))
    known = ~np.isnan(ti)
    if known.sum() < 50:
        return np.nan, np.nan, False
    kidx = np.flatnonzero(known)
    anchor_tvt = ti[kidx[-1]]
    # known rows within window_ft below the anchor
    lo_tvt = anchor_tvt - window_ft
    sel = kidx[(ti[kidx] >= lo_tvt) & (ti[kidx] <= anchor_tvt)]
    if len(sel) < 30:
        return np.nan, np.nan, False
    # must sit inside typewell coverage (with lag headroom)
    if (ti[sel].min() - MAX_LAG_FT < tvt_grid[0]) or (ti[sel].max() + MAX_LAG_FT > tvt_grid[-1]):
        return np.nan, np.nan, False
    # resample lateral GR onto typewell grid over [lo, anchor]
    grid = np.arange(np.ceil(ti[sel].min()), np.floor(ti[sel].max()) + GRID_STEP, GRID_STEP)
    if len(grid) < 20:
        return np.nan, np.nan, False
    order = np.argsort(ti[sel])
    lat_on_grid = np.interp(grid, ti[sel][order], gr[sel][order])
    lat_z = _zscore(lat_on_grid)
    # slide lateral against typewell GR sampled at grid+lag
    lags = np.arange(-MAX_LAG_FT, MAX_LAG_FT + GRID_STEP, GRID_STEP)
    best_corr, best_lag = -np.inf, 0.0
    for lag in lags:
        tw_at = np.interp(grid + lag, tvt_grid, gr_grid)
        tw_z = _zscore(tw_at)
        denom = np.sqrt((lat_z**2).sum() * (tw_z**2).sum())
        if denom < 1e-9:
            continue
        c = float((lat_z * tw_z).sum() / denom)
        if c > best_corr:
            best_corr, best_lag = c, float(lag)
    return best_lag, best_corr, True


def build(wells, typewells):
    rows = []
    for wid in sorted(wells.keys()):
        w = wells[wid]
        tw = typewells.get(wid)
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
        offset = float((true_eval[ok] - anchor_tvt).mean())
        tvt_grid, gr_grid = _typewell_grid(tw) if tw is not None else (np.array([]), np.array([]))
        # coverage: does eval band sit inside typewell TVT range?
        eval_tvt = tv[ev][ok]
        cov = len(tvt_grid) > 0 and eval_tvt.min() >= tvt_grid[0] and eval_tvt.max() <= tvt_grid[-1]
        lag100, corr100, v100 = alignment_lag(w, tvt_grid, gr_grid, 100.0)
        lag200, corr200, v200 = alignment_lag(w, tvt_grid, gr_grid, 200.0)
        c = compute_well_constants_v2(w, tw, compute_matcher_sim=True)
        rows.append(
            dict(
                wid=wid,
                offset=offset,
                coverage=cov,
                lag100=lag100,
                corr100=corr100,
                v100=v100,
                lag200=lag200,
                corr200=corr200,
                v200=v200,
                **{f: c[f] for f in V2_FEATS},
                tv=tv,
                ti=ti,
                anchor_tvt=anchor_tvt,
            )
        )
    return rows


def score_predictions(rows, pred_off):
    yt, yp, m = [], [], []
    for r, off in zip(rows, pred_off, strict=False):
        pred = r["ti"].copy()
        ev = np.isnan(r["ti"])
        pred[ev] = r["anchor_tvt"] + off
        yt.append(r["tv"])
        yp.append(pred)
        m.append(eval_mask(r["ti"]))
    return masked_rmse(np.concatenate(yt), np.concatenate(yp), np.concatenate(m))


def run_ridge(X, y, rows, wids, n_splits=5):
    pred = np.zeros(len(y))
    for tr, va in grouped_well_splits(np.array(wids), n_splits=n_splits):
        Xtr, Xva = X[tr].copy(), X[va].copy()
        for col in range(X.shape[1]):  # train-median impute each column
            med = np.nanmedian(Xtr[:, col])
            if not np.isfinite(med):
                med = 0.0
            Xtr[np.isnan(Xtr[:, col]), col] = med
            Xva[np.isnan(Xva[:, col]), col] = med
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(Xtr, y[tr])
        pred[va] = model.predict(Xva)
    rmse = score_predictions(rows, pred)
    ss_res = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r2


def main():
    ids = [w for w in list_wells("train") if w not in TEST_WELL_IDS]
    print(f"Loading {len(ids)} wells + typewells...")
    dh = load_horizontal("train", well_ids=ids, source="parquet")
    dt = load_typewell("train", well_ids=ids, source="parquet")
    wells = {w: g.reset_index(drop=True) for w, g in dh.groupby("well_id", sort=True)}
    typewells = {w: g.reset_index(drop=True) for w, g in dt.groupby("well_id", sort=True)}

    rows = build(wells, typewells)
    n = len(rows)
    y = np.array([r["offset"] for r in rows])
    wids = [r["wid"] for r in rows]

    # ---- PART A: coverage ----
    cov = np.array([r["coverage"] for r in rows])
    v100 = np.array([r["v100"] for r in rows])
    v200 = np.array([r["v200"] for r in rows])
    print(f"\n=== PART A: COVERAGE ({n} wells) ===")
    print(f"  eval band inside typewell TVT range:   {cov.mean():.1%}  ({cov.sum()}/{n})")
    print(f"  alignment computable, 100ft window:    {v100.mean():.1%}  ({v100.sum()}/{n})")
    print(f"  alignment computable, 200ft window:    {v200.mean():.1%}  ({v200.sum()}/{n})")
    if cov.mean() < 0.5:
        print("  *** Under half the wells have eval-band coverage. Alignment AND a CNN")
        print("      are structurally limited; interpret Part B as a partial-population result.")

    # how well does the raw lag correlate with the offset, on valid wells?
    for tag, vflag, lagkey, corrkey in [
        ("100ft", v100, "lag100", "corr100"),
        ("200ft", v200, "lag200", "corr200"),
    ]:
        if vflag.sum() < 20:
            print(f"\n  [{tag}] too few valid wells to correlate.")
            continue
        lag = np.array([r[lagkey] for r in rows])[vflag]
        off = y[vflag]
        cc = np.corrcoef(lag, off)[0, 1]
        mc = np.array([r[corrkey] for r in rows])[vflag]
        print(
            f"\n  [{tag}] valid={int(vflag.sum())}  "
            f"corr(lag, offset)={cc:+.3f}  median max-xcorr={np.median(mc):.3f}  "
            f"lag std={lag.std():.2f}ft"
        )

    # ---- PART B: does alignment lag predict the offset, through the harness? ----
    print("\n=== PART B: OFFSET PREDICTION (770-well GroupKFold ridge) ===")
    print(f"{'feature set':34s} pooled_RMSE  offset_R2  delta_vs_CF")
    print(f"{'carry-forward (ref)':34s} {CF_REF:9.4f}     0.000      +0.0000")
    print(f"{'oracle offset floor':34s} {ORACLE_OFFSET_FLOOR:9.4f}     1.000")

    def feat_matrix(keys):
        return np.array([[r[k] for k in keys] for r in rows], float)

    sets = {
        "v2 four (reference)": V2_FEATS,
        "+ lag100": V2_FEATS + ["lag100"],  # noqa: RUF005
        "+ lag200": V2_FEATS + ["lag200"],  # noqa: RUF005
        "+ lag100 + corr100": V2_FEATS + ["lag100", "corr100"],  # noqa: RUF005
        "lag100 + corr100 ALONE": ["lag100", "corr100"],
    }
    for name, keys in sets.items():
        X = feat_matrix(keys)
        rmse, r2 = run_ridge(X, y, rows, wids)
        print(f"{name:34s} {rmse:9.4f}  {r2:8.4f}    {CF_REF - rmse:+.4f}")

    print("\nRead:")
    print("  If '+ lag' rows beat 'v2 four' and offset_R2 climbs off ~0 -> alignment shift")
    print("    carries offset signal; a CNN over raw GR/typewell sequences is justified.")
    print("  If '+ lag' rows ~ 'v2 four' (~15.9, R2~0) -> explicit cross-correlation found")
    print(
        "    no offset signal; a CNN over the same inputs is very unlikely to. Strong stop signal."
    )


if __name__ == "__main__":
    main()
