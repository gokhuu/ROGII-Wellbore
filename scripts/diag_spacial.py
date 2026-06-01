"""Probe: is the per-well OFFSET predictable from SPATIAL position / neighbors?

Last credible hypothesis. Every prior probe derived the offset from GR (lateral or
typewell) and found nothing. The offset is a geologic-column-position property, so
it may vary with geography. Test it.

Mechanism: a well's offset ~ the offset of its spatially nearest wells (geology is
spatially correlated). Feature = mean offset of K genuine nearest neighbors by
HEEL (anchor-row) X/Y.

LEAKAGE GUARDS (both are load-bearing):
  1. The 3 test wells have copies in the train dir at distance 0.0 (their own
     selves). ANY neighbor within MIN_DIST ft, OR sharing a well_id, is excluded.
     A 0-distance neighbor = reading your own answer. Reports drop count.
  2. Neighbor pool is built ONLY from same-CV-fold TRAIN wells. A held-out well
     never sees another held-out well's offset. This mirrors the real LB setup
     (test wells find neighbors among the 770 known-offset train wells).

Leads with the honest tell: raw corr(neighbor-mean-offset, own-offset) BEFORE any
model. If that's ~0, the hypothesis is dead (alignment-probe lesson).

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
from rogii_wellbore.features_v2 import compute_well_constants_v2

TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}
V2_FEATS = ["tw_slope_at_anchor", "gr_delta_eval_anchor", "calib_a", "matcher_sim"]
CF_REF = 15.9240
ORACLE_OFFSET_FLOOR = 9.0270
MIN_DIST = 1.0  # exclude neighbors closer than this (self-copies sit at 0.0)
K_NEIGHBORS = (3, 5, 10)


def build(wells, typewells):
    rows = []
    for wid in sorted(wells.keys()):
        w = wells[wid]
        ti = w["TVT_input"].to_numpy(float)
        tv = w["TVT"].to_numpy(float)
        x = w["X"].to_numpy(float)
        y = w["Y"].to_numpy(float)
        known = ~np.isnan(ti)
        ev = np.isnan(ti)
        if known.sum() < 2 or ev.sum() < 1:
            continue
        anchor_i = int(np.flatnonzero(known)[-1])
        anchor_tvt = float(ti[anchor_i])
        true_eval = tv[ev]
        ok = np.isfinite(true_eval)
        if not ok.any():
            continue
        offset = float((true_eval[ok] - anchor_tvt).mean())
        c = compute_well_constants_v2(w, typewells.get(wid), compute_matcher_sim=True)
        rows.append(
            dict(
                wid=wid,
                offset=offset,
                hx=float(x[anchor_i]),
                hy=float(y[anchor_i]),  # HEEL location
                tv=tv,
                ti=ti,
                anchor_tvt=anchor_tvt,
                **{f: c[f] for f in V2_FEATS},
            )
        )
    return rows


def neighbor_offset(hx, hy, pool_xy, pool_off, k):
    """Mean offset of k nearest pool wells, excluding any within MIN_DIST
    (self-copies) by masking them out. Returns (mean_off, nearest_dist, n_used)."""
    d = np.sqrt((pool_xy[:, 0] - hx) ** 2 + (pool_xy[:, 1] - hy) ** 2)
    valid = d > MIN_DIST
    if valid.sum() < k:
        return np.nan, np.nan, 0
    dv = d[valid]
    ov = pool_off[valid]
    order = np.argsort(dv)[:k]
    return float(ov[order].mean()), float(dv[order][0]), k


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
    xy = np.array([[r["hx"], r["hy"]] for r in rows])
    print(f"Built {n} wells.\n")

    # ---- TELL: raw corr(neighbor-mean-offset, own-offset), fold-internal ----
    print("=== TELL: does a well's offset match its neighbors' offsets? ===")
    print("(neighbor pool = same-fold TRAIN wells; self-copies excluded)\n")
    for k in K_NEIGHBORS:
        nbr = np.full(n, np.nan)
        ndist = np.full(n, np.nan)
        zero_hits = 0
        for tr, va in grouped_well_splits(np.array(wids), n_splits=5):
            pool_xy = xy[tr]
            pool_off = y[tr]
            # count would-be self-copy hits for reporting
            for i in va:
                d = np.sqrt((pool_xy[:, 0] - xy[i, 0]) ** 2 + (pool_xy[:, 1] - xy[i, 1]) ** 2)
                if (d <= MIN_DIST).any():
                    zero_hits += 1
                m, nd, nu = neighbor_offset(xy[i, 0], xy[i, 1], pool_xy, pool_off, k)
                nbr[i] = m
                ndist[i] = nd
        ok = np.isfinite(nbr)
        cc = np.corrcoef(nbr[ok], y[ok])[0, 1]
        print(
            f"  k={k:2d}: corr(neighbor_off, own_off)={cc:+.3f}  "
            f"valid={ok.sum()}  median nearest-dist={np.nanmedian(ndist):.0f}ft  "
            f"self-copy exclusions={zero_hits}"
        )

    # ---- harness: offset prediction with spatial features ----
    print("\n=== OFFSET PREDICTION (770-well GroupKFold ridge) ===")
    print(f"{'feature set':32s} pooled_RMSE  offset_R2  delta_vs_CF")
    print(f"{'carry-forward (ref)':32s} {CF_REF:9.4f}     0.000      +0.0000")
    print(f"{'oracle offset floor':32s} {ORACLE_OFFSET_FLOOR:9.4f}     1.000")

    def score(pred):
        yt, yp, m = [], [], []
        for r, off in zip(rows, pred, strict=False):
            p = r["ti"].copy()
            evm = np.isnan(r["ti"])
            p[evm] = r["anchor_tvt"] + off
            yt.append(r["tv"])
            yp.append(p)
            m.append(eval_mask(r["ti"]))
        return masked_rmse(np.concatenate(yt), np.concatenate(yp), np.concatenate(m))

    def run(make_feats, k=5):
        """make_feats(i, pool_xy, pool_off) -> feature row for well i."""
        pred = np.zeros(n)
        for tr, va in grouped_well_splits(np.array(wids), n_splits=5):
            pool_xy, pool_off = xy[tr], y[tr]
            Xtr = np.array([make_feats(i, pool_xy, pool_off, k) for i in tr], float)
            Xva = np.array([make_feats(i, pool_xy, pool_off, k) for i in va], float)
            for col in range(Xtr.shape[1]):
                med = np.nanmedian(Xtr[:, col])
                med = 0.0 if not np.isfinite(med) else med
                Xtr[np.isnan(Xtr[:, col]), col] = med
                Xva[np.isnan(Xva[:, col]), col] = med
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            mdl.fit(Xtr, y[tr])
            pred[va] = mdl.predict(Xva)
        rmse = score(pred)
        ss_res = float(((pred - y) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return rmse, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

    def f_nbr_only(i, pxy, poff, k):
        m, _, _ = neighbor_offset(xy[i, 0], xy[i, 1], pxy, poff, k)
        return [m]

    def f_xy(i, pxy, poff, k):
        return [xy[i, 0], xy[i, 1]]

    def f_nbr_xy(i, pxy, poff, k):
        m, _, _ = neighbor_offset(xy[i, 0], xy[i, 1], pxy, poff, k)
        return [m, xy[i, 0], xy[i, 1]]

    def f_all(i, pxy, poff, k):
        m, _, _ = neighbor_offset(xy[i, 0], xy[i, 1], pxy, poff, k)
        return [m, xy[i, 0], xy[i, 1]] + [rows[i][fe] for fe in V2_FEATS]

    for name, fn in [
        ("neighbor_off ALONE (k=5)", f_nbr_only),
        ("heel X/Y ALONE", f_xy),
        ("neighbor_off + X/Y", f_nbr_xy),
        ("neighbor + X/Y + v2four", f_all),
    ]:
        rmse, r2 = run(fn, k=5)
        print(f"{name:32s} {rmse:9.4f}  {r2:8.4f}    {CF_REF - rmse:+.4f}")

    # ---- LB applicability: genuine neighbor distances for the 3 REAL test wells ----
    print("\n=== LB CHECK: real test wells' GENUINE neighbors (self excluded) ===")
    te = load_horizontal("test", source="parquet")
    teg = te.groupby("well_id")
    train_xy = xy
    train_off = y
    for wid in sorted(TEST_WELL_IDS):
        g = teg.get_group(wid).reset_index(drop=True)
        ti = g["TVT_input"].to_numpy(float)
        known = ~np.isnan(ti)
        ai = int(np.flatnonzero(known)[-1])
        hx, hy = float(g["X"].to_numpy()[ai]), float(g["Y"].to_numpy()[ai])
        d = np.sqrt((train_xy[:, 0] - hx) ** 2 + (train_xy[:, 1] - hy) ** 2)
        keep = d > MIN_DIST
        near = np.sort(d[keep])[:5]
        nidx = np.argsort(d[keep])[:5]
        noff = train_off[keep][nidx]
        print(
            f"  {wid}: genuine nearest 5 dist={near.round(0)}  "
            f"their offsets={noff.round(1)}  mean={noff.mean():+.2f}"
        )
    print("\nRead:")
    print("  TELL corr >~0.3 AND harness beats CF beyond fold noise (~0.8) -> spatial")
    print("    signal is real; build a proper spatial offset model. Check LB wells' neighbor")
    print("    distances above are small enough for it to transfer.")
    print("  TELL ~0 / harness ~15.9 -> offset is not spatially predictable either. With GR,")
    print("    typewell, alignment, AND space all exhausted, CF is the honest stopping point.")


if __name__ == "__main__":
    main()
