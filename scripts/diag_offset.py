"""Diagnostic: is the recoverable typewell signal a per-well OFFSET, not a slope?

Motivation: the slope diagnostic showed the true eval tail spans ~0.7 TVT over
~4840 ft (median |true slope| 0.0024/ft). So the oracle affine's recovered 9 RMSE
is almost surely OFFSET, not trajectory. This probe asks, with NO model and NO
look-ahead:

  Decompose carry-forward error per well into:
    (a) the per-well MEAN eval error  (the constant offset CF gets wrong), and
    (b) the residual scatter around that mean (irreducible by any constant).

  If most of CF's pooled SSE is component (a), then the entire game is predicting
  ONE number per well (an anchor correction), and a sequence model is overkill —
  a per-well offset regressor would capture it. If most is (b), no constant
  correction helps and the signal (if any) really is per-row shape.

Then it computes the ORACLE offset ceiling (subtract each well's true mean eval
error — look-ahead, diagnostic only) to see how far below CF a perfect per-well
offset would reach. This is the honest ceiling for an "anchor + per-well constant"
architecture, the thing the typewell would need to predict.

Read-only. No MLflow. No files written.
"""

from __future__ import annotations

import numpy as np

from rogii_wellbore.data import list_wells, load_horizontal

TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}


def main():
    ids = [w for w in list_wells("train") if w not in TEST_WELL_IDS]
    df = load_horizontal("train", well_ids=ids, source="parquet")
    wells = {w: g.reset_index(drop=True) for w, g in df.groupby("well_id", sort=True)}

    cf_sse = 0.0  # carry-forward SSE (anchor vs truth)
    within_sse = 0.0  # SSE around each well's own mean eval error (oracle-offset residual)
    n_rows = 0
    per_well_mean_err = []  # the offset CF gets wrong, per well
    per_well_eval_span = []  # TVT span of the eval zone, per well

    for w in wells.values():
        ti = w["TVT_input"].to_numpy(float)
        tv = w["TVT"].to_numpy(float)
        known = ~np.isnan(ti)
        ev = np.isnan(ti)
        if known.sum() < 2 or ev.sum() < 1:
            continue
        anchor_tvt = ti[np.flatnonzero(known)[-1]]
        true_eval = tv[ev]
        ok = np.isfinite(true_eval)
        if not ok.any():
            continue
        true_eval = true_eval[ok]

        cf_err = anchor_tvt - true_eval  # what CF gets wrong, per row
        mean_err = cf_err.mean()  # per-well constant component
        within = cf_err - mean_err  # residual after perfect offset

        cf_sse += float((cf_err * cf_err).sum())
        within_sse += float((within * within).sum())
        n_rows += int(ok.sum())
        per_well_mean_err.append(mean_err)
        per_well_eval_span.append(float(true_eval.max() - true_eval.min()))

    cf_rmse = np.sqrt(cf_sse / n_rows)
    oracle_off_rmse = np.sqrt(within_sse / n_rows)  # floor if per-well offset were perfect
    pme = np.array(per_well_mean_err)
    span = np.array(per_well_eval_span)

    # How much of CF's pooled variance is the per-well offset vs within-well scatter?
    frac_offset = 1.0 - (within_sse / cf_sse)

    print(f"Wells: {len(pme)}   eval rows: {n_rows:,}\n")
    print(f"Carry-forward pooled RMSE:                 {cf_rmse:.4f}")
    print(f"Oracle per-well OFFSET pooled RMSE (floor): {oracle_off_rmse:.4f}")
    print(
        f"  -> a PERFECT per-well constant correction would cut CF by "
        f"{cf_rmse - oracle_off_rmse:+.4f} RMSE"
    )
    print(
        f"  -> fraction of CF's SSE that is per-well offset (not within-well scatter): "
        f"{frac_offset:.1%}\n"
    )
    print("Per-well CF mean error (the offset to predict), TVT units:")
    qs = np.percentile(pme, [5, 25, 50, 75, 95])
    print(
        f"  median={np.median(pme):+.3f}  p5/95=[{qs[0]:+.3f},{qs[4]:+.3f}]  "
        f"mean|.|={np.mean(np.abs(pme)):.3f}"
    )
    print("\nEval-zone TVT span per well (confirms narrow band):")
    qs2 = np.percentile(span, [5, 25, 50, 75, 95])
    print(f"  median={np.median(span):.2f} ft  p5/95=[{qs2[0]:.2f},{qs2[4]:.2f}]")
    print("\nRead:")
    print("  frac_offset HIGH (>~60%) -> the signal is a per-well CONSTANT. Phase 4 target")
    print("    is 'predict one offset per well from the typewell', NOT a sequence model.")
    print("  frac_offset LOW           -> offset doesn't help; residual is per-row scatter,")
    print("    and even a perfect constant barely beats CF (sequence model also unlikely to).")
    print("  Compare oracle_off_rmse here to your notebook-03 affine oracle (6.89 on 30 wells):")
    print("    if they're close, the affine oracle WAS mostly offset, confirming the reframe.")


if __name__ == "__main__":
    main()
