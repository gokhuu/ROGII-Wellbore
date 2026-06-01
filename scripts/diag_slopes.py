"""Diagnostic: what does the known-zone TVT~MD slope actually look like?

The lambda sweep showed RMSE ~ linear in lambda, ~1260 at lambda=1 — i.e. the
fitted slope is huge and extrapolation walks far from flat. Before concluding
anything about drift, SEE the slopes and the known-zone TVT shape.

No new harness. Loads the same 770 wells, fits the same full-known-zone line,
and reports:
  - distribution of fitted slopes (TVT units per ft)
  - how flat the LAST 200 known rows are vs the FULL known zone (curvature check)
  - on 5 sample wells: full-zone slope vs last-200 slope vs true eval slope
    (true eval slope is look-AHEAD, diagnostic only, never used in a predictor)
"""

from __future__ import annotations

import numpy as np

from rogii_wellbore.data import list_wells, load_horizontal

TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}


def fit_slope(md, tvt):
    if len(md) < 2:
        return np.nan
    return float(np.polyfit(md, tvt, 1)[0])


def main():
    ids = [w for w in list_wells("train") if w not in TEST_WELL_IDS]
    df = load_horizontal("train", well_ids=ids, source="parquet")
    wells = {w: g.reset_index(drop=True) for w, g in df.groupby("well_id", sort=True)}

    full_slopes, last200_slopes, true_eval_slopes = [], [], []
    for w in wells.values():
        ti = w["TVT_input"].to_numpy(float)
        md = w["MD"].to_numpy(float)
        tv = w["TVT"].to_numpy(float)
        known = ~np.isnan(ti)
        ev = np.isnan(ti)
        if known.sum() < 2 or ev.sum() < 2:
            continue
        mk, tk = md[known], ti[known]
        full_slopes.append(fit_slope(mk, tk))
        last200_slopes.append(fit_slope(mk[-200:], tk[-200:]))
        true_eval_slopes.append(fit_slope(md[ev], tv[ev]))  # diagnostic only

    full = np.array(full_slopes)
    last200 = np.array(last200_slopes)
    te = np.array(true_eval_slopes)

    def desc(name, a):
        qs = np.nanpercentile(a, [5, 25, 50, 75, 95])
        print(
            f"{name:20s} median={np.nanmedian(a):+.5f}  "
            f"p5/95=[{qs[0]:+.5f},{qs[4]:+.5f}]  "
            f"|median|*4840={abs(np.nanmedian(a))*4840:8.1f} TVT over median tail"
        )

    print(f"Wells analyzed: {len(full)}\n")
    print("Fitted slope (TVT units per ft):")
    desc("full known zone", full)
    desc("last 200 known", last200)
    desc("TRUE eval (lookahd)", te)
    print()
    print("Key comparison — does the TRUE eval tail actually slope, or is it flat?")
    print(f"  median |true eval slope|   = {np.nanmedian(np.abs(te)):.5f} per ft")
    print(f"  median |full-zone slope|   = {np.nanmedian(np.abs(full)):.5f} per ft")
    print(
        f"  ratio (full / true)        = {np.nanmedian(np.abs(full))/max(np.nanmedian(np.abs(te)),1e-9):.1f}x"
    )
    print(f"  median |last200 slope|     = {np.nanmedian(np.abs(last200)):.5f} per ft")
    print(
        f"  ratio (last200 / true)     = {np.nanmedian(np.abs(last200))/max(np.nanmedian(np.abs(te)),1e-9):.1f}x"
    )
    print()
    print("Read: if full-zone slope >> true-eval slope but last200 ~ true-eval,")
    print("the known-zone line is catching build-section curvature, NOT tail drift —")
    print("and a last-K fit (small K) is the right positional variable, not full-zone.")


if __name__ == "__main__":
    main()
