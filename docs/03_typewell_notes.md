# Phase 3: Typewell features (LGBM v2) notes

## Summary

Phase 3 done-when criterion (from `00_phases.md`): *GR slide-and-match against the
typewell beats both Phase 2 baselines on the LB.* **Not met.** Recorded as an honest
negative result with a diagnosis that redirects Phase 4.

The per-row GR matcher was shelved (geometry wall). The fallback — 4 weak per-well
typewell-derived scalars fed to LGBM v2 — trains and OOFs correctly but lands at pooled
OOF **15.7367**, statistically indistinguishable from carry-forward (15.9099) and LGBM
v1 (15.4199): all deltas are below the ~4.2 RMSE fold-variance noise floor. The typewell
features carry feature-importance gain but **no out-of-sample lift**.

This is *not* "the typewell is uninformative." The signal is real (acid test, affine
ceiling below). It is that *this class of feature* — per-well scalars + a tree — cannot
express the signal that exists, which is **low-frequency drift**, not per-row TVT.

Runs logged to MLflow experiment `phase3`. Report notebook: `03_typewell.ipynb`.
Exploration scratchpad: `03_typewell_explore.ipynb`.


## Q1. Is there recoverable typewell signal at all?

**Yes — and it is bounded.** Two exploration results establish this.

### Acid test (known-zone GR correlation)
In the known zone (true TVT available), lateral GR vs true TVT tracks the typewell
GR-vs-TVT profile bump-for-bump.

| stat                      | value |
|---------------------------|------:|
| median Pearson r (30 wells) | 0.83 |
| min Pearson r             | 0.66 |
| fraction of wells r > 0.5 | 1.00 |
| exploration well `015fe0d2` | 0.879 (over 494-ft known TVT span) |

### Affine ceiling (oracle, 30-well pooled)
Oracle fit `TVT_pred = anchor + offset + slope·(i − anchor_idx)` against *true* eval
TVT (cheating — a ceiling, not achievable):

| model          | pooled RMSE |
|----------------|------------:|
| carry-forward  |       16.29 |
| affine ceiling |        6.89 |

- **~9 RMSE of recoverable affine drift signal** sits between CF and the ceiling.
- 30/30 wells improve over CF; max single-well improvement 33.6 RMSE.

### Observation
The signal exists and is broadly available. The question for Phase 3 was never "is
there signal" — it was "can we recover it without true TVT." The answer split: the
*drift trend* is recoverable in principle; *per-row TVT* is not (Q2).


## Q2. Why was the per-row GR matcher shelved?

**Geometry, not algorithm.** Four matcher designs (row-space, TVT-space with constant
proxy, TVT-stretched, 2-param affine-trajectory by max Pearson r) all scored **38–56
RMSE vs CF's 23** on the exploration well.

### Root cause
GR carries TVT information only when the lateral *moves* in TVT.
- **Known zone:** ~494 ft of TVT traverse → r = 0.88, matching works.
- **Eval zone:** ~30–40 ft of TVT over thousands of rows → the typewell GR character
  in that narrow window is not unique enough to disambiguate `(offset, slope)`. The
  matcher maximizes correlation correctly, but the correlation maximum is not at the
  right TVT.

### Observations
- **Z is not a TVT proxy.** `dZ ≠ dTVT` — TVT is in the formation frame (accounts for
  structural dip), Z is true vertical depth. The flat constant proxy (`anchor_tvt`) is
  the only safe search center.
- The improvement *direction* across matcher attempts was right (sim scores rose,
  predictions less degenerate) but never crossed CF. Cleverness does not beat a data
  resolution wall.
- **Decision:** discard the matcher's predicted TVT. Keep only its max similarity score
  as a weak informational feature. Ship 4 per-well scalars, let LGBM combine them.


## Q3. The training-window bug (and the Phase 2 lesson, rotated)

The first v2 wiring trained on rows `anchor_idx < row ≤ last_known` — inside the known
segment only. The real eval zone is the masked tail (`row > last_known`), pure forward
extrapolation at large `dmd`. **v2 never trained on a single extrapolation row.**

### Symptom
Early-stopping val RMSE read ~3–4 while real eval-zone OOF RMSE was ~12–21 — a 4–7x gap.

### Diagnosis
This is the Phase 2 distribution-shift lesson rotated 90°: not a leaky constant feature,
but the *target's relationship to `dmd`* differing between the (interpolation) rows
trained on and the (extrapolation) rows scored.

### Fix
Mirror v1's `_assemble_training_matrix`: drop the `row_idx ≤ last_known` cap; train on
all post-anchor rows with finite true TVT (which, in training data, includes the tail).
After the fix the val and eval RMSE land in the same range; the 50-well smoke training
matrix grows ~34k → ~870k rows. Params also aligned to v1 exactly (`regression_l2`,
`lambda_l2=1.0`, `bagging_freq=1`) so feature set is the only v1↔v2 difference.

> **Lesson reinforced:** train and inference must occupy the *same* `dmd`/target regime.
> v1 got this right by filtering on `target_tvt.notna()` (tail included); v2's explicit
> cap silently broke it. Carry this into Phase 4: training windows must include the
> extrapolation tail.


## Q4. Did LGBM v2 beat the bar?

**No.** Full 770-well well-grouped 5-fold OOF, logged to `phase3`:

| model          | pooled OOF RMSE | Δ vs CF        | Δ vs v1        |
|----------------|----------------:|---------------:|---------------:|
| carry-forward  |         15.9099 |             —  |             —  |
| LGBM v1        |         15.4199 |  −0.49 (better) |             —  |
| **LGBM v2**    |     **15.7367** |  −0.17 (better) |  +0.32 (worse) |

Per-fold OOF RMSE: `[13.89, 16.71, 15.33, 15.24, 17.27]`. Best iters: `[1, 9, 6, 7, 9]`.

### Observations
- **Both deltas are below the ~4.2 RMSE noise floor and the 1.5-pooled significance
  bar.** v2 is statistically indistinguishable from CF *and* v1. The features did not
  help.
- **Best iterations collapsed to 1–9** (vs v1's 14–43). Fold 0 stopped at iteration
  **1** — one split captures everything learnable. Weak global structure, nothing
  well-specific.
- **tr/va inverted and swung across folds** (fold 0: tr 18.6 > va 14.5; fold 4: tr 14.8,
  va 21.5). The per-well features do not transfer across well groups.
- **Feature importance is a trap here.** `tw_slope_at_anchor` and `gr_delta_eval_anchor`
  top the gain table, yet OOF does not move. High gain + no lift = the model fitting
  per-well structure that does not generalize. `gr_roll_std_k` = 0 gain (consistent with
  the 30-well sandbox). Importance ≠ out-of-sample value.


## Q5. Submission decision

**Did not submit v2.** v2's OOF (15.74) is already worse than v1's OOF (15.42); the
OOF→LB gap is asymmetric (Phase 2); spending a submission on a model CV says is no
better than the one already submitted is low value. Submission budget preserved.
(One submission for an LB datapoint is defensible if wanted for completeness, but is not
expected to change the conclusion.)


## Conclusion

1. **The typewell signal is real** — known-zone r median 0.83, oracle affine recovers
   ~9 RMSE below CF.
2. **That signal is low-frequency drift (offset + slope), not per-row TVT** — the eval
   zone's ~30-ft TVT traverse is too narrow for per-row matching; no matcher overcame it.
3. **Per-well scalars + LGBM cannot express drift** — a tree predicts a constant per
   leaf; forward-extrapolating a per-well slope is structurally what it does worst.
   Features show gain but zero OOF lift, best-iters collapse to 1–9, per-well constants
   fail to transfer across folds.

Corroborates and sharpens Phase 2: architecture must be *anchor + small correction*, and
the correction is a **drift trend that must be extrapolated** — which a per-row tree on
per-well scalars cannot do.

**Phase 3 closed honestly. Tag `v0.3-typewell`.**


## What Phase 4 needs (entry plan)

The affine ceiling (6.89 vs CF 16.29) says ~9 RMSE of drift is unclaimed. Phase 4's job
is to capture drift a tree cannot.

- **Architecture:** sequence model (1D-CNN / small transformer) over a windowed lateral
  view (GR sequence + `dz`, `dmd` positional), predicting residual-from-anchor *as a
  function of distance down the eval tail*. It can represent a continuous drift slope and
  carry it forward. (Phase tracker Phase 4: "1D-CNN/Transformer with delta-TVT + masked
  input beats Phase 3 on OOF and LB.")
- **Carry over (validated):** residual-from-anchor target; inference anchor = last known
  row; multi-anchor training `fracs (0.95, 0.90, 0.85, 0.80)` **with the extrapolation
  tail included** (Q3 fix); causal/leakage-safe features; uniform 1.0-ft grid (Phase 1
  Q4); per-well GR interpolation of short NaN runs (Phase 1 Q2).
- **Use the typewell as an auxiliary input *sequence*, not a per-well scalar** — let the
  model attend to the GR-vs-TVT profile directly rather than pre-summarizing it.
- **De-risking step first (~1 hour, recommended):** run plain causal per-well affine
  extrapolation (slope fit on known-zone tail only, no peeking) through the existing OOF
  harness. The *ceiling* (oracle slope) is measured; the *achievable causal* version is
  not. If it beats CF on OOF → Phase 4 has a concrete target and strong prior. If not →
  drift is not cleanly recoverable causally and Phase 4's framing changes. Either outcome
  de-risks the most expensive phase.
- **Evaluation discipline:** well-grouped pooled is headline; fold variance ~4.2 RMSE;
  improvements < 1.5 pooled need multi-seed runs; the 3 LB wells are CF-friendly and
  OOF→LB is asymmetric — check pad-grouped CV before trusting public LB.
