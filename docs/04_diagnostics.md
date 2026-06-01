# ROGII Wellbore — Consolidated Diagnostics Report

A single place collecting every diagnostic probe run across the project, from the
Phase 1 baselines through the Phase 4 causal-offset analysis. Each section gives
the result, a brief breakdown of what it means, and the decision it drove. The
overall conclusion is at the end.

All OOF numbers are pooled RMSE under deterministic 5-fold GroupKFold by
`well_id`, scored on the real eval mask via `evaluate.masked_rmse` (the
competition metric). "Look-ahead" / "oracle" numbers use true eval-zone TVT and
are **ceilings, not achievable scores** — flagged wherever they appear.

---

## Headline numbers

| Quantity | Value | Type |
| --- | ---: | --- |
| Carry-forward pooled OOF RMSE (773-well) | 15.9099 | achievable, the bar |
| Carry-forward pooled OOF RMSE (770-well subset) | 15.9240 | achievable |
| LGBM v1 pooled OOF RMSE (4 features) | 15.4199 | achievable |
| Carry-forward public LB (3 test wells) | 14.505 | achievable |
| LGBM v1 public LB (3 test wells) | 15.142 | achievable |
| Affine ceiling (30-well, offset+slope vs true TVT) | 6.89 | **oracle ceiling** |
| Per-well offset ceiling (770-well, true mean eval error) | 9.0270 | **oracle ceiling** |
| Fraction of carry-forward error that is a per-well constant | 67.9% | diagnostic |

---

## Phase 1–2 — Baseline geometry probes

### Probe: three constant/geometry baselines

| Baseline | Pooled OOF RMSE | vs carry-forward |
| --- | ---: | ---: |
| **A. carry-forward** (anchor = last known TVT) | **15.91** | — |
| B. Z-shift (assume flat horizon, follow wellbore Z) | 111.32 | +95 (catastrophic) |
| C. linear-MD extrapolation (best K=20) | 107.77 | +92 (catastrophic) |
| D. smooth-anchor (mean of last K known), K=20 | ~15.99 | +0.08 (worse) |
| D. smooth-anchor, K=50 | ~16.13 | +0.22 (worse) |

**Breakdown.** Carry-forward wins by ~7× over any geometry- or trajectory-based
method. Two facts fall out of this:

- **The freshest known TVT is the best single estimate.** Smoothing over older
  known values hurts monotonically — older values are stale.
- **Trajectory-following is actively harmful.** Both Z-shift and MD-slope
  extrapolation inherit the wellbore's wobble and amplify it over thousands of
  eval rows.

**Conclusion / decision.** The wells are *geosteered*: the driller keeps the bit
inside the target formation, so TVT (depth relative to formation) stays roughly
constant across the lateral while wellbore Z wobbles around it. Any model must
**anchor on the latest known TVT and predict a small correction**, never a
slope. This reframed the entire modeling target as residual-from-anchor.

### Canary: semantics check

Carry-forward reproduced a prior independent attempt at **15.9099 exactly**,
confirming the eval mask, pooled metric, and GroupKFold harness are wired
correctly. Every downstream number depends on this passing.

---

## Phase 2 — First learned model (LGBM v1)

### Probe: 4-feature residual LGBM vs carry-forward, per fold

| | carry-forward | LGBM v1 | delta (CF − LGBM) |
| --- | ---: | ---: | ---: |
| fold 0 | 12.9932 | 12.8751 | +0.1182 |
| fold 1 | 17.0642 | 16.5626 | +0.5016 |
| fold 2 | 15.3917 | 15.0688 | +0.3229 |
| fold 3 | 16.5371 | 15.5605 | +0.9766 |
| fold 4 | 17.1707 | 16.7104 | +0.4603 |
| **pooled** | **15.9099** | **15.4199** | **+0.4900** |

LightGBM best iterations across folds: 43, 14, 25, 41, 27 — early stopping fires
cleanly within ~50 rounds, so the 4-feature signal saturates fast (no fold ran
away chasing noise).

**Breakdown.** The model beats carry-forward in **5/5 folds** on OOF. The
architecture works; the feature set (`dmd`, `dz`, `gr_roll_mean_k`,
`gr_roll_std_k`) is the minimum that can beat anchor-only.

### Probe: OOF → public LB transfer (the inversion)

| Submission | Pooled OOF | Public LB |
| --- | ---: | ---: |
| Carry-forward | 15.9099 | **14.505** |
| LGBM v1 | 15.4199 | **15.142** |
| Δ (LGBM − CF) | −0.49 (better) | +0.64 (**worse**) |

**Breakdown.** The model that wins on OOF *loses* on the LB. Neither number is
wrong — they measure different populations. OOF averages 773 wells across the
whole field; the LB is **3 specific test wells** in the densest NE cluster that
are unusually carry-forward-friendly (TVT even flatter than the cross-well
average). Carry-forward alone gains ~1.4 RMSE going from OOF (15.91) to LB
(14.505); LGBM v1 gains only 0.27 (15.42 → 15.14), so it is not differentiated
enough to exploit the easy wells, and its small corrections add noise where the
right answer is "predict ~zero correction."

**Conclusion / decision.** Every improvement claim from here must be read against
**both** OOF and LB, never one. We need features strong enough to distinguish
"stay flat" wells from "drift" wells. The unused **typewell** (per-well vertical
GR-vs-TVT reference profile) is the obvious lever.

### Caveat captured this phase

Fold variance is ~4.2 RMSE (773-well). **Any pooled improvement below ~1.5 is
below fold noise** and cannot be trusted on OOF alone.

---

## Phase 3 — Typewell signal: is it there, and is it usable?

### Probe: acid test (does the typewell carry TVT information at all?)

In the **known zone**, lateral GR plotted against true TVT tracks the typewell
GR-vs-TVT profile bump-for-bump. Across a 30-well stratified sample:

- median Pearson **r = 0.83**, min **r = 0.66**, **100%** of wells above 0.5.
- exploration well `015fe0d2`: r = 0.879 over a 494-ft known-zone TVT span.

**Breakdown.** The GR signal is broadly and strongly informative *where the well
moves through TVT*. The typewell is real signal, not noise.

### Probe: affine ceiling (how much is recoverable in principle?)

Oracle 2-parameter fit `TVT_pred = anchor + offset + slope·(i − anchor_idx)`
against *true* eval TVT (cheating — a ceiling): **6.89 pooled RMSE** on the
30-well sample vs carry-forward 16.29. 30/30 wells improve.

**Breakdown.** There are ~9 RMSE of recoverable signal in principle. But this is
an oracle — the question is how much is reachable *causally*.

### Probe: per-row GR matcher (can we extract it row-by-row?)

Three matcher designs (row-space, TVT-space, TVT-stretched) plus a 2-parameter
affine-trajectory matcher (max global Pearson r) all landed at **38–56 RMSE** vs
CF's 23 on the exploration well — i.e. far worse than just carrying forward.

**Breakdown — root cause is geometry, not algorithm.** GR carries TVT
information only when the lateral *moves* in TVT. The known zone traverses ~494
ft of TVT (why r=0.88 works there). The **eval zone traverses only ~30–40 ft over
thousands of rows** — the typewell GR character within that narrow band is not
unique enough to disambiguate position. Pearson r is also offset/scale-blind, so
the matcher optimizes shape, not absolute TVT position.

**Conclusion / decision.** The per-row matcher was shelved. Recoverable-in-
principle (oracle 6.89) and extractable-per-row (38–56) are very different things;
the gap *is* the real problem.

### Bug caught this phase

A feature that was constant within a well and whose distribution shifted between
training and inference sabotaged the trees (val_rmse ~3 but eval RMSE ~15). Fixed
by narrowing `anchor_fracs` to (0.95, 0.90, 0.85, 0.80) and dropping
`anchor_tvt`/`anchor_z`. **Generalizable lesson: any feature constant within a
group whose distribution shifts train→inference will sabotage tree models.**

---

## Phase 4 — Reframe: is the signal a trajectory or a constant?

### Probe: MD-slope causal recovery (`diag` causal-affine sweep)

Fit a causal affine (offset + slope from the known zone only, no look-ahead) over
a sweep of regularization `lambda`, full 770-well GroupKFold, with a canary that
`lambda=0` must reproduce carry-forward to the float.

**Breakdown.** The best causal-affine delta vs carry-forward sat **within fold
noise**. The MD-slope is not cleanly causally recoverable — killing it costs
nothing real. This directly motivated questioning whether there is any trajectory
to recover at all.

### Probe: offset-vs-scatter decomposition (`diag_offset.py`)

Decompose carry-forward's per-well error into (a) a per-well constant (the mean
eval error CF gets wrong) and (b) within-well scatter around that mean.

```
Wells: 770   eval rows: 3,769,838
Carry-forward pooled RMSE:                 15.9240
Oracle per-well OFFSET pooled RMSE (floor): 9.0270
  -> a PERFECT per-well constant correction would cut CF by +6.8970 RMSE
  -> fraction of CF's SSE that is per-well offset (not within-well scatter): 67.9%
Per-well CF mean error (the offset to predict), TVT units:
  median=-0.958  p5/95=[-21.520,+17.907]  mean|.|=9.167
Eval-zone TVT span per well:
  median=26.32 ft  p5/95=[11.91,54.49]
```

**Breakdown.** **67.9% of carry-forward's error is a per-well constant.** A
perfect anchor-offset correction takes CF from 15.92 → **9.03** — almost exactly
the Phase 3 affine ceiling (6.89 on 30 easier wells; 9.03 on the full 770). So
the "affine oracle" was **mostly offset all along**; the slope carried almost
nothing, consistent with the MD-slope probe above. (Correction to an earlier
loose statement: the eval tail does *wander* ~26 ft median; its *net* drift is
~0. The 0.7-figure was net drift, not span. Either way the recoverable signal is
a constant, not a slope.)

**Conclusion / decision.** The Phase 4 target is now exact: **predict one number
per well** — the anchor's bias — from inference-available data (typewell shape +
known-zone GR calibration). Not a trajectory, not a sequence model. This is a
much simpler, much lower-leakage-surface problem than the originally planned
transformer.

### Probe: causal predictability of the offset (`diag_offset_predictable.py`)

The decisive follow-up: with **no look-ahead**, fit a small model (ridge, then
depth-capped LGBM) to predict each well's offset from `compute_well_constants_v2`
features under the same 770-well GroupKFold, apply as `anchor + predicted_offset`,
and score on the real eval zone. Compared against CF (15.92) and the oracle floor
(9.03).

**Status.** This is the open frontier probe — it measures how much of the 6.9 RMSE
of offset headroom is actually *reachable* causally (the gap from 15.92 tells you
what's reachable; the gap to 9.03 tells you what stays unpredicted). Leakage
controls: a held-out well's own offset never trains it (GroupKFold target),
features never read TVT/TVT_input outside the known zone, and eval-zone GR is
legitimate input because GR is never masked.

---

## Overall conclusion

The diagnostics tell one coherent story, and each phase narrowed the problem:

1. **The wells are geosteered** (Phase 1–2). TVT is approximately flat through
   the eval tail; carry-forward at 15.91 OOF is a strong, hard-to-beat bar, and
   anything that follows a slope is catastrophic.

2. **A learned residual model works but the LB is treacherous** (Phase 2). LGBM
   beat CF 5/5 folds on OOF yet lost on the 3-well LB, which is unusually
   CF-friendly. Conclusion: trust OOF, sanity-check against LB, and treat
   sub-1.5 OOF gains as noise (fold variance ~4.2).

3. **The typewell carries real signal, but the eval geometry hides it**
   (Phase 3). Known-zone GR-to-typewell correlation is strong (median r 0.83),
   and the oracle affine ceiling is ~6.9–9.0 RMSE — but per-row matching fails
   (38–56 RMSE) because the eval zone traverses too little TVT to disambiguate
   position.

4. **The recoverable signal is a per-well constant, not a trajectory**
   (Phase 4). 67.9% of CF's error is a per-well offset; a perfect offset reaches
   9.03. The MD-slope is causally dead. So the right target is **one
   anchor-bias number per well**, predicted from typewell + known-zone features —
   a small regressor, not a sequence model.

**The honest ceiling for the current architecture is ~9.0 RMSE** (down from CF's
15.9), and it is reachable only to the extent the per-well offset is *causally*
predictable — which the final probe (`diag_offset_predictable.py`) is built to
measure. Within-well scatter below 9.03 is, on present evidence, irreducible by
any constant or slope correction and would require per-row signal the eval-zone
geometry does not appear to provide.
