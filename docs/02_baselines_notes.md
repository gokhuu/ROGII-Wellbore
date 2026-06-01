# Phase 2: Baselines + first model notes

## Summary

Phase 2 establishes the bar (carry-forward, pooled OOF RMSE **15.9099**) and ships
the first learned model (LGBM v1, 4 features, pooled OOF RMSE **15.4199**). The
learned model beats carry-forward on OOF in **5/5 folds** but *loses* to it on the
public LB (15.142 vs 14.505). The OOF–LB inversion is the central finding of the
phase and the reason every subsequent improvement claim must be read against both
numbers, not one.

All runs logged to MLflow experiment `phase2`. Code in `src/rogii_wellbore/`,
experiments in `scripts/run_*.py`. Notebook `02_baselines.ipynb` is the report.


## Q1. Do the constant baselines confirm the CV/eval semantics?

**Yes — carry-forward reproduces the prior attempt at 15.9099 exactly.** This was
the single most important check of the phase: it confirms `evaluate.masked_rmse`
matches competition semantics (eval-only rows, pooled across wells) and that the
GroupKFold harness is wired correctly. Everything downstream depends on this.

### Constant-baseline results (well-grouped 5-fold OOF)

| baseline            | pooled OOF RMSE | vs carry-forward |
|---------------------|----------------:|-----------------:|
| carry-forward       |         15.9099 |               —  |
| smooth-anchor k=20  |         ~15.99  |          +0.08 (worse) |
| smooth-anchor k=50  |         ~16.13  |          +0.22 (worse) |
| linear-extrap k=20  |         ~107.77 |        +91.9 (catastrophic) |

### Observations
- **The freshest known TVT is the best estimate.** Smoothing over the last K
  known values hurts (k=20 loses 0.08, k=50 loses 0.22). Older known values are
  stale. → A model should anchor on the *latest* known TVT, not a window mean.
- **Trajectory-following is catastrophic.** Linear extrapolation off the anchor
  slope scores ~107.77, ~7x worse than carry-forward. TVT is approximately flat
  through the eval tail; the anchor slope is noise, and extrapolating it amplifies
  error over thousands of rows. → The correct architecture is **anchor + small
  correction**, not free-form trajectory. This single result shaped every model
  design in Phases 2–4.
- **Pooled RMSE is identical across CV schemes** (well-grouped vs pad-grouped) for
  carry-forward — expected, since it's a per-well computation and reshuffling fold
  composition doesn't change which value lands where. Only per-fold *spread*
  changes (see Q2).


## Q2. What is the fold-variance noise floor?

**Well-grouped per-fold spread is ~4.2 RMSE.** This is the realistic noise floor:
any pooled improvement below it is not reliably distinguishable from luck-of-the-draw
fold composition without multi-seed runs.

### CV scheme comparison (carry-forward, per-fold RMSE)
- **Well-grouped:** spread (max − min) ≈ **4.2 RMSE**.
- **Pad-grouped** (KMeans on wellhead X/Y → 20 pads, GroupKFold by pad): spread is
  *smaller*, not larger.

### Observations
- **Surprise: pad-grouping did not produce a more pessimistic estimate.** We
  expected geographically clustered holdouts to be harder. They weren't. The field
  is homogeneous enough that pad assignment effectively reshuffles wells across
  folds without creating systematically harder holdouts; each pad-grouped fold is
  a more *representative* population sample than each well-grouped fold (where one
  fold can over-sample long-eval or short-eval wells).
- **Headline number is well-grouped pooled** — it is the more conservative of the
  two and the one we report going forward.

### Action items carried forward
- **Improvements < 1.5 RMSE pooled are below the noise floor** and require
  multi-seed runs before any claim. This rule was applied (correctly) in Phase 3.
- Report both CV schemes; lead with well-grouped pooled.


## Q3. Does the first LGBM beat the bar, and how was it un-broken?

**Yes — LGBM v1 reaches 15.4199, beating carry-forward by 0.49, after fixing a
catastrophic 80-RMSE distribution bug.** The fix is the generalizable lesson of
the phase.

### Design (LGBM v1)
- **Target:** residual from anchor, `y = TVT − anchor_TVT`. The model can predict
  zero correction and recover carry-forward exactly. Predictions added back to the
  anchor at inference.
- **Inference anchor:** last known `TVT_input` row.
- **Training anchor:** synthetic anchors inside the known segment at
  `frac ∈ {0.95, 0.90, 0.85, 0.80}` — multi-anchor, 4 training examples per well.
- **Features (4):** `dmd`, `dz`, `gr_roll_mean_k=50`, `gr_roll_std_k=50` (all
  causal, all vary row-to-row within a well).
- **CV:** GroupKFold by well_id, 5 splits; 10% of each fold's train wells held out
  for LightGBM early stopping.

### The 80-RMSE bug and its fix (the generalizable lesson)
The first version scored pooled OOF **80.80** — five times worse than carry-forward.
A targeted diagnostic found two coupled problems:

1. **Residual-distribution mismatch.** With the training anchor at frac=0.4,
   post-anchor training rows spanned the entire well, so training residuals had
   mean **+123.56** and range −6 to +305. At inference (anchor at 100% of known),
   residuals had mean **+3.06** and range −40 to +53. The model was trained on a
   target distribution it never sees at inference.
2. **Leaky well-identity features.** `anchor_tvt` and `anchor_z` were top features
   by gain, but they are *constant within a well* and take *different values*
   between training (frac=0.4) and inference (frac=1.0). The model partitioned on
   well identity, then applied the wrong partition's rule at inference.

**Fix:** (a) narrow `anchor_fracs` to {0.95, 0.90, 0.85, 0.80} so every training
anchor sits near the inference anchor (matching residual distributions), and
(b) drop `anchor_tvt`/`anchor_z`, keeping only row-varying features.

> **Lesson (carried into every later phase):** any feature constant within a group
> whose distribution shifts between training and inference will sabotage a tree
> model. This recurred in Phase 3 (in a rotated form — the *target's* relationship
> to `dmd` shifting because train rows were capped before the extrapolation tail).

### Result
- **5/5 per-fold win.** Every fold, regardless of holdout composition, LGBM v1
  beats carry-forward by a small consistent margin. The 0.49 pooled improvement is
  *below* the 4.2 noise floor, but the unanimous per-fold direction rules out luck.
- **Best iterations 43, 14, 25, 41, 27** — healthy early stopping within ~50
  rounds. The 4-feature signal saturates fast; the feature set is starved.


## Q4. What does the Kaggle submission tell us about OOF vs LB?

**The model that beats carry-forward on OOF loses to it on LB.** Both numbers are
correct measurements — of different populations.

### Submission results

| submission              | pooled OOF | public LB | Δ vs CF (OOF) | Δ vs CF (LB) |
|-------------------------|-----------:|----------:|--------------:|-------------:|
| carry-forward (Phase 1) |    15.9099 | **14.505**|             — |            — |
| LGBM v1 (this phase)    |    15.4199 | **15.142**|  −0.49 (better) | +0.64 (worse) |

### Interpretation
- **OOF** averages over 773 wells across the whole field. **LB** is 3 specific test
  wells, all interior to the densest NE cluster (per Phase 1 Q5).
- **The 3 test wells are unusually carry-forward-friendly** — flatter TVT through
  their eval tails than the cross-well average. When the truth is "no correction,"
  any non-zero LGBM residual *adds* error. The model's strength (small consistent
  corrections) becomes a liability where the right answer is zero.
- CF gains ~1.4 RMSE moving from its OOF (15.91) to LB (14.505): the test wells are
  easier than average *for CF alone*. LGBM v1 gains only ~0.27 (15.42 → 15.14) — it
  is not differentiated enough to exploit easy wells.

### Action items carried forward
- **The OOF→LB gap is real and asymmetric.** A model that wins on OOF by X may gain
  less (or more) on LB. Do not over-index on OOF. (This warning governed the
  Phase 3 submit/don't-submit decision.)
- Private LB may include sparser-region wells; pad-grouped CV is the leakage-safe
  generalization check before trusting public LB.


## Phase 3 entry plan (as set at end of Phase 2)

- **The gap is the typewell** — a per-well vertical GR-vs-TVT reference, currently
  unused. Plan: derive features from GR-matching the lateral against the typewell,
  A/B test against v1 on OOF, submit if ≥1 RMSE improvement.
- Secondary levers: per-well GR z-score; pad-cluster ID; multi-seed runs; then
  Phase 4 sequence model.

**Phase 2 closed. Tag `v0.2-baselines`.**
