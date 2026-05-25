# Phase 1: EDA notes

## Q1. Eval-zone size per well

**All 773 train wells have the same eval-zone structure: a contiguous block of `TVT_input.isna()` rows at the tail of the well.** The mask never appears mid-well; it always extends from some row `k` to the final row.

This is forward extrapolation along the lateral, not random infilling. Implication for Phase 2: causal/autoregressive models are natural; no future leak during training.

### Distribution (773 wells)

| stat              | total rows | known rows | eval rows | eval fraction |
|-------------------|-----------:|-----------:|----------:|--------------:|
| min               |      2,058 |        851 |       407 |         0.198 |
| 5th pct           |      4,652 |      1,346 |     2,947 |         0.625 |
| median            |      6,576 |      1,703 |     4,840 |         0.740 |
| mean              |      6,588 |      1,692 |     4,895 |         0.733 |
| 95th pct          |      8,614 |      2,053 |     6,918 |         0.819 |
| max               |     12,141 |      2,392 |    10,052 |         0.875 |

### Observations
- **Known segment is tight** (851-2392 rows, std 217). Each well gives the model roughly the same amount of context.
- **Eval segment is wide** (407-10052 rows, std 1301). 25x range. A model needs to handle both short and very long forwad predictions.
- **One low-mask outlier** (eval_frac ≈ 0.20) — flagged but not investigated.
- Submission `id` integer = horizontal_well CSV row index; eval rows are exactly the rows where `TVT_input` is NaN.

### Figure
`reports/figures/01_eval_zone_distributions.png`


## Q2. GR comparability across wells

**Per-well normalization is sensible (and probably necessary).** The ratio of across-well variation to within-well variation is ≈ 0.89 — wells differ in mean GR about as much as a single well varies internally. A raw `GR = 80` means different things in different wells.

### Per-well summary (773 wells)

| stat            | mean GR | std GR | nan rate |
|-----------------|--------:|-------:|---------:|
| min             |    37.2 |    8.3 |    0.7%  |
| 5th pct         |    65.4 |   12.3 |    3.8%  |
| median          |    86.6 |   17.3 |   27.7%  |
| 95th pct        |   118.2 |   25.3 |   60.3%  |
| max             |   130.5 |   35.1 |   80.1%  |

- Across-well std of per-well means: 15.4
- Median of per-well stds: 17.3
- Ratio (across / within): **0.89** → per-well z-score is the right default

### Bimodality
The distribution of per-well GR means is bimodal (peaks ~85 and ~115), suggesting two clusters of wells (likely different formations, fields, or tool calibrations). A naive per-well z-score erases this cluster signal. Worth testing typewell-conditioned normalization in Phase 2 as an alternative.

### Missing GR
- **29.6% of GR values are NaN overall.** Rate is similar in known (23.5%) and eval (31.7%) zones — GR is *not* cut off at the eval boundary; it's intermittent tool dropouts.
- Per-well NaN rate varies widely: 1% (best) to 80% (worst), median 28%, std 19%.
- **145 wells (19%) have > 50% NaN**, 4 wells > 70%. Phase 2 should evaluate whether these are usable, downweight them, or drop them.
- NaN runs are short (median 1, 99th pct 11, max 19 rows on the spot-check well). Linear interpolation per well is safe; cap on max run-length to be confirmed across all wells.

### Outliers
Spot-checked well `000d7d20` shows a single-row GR spike to ~210 around row 4400 (vs ~95 baseline). Likely sensor glitch. Phase 2 needs robust scaling or outlier clipping.

### Figures
- `reports/figures/02_gr_per_well_stats.png` — distributions of per-well mean/std/spread
- `reports/figures/02b_gr_one_well.png` — GR trace for well 000d7d20 (gaps + outlier visible)


## Q3. Typewell TVT coverage

**Typewell covers the lateral TVT range for 760 / 773 wells (98.3%).** Coverage is sufficient by default — the GR-template matching feature can find a valid TVT match for almost all eval rows.

### Coverage breakdown

| condition                                  | wells |
|--------------------------------------------|------:|
| typewell fully covers lateral TVT range    |   760 |
| typewell misses BELOW lat_min (gap > 0)    |    11 |
| typewell misses ABOVE lat_max (gap > 0)    |     2 |

- **Above gaps are negligible**: 2 wells with max 1.3 ft (less than typewell's 0.5 ft step).
- **Below gaps are real**: 11 wells with 117-645 ft of lateral depth not covered by the typewell (median 434 ft). For these wells, GR-template matching has no candidate match for the shallow-end rows. Falls back to anchor-based prediction is the right strategy.

### Typewell size distribution
- Median length 1874 rows (typewell step is 0.5 ft, so ~937 ft span)
- 5th-95th percentile: 1053-3918 rows
- Range: 636-10043 rows

### Span ratio
Across all wells, typewell span exceeds lateral span by a healthy margin (most points above y=x in the span plot). Cluster of wells at typewell_span ≈ 500 ft regardless of lateral span — looks like some typewells were clipped to a fixed window. Not a blocker.

### Action items for Phase 2
- The GR-template match feature must gracefully fall back when no typewell candidate exists in the search radius (the prior `predict_gr_match_v2` already does this — sets `pred = anchor`).
- Track the 11 below-coverage wells; their eval rows in the uncovered shallow zone will rely entirely on anchor/trajectory features, not typewell matching.

### Figure
`reports/figures/03_typewell_coverage.png` — coverage diagram + span comparison


## Q4. MD step uniformity

**Perfect uniformity across all 773 wells.**

- MD step is exactly **1.0** in every well, every row.
- 0 wells with non-monotonic MD (no negative steps).
- 0 wells with duplicate MD (no zero steps).
- 773 / 773 wells have constant within-well step.
- 773 / 773 wells have the same step value (1.0).

### Implications for Phase 2
- **Windowed features, convolutions, RNNs/transformers all work on the uniform grid directly.** No resampling needed.
- `row_idx` is equivalent to `MD - MD_start` (a constant offset per well).
- The typewell is on a **0.5 ft** grid; cross-correlation with the lateral requires upsampling lateral (2x interpolation) or downsampling typewell (`np.interp` to a 1.0 ft grid). The prior `predict_gr_match_v2` uses the latter.
- Gotcha #6 from the project plan is resolved cleanly — no contingency code needed.


## Q5. Train/test well overlap + geographic structure

### Overlap
- **Train wells:** 773. **Test wells:** 3. **Overlap:** 3.
- The 3 test wells (`000d7d20`, `00bbac68`, `00e12e8b`) are exactly subsets of the train wells.
- **Schema differs:** test wells expose only `MD, X, Y, Z, GR, TVT_input`. The 6 surface-boundary columns (`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`) and `TVT` are train-only. These cannot be used as direct input features at inference time — only as auxiliary training targets.
- The same eval-zone tail mask appears in test as in train (3836 NaN rows in `000d7d20` test, identical to train). Public-LB evaluation is the masked tail of these 3 wells; ground truth held server-side.

### Geographic structure
- All 773 wells lie within a single field complex, ~33mi by 25mi (X: 2.86M to 3.04M, Y: 1.01M to 1.14M; units = ft).
- Wells cluster in **linear streaks** — these are pads/leases (multiple laterals drilled from a common surface location, branching out radially).
- Two visually distinct main concentrations: a SW cluster around (X≈2.88M, Y≈1.02M) and a NE cluster around (X≈3.00M, Y≈1.10M).
- All 3 test wells sit in the densest part of the NE cluster.

### Bimodality follow-up (from Q2)
A crude median-X split does NOT cleanly explain the bimodality of per-well mean GR (both halves have similar means ≈ 88 and contain the secondary peak ~115). The cluster signal appears to be finer-grained — likely **per-pad** rather than regional. A pad-level (X/Y proximity) cluster ID is a candidate engineered feature for Phase 2.

### Action items for Phase 2
- **CV strategy:** GroupKFold by `well_id` is the floor. Consider also a pad-level grouped split (cluster wells by spatial proximity) for a more pessimistic, leakage-safe estimate of generalization to new pads/fields.
- **Coordinate features:** raw X/Y as inputs is sensible; consider adding pad-cluster ID as a categorical.
- **Public-LB caveat:** the 3 test wells are interior to the densest training region. Private LB may include wells from sparser regions; generalization should be checked via pad-grouped CV before trusting public-LB scores.

### Figures
- `reports/figures/05_well_locations.png` — well map with test wells overlaid
- `reports/figures/05b_gr_by_region.png` — GR-by-region check (null result)
