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
