# Phase Tracker

| # | Phase                                 | Status | OOF RMSE | LB RMSE | Notes |
|---|---------------------------------------|--------|----------|---------|-------|
| 0 | Scaffolding & env                     | ☑      | —        | —       |       |
| 1 | EDA                                   | ☐      | —        | —       |       |
| 2 | Naive baselines (constant + LGBM)     | ☐      |          |         |       |
| 3 | Domain baseline (GR slide-and-match)  | ☐      |          |         |       |
| 4 | Sequence model                        | ☐      |          |         |       |
| 5 | Iteration                             | ☐      |          |         |       |
| 6 | Ensembling + offline Kaggle notebook  | ☐      |          |         |       |

## Done-when criteria

- **Phase 0**: `make setup && make test && make lint` pass; MLflow UI loads; hello-run logged in MLflow; pre-commit installed; CI green.
- **Phase 1**: written answers in `docs/01_eda_notes.md` to: eval-zone size per well, GR comparability across wells, typewell coverage of TVT range, MD step uniformity.
- **Phase 2**: constant baseline on LB; LGBM beats constant on OOF eval-masked RMSE; CV–LB gap recorded.
- **Phase 3**: GR slide-and-match vs typewell beats both Phase 2 baselines on LB.
- **Phase 4**: 1D-CNN/Transformer with delta-TVT + masked input beats Phase 3 on OOF and LB.
- **Phase 5**: ≥5% RMSE reduction over Phase 4 with stable CV–LB.
- **Phase 6**: Kaggle offline notebook reproduces final submission within time limit.
