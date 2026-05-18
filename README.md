# ROGII Wellbore Geology Prediction

Kaggle competition: predict True Vertical Thickness (TVT) along horizontal
oil & gas wells from lateral gamma-ray logs, well trajectory, and a vertical
typewell reference.

## Quickstart

```bash
make setup          # conda env update + install pre-commit hooks
make test           # run pytest
make lint           # ruff + mypy
make mlflow-ui      # MLflow UI at http://localhost:5000
```

## Phase status

See [`docs/00_phases.md`](docs/00_phases.md).

| # | Phase                                 | Status |
|---|---------------------------------------|--------|
| 0 | Scaffolding & env                     | ☐      |
| 1 | EDA                                   | ☐      |
| 2 | Naive baselines                       | ☐      |
| 3 | Domain baseline (GR slide-and-match)  | ☐      |
| 4 | Sequence model                        | ☐      |
| 5 | Iteration                             | ☐      |
| 6 | Ensembling + offline notebook         | ☐      |

## Layout

`src/rogii_wellbore/` — package. `tests/` — pytest. `data/` — gitignored
raw/interim/processed. `mlruns/` — local MLflow store.
