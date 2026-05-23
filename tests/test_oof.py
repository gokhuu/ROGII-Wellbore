"""Tests for OOF harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii_wellbore.oof import run_oof_constant


def _make_well(n_known: int, n_eval: int, tvt_true_value: float) -> pd.DataFrame:
    n = n_known + n_eval
    tvt = np.full(n, tvt_true_value, dtype=float)
    tvt_input = tvt.copy()
    tvt_input[n_known:] = np.nan
    return pd.DataFrame({"MD": np.arange(n, dtype=float), "TVT": tvt, "TVT_input": tvt_input})


def test_oof_identity_predict_gives_zero_rmse() -> None:
    wells = {
        f"w{i:02d}": _make_well(n_known=10, n_eval=5, tvt_true_value=float(i)) for i in range(15)
    }
    # "Perfect" predictor: returns the true TVT for every row.
    result = run_oof_constant(
        wells, predict_fn=lambda w: w["TVT"].to_numpy(dtype=float), n_splits=3
    )
    assert result.pooled_rmse == 0.0
    assert result.n_eval_total == 15 * 5


def test_oof_constant_offset_gives_known_rmse() -> None:
    # All wells have TVT == 0; predictor returns 3.0 everywhere → RMSE = 3.
    wells = {f"w{i:02d}": _make_well(n_known=10, n_eval=5, tvt_true_value=0.0) for i in range(15)}
    result = run_oof_constant(wells, predict_fn=lambda w: np.full(len(w), 3.0), n_splits=3)
    assert result.pooled_rmse == 3.0
