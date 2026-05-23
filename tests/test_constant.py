"""Tests for constant baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii_wellbore.models.constant import (
    predict_carry_forward,
    predict_linear_extrap,
    predict_smooth_anchor,
)


def _well(tvt_input: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": np.arange(len(tvt_input), dtype=float),
            "TVT_input": tvt_input,
        }
    )


def test_carry_forward_fills_tail_with_anchor() -> None:
    well = _well([10.0, 11.0, 12.0, np.nan, np.nan, np.nan])
    pred = predict_carry_forward(well)
    # Known rows preserved, eval rows = 12.0 (anchor).
    np.testing.assert_array_equal(pred, [10.0, 11.0, 12.0, 12.0, 12.0, 12.0])


def test_carry_forward_handles_no_eval_rows() -> None:
    # Edge case: well with no NaNs (shouldn't happen in practice, but be safe).
    well = _well([10.0, 11.0, 12.0])
    pred = predict_carry_forward(well)
    np.testing.assert_array_equal(pred, [10.0, 11.0, 12.0])


def test_carry_forward_raises_when_no_known_rows() -> None:
    well = _well([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="no known"):
        predict_carry_forward(well)


def test_carry_forward_uses_last_known_not_first() -> None:
    # Sanity: anchor is the LAST known row, not the first.
    well = _well([10.0, 20.0, 30.0, np.nan, np.nan])
    pred = predict_carry_forward(well)
    assert pred[-1] == 30.0


def test_smooth_anchor_uses_mean_of_last_k() -> None:
    # Known TVT_input = [1, 2, 3, 4, 5], K=3 → anchor = mean(3,4,5) = 4.
    well = _well([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan])
    pred = predict_smooth_anchor(well, k=3)
    np.testing.assert_array_equal(pred[-2:], [4.0, 4.0])


def test_linear_extrap_recovers_exact_line() -> None:
    # TVT_input = 2*MD + 1 on known rows; extrapolation should be exact.
    md = np.arange(10, dtype=float)
    tvt_input = 2 * md + 1
    tvt_input[7:] = np.nan
    well = pd.DataFrame({"MD": md, "TVT_input": tvt_input})
    pred = predict_linear_extrap(well, k=5)
    np.testing.assert_allclose(pred[7:], 2 * md[7:] + 1, atol=1e-9)
