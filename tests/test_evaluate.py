"""Tests for masked RMSE."""

from __future__ import annotations

import numpy as np
import pytest

from rogii_wellbore.evaluate import eval_mask, masked_rmse


def test_masked_rmse_zero_when_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([False, True, True, False])
    assert masked_rmse(y, y.copy(), mask) == 0.0


def test_masked_rmse_only_uses_masked_rows() -> None:
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([100.0, 1.0, 1.0, 100.0])  # huge errors on UNmasked rows
    mask = np.array([False, True, True, False])
    # Only the middle two count: errors of 1 and 1 → RMSE = 1.
    assert masked_rmse(y_true, y_pred, mask) == pytest.approx(1.0)


def test_masked_rmse_empty_mask_raises() -> None:
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        masked_rmse(y, y, np.array([False, False]))


def test_eval_mask_marks_nan_rows() -> None:
    tvt_input = np.array([1.0, np.nan, 2.0, np.nan])
    np.testing.assert_array_equal(eval_mask(tvt_input), [False, True, False, True])


def test_masked_rmse_is_pooled_not_per_well_averaged() -> None:
    # Well A: 2 rows, errors of 10 each → per-well RMSE = 10.
    # Well B: 100 rows, errors of 1 each → per-well RMSE = 1.
    # Pooled SSE = 2*100 + 100*1 = 300 over 102 rows → sqrt(300/102) ≈ 1.715.
    # Mean of per-well RMSEs = 5.5. Pooled must NOT equal 5.5.
    y_true = np.concatenate([np.zeros(2), np.zeros(100)])
    y_pred = np.concatenate([np.full(2, 10.0), np.full(100, 1.0)])
    mask = np.ones(102, dtype=bool)
    pooled = masked_rmse(y_true, y_pred, mask)
    assert pooled == pytest.approx(np.sqrt(300 / 102))
    assert pooled != pytest.approx(5.5)


def test_masked_rmse_raises_on_nan_in_scored_rows() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 3.0])
    mask = np.array([True, True, True])
    with pytest.raises(ValueError, match="NaN"):
        masked_rmse(y_true, y_pred, mask)


def test_masked_rmse_ignores_nan_in_unscored_rows() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([np.nan, 2.0, 3.0])  # NaN is in an unscored row
    mask = np.array([False, True, True])
    assert masked_rmse(y_true, y_pred, mask) == 0.0
