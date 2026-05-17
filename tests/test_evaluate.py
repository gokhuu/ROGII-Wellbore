"""Tests for masked RMSE."""

from __future__ import annotations

import numpy as np
import pytest

from rogii_wellbore.evaluate import masked_rmse


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
