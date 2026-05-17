"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np

from rogii_wellbore.features import per_well_zscore


def test_per_well_zscore_centers_each_well(tiny_well_df) -> None:
    z = per_well_zscore(tiny_well_df, col="GR")
    # Each well's z-scores should have mean ~0.
    means = tiny_well_df.assign(z=z).groupby("well_id")["z"].mean()
    np.testing.assert_allclose(means.to_numpy(), 0.0, atol=1e-9)
