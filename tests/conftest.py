"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_well_df() -> pd.DataFrame:
    """3 wells, 10 rows each, with a fake GR column and a tail-masked TVT_input."""
    rng = np.random.default_rng(0)
    rows = []
    for well_id in ["W1", "W2", "W3"]:
        md = np.arange(10, dtype=float)
        gr = rng.normal(loc=50.0, scale=10.0, size=10)
        tvt = md * 0.5 + rng.normal(0, 0.1, size=10)
        tvt_input = tvt.copy()
        tvt_input[-3:] = np.nan  # last 3 rows are the eval zone
        rows.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "MD": md,
                    "GR": gr,
                    "TVT": tvt,
                    "TVT_input": tvt_input,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
