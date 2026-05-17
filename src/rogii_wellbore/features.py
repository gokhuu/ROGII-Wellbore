"""Feature engineering. Stubs for now."""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_well_zscore(df: pd.DataFrame, col: str, group_col: str = "well_id") -> pd.Series:
    """Per-well z-score normalization for a column (e.g. gamma-ray).

    Phase 1 gotcha: raw GR is not comparable across wells (different tools/calibrations).
    """
    grouped = df.groupby(group_col)[col]
    return (df[col] - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
