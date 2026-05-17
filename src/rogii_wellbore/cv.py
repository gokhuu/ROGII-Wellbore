"""Grouped + tail-masked cross-validation.

Per Phase 0 gotcha #1: random KFold leaks because rows along MD are correlated.
Split by well_id; score OOF only on rows where TVT_input is NaN.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold


def grouped_well_splits(
    well_ids: np.ndarray, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) splitting by well_id."""
    gkf = GroupKFold(n_splits=n_splits)
    # Dummy X/y of correct length; GroupKFold only uses groups.
    n = len(well_ids)
    yield from gkf.split(np.zeros(n), np.zeros(n), groups=well_ids)
