"""Tests for cross-validation splitter."""

from __future__ import annotations

import numpy as np

from rogii_wellbore.cv import grouped_well_splits


def test_grouped_splits_dont_share_wells(tiny_well_df) -> None:
    wells = tiny_well_df["well_id"].to_numpy()
    for train_idx, val_idx in grouped_well_splits(wells, n_splits=3):
        train_wells = set(wells[train_idx])
        val_wells = set(wells[val_idx])
        assert train_wells.isdisjoint(val_wells)


def test_grouped_splits_cover_all_rows(tiny_well_df) -> None:
    wells = tiny_well_df["well_id"].to_numpy()
    seen = np.zeros(len(wells), dtype=bool)
    for _, val_idx in grouped_well_splits(wells, n_splits=3):
        seen[val_idx] = True
    assert seen.all()
