"""Tests for pad clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii_wellbore.pads import assign_pads, wellhead_xy


def _well_at(x: float, y: float, n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"MD": np.arange(n, dtype=float), "X": np.full(n, x), "Y": np.full(n, y)})


def test_wellhead_xy_uses_min_md() -> None:
    well = pd.DataFrame({"MD": [10.0, 5.0, 20.0], "X": [1.0, 2.0, 3.0], "Y": [10.0, 20.0, 30.0]})
    out = wellhead_xy({"w0": well})
    assert out.loc["w0", "X"] == 2.0
    assert out.loc["w0", "Y"] == 20.0


def test_assign_pads_groups_nearby_wells() -> None:
    # Two tight clusters far apart → must end up in different pads.
    wells = {
        "a0": _well_at(0.0, 0.0),
        "a1": _well_at(0.1, 0.1),
        "b0": _well_at(100.0, 100.0),
        "b1": _well_at(100.1, 100.1),
    }
    pads = assign_pads(wells, n_pads=2)
    assert pads["a0"] == pads["a1"]
    assert pads["b0"] == pads["b1"]
    assert pads["a0"] != pads["b0"]
