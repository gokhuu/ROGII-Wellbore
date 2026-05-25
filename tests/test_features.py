import numpy as np
import pandas as pd
import pytest

import rogii_wellbore.features


def _well_for_features(n_known: int = 10, n_eval: int = 5) -> pd.DataFrame:
    n = n_known + n_eval
    return pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "X": np.zeros(n),
            "Y": np.zeros(n),
            "Z": np.arange(n, dtype=float) * 0.1,
            "GR": np.arange(n, dtype=float) + 50.0,
            "TVT": np.full(n, 100.0),
            "TVT_input": np.concatenate([np.full(n_known, 100.0), np.full(n_eval, np.nan)]),
        }
    )


def test_pick_training_anchor_lands_inside_known_segment() -> None:
    w = _well_for_features(n_known=10, n_eval=5)
    a = rogii_wellbore.features.pick_training_anchor(w, frac=0.4)
    assert 1 <= a <= 8  # inside known (last_known = 9), strictly before last_known


def test_pick_inference_anchor_is_last_known() -> None:
    w = _well_for_features(n_known=10, n_eval=5)
    assert rogii_wellbore.features.pick_inference_anchor(w) == 9


def test_build_features_dmd_is_zero_at_anchor() -> None:
    w = _well_for_features(n_known=10, n_eval=5)
    feats = rogii_wellbore.features.build_features_for_well(w, anchor_idx=5)
    assert feats.loc[5, "dmd"] == 0.0
    assert feats.loc[5, "dz"] == 0.0
    assert feats.loc[5, "anchor_tvt"] == 100.0


def test_build_features_dmd_is_signed() -> None:
    w = _well_for_features(n_known=10, n_eval=5)
    feats = rogii_wellbore.features.build_features_for_well(w, anchor_idx=5)
    assert feats.loc[0, "dmd"] == -5.0  # before anchor
    assert feats.loc[10, "dmd"] == 5.0  # after anchor


def test_build_features_raises_on_nan_anchor() -> None:
    w = _well_for_features(n_known=10, n_eval=5)
    with pytest.raises(ValueError, match="NaN"):
        rogii_wellbore.features.build_features_for_well(
            w, anchor_idx=12
        )  # eval row → NaN TVT_input
