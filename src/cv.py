"""Cross-validation helpers.

The project uses GroupKFold on `well` so entire wells are held out per fold —
honest cross-well generalization, no within-well leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def assign_folds(features_df: pd.DataFrame, n_splits: int = 5,
                 group_col: str = 'well') -> np.ndarray:
    """Returns fold_arr (length len(features_df)) with the fold index per row.

    Asserts every row gets a fold and no group appears in two folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_arr = np.full(len(features_df), -1, dtype=np.int8)
    for fold, (_, va_idx) in enumerate(gkf.split(features_df, groups=features_df[group_col].values)):
        fold_arr[va_idx] = fold
    assert (fold_arr >= 0).all(), 'some rows unassigned'

    # Cross-check group disjointness
    for fold in range(n_splits):
        tr = set(features_df.loc[fold_arr != fold, group_col])
        va = set(features_df.loc[fold_arr == fold, group_col])
        assert not (tr & va), f'fold {fold} has group overlap'

    return fold_arr
