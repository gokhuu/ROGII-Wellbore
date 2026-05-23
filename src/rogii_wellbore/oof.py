"""OOF harness for per-well baselines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cv import grouped_well_splits
from .evaluate import eval_mask, masked_rmse


@dataclass
class OOFResult:
    pooled_rmse: float
    per_fold_rmse: list[float]
    n_eval_total: int


def run_oof_constant(
    wells: dict[str, pd.DataFrame],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    n_splits: int = 5,
    well_to_group: dict[str, str | int] | None = None,
) -> OOFResult:
    """Run a per-well, non-fitting baseline through GroupKFold folds.

    well_to_group: optional mapping from well_id to a group label. Default is identity
    (each well is its own group → well-grouped CV). Pass pad_id mapping for pad-grouped.
    """
    well_ids = sorted(wells.keys())
    if well_to_group is None:
        well_to_group = {wid: wid for wid in well_ids}

    row_well_ids = np.concatenate([np.full(len(wells[wid]), wid) for wid in well_ids])
    row_groups = np.array([well_to_group[wid] for wid in row_well_ids])

    all_y_true, all_y_pred, all_mask = [], [], []
    per_fold_rmse: list[float] = []

    for _, val_idx in grouped_well_splits(row_groups, n_splits=n_splits):
        val_wells = sorted(set(row_well_ids[val_idx]))
        f_yt, f_yp, f_m = [], [], []
        for wid in val_wells:
            w = wells[wid]
            f_yt.append(w["TVT"].to_numpy(dtype=float))
            f_yp.append(predict_fn(w))
            f_m.append(eval_mask(w["TVT_input"].to_numpy(dtype=float)))
        fy, fp, fm = np.concatenate(f_yt), np.concatenate(f_yp), np.concatenate(f_m)
        per_fold_rmse.append(masked_rmse(fy, fp, fm))
        all_y_true.append(fy)
        all_y_pred.append(fp)
        all_mask.append(fm)

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    mask = np.concatenate(all_mask)
    return OOFResult(
        pooled_rmse=masked_rmse(y_true, y_pred, mask),
        per_fold_rmse=per_fold_rmse,
        n_eval_total=int(mask.sum()),
    )
