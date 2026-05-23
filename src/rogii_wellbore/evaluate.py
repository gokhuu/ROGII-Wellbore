"""OOF evaluation utilities. Score only on eval-masked rows (TVT_input.isna())."""

from __future__ import annotations

import numpy as np


def eval_mask(tvt_input: np.ndarray) -> np.ndarray:
    """Eval rows are where TVT_input is NaN (the masked tail)."""
    return np.isnan(np.asarray(tvt_input, dtype=float))


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """RMSE computed only on rows where mask is True. Pooled, not per-well averaged."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        raise ValueError("Empty mask — nothing to score.")
    yt, yp = y_true[mask], y_pred[mask]
    n_nan = int(np.isnan(yp).sum() + np.isnan(yt).sum())
    if n_nan:
        raise ValueError(f"{n_nan} NaN value(s) in scored rows; fill predictions before scoring.")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))
