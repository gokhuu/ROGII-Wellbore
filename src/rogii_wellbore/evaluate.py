"""OOF evaluation utilities. Score only on eval-masked rows (TVT_input.isna())."""

from __future__ import annotations

import numpy as np


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """RMSE computed only on rows where mask is True."""
    if mask.sum() == 0:
        raise ValueError("Empty mask — nothing to score.")
    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(diff**2)))
