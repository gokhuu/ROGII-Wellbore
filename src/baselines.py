"""Heuristic baselines from notebooks 02-03.

Each baseline implements `predict(g, tw) -> np.ndarray` of length `len(g)`. The
typewell argument is accepted for API consistency even when unused.
"""
import numpy as np

from .utils import COL_TVT_INPUT, COL_Z, COL_MD


def predict_carry_forward(g, tw=None):
    """A — predict TVT_eval = TVT_boundary. Pooled RMSE 15.910."""
    return g[COL_TVT_INPUT].ffill().bfill().values.astype(float)


def predict_z_shift(g, tw=None):
    """B — TVT_eval = Z_eval + (TVT_boundary - Z_boundary). Pooled RMSE 111.32."""
    pred = g[COL_TVT_INPUT].values.astype(float).copy()
    eval_mask = np.isnan(pred)
    if not eval_mask.any():
        return pred
    known_idx = np.where(~eval_mask)[0]
    if len(known_idx) == 0:
        return g[COL_Z].values.astype(float)
    last_k = known_idx[-1]
    z_vals = g[COL_Z].values
    offset = pred[last_k] - z_vals[last_k]
    pred[eval_mask] = z_vals[eval_mask] + offset
    return pred


def make_linear_extrap_md(K: int):
    """C — fit line on last K known TVT vs MD, extrapolate. Best K=20: pooled 107.77."""
    def predict(g, tw=None):
        pred = g[COL_TVT_INPUT].values.astype(float).copy()
        eval_mask = np.isnan(pred)
        if not eval_mask.any():
            return pred
        known_idx = np.where(~eval_mask)[0]
        if len(known_idx) < 2:
            if len(known_idx) == 1:
                pred[eval_mask] = pred[known_idx[0]]
            return pred
        idx_fit = known_idx[-K:] if len(known_idx) >= K else known_idx
        slope, intercept = np.polyfit(g[COL_MD].values[idx_fit], pred[idx_fit], 1)
        pred[eval_mask] = slope * g[COL_MD].values[eval_mask] + intercept
        return pred
    return predict


def make_smooth_anchor(K: int):
    """D — anchor = mean of last K known TVT values. K=1 reproduces A."""
    def predict(g, tw=None):
        pred = g[COL_TVT_INPUT].values.astype(float).copy()
        eval_mask = np.isnan(pred)
        if not eval_mask.any():
            return pred
        known_idx = np.where(~eval_mask)[0]
        if len(known_idx) == 0:
            return pred
        anchor = pred[known_idx[-K:]].mean() if len(known_idx) >= 1 else pred[known_idx[-1]]
        pred[eval_mask] = anchor
        return pred
    return predict
