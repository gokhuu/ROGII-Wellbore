"""Validation harness from notebooks 02–04.

Used for evaluating heuristic baselines (those defined in `src.baselines`).
Trained ML models use a different evaluation path via `src.training` (OOF on folds).
"""
import numpy as np
import pandas as pd

from .utils import COL_TVT_INPUT, COL_TVT


def evaluate_baseline(predict_fn, train_df: pd.DataFrame, typewells: dict,
                      label: str = '', progress: int = 0):
    """Score a baseline against simulated-eval-zone ground truth.

    Args:
        predict_fn: function(g, tw) -> array of length len(g).
        train_df:   concatenated train DataFrame (from io.load_train_horizontal()).
        typewells:  dict mapping well_id -> typewell DataFrame.
        label:      optional headline label printed alongside the pooled RMSE.
        progress:   print progress every N wells (0 = silent).

    Returns:
        per_well: DataFrame, one row per well with [well, rmse, n_eval, mae, max_err].
        pooled:   sqrt(SSE_total / N_total) — matches leaderboard scoring.
    """
    rows, sse_total, n_total = [], 0.0, 0
    for i, (well, g) in enumerate(train_df.groupby('well', sort=False)):
        if progress and i and i % progress == 0:
            print(f'  ... {i} wells', flush=True)
        eval_mask = g[COL_TVT_INPUT].isna().values
        if not eval_mask.any():
            continue
        truth = g.loc[eval_mask, COL_TVT].values
        tw = typewells.get(well)
        pred = np.asarray(predict_fn(g, tw), dtype=float)
        if pred.shape[0] != len(g):
            raise ValueError(
                f'predict_fn returned {pred.shape[0]} preds for well {well} of length {len(g)}'
            )
        diff = pred[eval_mask] - truth
        sq = diff ** 2
        rows.append({
            'well':    well,
            'rmse':    float(np.sqrt(sq.mean())),
            'n_eval':  int(eval_mask.sum()),
            'mae':     float(np.abs(diff).mean()),
            'max_err': float(np.abs(diff).max()),
        })
        sse_total += float(sq.sum())
        n_total   += int(len(sq))
    res = pd.DataFrame(rows)
    pooled = float(np.sqrt(sse_total / n_total)) if n_total else float('nan')
    if label:
        print(f'{label:35s}  pooled RMSE = {pooled:8.3f}')
    return res, pooled
