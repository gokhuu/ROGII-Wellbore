"""Test prediction (fold-ensemble) and submission writing."""
import numpy as np
import pandas as pd

from .utils import RAW_DIR, SUB_DIR


def predict_test_fold_ensemble(fold_models, test_features: pd.DataFrame,
                                feature_cols: list, anchor_col: str = None) -> np.ndarray:
    """Mean of fold-model predictions on test_features.

    If `anchor_col` is given, treat predictions as residual and add the anchor
    back to recover absolute target values.
    """
    X_test = test_features[feature_cols].astype('float32').values
    preds = np.mean([m.predict(X_test) for m in fold_models], axis=0)
    if anchor_col is not None:
        preds = preds + test_features[anchor_col].astype('float64').values
    return preds


def write_submission(predictions: np.ndarray, test_features: pd.DataFrame,
                     experiment_name: str, sample_submission_path=None):
    """Map predictions onto sample_submission row order; write CSV to submissions/."""
    if sample_submission_path is None:
        sample_submission_path = RAW_DIR / 'sample_submission.csv'
    ss = pd.read_csv(sample_submission_path)
    id_col, val_col = ss.columns[0], ss.columns[1]

    pred_dict = {f'{w}_{int(r)}': float(p) for w, r, p in
                 zip(test_features['well'].values,
                     test_features['row_idx'].values,
                     predictions)}

    out = ss.copy()
    out[val_col] = out[id_col].map(pred_dict)
    n_missing = int(out[val_col].isna().sum())
    if n_missing:
        raise ValueError(
            f'{n_missing} sample_submission rows have no prediction. '
            f'Check well/row_idx alignment.'
        )

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUB_DIR / f'{experiment_name}.csv'
    out.to_csv(sub_path, index=False)
    print(f'Saved: {sub_path}')
    return sub_path
