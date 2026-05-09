"""Generic OOF training loop.

Model-class agnostic: pass `model_class='xgboost'` (or 'lightgbm' once added) and
its constructor params via the config. Saves all artifacts under
`experiments/<experiment_name>/`.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import EXPERIMENTS_DIR


def _get_constructor(model_class: str):
    if model_class == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor
    elif model_class == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMRegressor
    raise ValueError(f'Unknown model_class: {model_class!r}')


def train_oof(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    fold_arr: np.ndarray,
    model_class: str,
    model_params: dict,
    fit_params: dict,
    experiment_name: str,
    config_path: Path = None,
):
    """Train fold-wise models with early stopping. Save artifacts to experiments/<name>/.

    Artifacts saved:
      - oof_preds.npy        per-row OOF prediction (residual or absolute, matches target_col)
      - fold_arr.npy         the fold assignments used
      - fold_models.pkl      list of trained models
      - metrics.json         pooled OOF RMSE + per-fold breakdown + config snapshot
      - feature_importance.csv  (if model exposes feature_importances_)
      - config.yaml          copy of the config that produced this run

    Returns:
        fold_models: list[Model]
        oof_preds:   np.ndarray of shape (len(features_df),)
        metrics:     dict
    """
    n_folds = int(fold_arr.max()) + 1
    X = features_df[feature_cols].astype('float32').values
    y = features_df[target_col].astype('float32').values

    oof_preds = np.zeros(len(features_df), dtype=np.float64)
    fold_models = []
    fold_metrics = []

    exp_dir = EXPERIMENTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    Constructor = _get_constructor(model_class)
    base_seed = model_params.get('random_state', 42)

    for fold in range(n_folds):
        tr_mask = fold_arr != fold
        va_mask = fold_arr == fold
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_va, y_va = X[va_mask], y[va_mask]

        params = dict(model_params)
        params['random_state'] = base_seed + fold
        model = Constructor(**params)

        print(f'\n--- Fold {fold}: train={len(y_tr):,} val={len(y_va):,} ---')
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], **fit_params)

        pred_va = model.predict(X_va)
        oof_preds[va_mask] = pred_va
        rmse = float(np.sqrt(np.mean((pred_va - y_va) ** 2)))
        best_iter = getattr(model, 'best_iteration', None)
        fold_metrics.append({
            'fold': fold,
            'rmse': rmse,
            'best_iteration': int(best_iter) if best_iter is not None else None,
            'n_train': int(tr_mask.sum()),
            'n_val':   int(va_mask.sum()),
        })
        fold_models.append(model)
        print(f'Fold {fold}: RMSE={rmse:.3f}  best_iter={best_iter}')

    pooled_rmse = float(np.sqrt(np.mean((oof_preds - y) ** 2)))

    # Save artifacts
    np.save(exp_dir / 'oof_preds.npy', oof_preds)
    np.save(exp_dir / 'fold_arr.npy',  fold_arr)
    pd.to_pickle(fold_models, exp_dir / 'fold_models.pkl')
    if config_path is not None:
        shutil.copy2(config_path, exp_dir / 'config.yaml')

    metrics = {
        'experiment_name': experiment_name,
        'model_class':     model_class,
        'n_features':      len(feature_cols),
        'features':        list(feature_cols),
        'target_col':      target_col,
        'n_folds':         n_folds,
        'pooled_oof_rmse': pooled_rmse,
        'per_fold':        fold_metrics,
    }
    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    if hasattr(fold_models[0], 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': feature_cols,
            **{f'fold_{i}': fold_models[i].feature_importances_ for i in range(n_folds)},
        })
        imp['mean'] = imp[[f'fold_{i}' for i in range(n_folds)]].mean(axis=1)
        imp['std']  = imp[[f'fold_{i}' for i in range(n_folds)]].std(axis=1)
        imp = imp.sort_values('mean', ascending=False)
        imp.to_csv(exp_dir / 'feature_importance.csv', index=False)

    print(f'\n=== Pooled OOF RMSE: {pooled_rmse:.3f} ===')
    print(f'Saved to: {exp_dir}')
    return fold_models, oof_preds, metrics
