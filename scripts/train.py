#!/usr/bin/env python
"""Train a model from a config.

Usage:
    python scripts/train.py --config configs/xgb_baseline.yaml

Writes:
    experiments/<name>/oof_preds.npy
    experiments/<name>/fold_arr.npy
    experiments/<name>/fold_models.pkl
    experiments/<name>/metrics.json
    experiments/<name>/feature_importance.csv
    experiments/<name>/config.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import featurize_all_train
from src.cv import assign_folds
from src.training import train_oof
from src.utils import ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()

    ensure_dirs()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"=== Experiment: {config['name']} ===")
    if config.get('description'):
        print(config['description'].strip())

    # Load features
    train_feats = featurize_all_train()

    # Build target (residual or absolute)
    target_cfg = config['target']
    source = target_cfg['source_col']
    anchor = target_cfg.get('residual_anchor_col')
    if anchor:
        train_feats['_target'] = train_feats[source] - train_feats[anchor]
        target_col = '_target'
        # Sanity: predicting 0 residual should match baseline A
        rmse_zero = float(np.sqrt(np.mean(train_feats[target_col].astype('float64') ** 2)))
        print(f"Target: {source} - {anchor} (residual)")
        print(f"Sanity (predict 0 residual): pooled RMSE = {rmse_zero:.3f}  "
              f"(expect 15.910 for baseline A)")
    else:
        target_col = source
        print(f"Target: {source} (absolute)")

    # CV folds
    fold_arr = assign_folds(train_feats,
                            n_splits=config['cv']['n_folds'],
                            group_col=config['cv']['group_col'])
    print(f"Folds: {config['cv']['n_folds']} "
          f"(GroupKFold by {config['cv']['group_col']})")

    # Train
    train_oof(
        features_df=train_feats,
        feature_cols=config['features'],
        target_col=target_col,
        fold_arr=fold_arr,
        model_class=config['model']['class'],
        model_params=config['model']['params'],
        fit_params=config['model'].get('fit_params', {}),
        experiment_name=config['name'],
        config_path=args.config,
    )


if __name__ == '__main__':
    main()
