#!/usr/bin/env python
"""Generate test predictions and submission from a trained experiment.

Usage:
    python scripts/predict.py --config configs/xgb_baseline.yaml

Reads:
    experiments/<name>/fold_models.pkl
    cache/test_features.pkl
    data/raw/sample_submission.csv

Writes:
    submissions/<name>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import featurize_all_test
from src.inference import predict_test_fold_ensemble, write_submission
from src.utils import EXPERIMENTS_DIR, ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()

    ensure_dirs()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    name = config['name']
    exp_dir = EXPERIMENTS_DIR / name
    if not exp_dir.exists():
        raise FileNotFoundError(
            f'Experiment dir not found: {exp_dir}. '
            f'Run scripts/train.py --config {args.config} first.'
        )

    fold_models = pd.read_pickle(exp_dir / 'fold_models.pkl')
    test_feats = featurize_all_test()

    anchor_col = config['target'].get('residual_anchor_col')
    preds = predict_test_fold_ensemble(fold_models, test_feats,
                                        config['features'], anchor_col)

    write_submission(preds, test_feats, name)


if __name__ == '__main__':
    main()
