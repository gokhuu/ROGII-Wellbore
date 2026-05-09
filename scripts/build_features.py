#!/usr/bin/env python
"""Build feature caches.

Usage:
    python scripts/build_features.py            # use cache if present
    python scripts/build_features.py --force    # re-featurize from scratch

Reads:
    data/raw/train/*__horizontal_well.csv
    data/raw/train/*__typewell.csv
    data/raw/test/*__horizontal_well.csv
    data/raw/test/*__typewell.csv

Writes:
    cache/train_wells.pkl
    cache/train_typewells.pkl
    cache/train_features.pkl       (~3.78M rows x 22 cols, ~380 MB)
    cache/test_features.pkl        (14,151 rows x 21 cols)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import featurize_all_train, featurize_all_test
from src.utils import ensure_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--force', action='store_true',
                        help='Re-featurize even if cache exists')
    args = parser.parse_args()

    ensure_dirs()

    print('=== Train features ===')
    train_feats = featurize_all_train(force_reload=args.force)
    print(f'\n=== Test features ===')
    test_feats = featurize_all_test(force_reload=args.force)

    print(f'\nDone.')
    print(f'  Train: {train_feats.shape} ({train_feats["well"].nunique()} wells)')
    print(f'  Test:  {test_feats.shape}  ({test_feats["well"].nunique()} wells)')


if __name__ == '__main__':
    main()
