"""Train 5 OOF fold LGBM models and pickle them for Kaggle inference.

Mirrors the RNG order in run_oof_lgbm so saved models match the 15.42 OOF run.
"""

from __future__ import annotations

import pickle

import numpy as np

from rogii_wellbore.config import MODELS_DIR, RANDOM_SEED
from rogii_wellbore.cv import grouped_well_splits
from rogii_wellbore.data import list_wells, load_horizontal
from rogii_wellbore.models.lgbm import train_lgbm


def main() -> None:
    well_ids = list_wells("train")
    print(f"Loading {len(well_ids)} train wells from parquet…")
    df = load_horizontal("train", well_ids=well_ids, source="parquet")
    wells = {wid: g.reset_index(drop=True) for wid, g in df.groupby("well_id", sort=True)}

    row_well_ids = np.concatenate([np.full(len(wells[w]), w) for w in sorted(wells)])
    rng = np.random.default_rng(RANDOM_SEED)

    fold_models = []
    for fold_i, (train_idx, _) in enumerate(grouped_well_splits(row_well_ids, n_splits=5)):
        train_wells_in_fold = sorted(set(row_well_ids[train_idx]))
        n_es = max(1, int(0.1 * len(train_wells_in_fold)))
        es_wells = rng.choice(train_wells_in_fold, size=n_es, replace=False).tolist()
        train_only = [w for w in train_wells_in_fold if w not in set(es_wells)]
        print(f"  Fold {fold_i}: train={len(train_only)} es={len(es_wells)}")
        model = train_lgbm(wells, train_only, es_wells)
        print(f"    best_iter={model.best_iteration}")
        fold_models.append(model)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "lgbm_v1_fold_models.pkl"
    with out_path.open("wb") as f:
        pickle.dump(fold_models, f)
    print(f"\nSaved {len(fold_models)} models to {out_path}")
    print("LightGBM version (note for Kaggle):")
    import lightgbm

    print(f"  {lightgbm.__version__}")


if __name__ == "__main__":
    main()
