"""Cache raw CSVs as parquet for fast loading in later phases.

One-shot. Run once after raw data is in place:
    python scripts/cache_parquet.py
    python scripts/cache_parquet.py --force    # overwrite existing parquets

Layout produced:
    data/interim/parquet/train_horizontal.parquet
    data/interim/parquet/train_typewell.parquet
    data/interim/parquet/test_horizontal.parquet
    data/interim/parquet/test_typewell.parquet
"""

from __future__ import annotations

import argparse
import time

from rogii_wellbore import data
from rogii_wellbore.config import RAW_DIR

PARQUET_DIR = RAW_DIR.parent / "interim" / "parquet"


def cache_one(name: str, loader, split: data.Split, force: bool) -> None:
    out = PARQUET_DIR / f"{name}.parquet"
    if out.exists() and not force:
        print(f"  skip {out.name}  (exists; use --force to overwrite)")
        return
    t0 = time.perf_counter()
    df = loader(split)
    df.to_parquet(out, index=False)
    dt = time.perf_counter() - t0
    print(f"  wrote {out.name}: shape={df.shape}, {dt:.1f}s, {out.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing parquets")
    args = ap.parse_args()

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {PARQUET_DIR}")

    jobs = [
        ("train_horizontal", data.load_horizontal, "train"),
        ("train_typewell", data.load_typewell, "train"),
        ("test_horizontal", data.load_horizontal, "test"),
        ("test_typewell", data.load_typewell, "test"),
    ]
    for name, loader, split in jobs:
        print(f"\n[{name}] from data/raw/{split}/")
        cache_one(name, loader, split, args.force)

    print("\ndone.")


if __name__ == "__main__":
    main()
