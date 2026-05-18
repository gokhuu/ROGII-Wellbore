"""Data loaders for ROGII wellbore competition.

Layout:
    data/raw/{split}/{well_id}__horizontal_well.csv
    data/raw/{split}/{well_id}__typewell.csv
    data/raw/{split}/{well_id}.png            (train only, ignored)
    data/raw/sample_submission.csv

`split` is "train" or "test".
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from rogii_wellbore.config import RAW_DIR

Split = Literal["train", "test"]

_HORIZONTAL_SUFFIX = "__horizontal_well.csv"
_TYPEWELL_SUFFIX = "__typewell.csv"


def _split_dir(split: Split, raw_dir: Path) -> Path:
    d = raw_dir / split
    if not d.is_dir():
        raise FileNotFoundError(f"Expected directory: {d}")
    return d


def list_wells(split: Split, raw_dir: Path = RAW_DIR) -> list[str]:
    """Sorted well IDs in `split` for which both CSVs exist."""
    d = _split_dir(split, raw_dir)
    h = {p.name.removesuffix(_HORIZONTAL_SUFFIX) for p in d.glob(f"*{_HORIZONTAL_SUFFIX}")}
    t = {p.name.removesuffix(_TYPEWELL_SUFFIX) for p in d.glob(f"*{_TYPEWELL_SUFFIX}")}
    return sorted(h & t)


def load_horizontal(
    split: Split,
    well_ids: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Concatenated horizontal_well CSVs as a long DataFrame.

    Adds two leading columns:
        well_id: str  — 8-char hex prefix from the filename.
        row_idx: int  — within-well row index. Matches the integer in
                        sample_submission `id` (`{well_id}_{row_idx}`).

    Schema notes:
        Train wells:  MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA,
                      TVT, GR, TVT_input
        Test wells:   MD, X, Y, Z, GR, TVT_input
    """
    d = _split_dir(split, raw_dir)
    ids = well_ids if well_ids is not None else list_wells(split, raw_dir)
    frames: list[pd.DataFrame] = []
    for wid in ids:
        df = pd.read_csv(d / f"{wid}{_HORIZONTAL_SUFFIX}")
        df.insert(0, "row_idx", df.index.to_numpy())
        df.insert(0, "well_id", wid)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["well_id", "row_idx"])
    return pd.concat(frames, ignore_index=True)


def load_typewell(
    split: Split,
    well_ids: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Concatenated typewell CSVs as a long DataFrame.

    Adds `well_id` and `row_idx` leading columns.
    Schema: TVT, GR, Geology.
    """
    d = _split_dir(split, raw_dir)
    ids = well_ids if well_ids is not None else list_wells(split, raw_dir)
    frames: list[pd.DataFrame] = []
    for wid in ids:
        df = pd.read_csv(d / f"{wid}{_TYPEWELL_SUFFIX}")
        df.insert(0, "row_idx", df.index.to_numpy())
        df.insert(0, "well_id", wid)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["well_id", "row_idx"])
    return pd.concat(frames, ignore_index=True)


def load_submission_template(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load sample_submission.csv. Columns: id, tvt."""
    return pd.read_csv(raw_dir / "sample_submission.csv")
