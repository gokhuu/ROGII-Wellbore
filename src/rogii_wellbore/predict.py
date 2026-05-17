"""Build submission CSV. Stub for now."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rogii_wellbore.config import SUBMISSIONS_DIR


def write_submission(preds: pd.DataFrame, name: str) -> Path:
    """Write submission CSV to submissions/<name>.csv."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / f"{name}.csv"
    preds.to_csv(out, index=False)
    return out
