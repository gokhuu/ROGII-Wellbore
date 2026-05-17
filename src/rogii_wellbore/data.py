"""Data loaders. Stubs for now — filled in during Phase 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rogii_wellbore.config import RAW_DIR


def load_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load training data. Implementation pending Phase 1 EDA."""
    raise NotImplementedError("Wire up after Phase 1 EDA confirms file layout.")


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load test data. Implementation pending Phase 1 EDA."""
    raise NotImplementedError("Wire up after Phase 1 EDA confirms file layout.")
