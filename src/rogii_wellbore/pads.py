"""Pad clustering — group wells by wellhead X/Y proximity for pessimistic CV."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from .config import RANDOM_SEED


def wellhead_xy(wells: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per well with X/Y at minimum MD (the wellhead/surface position)."""
    rows = []
    for wid, w in wells.items():
        idx = w["MD"].idxmin()
        rows.append({"well_id": wid, "X": float(w.loc[idx, "X"]), "Y": float(w.loc[idx, "Y"])})
    return pd.DataFrame(rows).set_index("well_id")


def assign_pads(
    wells: dict[str, pd.DataFrame],
    n_pads: int = 20,
    seed: int = RANDOM_SEED,
) -> dict[str, int]:
    """Cluster wells by wellhead X/Y into n_pads groups. Returns well_id → pad_id."""
    xy = wellhead_xy(wells)
    km = KMeans(n_clusters=n_pads, random_state=seed, n_init=10)
    labels = km.fit_predict(xy[["X", "Y"]].to_numpy())
    return {wid: int(lab) for wid, lab in zip(xy.index, labels, strict=True)}
