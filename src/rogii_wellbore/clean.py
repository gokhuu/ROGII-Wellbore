"""clean.py — deterministic, config-driven cleaning for ROGII wellbore data.

Single source of truth is data/interim/clean_config.json (written by 01_eda).
This module APPLIES the recorded decisions; it makes no new judgment calls.
Every threshold, fill rule, and canonical mapping comes from the config so the
EDA and the cleaner can never silently diverge.

Public API
----------
load_config(path) -> dict
clean_horizontal(df, cfg, split) -> DataFrame      # one or many wells
clean_typewell(df, cfg) -> DataFrame
clean_split(raw_dir, out_dir, cfg, split)          # per-well files in -> out
assert_no_leakage(feature_cols, cfg)               # guard for the model step
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# helpers (each maps to one recorded EDA decision)
# ----------------------------------------------------------------------
def _clip_gr(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Section 2: clip GR spikes to the physical valid range (not delete)."""
    return s.clip(lower=lo, upper=hi)


def _blank_flatlines(df: pd.DataFrame, gr_col: str, thr: int) -> pd.Series:
    """Section 2: set stuck-tool runs (>= thr identical consecutive) to NaN so the
    fill policy handles them rather than the model learning a fake flat zone.
    Operates within the single well represented by df (already row-ordered)."""
    g = df[gr_col]
    if thr is None or thr <= 0 or len(g) == 0:
        return g
    # run id increments whenever the value changes
    same = (g == g.shift()) & g.notna()
    run_id = (~same).cumsum()
    run_len = run_id.map(run_id.value_counts())
    flat = (run_len >= thr) & g.notna()
    out = g.mask(flat, np.nan)
    return out


def _interp_capped(s: pd.Series, max_run: int) -> tuple[pd.Series, pd.Series]:
    """Section 2 nan_fill: linear-interpolate NaN runs <= max_run; leave longer
    gaps as NaN. Returns (filled, was_missing_flag). Assumes within-well order.
    Never zero-fills (zero is a real low-radioactivity reading)."""
    was_missing = s.isna()
    if not was_missing.any():
        return s, was_missing
    # identify NaN run lengths
    m = s.isna().values.astype(int)
    # run length for each NaN position
    run_len = np.zeros(len(m), dtype=int)
    i = 0
    while i < len(m):
        if m[i] == 0:
            i += 1
            continue
        j = i
        while j < len(m) and m[j] == 1:
            j += 1
        run_len[i:j] = j - i
        i = j
    short = (run_len > 0) & (run_len <= max_run)
    filled = s.copy()
    # interpolate everything, then only keep interpolated values on short gaps
    interp = s.interpolate(method="linear", limit_direction="both")
    filled[short] = interp[short]
    # long gaps remain NaN; the missing flag marks ALL originally-missing rows
    return filled, was_missing


def _per_well_zscore(s: pd.Series) -> tuple[pd.Series, float, float]:
    """Section 3: per-well z-score. Fit on this well's finite values only.
    Rows where GR is still NaN (long gaps left unfilled) get GR_z = 0.0 (the
    well mean in z-space); the companion gr_missing flag is what tells the model
    those values are imputed. This keeps GR_z finite everywhere so downstream
    tensors never carry NaN. Returns (z, mu, sd)."""
    finite = s[np.isfinite(s)]
    mu = float(finite.mean()) if len(finite) else 0.0
    sd = float(finite.std()) if len(finite) else 1.0
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    z = (s - mu) / sd
    return z.fillna(0.0), mu, sd


def _flag_teleports(df: pd.DataFrame, thr: float) -> pd.Series:
    """Section 4: flag rows whose 3D displacement from the previous row exceeds
    thr (bad survey rows). Within-well order assumed."""
    d = np.sqrt(df["X"].diff() ** 2 + df["Y"].diff() ** 2 + df["Z"].diff() ** 2)
    return (d > thr).fillna(False)


def _canonicalize_geology(s: pd.Series, canonical_set: set) -> pd.Series:
    """Section 3b.3: collapse raw Geology labels to the canonical set; everything
    rare/free-text -> 'OTHER'; NaN stays NaN (unlabeled interval)."""
    return pd.Series(
        np.where(
            s.isna(),
            np.nan,
            np.where(s.isin(canonical_set), s, "OTHER"),
        ),
        index=s.index,
    )


# ----------------------------------------------------------------------
# per-frame cleaners
# ----------------------------------------------------------------------
def clean_horizontal(df: pd.DataFrame, cfg: dict, split: str) -> pd.DataFrame:
    """Apply all horizontal-well cleaning decisions. Works on one or many wells
    (grouped by well_id). Adds: GR (cleaned), GR_raw, GR_z, gr_missing,
    gr_flatline, traj_teleport. Leaves target/boundary columns untouched in
    train; they are simply absent in test."""
    gr_cfg = cfg["gr"]
    cfg.get("gr_normalization", {})
    traj_cfg = cfg["trajectory"]
    flatline_thr = gr_cfg.get("flatline_run_flag_threshold")
    max_run = gr_cfg["nan_fill"]["max_interp_run"]

    out_parts = []
    for _well_id, g in df.sort_values(["well_id", "row_idx"]).groupby("well_id", sort=False):
        g = g.copy()
        g["GR_raw"] = g["GR"]
        # 1. clip spikes
        if gr_cfg.get("clip_to_valid_range", True):
            g["GR"] = _clip_gr(g["GR"], gr_cfg["valid_min"], gr_cfg["valid_max"])
        # 2. blank flatlines (stuck tool) so they get treated as gaps
        before_flat = g["GR"].isna()
        g["GR"] = _blank_flatlines(g, "GR", flatline_thr)
        g["gr_flatline"] = g["GR"].isna() & ~before_flat
        # 3. capped interpolation + missing flag
        g["GR"], g["gr_missing"] = _interp_capped(g["GR"], max_run)
        # 4. per-well z-score (baked in; raw kept above)
        g["GR_z"], _mu, _sd = _per_well_zscore(g["GR"])
        # 5. trajectory teleport flag
        g["traj_teleport"] = _flag_teleports(g, traj_cfg["teleport_step_threshold"])
        out_parts.append(g)

    cleaned = pd.concat(out_parts, ignore_index=True)
    return cleaned


def clean_typewell(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply typewell cleaning: GR clip + per-well z-score (same recipe as the
    lateral, fit independently per well, per source) and Geology canonicalization.
    Typewell GR has no NaNs in practice but we guard anyway."""
    gr_cfg = cfg["gr"]
    geo_cfg = cfg.get("geology", {})
    # canonical_set is written by 01_eda Section 3b.3; fall back to the six
    # boundary formations + consistent units if an older config lacks it.
    canonical = set(
        geo_cfg.get(
            "canonical_set",
            [
                "ANCC",
                "ASTNU",
                "ASTNL",
                "EGFDU",
                "EGFDL",
                "BUDA",
                "OLMOS",
                "MNSS",
                "UPSN",
                "LBHL",
                "LTHL",
                "LTGT",
            ],
        )
    )

    out_parts = []
    for _well_id, g in df.groupby("well_id", sort=False):
        g = g.copy()
        g["GR_raw"] = g["GR"]
        if gr_cfg.get("clip_to_valid_range", True):
            g["GR"] = _clip_gr(g["GR"], gr_cfg["valid_min"], gr_cfg["valid_max"])
        # typewell gaps are rare; interpolate fully (it is a reference curve)
        g["GR"] = g["GR"].interpolate(method="linear", limit_direction="both")
        g["GR_z"], _mu, _sd = _per_well_zscore(g["GR"])
        if "Geology" in g.columns:
            g["Geology_canon"] = _canonicalize_geology(g["Geology"], canonical)
        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


# ----------------------------------------------------------------------
# leakage guard (imported by the model step)
# ----------------------------------------------------------------------
def assert_no_leakage(feature_cols, cfg: dict) -> None:
    forbidden = set(cfg["schema"]["forbidden_features"])
    bad = set(feature_cols) & forbidden
    if bad:
        raise ValueError(
            f"Leakage: train-only columns used as features: {sorted(bad)}. "
            f"Forbidden set: {sorted(forbidden)}"
        )


# ----------------------------------------------------------------------
# IO: per-well files in -> per-well cleaned files out (mirrors raw layout)
# ----------------------------------------------------------------------
_SUFFIX = {"horizontal": "horizontal_well", "typewell": "typewell"}


def _well_id_from(path: Path) -> str:
    return path.name.split("__")[0]


def clean_split(raw_dir: str | Path, out_dir: str | Path, cfg: dict, split: str) -> dict:
    """Read every per-well CSV under raw_dir/<split>, clean it, and write a
    matching cleaned CSV under out_dir/<split>. Returns a small report dict."""
    raw_dir = Path(raw_dir) / split
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    hz_files = sorted(raw_dir.glob(f"*__{_SUFFIX['horizontal']}.csv"))
    tw_files = sorted(raw_dir.glob(f"*__{_SUFFIX['typewell']}.csv"))
    report = {"split": split, "n_horizontal": 0, "n_typewell": 0, "rows_in": 0, "rows_out": 0}

    for p in hz_files:
        wid = _well_id_from(p)
        df = pd.read_csv(p)
        df.insert(0, "well_id", wid)
        df = df.sort_values("MD").reset_index(drop=True)
        df["row_idx"] = np.arange(len(df))
        n_in = len(df)
        cleaned = clean_horizontal(df, cfg, split)
        assert len(cleaned) == n_in, f"row count changed for {wid}: {n_in}->{len(cleaned)}"
        cleaned.to_csv(out_dir / p.name, index=False)
        report["n_horizontal"] += 1
        report["rows_in"] += n_in
        report["rows_out"] += len(cleaned)

    for p in tw_files:
        wid = _well_id_from(p)
        df = pd.read_csv(p)
        df.insert(0, "well_id", wid)
        df["row_idx"] = np.arange(len(df))
        cleaned = clean_typewell(df, cfg)
        cleaned.to_csv(out_dir / p.name, index=False)
        report["n_typewell"] += 1

    return report
