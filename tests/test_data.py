"""Tests for data loaders. Uses synthetic fixtures (CI has no real data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rogii_wellbore import data

TRAIN_COLS = [
    "MD",
    "X",
    "Y",
    "Z",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
    "TVT",
    "GR",
    "TVT_input",
]
TEST_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
TYPEWELL_COLS = ["TVT", "GR", "Geology"]


@pytest.fixture
def fake_raw(tmp_path: Path) -> Path:
    """Mirror the real raw/ layout: 2 train wells, 1 test well."""
    raw = tmp_path / "raw"
    (raw / "train").mkdir(parents=True)
    (raw / "test").mkdir(parents=True)

    for wid, n in [("aaa", 5), ("bbb", 4)]:
        pd.DataFrame({c: range(n) for c in TRAIN_COLS}).to_csv(
            raw / "train" / f"{wid}__horizontal_well.csv", index=False
        )
        pd.DataFrame({c: range(n + 1) for c in TYPEWELL_COLS}).to_csv(
            raw / "train" / f"{wid}__typewell.csv", index=False
        )

    pd.DataFrame({c: range(3) for c in TEST_COLS}).to_csv(
        raw / "test" / "ccc__horizontal_well.csv", index=False
    )
    pd.DataFrame({c: range(3) for c in TYPEWELL_COLS}).to_csv(
        raw / "test" / "ccc__typewell.csv", index=False
    )

    pd.DataFrame({"id": ["aaa_2", "aaa_3"], "tvt": [0.0, 0.0]}).to_csv(
        raw / "sample_submission.csv", index=False
    )
    return raw


def test_list_wells_train(fake_raw: Path) -> None:
    assert data.list_wells("train", raw_dir=fake_raw) == ["aaa", "bbb"]


def test_list_wells_test(fake_raw: Path) -> None:
    assert data.list_wells("test", raw_dir=fake_raw) == ["ccc"]


def test_load_horizontal_train_shape(fake_raw: Path) -> None:
    df = data.load_horizontal("train", raw_dir=fake_raw)
    # 5 + 4 = 9 rows; 13 data cols + well_id + row_idx = 15
    assert df.shape == (9, 15)
    assert df.columns[:2].tolist() == ["well_id", "row_idx"]


def test_load_horizontal_test_has_reduced_schema(fake_raw: Path) -> None:
    df = data.load_horizontal("test", raw_dir=fake_raw)
    assert df.shape == (3, 8)  # 6 data cols + well_id + row_idx
    assert "TVT" not in df.columns
    assert "ANCC" not in df.columns


def test_load_horizontal_row_idx_resets_per_well(fake_raw: Path) -> None:
    df = data.load_horizontal("train", raw_dir=fake_raw)
    for _, grp in df.groupby("well_id"):
        assert grp["row_idx"].tolist() == list(range(len(grp)))


def test_load_horizontal_well_id_filter(fake_raw: Path) -> None:
    df = data.load_horizontal("train", well_ids=["bbb"], raw_dir=fake_raw)
    assert df["well_id"].unique().tolist() == ["bbb"]
    assert len(df) == 4


def test_load_typewell(fake_raw: Path) -> None:
    df = data.load_typewell("train", raw_dir=fake_raw)
    # well aaa has 6 rows, bbb has 5 → 11 total; 3 data + well_id + row_idx
    assert df.shape == (11, 5)
    assert set(df.columns) >= {"well_id", "row_idx", "TVT", "GR", "Geology"}


def test_load_submission_template(fake_raw: Path) -> None:
    sub = data.load_submission_template(raw_dir=fake_raw)
    assert sub.columns.tolist() == ["id", "tvt"]
    assert len(sub) == 2


def test_missing_split_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        data.list_wells("train", raw_dir=tmp_path / "nonexistent")
