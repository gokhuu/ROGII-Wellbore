"""Sanity tests for data module."""

from __future__ import annotations

import pytest

from rogii_wellbore import data


def test_load_train_is_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        data.load_train()


def test_load_test_is_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        data.load_test()
