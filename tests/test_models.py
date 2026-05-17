"""Smoke test that model modules import."""

from __future__ import annotations


def test_models_import() -> None:
    from rogii_wellbore.models import constant, lgbm, sequence  # noqa: F401
