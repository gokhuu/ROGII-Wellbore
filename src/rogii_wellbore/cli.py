"""Typer CLI: `rogii <command>`."""

from __future__ import annotations

import typer

from rogii_wellbore.logging_config import logger

app = typer.Typer(help="ROGII Wellbore Geology CLI.")


@app.command()
def hello(name: str = "world") -> None:
    """Sanity check command."""
    logger.info(f"hello, {name}")


@app.command("train-constant")
def train_constant() -> None:
    """Phase 2 — constant baseline."""
    raise NotImplementedError("Phase 2.")


@app.command("train-lgbm")
def train_lgbm() -> None:
    """Phase 2/3 — LightGBM tabular baseline."""
    raise NotImplementedError("Phase 2/3.")


@app.command("train-seq")
def train_seq() -> None:
    """Phase 4 — sequence model."""
    raise NotImplementedError("Phase 4.")


@app.command()
def submit() -> None:
    """Build submission CSV from latest model."""
    raise NotImplementedError("Phase 2+.")


if __name__ == "__main__":
    app()
