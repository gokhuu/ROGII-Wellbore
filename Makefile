.PHONY: setup lint format test eda mlflow-ui train-constant train-lgbm train-seq submit clean

setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest

eda:
	uv run jupyter lab

mlflow-ui:
	uv run mlflow ui --backend-store-uri ./mlruns --port 5000

train-constant:
	uv run rogii train-constant

train-lgbm:
	uv run rogii train-lgbm

train-seq:
	uv run rogii train-seq

submit:
	uv run rogii submit

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
	rm -rf build dist *.egg-info
