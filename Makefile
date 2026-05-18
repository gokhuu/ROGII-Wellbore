.PHONY: setup lint format test eda mlflow-ui train-constant train-lgbm train-seq submit clean

setup:
	conda env update -f environment.yml --prune
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests
	mypy

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

eda:
	jupyter lab

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

train-constant:
	rogii train-constant

train-lgbm:
	rogii train-lgbm

train-seq:
	rogii train-seq

submit:
	rogii submit

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
	rm -rf build dist *.egg-info
