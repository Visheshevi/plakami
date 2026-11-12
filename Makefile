.PHONY: setup data data-combine holdout-prepare holdout-finalize baseline test lint

DATA_CONFIG ?= configs/data/open_images.yaml

setup:
	uv sync --locked

data:
	PYTHONPATH=src uv run --locked python -m platedet.data.build --config $(DATA_CONFIG) $(if $(MAX_IMAGES),--max-images $(MAX_IMAGES),)

data-combine:
	PYTHONPATH=src uv run --locked python -m platedet.data.combine --config configs/data/combined.yaml

holdout-prepare:
	PYTHONPATH=src uv run --locked python -m platedet.data.holdout prepare --config configs/holdout/open_images_curated.yaml

holdout-finalize:
	PYTHONPATH=src uv run --locked python -m platedet.data.holdout finalize --config configs/holdout/open_images_curated.yaml

baseline:
	PYTHONPATH=src uv run --locked python -m platedet.eval.baseline --config $(CONFIG)

test:
	PYTHONPATH=src uv run --locked pytest

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .