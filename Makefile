.PHONY: parity audit-prepare audit-finalize audit-corrections pseudo-labels train-data train export eval colab-bundle setup data data-combine holdout-prepare holdout-finalize baseline test lint

DATA_CONFIG ?= configs/data/open_images.yaml

setup:
	uv sync --locked
	PYTHONPATH=src uv run --locked python -m platedet.train.yolox_src

data:
	PYTHONPATH=src uv run --locked python -m platedet.data.build --config $(DATA_CONFIG) $(if $(MAX_IMAGES),--max-images $(MAX_IMAGES),)

data-combine:
	PYTHONPATH=src uv run --locked python -m platedet.data.combine --config configs/data/combined.yaml

holdout-prepare:
	PYTHONPATH=src uv run --locked python -m platedet.data.holdout prepare --config configs/holdout/open_images_curated.yaml

audit-prepare:
	PYTHONPATH=src uv run --locked python -m platedet.eval.label_audit prepare

audit-finalize:
	PYTHONPATH=src uv run --locked python -m platedet.eval.label_audit finalize

audit-corrections:
	PYTHONPATH=src uv run --locked python -m platedet.eval.label_audit corrections

holdout-finalize:
	PYTHONPATH=src uv run --locked python -m platedet.data.holdout finalize --config configs/holdout/open_images_curated.yaml

pseudo-labels:
	PYTHONPATH=src uv run --locked python -m platedet.train.pseudo_labels --config configs/train/pseudo_labels.yaml

train-data:
	PYTHONPATH=src uv run --locked python -m platedet.train.dataset --config $(CONFIG)

# Keeps a Mac awake while training (caffeinate); rerun with the same RUN_ID to resume.
KEEP_AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i -s)
train:
	PYTHONPATH=src $(KEEP_AWAKE) uv run --locked python -m platedet.train.train --config $(CONFIG) $(if $(RUN_ID),--run-id $(RUN_ID))

export:
	PYTHONPATH=src uv run --locked python -m platedet.export.onnx_export --run-dir runs/$(RUN)
	PYTHONPATH=src uv run --locked python -m platedet.export.tflite_export --run-dir runs/$(RUN)

parity:
	PYTHONPATH=src uv run --locked python -m platedet.parity.parity --run $(RUN) $(if $(LIMIT),--limit $(LIMIT))

eval:
	PYTHONPATH=src uv run --locked python -m platedet.eval.evaluate_run --run $(RUN) $(if $(TILES),--tiles $(TILES)) $(if $(SPLITS),--splits $(SPLITS)) $(if $(RESCORE),--rescore)

colab-bundle:
	PYTHONPATH=src uv run --locked python -m platedet.train.colab_bundle

baseline:
	PYTHONPATH=src uv run --locked python -m platedet.eval.baseline --config $(CONFIG)

test:
	PYTHONPATH=src uv run --locked python -m platedet.train.yolox_src >/dev/null
	PYTHONPATH=src uv run --locked pytest

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .