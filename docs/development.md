# Development notes

Moved from the original README (T1). The pipeline status below is historical; current status is in [PROGRESS.md](PROGRESS.md) and the command list in [CLAUDE.md](../CLAUDE.md). The licence is now decided: MIT (see [LICENSE](../LICENSE)).

Training project for a licence-clean detector of licence plates and vehicles. The technical source of truth is [docs/SPEC.md](SPEC.md); project workflow and safety constraints are in [CLAUDE.md](../CLAUDE.md).

## Development setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
make setup
make lint
make test
```

The initial scaffold started with no ML runtime dependencies. The data-ingestion dependencies are now installed and the T1 data pipeline is in progress; training, evaluation, export, and parity dependencies remain deferred until their project steps. No data, model weights, or generated run artifacts belong in Git.

## Data pipeline (T1 in progress)

The first data source is the documented Open Images V6 endpoints. Run `make data MAX_IMAGES=100` for a small bounded sample; the command refuses to run without an explicit positive cap. Even a small sample currently requires fetching the upstream annotation and metadata CSVs, which are large (the training box-annotation CSV alone is about 2.2 GB). The pipeline does not download the full image dataset unless the cap is set accordingly. It retains only records with the exact CC BY 2.0 metadata URL, an author, and a source landing URL; the upstream licence disclaimer and owner review requirement are recorded in [docs/adr/ADR-0002-open-images-v6-source.md](adr/ADR-0002-open-images-v6-source.md).

The generated datasets and manifests stay under the ignored `data/` directory. The confirmed split ratios are 80/10/10. T1 remains in progress: Open Images V6 and Kaggle adapters are implemented, but the remaining allowlisted source and combined dataset build are not complete.

The allowlisted Kaggle car-plate dataset adapter is also available. Example: `make data DATA_CONFIG=configs/data/kaggle_car_plate_detection.yaml MAX_IMAGES=100`. The image cap limits processing, not the archive download (about 213 MB). Kaggle lists the dataset as CC0 1.0; provenance and safeguards are in [docs/adr/ADR-0003-kaggle-car-plate-detection.md](adr/ADR-0003-kaggle-car-plate-detection.md). A bounded real-data run produced 91 usable images from 100 selected annotations after 9 near-duplicates were removed. The source provides plate boxes only—no vehicle boxes—so these images must not be treated as labeled vehicle negatives when training the two-class model.

CCPD processing expects the official archive to be extracted under `data/raw/ccpd/extracted/`; the local adapter does not download the full archive. Google Drive identifies the official archive as approximately 12 GB. For bounded Colab preparation, open a blank notebook and upload [scripts/colab_ccpd_sample.py](../scripts/colab_ccpd_sample.py) with the **folder-shaped Files pane > Upload to session storage** (not File > Upload notebook, which expects `.ipynb`). Run `!python /content/colab_ccpd_sample.py --max-images 100`. Before reading archive bytes, the helper checks the final response's exact size and runtime disk space; it refuses unknown or oversized responses and requires archive size plus an 8 GiB reserve. It streams the tar.xz and retains at most 100 deterministically selected JPEGs. After it finishes, run `from google.colab import files; files.download('/content/ccpd_work/ccpd_sample.tar.gz')` in a notebook code cell. Extract that bundle into `data/raw/ccpd/` so it creates `data/raw/ccpd/extracted/`, then run `make data DATA_CONFIG=configs/data/ccpd.yaml MAX_IMAGES=100` locally.

The confirmed combined-data policy caps CCPD at 30% of plate boxes. After all source subsets are built, `make data-combine` globally deduplicates them, enforces that cap, creates deterministic splits, and writes `data/processed/combined/stats.json`. Per-image `annotated_classes` records whether a source actually labels vehicles; CCPD and Kaggle are plate-only and are not vehicle negatives. The first Colab archive/sample run succeeded, but a later retry got an HTML response; current local code has synthetic coverage for bounded confirmation retries and quota diagnostics. No CCPD bundle is present in the workspace yet. If a browser-downloaded bundle already exists from the first run, use it rather than downloading the 12 GB archive again. See [docs/adr/ADR-0004-ccpd-dataset.md](adr/ADR-0004-ccpd-dataset.md) for source and storage safeguards.

