# CLAUDE.md — Plate & Vehicle Detector (training project)

This project trains and publishes a model that **finds licence plates and vehicles in a photo**. Given an image, it returns a box around each plate and each vehicle, with a confidence score. Typical uses are blurring plates for privacy and checking that a photo contains a vehicle.

- **Why build our own:** existing open plate detectors are either GPL/AGPL-derived, trained on undocumented or non-commercial data, or too large for phones.
- **What we build:** one model (YOLOX-Tiny, Apache-2.0), trained only on commercially usable images, exported as **TFLite** (phones and edge devices) and **ONNX** (servers and desktops).
- **What we deliver:** versioned GitHub releases with the model files, reference pre/post-processing code with test vectors, accuracy numbers and licence records. Anyone can use a release without this repo's training code or data.

## Source of truth

- **`docs/SPEC.md`** is the full spec. Read the sections relevant to your task before writing code. When the spec and this file disagree, the spec wins.
- **`docs/PROGRESS.md`** holds the current step, the task checklist, the features log and session notes. **Read it at the start of every session.** (Created in T0.)
- **`docs/adr/`** holds architecture decision records. Any deviation from the spec, and any new data source, needs a short ADR (context, decision, consequences).

## Build order (SPEC §10)

T0 Scaffold → T1 Data → T2 Baseline → T3 Train v0 → T4 Iterate → T5 Export + parity → T6 Release v1.0.0.

## Working agreement

1. **One task at a time.** Work only on the current task in `docs/PROGRESS.md`. Don't start the next one without the owner's go-ahead.
2. **Plan before coding** at the start of each step (T0, T1, …) and for any task touching more than ~5 files: list the steps, the files and your open questions, then wait for approval.
3. **Ask, don't guess,** on owner decisions: costs, accounts, data sources, licences, the targets in SPEC §6, and anything in SPEC §11. For small technical choices, pick the simplest option, write an ADR, and continue.
4. **Tests come with the code.** A task is done only when `make test` and `make lint` pass.
5. **Commit after each task** with Conventional Commits (`feat(data): …`, `fix(export): …`) on a branch named `<task-id>-<slug>` (e.g. `t1-open-images-downloader`).
6. **End of task / session:** tick the box in `docs/PROGRESS.md`, update "Next up", and add a dated line to "Session notes".
7. **Features log:** after each feature is built, add an entry to the "Features" section of `docs/PROGRESS.md`: the feature, what it does, and the branch it started on.

## About the owner

- Background: C++/Java and warehouse automation; now manages a data science team. Comfortable with ML concepts and metrics, so keep ML explanations brief and focus on decisions.
- Prefers direct, concise summaries: what changed, how it was verified, what's next.
- For every training run, report: config used, metrics vs targets (SPEC §6), and the worst misses (with image paths).

## Commands (created in T0; keep this list current)

```
make setup      # uv sync: create the venv with pinned dependencies
make data       # download allowlisted sources, build COCO dataset + manifest, licence check
make holdout-prepare   # download holdout candidates + review sheets (ADR-0008)
make holdout-finalize  # build the holdout from configs/holdout/review.csv
make baseline CONFIG=configs/baseline/<name>.yaml  # evaluate an off-the-shelf ONNX model on val/test
make pseudo-labels     # vehicle pseudo-labels for plate-only images (ADR-0011)
make train-data CONFIG=configs/train/<name>.yaml  # YOLOX training labels at that size
make colab-bundle      # tar for Colab training (code, YOLOX, checkpoint, data); needs a clean commit
make train CONFIG=configs/<name>.yaml RUN_ID=<id>   # train into runs/<id>/; Apple GPU locally (ADR-0012) or CUDA; rerun to resume
make eval RUN=<run_id> [TILES=2] [SPLITS=val] [RESCORE=1]  # metrics + error gallery on val/test/holdout; TILES = tiled detection (ADR-0013); RESCORE reuses saved predictions
make audit-prepare     # plate label audit sheets for test/holdout (ADR-0014)
make audit-finalize    # check the review and write configs/audit/plate_labels.csv
make export RUN=<run_id>                # ONNX (checked against PyTorch) + TFLite FP16/INT8 via pinned onnx2tf (ADR-0016)
make test-vectors [CHECK=1]              # porting test vectors for the release candidate (SPEC §8)
make try IMAGES=<folder> [OUT=<folder>]  # run the release candidate on your own photos, save boxed copies
make parity RUN=<run_id> [LIMIT=n]      # ONNX vs TFLite parity (single pass + 2x2 tiles), speed, size
make test       # pytest (unit + smoke train/export on tests/data/smoke)
make lint       # ruff check + ruff format --check
```

## Conventions

- **Python 3.12 + uv.** Add a dependency only with a one-line justification in the commit, and check its licence first (no GPL/AGPL).
- **Configs, not code edits,** for experiments: every run is driven by a YAML file in `configs/`, copied into `runs/<run_id>/`.
- **Reproducibility:** fixed seeds; every run records the git commit, config, dataset manifest hash and checkpoint SHA-256.
- **Pre/post-processing** lives once in `src/platedet/postprocess.py` and is documented in `docs/preprocessing.md` with test vectors. Never duplicate it with small differences.
- **Splits are deterministic** (hash of `image_id`) and never change between runs.
- Long jobs (downloads, training) run in the background with logs in `runs/` or `data/logs/`. Report progress; don't block.

## Hard rules: never do these

- **Never use an image, dataset, checkpoint or library that isn't on the SPEC §4.1 allowlist or has an unknown licence.** Never use Ultralytics (AGPL), WongKinYiu YOLOv9 (GPL), DEIMv2, InsightFace, WIDER FACE, Objects365, UFPR/RodoSol-ALPR, platesmania or unlicensed Roboflow data.
- **Never train, tune thresholds or calibrate on `test` or `holdout`.** They are evaluation-only.
- **Never commit** datasets, images, annotations, checkpoints, model files or TensorBoard logs. They stay in `data/` and `runs/` (git-ignored).
- **Never sign up for, enable or pay for** compute, storage or any service (incl. Colab Pro, cloud GPUs, paid datasets). Ask the owner.
- **Never read, print or commit secrets** (`.env*`, API tokens, Kaggle credentials). Use `.env.example` with placeholders.
- **Never publish a release** or push a tag without the owner approving the evaluation report.
- **Never weaken a test or lower a target** to make it pass. Report the gap and ask.
- **Never edit `docs/SPEC.md`** without the owner asking. Propose changes in your summary instead.
- **Always create a new branch for a feature.** Never develop on `main`.

## Definition of done (every task)

- [ ] `make lint` is clean and `make test` passes
- [ ] New data sources have an ADR, an allowlist row and manifest licence checks
- [ ] Runs record commit, config, manifest hash and checkpoint hash
- [ ] `docs/PROGRESS.md` is updated (checkbox, Next up, features log, session note)
- [ ] Committed on the task branch with a Conventional Commit message
