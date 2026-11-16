# Project Progress

## Current step: T6 — Release v1.0.0 (branch `t6-release`)

### T0 — Scaffold

- [x] Project layout, uv, Makefile, Ruff, pytest, and CI workflow created
- [x] `make setup`, `make lint`, and `make test` pass locally
- [ ] Hosted CI run (requires Git and a remote; deferred by owner)

### T1 — Data (done)

- [x] Confirmed 80/10/10 deterministic split ratios
- [x] Implemented first source: capped Open Images V6 parsing, download, COCO conversion, manifest, attribution, and pHash deduplication
- [x] Implemented Kaggle CC0 car-plate source adapter with capped processing and safe archive/XML validation
- [x] Implemented CCPD filename parser and capped builder for an already-extracted official dataset
- [x] Added combined dataset builder with global pHash deduplication, deterministic resplitting, and confirmed 30% CCPD plate-box cap
- [x] Track per-image annotated classes so plate-only sources are not mislabeled as vehicle negatives
- [x] Added asset-free synthetic tests and checked dependencies for permissive licences
- [x] Owner approved metadata-listed CC BY 2.0 for the bounded sample; upstream disclaimer remains documented
- [x] Implement public allowlisted source adapters (Open Images, Kaggle, CCPD); owner-photo ingestion remains
- [x] Run a bounded real-data build and review generated stats/manifest
- [x] Download and validate a bounded Kaggle real-data sample (about 213 MB archive download)
- [x] Run guarded Colab download and select a deterministic 100-image CCPD sample
- [x] Obtain a CCPD sample and validate it with the local CCPD builder (used CCPD2020 while CCPD2019 is quota-blocked; see ADR-0004 addendum)
- [x] Fix COCO `file_name` dropping subfolders and Open Images `.jpg` lookup in combine
- [x] Plate-first Open Images sampling (config-driven plate/vehicle mix), rebuild capped sample
- [x] Build and review the combined dataset and stats report (100-image samples per source)
- [x] Full-size data build and stats report (Open Images 7,600 via CVDF, Kaggle all, CCPD2020 6,000 before cap)
- [x] Owner reviewed and approved the full combined dataset (2026-11-12)

Bounded sample results (`MAX_IMAGES=100`): 83 usable images; 17 image URLs returned HTTP 404/410; 0 rows excluded for missing/unapproved licence or attribution; 0 near-duplicates; 65 train / 7 val / 11 test; 1 plate box and 186 vehicle boxes. All manifest rows have CC BY 2.0 metadata and attribution fields, and all referenced files exist. This sample is for pipeline validation only, not for training-quality conclusions.

### T2 — Baseline (done)

- [x] Owner approved the T2 plan (2026-11-12): no external plate baseline (ADR-0007), curated holdout (option 2a), small boxes kept in eval and dropped from training only
- [x] Shared pre- and post-processing in `src/platedet/postprocess.py`, documented in `docs/preprocessing.md`
- [x] SPEC §6 metrics in `src/platedet/eval/metrics.py` (recall at threshold, small-plate recall, FP/image, AP@0.5, annotated-class scoping)
- [x] `make baseline` runner: hash-checked checkpoint, summary.json with commit, manifest and checkpoint hashes, predictions and miss gallery
- [x] YOLOX-Tiny COCO vehicle baseline on val/test, results in `docs/baseline.md`
- [x] Curated `holdout` set (ADR-0008): 244 Open Images test-split images, 364 plates, every image reviewed; `make holdout-prepare` / `make holdout-finalize`; evaluation runner scores it via `holdout_dir`

### T3 — Train v0 (current)

- [x] Owner approved the T3 plan (2026-11-12): vehicle pseudo-labels from YOLOX-L (ADR-0011), BGR input (ADR-0010), training on free Colab via a bundle and notebook, YOLOX default augmentation for v0
- [x] Pinned YOLOX source and `train` dependency group (ADR-0009); licences checked
- [x] Preprocessing now matches YOLOX/OpenCV within 1 grey level (ADR-0010); baseline rerun
- [x] Vehicle pseudo-labels: 3,805 plate-only train/val images, 4,526 boxes at score ≥ 0.5 (845 images got none, mostly CCPD close-ups)
- [x] Training data at 640 and 416 (`make train-data`): boxes under 6 px dropped from training only (640: 652 plates and 270 vehicles in train; 416: 1,886 and 720)
- [x] `make train` (CUDA, resumable, run record), `make export` (ONNX + PyTorch check), `make eval` (val/test/holdout report against SPEC §6), `make colab-bundle`, `notebooks/colab_train.ipynb`
- [x] Local Apple-GPU training path (ADR-0012): timing test, `local_trainer.py`, batch 16 configs, `make train RUN_ID=` with caffeinate; CPU and MPS tests
- [x] Train 640 (`runs/yolox_tiny_640_v0`, ~11 h), then 416 (`runs/yolox_tiny_416_v0`, ~8.7 h) on the Mac
- [x] Export, evaluate and report v0 to the owner (config, metrics against targets, worst misses)
- [x] Owner reviews the v0 report and picks the T4 direction: keep 640, drop 416, focus on small plates (2026-11-13)

### T4 — Small plates (current, branch `t4-small-plates`)

- [x] Owner approved the plan (2026-11-13): one photo at a time (tiling OK), unreadable plates ignored, run step B
- [x] Step B: retrain 640 for small plates (`configs/train/yolox_tiny_640_v1.yaml`, `runs/yolox_tiny_640_v1`)
- [x] Step A: tiled detection in `postprocess.py`, evaluation option, readable-size rule in metrics; tune on val
- [x] Evaluate v0/v1 with and without tiles; report to the owner
- [x] Owner picks the next T4 step: label audit, agent review + owner spot check (2026-11-14)
- [x] Audit sheets, agent review of all 1,683 test/holdout plate labels, audited scores (ADR-0014)
- [x] Owner spot check (`data/processed/label_audit/spot_check/`): 50/50 agreed (2026-11-14); `configs/audit/` committed
- [x] Owner rules (2026-11-14): blanked plates and dealer/show inserts count as plates (applied, ADR-0014)
- [x] Owner reviewed the 31 loose boxes: 25 accepted, 6 redrawn (`corrections.json`, hash in meta)
- [x] All 25 accepted loose boxes redrawn (owner go-ahead 2026-11-14); 31 corrections in total
- [x] Owner accepts the model for v1 (2026-11-14); gap recorded in ADR-0015

### T5 — Export + parity (done)

- [x] Owner approved the T5 plan (2026-11-14)
- [x] TFLite FP16 and INT8 via pinned onnx2tf (ADR-0016); calibration list recorded
- [x] Parity on test + holdout, single pass and 2×2 tiles; speed and size (`make parity`)
- [x] INT8 fails parity; owner: ship FP16 only (2026-11-15)
- [x] Recommended thresholds plate 0.10 / vehicle 0.30 (ADR-0017, `configs/release/v1.yaml`)
- [x] `make try` and `make test-vectors`; tile-edge fix (ADR-0013 addendum)
- [x] Owner tested 10 own photos: 8 fully right; 2 dark-garage, steep-angle close-ups missed. Contrast boost fixes both and is harmless on val; owner chose to ship v1 without it (known limitation)

### T6 — Release v1.0.0 (current, branch `t6-release`)

- [x] Owner decisions (2026-11-15): build the release package locally first, push to GitHub later; Apache-2.0 for code and weights; mobile `README.md` is part of T6
- [ ] Release package builder (`make release`): all SPEC §8 assets, FP16 only, `SHA256SUMS`, verify step
- [ ] `MODEL_CARD.md` and mobile integration `README.md`
- [ ] Owner approves the evaluation report and package
- [ ] GitHub repo, CI release from tag (deferred by owner)

### Next up

T6: build the v1.0.0 release package locally (`make release`), model card and mobile README; owner review. GitHub push after that.

### Open owner decisions

- SPEC §6 readable-size wording (ADR-0013): plates under 12 px wide excluded; small = 12–32 px. Proposed, not applied.

- 2026-11-15: Owner chose Apache-2.0 for this repo's code and the released weights (SPEC §11.3).
- GitHub repository creation and remote setup are deferred.
- Proposed data and evaluation targets in SPEC §§4–6 remain unconfirmed.
- ~~2026-11-12: Owner wants a configurable redaction (blur) tool after T3.~~ Dropped by the owner on 2026-11-14: apps blur the returned boxes themselves. Instead, once the owner has tested and validated the model, write a `README.md` explaining how to use the model in mobile apps (pre/post-processing, tiling, outputs, blurring the boxes).
- 2026-11-12: Owner wants face blurring added after T3 (a third class from Open Images "Human face" labels; plan to be written before T4). Phone-number blurring is a possible later version (needs text detection plus OCR).
- 2026-11-12: Proposed SPEC §5 wording change for BGR input and OpenCV-style resize (ADR-0010), awaiting the owner.
- 2026-11-12: Proposed SPEC §6 wording change for the dropped plate baseline (ADR-0007), awaiting the owner. Channel order for our own models (SPEC §5 says RGB; the official YOLOX trainer reads BGR) is to be fixed in the T3 plan.
- Google Colab free tier is approved for T3 but no longer needed: owner chose local Apple-GPU training (2026-11-12, ADR-0012). No compute has been started or paid for.
- Confirmed T1 split ratios: train 80%, val 10%, test 10%.
- 2026-11-09: Owner confirmed the model will be used worldwide, and that no owner photos are available (no owner-photo training data or `holdout` set). The SPEC §6 holdout recall target therefore can't be measured as written; proposed substitute to be agreed before T3.

## Features

| Feature | What it does | Branch |
|---|---|---|
| Initial project scaffold | Establishes the Python package, developer checks, CI, and artifact ignore rules. | Not available (local Git not initialized; owner chose to proceed without Git). |
| Open Images V6 ingestion (partial T1) | Parses allowed labels, downloads explicitly capped images, validates metadata, builds COCO splits and licence manifest, and deduplicates images. | Not available (local Git not initialized; owner chose to proceed without Git). |
| Kaggle CC0 plate-source adapter (partial T1) | Adds public archive ingestion, Pascal VOC plate parsing, CC0 manifest records, COCO output, pHash deduplication, and deterministic splits; validated on a capped real sample. | Not available (local Git not initialized; owner chose to proceed without Git). |
| CCPD + combined dataset pipeline (partial T1) | Parses filename boxes from extracted CCPD data; globally deduplicates source manifests, caps CCPD at 30% of plate boxes, tracks annotation scope, and writes merged COCO/manifest/stats. Synthetic tests only; no CCPD archive downloaded. | Not available (local Git not initialized; owner chose to proceed without Git). |
| COCO image-path fix (T1) | One shared helper maps manifest IDs to image paths (keeps subfolders, adds `.jpg` to bare Open Images IDs); COCO files and the combined merge now always point at real images. | `t1-fix-coco-paths` |
| Plate-first Open Images sampling (T1) | Reserves a configurable share (`plate_image_fraction`, 0.8) of the Open Images cap for images with plate boxes, so samples are no longer almost plate-free; reports the mix in `stats.json`. ADR-0005. | `t1-open-images-plate-first` |
| Combined dataset stats report (T1) | `stats.json` now breaks counts down by split and source, and reports plate widths (p10/p50/p90), plates under 32 px wide (SPEC §6) and boxes under 6 px after letterboxing to 640/416 (SPEC §4.4). | `t1-combined-dataset-report` |
| Split-safe sampling + download retries (T1) | Salts CCPD/Kaggle sampling and the CCPD cap so they don't correlate with the split hash; Open Images downloads retry 429/5xx with Retry-After or exponential backoff. | `t1-fix-sampling-and-rate-limits` |
| Open Images CVDF image host (T1) | `image_host: cvdf` downloads images from the official CVDF mirror by split and ID instead of Flickr (no throttling, no deleted photos); attribution still from the metadata CSV. ADR-0006. | `t1-open-images-cvdf-host` |
| Pre/post-processing, metrics and baseline runner (T2) | One reference letterbox/decode/NMS module; SPEC §6 metrics scored only where a class is annotated; `make baseline` evaluates an ONNX YOLOX model on val/test with a hash-checked checkpoint, run record and miss gallery. ADR-0007. | `t2-baseline` |
| Curated plate holdout (T2) | Builds an evaluation-only plate set from the Open Images test split (never used in training): downloads candidates, removes near-duplicates of the dataset, draws review sheets, and keeps only images accepted in the committed `configs/holdout/review.csv`. ADR-0008. | `t2-curated-holdout` |
| YOLOX training pipeline (T3) | Pinned YOLOX source; vehicle pseudo-labels for plate-only images; YOLOX training data with small-box dropping; resumable CUDA training with a run record; ONNX export checked against PyTorch; run evaluation report against SPEC §6; Colab bundle and notebook. ADR-0009/0010/0011. | `t3-train-v0` |
| Small-plate improvements (T4) | Tiled detection (whole image + 2×2 tiles) in `postprocess.py`; unreadable-plate ignore rule; v1 training config keeping every readable plate; plate label audit for test/holdout with owner-checked decisions and redrawn boxes; audited scores and `RESCORE`. ADR-0013/0014/0015. | `t4-small-plates` |
| Local Apple-GPU training (T3) | `make train` runs on the Mac GPU (MPS) when no CUDA GPU exists: a YOLOX `Trainer` subclass replacing only the CUDA-specific parts, with the same schedule, EMA, resume and checkpoints; keeps the Mac awake; fixed `RUN_ID` for resume. ADR-0012. | `t3-local-mps-training` |
| Guarded Colab CCPD sampler (partial T1) | Checks archive size and free runtime disk, streams a deterministic cap of 100 JPEGs, and bundles them for transfer. Initial Colab run succeeded; retry diagnostics have synthetic coverage. | Not available (local Git not initialized; owner chose to proceed without Git). |

## Session notes

- 2026-11-01: `make setup`, `make lint`, and `make test` passed locally with Python 3.12.16; hosted CI remains unverified because no Git remote exists.
- 2026-11-01: Owner confirmed 80/10/10 splits and approved Open Images as the first T1 source; owner also approved free Colab for T3. Official source docs identify V6, not V7, so ADR-0002 records the V6 endpoint adjustment. Implemented the bounded V6 pipeline and synthetic tests; `make lint`, `make test` (9 passed), and `uv lock --check` pass. T1 is not complete; additional allowlisted sources remain outstanding.
- 2026-11-02: Owner approved metadata-listed CC BY 2.0 for a bounded Open Images sample. Ran `make data MAX_IMAGES=100`; 83 images were usable, 17 URLs returned 404/410, no licence/attribution rows were excluded, and no near-duplicates were found. Reviewed manifest and COCO outputs: all 83 image files exist, all manifest licence values are CC BY 2.0, and the sample contains 1 plate and 186 vehicle boxes (too imbalanced/small to train). No training started. Full T1 remains in progress.
- 2026-11-02: Owner selected Kaggle `andrewmvd/car-plate-detection`. Verified the Kaggle dataset card's CC0 1.0 declaration, 433-image/PASCAL VOC description, Make ML acknowledgement, and anonymous public archive redirect (no credentials used). Implemented the capped adapter and synthetic tests; `make lint` and `make test` pass (15 tests). The ~213 MB archive has not been downloaded. Recorded that the source has plate-only labels and cannot safely imply vehicle negatives. T1 remains in progress.
- 2026-11-02: Owner approved downloading the ~213 MB public Kaggle archive. First real run revealed the source's exact object class is `licence`; added explicit normalization and regression coverage. `make data DATA_CONFIG=configs/data/kaggle_car_plate_detection.yaml MAX_IMAGES=100` processed 100 selected annotations: 91 images usable, 9 near-duplicates removed, 0 failures; 75 train / 9 val / 7 test; 108 plate boxes, 0 vehicle boxes. Reviewed manifest: all 91 rows are CC0 1.0 with source attribution, and every image exists. COCO output contains 108 plate boxes. `make lint` and `make test` pass (16 tests). This plate-only source must not count as vehicle negatives. T1 remains in progress pending CCPD and full combined build/report.
- 2026-11-02: Owner confirmed a 30% CCPD plate-box cap and approved a small real sample. Verified official CCPD README's MIT dataset statement, filename-embedded bounding boxes, train/validation and challenge splits, and Google Drive bulk archive link. Implemented filename parsing, capped processing of an already-extracted tree, manifest/COCO generation, global source merge, deterministic resplitting, pHash dedupe, and the 30% cap. Added per-image `annotated_classes` to prevent plate-only sources being treated as vehicle negatives. `make lint`, `make test` (22 passed), and `uv lock --check` pass. Did not download CCPD: the official Drive endpoint exposes no archive size, while the Mac has only about 29 GiB free; bulk download/extraction risks exhausting storage. ADR-0004 records this blocker. Combined real-data report remains pending local CCPD data.
- 2026-11-02: Owner approved a conditional free-Colab CCPD prep workflow. The public Google Drive confirmation page reports the archive is approximately 12 GB. Added `scripts/colab_ccpd_sample.py`: it requires a known final response size, rejects archives above 16 GiB, requires the archive size plus an 8 GiB reserve in the runtime, streams only the 100-image cap from the tar.xz, bundles the sample, and deletes the large archive after a successful bundle. Unit tests cover size/disk guards and deterministic streaming extraction. `make lint`, `make test` (28 passed), and `uv lock --check` pass. No Colab runtime started and no archive downloaded; owner must upload/run the script in free Colab and transfer the bounded bundle back.
- 2026-11-07: Owner reports guarded Colab run succeeded: downloaded 12.26 GiB CCPD2019 archive and retained 100 deterministic images from 355,013 JPEGs. Bundle transfer/local validation remains pending; no raw CCPD data is yet present under `data/raw/ccpd/`.
- 2026-11-08: Owner reports a later Colab retry returned HTML instead of the CCPD archive. The prior generic error did not preserve enough safe diagnostics to confirm whether this was quota/rate limiting or an extra confirmation page. Updated the helper to follow a bounded chain of official confirmation forms, reject non-Google form targets, and classify likely quota, sign-in, or virus-scan responses without printing hidden values. `make lint`, `make test` (30 passed), and `uv lock --check` pass. Check local Downloads for the bundle from the successful first run before attempting another 12 GB download; if Drive reports quota, stop retrying and wait.
- 2026-11-09: Owner requested local CCPD sampling in `data/raw/ccpd_work/`. Disk preflight showed about 37.5 GiB free, enough for the helper's required 12.26 GiB plus 8 GiB reserve. Attempted the guarded run; Google Drive returned the same quota/rate-limit response before any archive bytes were downloaded. The work directory is empty and no partial archive remains. Further retries should wait for quota reset or use an owner-provided alternate copy.
- 2026-11-09: Searched `~/Downloads`, `~/Desktop`, `~/Documents` and the home folder (Spotlight + `find`) for the first Colab run's `ccpd_sample.tar.gz`; not found on this Mac. Did not retry Drive (last quota hit was 22:46 the same day). Options for the owner: check the earlier Colab runtime's `/content/ccpd_work/`, copy the official file into their own Google Drive and download from there, or retry after the quota resets.

- 2026-11-01: Began T0 scaffold after owner approval. No local Git repository was present; owner chose implementation without Git. Code licence deferred; no GitHub repository created.- 2026-11-09: CCPD2019 Drive retry hit the quota again (no bytes downloaded). Downloaded the official CCPD2020 archive instead (866 MiB, same README and MIT licence; ADR-0004 addendum), extracted a deterministic 100-image sample, and ran `make data DATA_CONFIG=configs/data/ccpd.yaml MAX_IMAGES=100`: 99 images, 1 near-duplicate removed, 0 failures; 83 train / 9 val / 7 test; 99 plate boxes; all rows MIT and plate-only; all boxes inside image bounds and visually on the plate. Found that all 99 COCO `file_name` values point to missing paths (subfolders dropped by `_coco_dataset`), so the combined build is blocked until that bug is fixed. CCPD2020 is green plates only; add CCPD2019 later.
- 2026-11-09: Initialised local Git (no remote); baseline commit on `main`, task branch `t1-fix-coco-paths`. Fixed COCO `file_name` dropping subfolders and the combine step's missing `.jpg` for Open Images via one shared `image_relative_path` helper; added `tests/test_coco_paths.py` plus a CCPD path assertion (confirmed it fails on the old code). `make lint` clean, `make test` 32 passed. Rebuilt CCPD and ran `make data-combine` as a smoke check: every COCO path in all three sources and the 220-image combined set resolves to a real file.
- 2026-11-09: Added plate-first Open Images sampling (ADR-0005, `plate_image_fraction: 0.8`) on `t1-open-images-plate-first`. Rebuilt `MAX_IMAGES=100`: 80 plate + 20 vehicle-only selected; 87 usable (13 dead URLs); 69 images with plates; 110 plate boxes (was 1) and 216 vehicle boxes; 66/12/9 split. Visual check: boxes correct; plates from US, UK and continental Europe. Data-quality caveats: some visible plates are unlabelled (Open Images labels are incomplete) and at least one plate label looks wrong (neon-sign image). `make lint` clean, `make test` 34 passed.
- 2026-11-09: Built the combined dataset from the three 100-image samples on `t1-combined-dataset-report`: 271 images (224/30/17), 311 plate and 216 vehicle boxes; 0 cross-source duplicates; CCPD 29.9% of plates (6 images dropped by the 30% cap); 184 plate-only and 87 fully labelled images; all licences allowlisted, none empty. Plate width p10/p50/p90 = 25/140/407 px; 47 plates under 32 px wide; boxes under 6 px short side: 12 at 640, 37 at 416 (reported, not dropped). Sample is too small for training conclusions (e.g. only 1 CCPD image in test). `make lint` clean, `make test` 35 passed.
- 2026-11-10: First full build (Kaggle 433, CCPD 3000, Open Images 2500) exposed two bugs: all CCPD images landed in `train` (sampling and cap hashed the same key as the split) and Flickr returned HTTP 429 for 2,359 of 2,500 images. Fixed both on `t1-fix-sampling-and-rate-limits` (4 new tests, confirmed failing on old code; `make test` 39 passed, lint clean). Kaggle's 136 removed images are exact pHash duplicates (dataset repeats 86 images), so dedupe is correct. Rebuild: CCPD 2,990 images split 2428/279/283; Kaggle 297; Open Images still 122 (Flickr throttling plus Mac sleep/network drop at 01:38). Combined: 627 images, 695 plate / 310 vehicle boxes, CCPD 29.9% (2,782 CCPD images dropped by the cap because Open Images is small). Open Images' official download page points to CVDF hosting (`downloader.py`, Apache-2.0, bucket `open-images-dataset`); a HEAD check of one image returned 200, 166 KB vs 1 MB from Flickr. Switching the image host needs owner approval and an ADR.
- 2026-11-11: Switched Open Images image downloads to CVDF (ADR-0006, owner approved) on `t1-open-images-cvdf-host`; `make test` 40 passed, lint clean. Full build (kept awake with `caffeinate`, log `data/logs/full-build-20261111-2252.log`): Open Images 7,599 of 7,600 (1 network error), 6,079 with plates, 8,819 plate / 21,156 vehicle boxes, 2.2 GB, images max 1,024 px; CCPD2020 5,967 (30 near-duplicates); Kaggle 297. Combined: 11,815 images (9,503 / 1,121 / 1,191), 13,067 plate / 21,155 vehicle boxes, CCPD 30.0% (2,047 dropped by cap), 1 cross-source duplicate, all licences allowlisted with authors. Plate width p10/p50/p90 = 23/108/371 px; 2,445 plates under 32 px wide; under 6 px short side: 867 at 640, 2,443 at 416. Hand check of 60 random Open Images plate crops: 10 US, 8 UK, 16 continental Europe, 2 other (Brazil, Belize), 23 too small/blurry to identify, 1 not a plate (hazard placard). Asia is represented almost only by CCPD (green plates). Open items before T3: owner review of this dataset, holdout substitute, small-box handling, CCPD2019 when Drive allows.
- 2026-11-12: Owner reviewed the full T1 dataset and approved it; T1 closed. Traced the 10 US plates from the spot-check to image IDs (one tile had been mislabelled in my notes and was corrected). Proposed the T2 plan. A web search found no RT-DETRv2-R18 plate model trained on Open Images: RT-DETRv2-R18 checkpoints are COCO-only, and the public plate fine-tunes are R50 trained on Roboflow data of unknown licence.
- 2026-11-12: T2 on `t2-baseline`. Added `postprocess.py`, `eval/metrics.py`, `eval/baseline.py`, `make baseline`, ADR-0007, `docs/preprocessing.md`, `docs/baseline.md`; new deps numpy (BSD-3) and onnxruntime (MIT). `make lint` clean, `make test` 56 passed. YOLOX-Tiny COCO at 416 (official ONNX, SHA-256 427cc366…): test vehicle recall 0.687 @0.3 (target 0.90), FP/image 1.49, AP@0.5 0.582; val 0.703 / 1.33 / 0.596. Recall is 0.45 for vehicles under 1% of image area (a third of boxes) and over 0.8 for larger ones. The largest misses are Open Images full-frame vehicle labels (plate close-ups, in-car shots). Run `runs/baseline-yolox_tiny_coco-20261112-221740`.
- 2026-11-12: Built the curated holdout on `t2-curated-holdout` (ADR-0008). Unused plate images in Open Images train/validation ran out (12 left; the training sample uses about 6,080 of 6,090), so candidates come from the official Open Images test split. 300 candidates, 0 near-duplicates of the dataset; reviewed by 4 parallel reviewer agents with fixed criteria plus my re-check of 24 accepts (2 overruled). Result: 244 images, 364 plates, 98 under 32 px wide. Rejects: 34 display or dealer plates labelled as plates, 13 unclear, 8 unlabelled plates, 1 redacted. `make lint` clean, `make test` 63 passed.
- 2026-11-12: T3 pipeline on `t3-train-v0`. Owner approved all recommended options. Pseudo-labels: 3,805 images and 4,526 vehicle boxes (spot check: boxes correct; empty cases are close-ups). Training data built at 640 and 416. Preprocessing switched to OpenCV-style resize (within 1 grey level of YOLOX); baseline rerun gives test vehicle recall 0.693 (was 0.687). CPU smoke test runs a real YOLOX training step and ONNX export. `make lint` clean, `make test` 70 passed. GPU training can't run locally (YOLOX's trainer needs CUDA), so the owner runs it on free Colab.
- 2026-11-12: Owner chose local training over Colab. Timing test on the M2 Pro (200 steps, real 640 pipeline): batch 16 fp32 0.66 s/step, about 11 h per 100 epochs; fp16 no faster; batch 32 hits memory pressure (31–54 h). Added `local_trainer.py` (ADR-0012), batch 16 / fp32 / 6 workers in the v0 configs, `make train RUN_ID=` with caffeinate. YOLOX's FLOPs logger (`thop`) needs distutils, which Python 3.12 lacks, so the local trainer logs the parameter count instead. Tests: CPU train, eval and resume, plus a 1-epoch MPS run.
- 2026-11-12: First local 640 run stalled at epoch 1, iteration 200: the MPS cache grew with each multi-scale size until macOS swapped (14 GB). Fixed by releasing the MPS cache at each size change (ADR-0012), then restarted. Memory now follows the input size (4.4–8.6 GB); ETA about 10.5 h.
- 2026-11-12: The 640 run crashed at its first evaluation (epoch 10): YOLOX's fast COCO scorer needs a C++ build (ninja). The local trainer now uses pycocotools (same metrics), with a test that scores real detections. The epoch-10 checkpoint was saved before the crash; its val score is AP50 0.809 and AP50:95 0.513 (plate AP 0.561, vehicle 0.465). Resumed from epoch 11.
- 2026-11-13: 640 run finished (`runs/yolox_tiny_640_v0`, 100 epochs on the Mac GPU, ~11 h, best val AP@0.5:0.95 0.598 at the last epochs; checkpoint SHA-256 ffaa8c75…). ONNX export matches PyTorch (max error 4.5e-6). Test vs SPEC §6: plate recall 0.885 (target 0.95), small plates 0.652 (0.85), vehicles 0.684 (0.90; off-the-shelf baseline 0.693), plate FP/image 0.21 (≤1.0, met); holdout plate recall 0.755, small 0.265. Of 156 missed test plates: 64 not found (48 under 32 px), 51 found below the 0.2 threshold, 41 found with box overlap under 0.5 (many loose Open Images labels). Of vehicle misses, half are found below the 0.3 threshold; many are Open Images full-frame or close-up "vehicle" labels. Val sweep: plate threshold 0.05 gives recall 0.90 at 0.30 FP/image. 416 run started automatically after 640.
- 2026-11-13: 416 run finished (`runs/yolox_tiny_416_v0`, ~8.7 h, data loading was the bottleneck; best val AP@0.5:0.95 0.608, checkpoint SHA-256 73535a13…; ONNX matches PyTorch, max error 7.3e-6). Test: plate recall 0.747, small plates 0.232, plates ≥32 px ~0.88 (640: ~0.94), vehicles 0.640, plate FP/image 0.11; holdout plate recall 0.698. The val AP during training was higher than 640's because it is scored on the 416 labels, which drop the smallest plates; the full evaluation shows 416 is worse at every plate size. Recommendation to the owner: ship 640, drop 416.
- 2026-11-13: T4 started on `t4-small-plates` (owner: keep 640, small plates first; one photo at a time; ignore unreadable plates; run step B). T3 merged to `main`. ADR-0013. Training labels `data/processed/yolox_640_v1` keep plates ≥ 12 px wide (117 dropped vs 652 in v0). `make lint` clean, `make test` 75 passed. v1 training started.
- 2026-11-13: Step A built on `t4-small-plates`: `detect_tiled` in `postprocess.py`, ignore rule for plates under 12 px in metrics, `make eval TILES=N SPLITS=…`; `make test` 80 passed. Val, v0 model: plate recall 0.882 → 0.920 with 2×2 tiles (3×3: 0.929), small plates 0.580 → 0.771 (0.776), plate FP/image 0.14 → 0.35. Chose 2×2 (ADR-0013). The v0 `eval/` report now holds val only; test/holdout are rerun in the final comparison. v1 training at epoch 3, ETA ~13:00 on 2026-11-14.
- 2026-11-14: v1 finished (~13 h, checkpoint 48086beb…, ONNX matches PyTorch). Test (readable plates, ADR-0013): v0 0.892 / small 0.674; v0+2×2 tiles 0.926 / 0.822; v1 0.891 / 0.682; v1+tiles 0.933 / 0.841 (targets 0.95 / 0.85), plate FP/image 0.43, vehicles 0.741. Holdout v1+tiles 0.886, small 0.567. Val agrees (v1+tiles best: 0.933 / 0.796). Of 90 remaining test plate misses, 57 have a confident box with IoU 0.1–0.5 (gallery: mostly loose or offset labels), 18 are below 0.2, 15 not found. A first evaluation chain failed on a shell word-splitting bug (my script); rerun fine.
- 2026-11-14: Label audit (ADR-0014). 5 reviewer agents, model-blind, all 1,683 readable test/holdout plate labels: test 1,283 ok / 27 loose / 40 not_plate; holdout 311 / 2 / 20. Not_plate is mostly deliberately blanked plates (Open Images anonymisation) and dealer/show inserts. I re-checked 2 random all-ok CCPD sheets (40 boxes, all ok). Audited v1 + 2×2 tiles: test plate recall 0.947 (0.95), small 0.849 (0.85), holdout 0.900. Fixed a missing report field found when re-scoring; `make test` 85 passed.
- 2026-11-14: Owner spot check of the label audit: 50/50 agreed (two comments, no changes). Committed `configs/audit/plate_labels.csv` and meta (IDs and decisions only).
- 2026-11-14: Owner rules: blanked plates and dealer/show inserts count as plates. Of 52 such labels, 49 became ok, 2 loose, 1 kept not_plate (checked by eye). Audit now: test 1,316 / 28 / 6, holdout 327 / 3 / 3. Audited v1 + tiles: test 0.948, small 0.849, holdout 0.902. Loose-box sheets for the owner in `data/processed/label_audit/loose/`.
- 2026-11-14: Owner reviewed the 31 loose boxes: 25 accepted, 6 redrawn by me from the photos and checked on a before/after sheet (`make audit-corrections`; boxes stay in `data/`). Accepted oversized labels make 15 tight model boxes count as misses (14 of 15 overlap the plate). Audited v1 + tiles: test 0.938, holdout 0.900. `make test` 86 passed.
- 2026-11-14: Redrew all 25 accepted loose boxes from the photos (model-blind; all, not only the 15 the model missed). Audited v1 + tiles: test 0.947, small 0.849, holdout 0.900, FP/image 0.418.
- 2026-11-14: Owner accepted v1 + 2×2 tiles for release (test 0.947, small 0.849, holdout 0.900, vehicles 0.741; FP 0.42). ADR-0015 records the gap; targets unchanged. T4 closed and merged to `main`.
- 2026-11-14: Owner dropped the blur tool (apps blur boxes themselves). After T5/T6 and the owner's own testing, write a mobile-integration `README.md` on request.
- 2026-11-15: TFLite export (`make export`): FP16 9.7 MB, INT8 5.1 MB, calibration list recorded. Parity on 1,435 test+holdout images: FP16 single pass 5,159/5,159 within tolerance (Δconf ≤ 0.003); with tiles 16 of 6,442 boxes differ (merge tie-breaks; plates still boxed). INT8 fails (16 plates missing single pass). Owner: ship FP16 only. Mac CPU per tiled photo: ONNX 381 ms, FP16 441 ms. ADR-0016.
- 2026-11-15: Owner approved thresholds plate 0.10 / vehicle 0.30 (ADR-0017, `configs/release/v1.yaml`). Added `make try` (owner validation) and `make test-vectors` (5 vectors, `CHECK=1` reproduces). `make try` exposed tile fragment boxes: edge margin now 2% of tile (val: vehicle FP 1.45 → 1.22, no recall loss). Re-run: audited test 0.948, small 0.849, holdout 0.900, plate FP 0.39; FP16 parity single pass all within tolerance, tiled 3 near-tie differences, 0 missing. `make test` 94 passed.
- 2026-11-15: Owner tested 10 own photos with `make try` (`data/my_photos/`, now the default folder): 8 fully right; 2 dark-garage close-ups at steep angles missed the plate (score ~0.02, not a threshold issue). An autocontrast step on dark photos (mean grey < 70) found both (0.61, 0.31) and was harmless on val (plate recall 0.940 → 0.941, FP unchanged). Owner chose to ship v1 without it; listed as a known limitation. T5 closed and merged to `main`; T6 started.
