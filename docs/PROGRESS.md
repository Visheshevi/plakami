# Project Progress

## Current step: T1 — Data (Open Images source implementation)

### T0 — Scaffold

- [x] Project layout, uv, Makefile, Ruff, pytest, and CI workflow created
- [x] `make setup`, `make lint`, and `make test` pass locally
- [ ] Hosted CI run (requires Git and a remote; deferred by owner)

### T1 — Data (current)

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
- [ ] Build and review the combined dataset and stats report
- [ ] Complete full T1 data pipeline and dataset stats report

Bounded sample results (`MAX_IMAGES=100`): 83 usable images; 17 image URLs returned HTTP 404/410; 0 rows excluded for missing/unapproved licence or attribution; 0 near-duplicates; 65 train / 7 val / 11 test; 1 plate box and 186 vehicle boxes. All manifest rows have CC BY 2.0 metadata and attribution fields, and all referenced files exist. This sample is for pipeline validation only, not for training-quality conclusions.

### Next up

Plate-first Open Images sampling (task 2), then the combined build and stats report (task 3). Earlier context:

Open Images V6 and Kaggle CC0 have bounded real-data samples; CCPD parsing/build support and the combined merge/cap pipeline are implemented and covered by synthetic tests. The guarded Colab helper has now downloaded the approximately 12.26 GiB CCPD archive and retained 100 deterministic images from 355,013 JPEGs. Full T1 still needs the sample bundle transferred into the workspace, local CCPD validation, and the combined dataset build/report. The owner approved metadata-listed CC BY 2.0 for the capped Open Images sample; the upstream per-image licence disclaimer remains recorded in ADR-0002.

### Open owner decisions

- Repository code licence is deferred; no `LICENSE` file is included.
- GitHub repository creation and remote setup are deferred.
- Local Git is not initialized; no task branch or commit exists.
- Proposed data and evaluation targets in SPEC §§4–6 remain unconfirmed.
- Google Colab free tier is approved for the eventual T3 training run; no compute has been started or paid for.
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
