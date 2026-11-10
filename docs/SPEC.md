# Plate & Vehicle Detector — Training Spec

> Build instructions for an AI coding agent. Project folder: `training_resource`; Python package: `platedet`.
> This project trains and publishes a model that **finds licence plates and vehicles in photos**. Read this whole document before writing code.
> Status: **draft v0.1 (2026-11-01).** Numbers marked *(proposed)* need the owner's confirmation.

---

## 1. What this project does and why

**What it does:** trains a model that looks at a photo and returns a box around every **licence plate** and every **vehicle** in it, each with a confidence score. Typical use: blurring plates in photos for privacy, or checking that a photo contains a vehicle.

**What it delivers:** versioned releases of the trained model in two formats of the same model:

- **TFLite** (FP16 and INT8) for phones and edge devices;
- **ONNX** (FP32) for servers and desktops.

Each release also carries a reference implementation of the pre- and post-processing, accuracy numbers, and licence records, so any app can use the model correctly without access to this repo's training code or data.

**Why build our own:** the model must:

1. be **legally safe for commercial use**: permissively licensed weights *and* commercially usable training data;
2. run on **phones and servers** with matching results;
3. be **small and fast** enough for mid-range phones.

Existing open plate detectors fail at least one of these:
- the popular ones are derived from GPL/AGPL code (YOLOv5/v8/v9/11 families), or trained on undocumented or non-commercial data;
- the few licence-clean ones are too large for phones and unproven.

**Scope**

| In scope | Out of scope |
|---|---|
| One detector, two classes: `license_plate`, `vehicle` | Faces (use an existing licence-clean face detector) |
| Data download, cleaning, licence tracking | Reading plate text (OCR) |
| Training, evaluation, export to ONNX + TFLite, parity checks | Blurring, serving or app integration (done by whoever uses the model) |
| Versioned, self-describing releases | |

---

## 2. Guiding principles

1. **Licence-clean or nothing.** Every architecture, pretrained checkpoint, dataset and image has a recorded, commercially usable licence. Unknown = excluded.
2. **Recall over precision for plates.** A missed plate is a privacy leak; an extra box is harmless. Thresholds and metrics are chosen with this in mind.
3. **Reproducible.** Pinned dependencies, fixed seeds, config files in Git, and a dataset manifest, so any release can be rebuilt.
4. **Free first.** Train on the owner's Mac or free tiers (e.g. Google Colab free). No paid compute or storage without the owner's approval.
5. **Small, boring tools.** Plain PyTorch + the official YOLOX code, scripts driven by `make`. No MLOps platform.
6. **The release is the product.** A release must be usable on its own: model files plus everything needed to run them correctly.

---

## 3. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language / env | **Python 3.12**, managed with **uv** (`pyproject.toml` + `uv.lock`) | Fast, reproducible installs |
| Model | **YOLOX-Tiny** (Megvii, Apache-2.0), starting from the official COCO-pretrained checkpoint. **YOLOX-Nano** as the smaller fallback | Permissive, plain convolutions, converts cleanly to TFLite, NMS outside the graph |
| Training | Official YOLOX code (pinned commit) or a thin wrapper around it | Don't reinvent the trainer |
| Hardware | Apple Silicon (MPS) or CPU locally; **Google Colab free GPU** for full runs | $0 |
| Experiment tracking | **TensorBoard** logs + a `runs/<run_id>/summary.json` per run | Open source, local |
| Data tooling | `fiftyone` (Apache-2.0) or plain scripts for Open Images download; `imagehash` for duplicates | |
| Export | PyTorch → **ONNX** (opset 17) → **onnx2tf** (MIT) → TFLite FP16 + INT8 | Proven path for YOLOX |
| Inference for checks | `onnxruntime`, `ai-edge-litert` (TFLite runtime) | The runtimes users of the model will use |
| Tests / lint | `pytest`, `ruff` | |
| CI | GitHub Actions: lint, unit tests, a tiny smoke train + export on a few images | |

Any change to this table needs a short ADR in `docs/adr/`.

---

## 4. Data

### 4.1 Allowed sources (allowlist)

| Source | What we use | Licence | Notes |
|---|---|---|---|
| **Open Images V7** | Images with boxes for `Vehicle registration plate` → `license_plate`; `Car`, `Truck`, `Bus`, `Van`, `Motorcycle`, `Land vehicle` → `vehicle` | Annotations **CC BY 4.0** (Google); images "listed as" **CC BY 2.0** | Primary source. Attribution required: keep each image's ID, author and licence URL |
| **CCPD** (2019/2020) | Plate boxes | **MIT** (per its README) | Chinese parking-lot plates; cap its share so the model doesn't over-fit to one plate style *(proposed: ≤ 30% of plate boxes)* |
| **Kaggle `andrewmvd/car-plate-detection`** | Plate boxes | **CC0** | Small (433 images) |
| **Owner's own photos** | Plates + vehicles (labelled by us) | Owned by the owner | Closest to real-world phone photos |

Adding any new source needs an ADR plus a row here.

### 4.2 Banned sources (denylist)

- Anything **CC BY-NC**, **NC-ND**, "academic / research only", or with no licence: e.g. **UFPR-ALPR**, **RodoSol-ALPR**, **platesmania** scrapes, unlicensed **Roboflow Universe** sets, **WIDER FACE**, **Objects365**.
- Any pretrained weights or code from **Ultralytics** (AGPL), **WongKinYiu YOLOv9** (GPL), **DEIMv2** (non-commercial), **InsightFace**.
- Outputs of other models with unclear licences used as pseudo-labels.

### 4.3 Licence manifest (MUST)

Every image used anywhere (train, val, test, calibration) has a row in `data/manifest.parquet` (also exported as CSV for review):

`image_id, source, source_image_id, url, author, licence, licence_url, sha256, phash, width, height, split, n_plates, n_vehicles`

- `make data` builds it. It fails if any row has an empty or non-allowlisted `licence`.
- `ATTRIBUTION.md` is generated from it for CC BY sources and shipped with each release.
- Images and annotations are **never committed to Git**. Only the scripts and the manifest schema are.

### 4.4 Processing

1. Download each source into `data/raw/<source>/` (git-ignored).
2. Convert to one **COCO-format** dataset with the two classes.
3. **Deduplicate** across sources with perceptual hashes (exact + near duplicates), so the same photo can't land in both train and test.
4. **Split** into train / val / test *(proposed: 80 / 10 / 10)*, stratified by source. The split is deterministic (hash of `image_id`), so it never changes between runs.
5. **Hold-out sets that are never trained on:**
   - `test`: the 10% split above.
   - `holdout`: an optional extra folder of real-world photos (e.g. the owner's own), set by path in the config and never copied into training data.
6. Drop boxes smaller than *(proposed)* 6 px on the shorter side after resizing; count how many were dropped.

---

## 5. Model

- **Architecture:** YOLOX-Tiny, 2 classes, **input 640×640** *(proposed; plates in street photos are small)*. Also train and evaluate 416×416 to measure the speed/accuracy trade-off.
- **Start point:** official YOLOX-Tiny COCO checkpoint (Apache-2.0). Record its URL + SHA-256 in the model card.
- **Pre-processing:** letterbox resize to the input size, pad with value 114, RGB, float32 in 0–255 (YOLOX default, no mean/std).
- **Post-processing:** **outside the model graph.** Decode + class-aware NMS (IoU 0.45 *(proposed)*).
- Both are implemented once in `src/platedet/postprocess.py` and documented in `docs/preprocessing.md` with **test vectors** (input image → expected boxes), so anyone porting them to another language (Dart, Kotlin, Swift, Go…) can verify their port.
- **Augmentation:** YOLOX defaults (mosaic, mixup, HSV, flips), plus motion blur, JPEG compression, low light and small-scale crops to mimic phone photos. Mosaic is turned off for the last epochs (YOLOX default).

---

## 6. Evaluation and acceptance

Every run writes `runs/<run_id>/summary.json` with all metrics below on `val`, `test` and `holdout` (if set).

| Metric | Target for v1.0 *(proposed)* |
|---|---|
| **Plate recall** at confidence 0.2, IoU 0.5, on `test` | **≥ 0.95** |
| Plate recall at confidence 0.2 on `holdout` | **≥ 0.95** (every miss listed in the report with the image) |
| Plate recall for **small plates** (< 32 px wide) | Reported; ≥ 0.85 |
| Vehicle recall at confidence 0.3 on `test` | ≥ 0.90 |
| mAP@0.5 per class | Reported (no hard target) |
| False positives per image (plates) | Reported; ≤ 1.0 |
| **Beat the baseline:** best licence-clean open plate model found (RT-DETRv2-R18 trained on Open Images, Apache-2.0) | Higher plate recall at the same false-positive rate |

The **recommended operating threshold** per class is chosen from the precision/recall curve to meet the recall target and written into `release.json`.

---

## 7. Export and parity

For every release candidate:

1. Export **ONNX** (FP32, opset 17, fixed input shape, no NMS in the graph).
2. Convert with **onnx2tf** to **TFLite FP16** and **TFLite INT8** (INT8 calibration on ~300 *train* images, recorded in the manifest).
3. **Parity check** against ONNX FP32 on `test` + `holdout`, after the shared post-processing:
   - matched boxes: **IoU ≥ 0.9** and **|Δconfidence| ≤ 0.05**;
   - **no plate detected by ONNX above the operating threshold may be missing** in TFLite.
   - FP16 must pass. INT8 is published only if it also passes.
4. **Speed:** median latency on the owner's Mac for ONNX Runtime (CPU) and TFLite (CPU, 4 threads).
5. **Size:** report file sizes. Target *(proposed)*: FP16 ≤ 12 MB, INT8 ≤ 6 MB.

---

## 8. Releases

- Versioned with **semver** tags `vMAJOR.MINOR.PATCH`:
  - MAJOR: input size, classes, pre/post-processing or output format change (users must update their code).
  - MINOR: retrained / better weights, same interface.
  - PATCH: metadata or docs only.
- Published as a **GitHub Release** with these assets:

| File | Contents |
|---|---|
| `detector.onnx` | FP32 ONNX |
| `detector_fp16.tflite`, `detector_int8.tflite` | TFLite (INT8 only if it passed parity) |
| `labels.txt` | `license_plate`, `vehicle` (index order) |
| `release.json` | version, input size, pre/post-processing parameters, recommended thresholds per class, NMS IoU, metrics summary, parity results, source commit, checkpoint SHA-256 |
| `postprocess.py` + `test_vectors/` | Reference pre/post-processing and expected outputs for porting |
| `MODEL_CARD.md` | intended use, limits, training data summary, metrics, licences |
| `ATTRIBUTION.md` | CC BY attributions generated from the manifest |
| `SHA256SUMS` | checksums of all the above |

- Consumers download a release by URL and verify it with `SHA256SUMS`.
- Releases are created by CI from a tag, never uploaded by hand. A release is tagged only after the owner approves the evaluation report.

---

## 9. Repository layout

```
/configs/               YAML configs for data, training and export (one per experiment)
/src/platedet/
    data/               downloaders per source, COCO conversion, dedupe, split, manifest
    train/              YOLOX experiment definition and train entrypoint
    eval/               metrics, PR curves, error galleries, baseline comparison
    export/             ONNX export, onnx2tf conversion, INT8 calibration
    parity/             ONNX vs TFLite parity checks
    postprocess.py      reference pre/post-processing (shipped with each release)
/tests/                 pytest: unit tests + tiny smoke dataset
/tests/data/smoke/      ~10 CC0 images with boxes, for CI
/data/                  git-ignored: raw downloads, processed datasets, manifest
/runs/                  git-ignored: checkpoints, TensorBoard logs, summaries
/docs/SPEC.md           this file
/docs/PROGRESS.md       task checklist, features log, session notes
/docs/adr/              architecture decision records
/docs/preprocessing.md  exact pre/post-processing contract + test vectors
Makefile, pyproject.toml, uv.lock, CLAUDE.md, README.md, LICENSE
```

---

## 10. Build order

| Step | Goal | Done when |
|---|---|---|
| **T0 Scaffold** | Repo layout, uv project, Makefile, ruff/pytest, CI, `.gitignore` (data, runs, weights), `docs/PROGRESS.md` | `make test` and `make lint` are green in CI |
| **T1 Data** | Downloaders for each allowlisted source, COCO conversion, dedupe, deterministic split, manifest + licence check, `ATTRIBUTION.md` | `make data` builds the dataset; the manifest has no empty or banned licence; dataset stats report |
| **T2 Baseline** | Evaluate the RT-DETRv2 Open Images plate model and COCO YOLOX-Tiny (vehicles only) with our metrics | Baseline numbers in `docs/baseline.md` |
| **T3 Train v0** | First YOLOX-Tiny run at 640 (and 416), full eval report with error gallery | Report reviewed by the owner |
| **T4 Iterate** | Data fixes, augmentation, threshold choice, until the §6 targets are met (or the owner accepts a gap) | Targets met, or an ADR records the accepted gap |
| **T5 Export + parity** | ONNX, TFLite FP16/INT8, parity, speed, size | All §7 checks pass |
| **T6 Release v1.0.0** | CI release from tag, all §8 assets | A fresh checkout can download a release, verify it, and run the test vectors |

---

## 11. Open decisions for the owner

1. GitHub repo name and visibility.
2. Confirm the *(proposed)* numbers: input size, recall targets, size targets, CCPD cap, split ratios.
3. Licence for this repo's code and for the released weights (they derive from Apache-2.0 YOLOX + CC BY data; attribution ships with releases).
4. Whether to use Google Colab (free tier, needs a Google account) for full training runs.
5. How many of your own photos you can provide and label for the `holdout` set and training, and whether to label them yourself or have the agent pre-label them for you to correct.
