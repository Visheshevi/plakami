# ADR-0011: Vehicle pseudo-labels for plate-only training images

- **Status:** Accepted (owner approved on 2026-11-12)
- **Date:** 2026-11-12

## Context

CCPD and Kaggle images (about 4,200 of 11,815) carry plate boxes only, but most of them show cars. YOLOX has one objectness score shared by both classes, so the trainer would treat every unlabelled car in those images as background. That teaches the model to ignore vehicles. Masking the loss per class isn't possible without rewriting YOLOX's loss.

## Decision

Run the official **YOLOX-L COCO** model on every plate-only image in `train` and `val`. It is Apache-2.0, so its outputs have a clear licence, which SPEC §4.2 requires for pseudo-labels.
- **Settings:** ONNX, 640 px, BGR, NMS 0.45, with COCO car, motorcycle, bus and truck mapped to `vehicle`. Only boxes scoring **≥ 0.5** are kept as `vehicle` training labels.
- **Output:** `data/processed/pseudo_labels/yolox_l_vehicles.json` (git-ignored). It records the model URL and SHA-256 (`7860ae79…a411`), the threshold, and the combined manifest hash.
- **Stale-label check:** `make train-data` refuses pseudo-labels built for a different dataset.
- **Marking:** pseudo boxes carry `"pseudo": true` in the training annotations.
- **Scope:** pseudo-labels are used only in the YOLOX **training** files. Evaluation (`make eval`, baselines) uses the original labels and `annotated_classes`, so plate-only images still don't score vehicles.

## Consequences

- **Missed vehicles.** YOLOX-L misses some vehicles, especially small and distant ones. Those stay unlabelled and are still learned as background, though far fewer than before.
- **Wrong boxes.** A few pseudo-boxes will be wrong. The 0.5 threshold favours precision.
- **Val selection.** The val AP that YOLOX uses to pick `best_ckpt` includes pseudo-labelled vehicles, which is acceptable for choosing a checkpoint.
- **Reversible.** The step can be dropped in T4 by building training data without pseudo-labels, if they turn out to hurt.
