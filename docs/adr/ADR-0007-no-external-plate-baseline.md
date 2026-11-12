# ADR-0007: No external plate baseline; our first trained model is the plate reference

- **Status:** Accepted (owner approved 2026-11-12)
- **Date:** 2026-11-12

## Context

SPEC §6 and §10 (T2) say to beat "the best licence-clean open plate model found (RT-DETRv2-R18 trained on Open Images, Apache-2.0)". A search on 2026-11-12 found no such model. The published RT-DETRv2-R18 checkpoints (`PekingU/rtdetr_v2_r18vd`, Apache-2.0) are trained on COCO, which has no licence-plate class. The public RT-DETRv2 plate detectors (for example `justjuu/rtdetr-v2-license-plate-detection` and `Garon16/rtdetr_r50vd_russia_plate_detector_lightning`) are R50 models fine-tuned on Roboflow or other data with no documented licence. CLAUDE.md forbids using any checkpoint with an unknown licence, even for comparison.

## Decision

T2 evaluates only the official YOLOX-Tiny COCO checkpoint (Apache-2.0) for the `vehicle` class, using the COCO classes car, motorcycle, bus and truck. There is no external plate baseline. The plate reference for T4 is our first trained model from T3 (`v0`), measured with the same metrics code. Fine-tuning RT-DETRv2-R18 ourselves as a second model was considered and rejected for now: it is a full extra training job on free compute.

## Consequences

The SPEC §6 row "Beat the baseline … higher plate recall at the same false-positive rate" cannot be measured as written. Plate acceptance relies on the absolute targets (plate recall ≥ 0.95, small-plate recall ≥ 0.85, FP/image ≤ 1.0). Proposed SPEC wording for the owner to approve: *"Beat the baseline: vehicle recall/AP above the COCO YOLOX-Tiny baseline (T2); plate metrics improve on T3 v0 or meet the absolute targets."* If a licence-clean plate model appears later, it can be added as another baseline config without code changes.
