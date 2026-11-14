# T2 Baseline

Run: `runs/baseline-yolox_tiny_coco-20261112-234836/` (git-ignored; rerun after ADR-0010 switched preprocessing to OpenCV-style resize). The first run, `…-221740`, used Pillow's resize. Command: `make baseline CONFIG=configs/baseline/yolox_tiny_coco.yaml`.

| Item | Value |
|---|---|
| Model | Official YOLOX-Tiny COCO checkpoint, ONNX, Apache-2.0 ([release 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx)) |
| Checkpoint SHA-256 | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` |
| Input | 416×416 (the official ONNX file is fixed), BGR, top-left letterbox with 114 padding, OpenCV-style resize (ADR-0010) |
| Classes | COCO car, motorcycle, bus and truck → `vehicle`. No plate class (COCO has none). |
| Dataset | `data/processed/combined`, manifest SHA-256 `b0506a1b1944…` (full hash in `summary.json`) |
| Code | commit `4405076` plus the uncommitted T2 work tree, committed as the `t2-baseline` merge |
| Thresholds | Vehicle recall at confidence 0.3, IoU 0.5, NMS IoU 0.45. Nothing tuned. |

There is no plate baseline. The model named in SPEC §6 doesn't exist with a known licence (see ADR-0007), so our T3 v0 model becomes the plate reference.

## Vehicle results

Vehicles are scored only on Open Images images, the only source with vehicle labels.

| Split | Images scored | Vehicle boxes | Recall @0.3 | FP / image | AP@0.5 |
|---|---|---|---|---|---|
| val | 706 | 1,923 | 0.705 | 1.34 | 0.596 |
| test | 779 | 2,302 | **0.693** | 1.53 | 0.585 |

SPEC §6 target for v1.0: vehicle recall ≥ 0.90 on `test`. The baseline is 0.21 below it. The first run, with Pillow's resize, gave test 0.687 / 1.49 / 0.582, so the resize change matters little. The size breakdown below is from the first run.

### Recall by box size (share of image area)

| Split | < 1% | 1–10% | 10–50% | 50–90% | ≥ 90% |
|---|---|---|---|---|---|
| val | 0.44 (617) | 0.82 (773) | 0.83 (291) | 0.92 (201) | 0.51 (41) |
| test | 0.45 (791) | 0.81 (879) | 0.83 (368) | 0.91 (215) | 0.39 (49) |

The box count for each size is in brackets.

## What the misses show

- **Small, distant vehicles are the main gap.** They are about a third of all vehicle boxes, and the model finds under half of them at 416. Training at 640 (SPEC §5) should help most here.
- **Labelling differences account for the largest misses** (gallery `misses_test_vehicle.jpg`). Open Images labels the whole frame as a vehicle in plate close-ups, in shots from inside a car (the car you're sitting in), in a horse-drawn carriage and in a motorbike tank close-up. A COCO model doesn't box these. They are only about 2% of vehicle boxes, but our own model will learn the Open Images convention.
  - Examples from `test`: `images/open_images_v6/818e4739ac9a0a98.jpg`, `8e6b94bc1e579dad.jpg`, `3092ab573617b501.jpg`, `ec56806003da28f2.jpg`, `39c5353896b2447c.jpg`.
- **False positives are partly label noise.** Open Images often leaves visible vehicles unlabelled, so some of the 1.3–1.5 false positives per image are real vehicles.

## Reproducing

`make baseline CONFIG=configs/baseline/yolox_tiny_coco.yaml` downloads the checkpoint into `data/models/` if it is missing, refuses it if the SHA-256 differs, and writes these files to `runs/<run_id>/`:

- `config.yaml`;
- `summary.json`, with the git state, manifest hash, checkpoint hash, metrics and worst misses;
- `predictions_<split>.jsonl`;
- `misses_<split>_vehicle.jpg`.
