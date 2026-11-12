# Pre- and post-processing contract

The reference implementation is [`src/platedet/postprocess.py`](../src/platedet/postprocess.py). It is shipped with every release. Ports to other languages must match it. Test vectors (an input image and its expected boxes) are added in T5, once the release model exists.

## Pre-processing (`letterbox`)

1. Decode the image, apply EXIF orientation if your platform doesn't do it automatically, and convert it to 3-channel RGB.
2. `ratio = min(S / width, S / height)`, where `S` is the model input size (from `release.json`).
3. Resize to `round(width * ratio) × round(height * ratio)` with bilinear filtering (Pillow `BILINEAR`).
4. Paste the resized image at the **top-left** of an `S × S` canvas filled with **114** on all channels. This matches YOLOX: the padding goes on the right and bottom, not centred.
5. Use float32 values in 0–255, with **no mean/std normalisation**, laid out as NCHW `(1, 3, S, S)`.
6. Channel order: the model's `channel_order`. The official YOLOX COCO weights expect **BGR**. The SPEC §5 contract is RGB; our own models' order is fixed in T3 and recorded in `release.json`.

## Post-processing (`postprocess`)

The model outputs `(1, N, 5 + C)` raw head predictions, with `N = Σ (S / s)²` for strides `s` = 8, 16 and 32. For example, `N` = 3,549 at 416 and 8,400 at 640. Columns are `dx, dy, log_w, log_h, objectness, class_1 … class_C`. Objectness and class values are already probabilities.

1. **Decode.** Grid cells are ordered row-major within each stride, with strides ascending: `cx = (dx + grid_x) * s`, `cy = (dy + grid_y) * s`, `w = exp(log_w) * s`, `h = exp(log_h) * s`.
2. **Score.** `score = objectness × max(class probabilities)`, and the class is the argmax.
3. **Class map (optional).** Remap model classes to output classes and drop unmapped ones before NMS. This is used only by baselines, for example COCO car, motorcycle, bus and truck → `vehicle`.
4. **Threshold.** Keep boxes with `score ≥ conf_threshold`.
5. **Convert and scale.** Convert to `x1, y1, x2, y2`, divide by `ratio`, then clip to the original image size.
6. **NMS.** Class-aware greedy NMS with IoU > `nms_iou` (default **0.45**) suppressed, highest score first.
7. Output class IDs follow `labels.txt`: 1 = `license_plate`, 2 = `vehicle`.

## Evaluation conventions (`src/platedet/eval/metrics.py`)

- A detection matches a ground-truth box at IoU ≥ 0.5. Matching is greedy by score and one-to-one.
- Recall is computed at the SPEC §6 thresholds: plates 0.2, vehicles 0.3.
- Small plates are those whose ground truth is under 32 px wide in original pixels.
- A class is scored only on images where it is annotated (`annotated_classes`). Plate-only sources (CCPD, Kaggle) don't count vehicle misses or vehicle false positives.
- AP@0.5 is the all-point interpolated area under the precision/recall curve, using detections down to `collect_conf` (0.01).
