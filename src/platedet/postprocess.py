"""Reference pre- and post-processing for YOLOX detectors (documented in docs/preprocessing.md).

This module is the single implementation shipped with releases. Keep it dependency-light
(NumPy + Pillow) so it is easy to port.

Pre-processing follows YOLOX `ValTransform(legacy=False)`:
  - resize keeping the aspect ratio so the longer side fits `size` (bilinear),
  - paste at the top-left of a `size` x `size` canvas filled with 114,
  - float32 values in 0-255, no mean/std, channels-first, batch of 1.

Post-processing:
  - decode raw head outputs (grid offsets, log-space sizes) for strides 8/16/32,
  - score = objectness x best class probability,
  - optional class remapping (e.g. COCO car/truck/bus -> vehicle) before NMS,
  - class-aware NMS, then boxes scaled back to original image pixels and clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

PAD_VALUE = 114
STRIDES = (8, 16, 32)
DEFAULT_NMS_IOU = 0.45
CHANNEL_ORDERS = ("rgb", "bgr")


@dataclass(frozen=True)
class Detection:
    """One box in original-image pixels (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


def letterbox(
    image: Image.Image, size: int, channel_order: str = "rgb"
) -> tuple[np.ndarray, float]:
    """Return a (1, 3, size, size) float32 tensor and the resize ratio."""
    if channel_order not in CHANNEL_ORDERS:
        raise ValueError(f"channel_order must be one of {CHANNEL_ORDERS}")
    image = image.convert("RGB")
    ratio = min(size / image.width, size / image.height)
    new_width = max(1, round(image.width * ratio))
    new_height = max(1, round(image.height * ratio))
    resized = np.asarray(image.resize((new_width, new_height), Image.Resampling.BILINEAR))
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.float32)
    canvas[:new_height, :new_width] = resized
    if channel_order == "bgr":
        canvas = canvas[:, :, ::-1]
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), ratio


def decode_outputs(raw: np.ndarray, size: int, strides: tuple[int, ...] = STRIDES) -> np.ndarray:
    """Decode (N, 5 + C) raw outputs into (N, 5 + C) with boxes as cx, cy, w, h in input pixels.

    Objectness and class columns are expected to be probabilities already (YOLOX applies
    the sigmoid inside the graph when `decode_in_inference` is off).
    """
    grids = []
    stride_column = []
    for stride in strides:
        cells = size // stride
        ys, xs = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
        grids.append(np.stack((xs, ys), axis=-1).reshape(-1, 2))
        stride_column.append(np.full((cells * cells, 1), stride))
    grid = np.concatenate(grids).astype(np.float32)
    stride_values = np.concatenate(stride_column).astype(np.float32)
    if raw.shape[0] != grid.shape[0]:
        raise ValueError(
            f"Expected {grid.shape[0]} predictions for size {size}, got {raw.shape[0]}"
        )
    decoded = raw.astype(np.float32).copy()
    decoded[:, :2] = (decoded[:, :2] + grid) * stride_values
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * stride_values
    return decoded


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy arrays."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS; returns kept indices in descending score order (ties keep input order)."""
    order = list(np.argsort(-scores, kind="stable"))
    keep: list[int] = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        if not order:
            break
        overlaps = iou_matrix(boxes[best : best + 1], boxes[order])[0]
        order = [
            index
            for index, overlap in zip(order, overlaps, strict=True)
            if overlap <= iou_threshold
        ]
    return keep


def postprocess(
    raw: np.ndarray,
    *,
    size: int,
    ratio: float,
    image_width: int,
    image_height: int,
    conf_threshold: float,
    nms_iou: float = DEFAULT_NMS_IOU,
    class_map: dict[int, int] | None = None,
) -> list[Detection]:
    """Turn one image's raw (N, 5 + C) model output into detections in original pixels.

    `class_map` maps model class indices to output class IDs; unmapped classes are dropped.
    Without it, output class IDs are model index + 1 (COCO category IDs: 1 = license_plate).
    """
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise ValueError("postprocess handles one image at a time")
        raw = raw[0]
    decoded = decode_outputs(raw, size)
    class_scores = decoded[:, 5:]
    best_class = class_scores.argmax(axis=1)
    scores = decoded[:, 4] * class_scores[np.arange(len(decoded)), best_class]
    if class_map is None:
        class_ids = best_class + 1
        keep_mask = scores >= conf_threshold
    else:
        lookup = np.full(class_scores.shape[1], -1)
        for source, target in class_map.items():
            lookup[source] = target
        class_ids = lookup[best_class]
        keep_mask = (scores >= conf_threshold) & (class_ids >= 0)
    if not keep_mask.any():
        return []
    centres = decoded[keep_mask, :4]
    boxes = (
        np.stack(
            (
                centres[:, 0] - centres[:, 2] / 2,
                centres[:, 1] - centres[:, 3] / 2,
                centres[:, 0] + centres[:, 2] / 2,
                centres[:, 1] + centres[:, 3] / 2,
            ),
            axis=1,
        )
        / ratio
    )
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_height)
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]
    detections: list[Detection] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        for kept in nms(boxes[indices], scores[indices], nms_iou):
            index = indices[kept]
            detections.append(
                Detection(*map(float, boxes[index]), float(scores[index]), int(class_id))
            )
    return sorted(detections, key=lambda detection: -detection.score)
