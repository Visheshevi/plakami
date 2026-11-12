"""Detection metrics from SPEC §6: recall at a fixed confidence, small-plate recall,
false positives per image, and AP@0.5 per class.

A class is scored only on images where it is annotated (`annotated_classes`), so
plate-only sources neither count missed vehicles nor punish vehicle detections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from platedet.postprocess import Detection, iou_matrix

CLASS_NAMES = {1: "license_plate", 2: "vehicle"}
SMALL_PLATE_WIDTH = 32.0


@dataclass(frozen=True)
class GroundTruth:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1


@dataclass
class ImageResult:
    """Ground truth and detections for one image."""

    file_name: str
    annotated_classes: frozenset[int]
    ground_truth: list[GroundTruth]
    detections: list[Detection] = field(default_factory=list)


def match_image(
    ground_truth: list[GroundTruth],
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> tuple[list[bool], list[bool]]:
    """Greedy COCO-style matching for one class in one image.

    Detections are taken in descending score order; each claims the unmatched ground
    truth box with the highest IoU at or above the threshold. Returns (gt_matched,
    detection_is_true_positive), the latter in the given detection order.
    """
    gt_matched = [False] * len(ground_truth)
    detection_tp = [False] * len(detections)
    if not ground_truth or not detections:
        return gt_matched, detection_tp
    gt_boxes = np.array([[g.x1, g.y1, g.x2, g.y2] for g in ground_truth], dtype=np.float64)
    det_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float64)
    overlaps = iou_matrix(det_boxes, gt_boxes)
    for det_index in np.argsort([-d.score for d in detections], kind="stable"):
        candidates = np.where(gt_matched, -1.0, overlaps[det_index])
        best = int(candidates.argmax())
        if candidates[best] >= iou_threshold:
            gt_matched[best] = True
            detection_tp[det_index] = True
    return gt_matched, detection_tp


def _class_items(
    image: ImageResult, class_id: int, min_score: float
) -> tuple[list[GroundTruth], list[Detection]]:
    return (
        [g for g in image.ground_truth if g.class_id == class_id],
        [d for d in image.detections if d.class_id == class_id and d.score >= min_score],
    )


def recall_at(
    images: list[ImageResult],
    class_id: int,
    conf_threshold: float,
    *,
    iou_threshold: float = 0.5,
    max_width: float | None = None,
) -> dict[str, float | int | None]:
    """Recall, missed count and false positives per image at one confidence threshold.

    `max_width` restricts recall to ground-truth boxes narrower than that many pixels
    (false positives are then not reported, since detections have no size bucket).
    """
    total = matched = false_positives = scored_images = 0
    for image in images:
        if class_id not in image.annotated_classes:
            continue
        scored_images += 1
        ground_truth, detections = _class_items(image, class_id, conf_threshold)
        gt_matched, detection_tp = match_image(ground_truth, detections, iou_threshold)
        for box, hit in zip(ground_truth, gt_matched, strict=True):
            if max_width is not None and box.width >= max_width:
                continue
            total += 1
            matched += hit
        false_positives += detection_tp.count(False)
    return {
        "recall": matched / total if total else None,
        "ground_truth": total,
        "missed": total - matched,
        "false_positives_per_image": (
            false_positives / scored_images if scored_images and max_width is None else None
        ),
        "images": scored_images,
    }


def average_precision(
    images: list[ImageResult], class_id: int, *, iou_threshold: float = 0.5
) -> float | None:
    """All-point interpolated AP (VOC 2010+ / COCO-style area under the PR curve)."""
    scores: list[float] = []
    hits: list[bool] = []
    total = 0
    for image in images:
        if class_id not in image.annotated_classes:
            continue
        ground_truth, detections = _class_items(image, class_id, 0.0)
        total += len(ground_truth)
        _, detection_tp = match_image(ground_truth, detections, iou_threshold)
        scores.extend(d.score for d in detections)
        hits.extend(detection_tp)
    if total == 0:
        return None
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.cumsum(np.asarray(hits, dtype=np.float64)[order])
    fp = np.cumsum(1.0 - np.asarray(hits, dtype=np.float64)[order])
    recall = np.concatenate(([0.0], tp / total, [tp[-1] / total]))
    precision = np.concatenate(([1.0], tp / np.maximum(tp + fp, 1e-12), [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def missed_boxes(
    images: list[ImageResult], class_id: int, conf_threshold: float, iou_threshold: float = 0.5
) -> list[tuple[ImageResult, GroundTruth]]:
    """Ground-truth boxes not found at the threshold, largest (most obvious) first."""
    misses: list[tuple[ImageResult, GroundTruth]] = []
    for image in images:
        if class_id not in image.annotated_classes:
            continue
        ground_truth, detections = _class_items(image, class_id, conf_threshold)
        gt_matched, _ = match_image(ground_truth, detections, iou_threshold)
        misses.extend(
            (image, box) for box, hit in zip(ground_truth, gt_matched, strict=True) if not hit
        )
    return sorted(misses, key=lambda item: -(item[1].width * (item[1].y2 - item[1].y1)))


def summarize(
    images: list[ImageResult],
    *,
    plate_conf: float = 0.2,
    vehicle_conf: float = 0.3,
    iou_threshold: float = 0.5,
    model_classes: frozenset[int] = frozenset(CLASS_NAMES),
) -> dict[str, object]:
    """All SPEC §6 metrics for one split. Classes the model can't predict are reported as None."""
    summary: dict[str, object] = {"images": len(images)}
    thresholds = {1: plate_conf, 2: vehicle_conf}
    for class_id, name in CLASS_NAMES.items():
        if class_id not in model_classes:
            summary[name] = None
            continue
        entry: dict[str, object] = {
            "conf_threshold": thresholds[class_id],
            **recall_at(images, class_id, thresholds[class_id], iou_threshold=iou_threshold),
            "ap50": average_precision(images, class_id, iou_threshold=iou_threshold),
        }
        if class_id == 1:
            entry["small_plates"] = recall_at(
                images,
                1,
                plate_conf,
                iou_threshold=iou_threshold,
                max_width=SMALL_PLATE_WIDTH,
            )
        summary[name] = entry
    return summary
