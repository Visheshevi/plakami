"""Tiled detection (ADR-0013) and the unreadable-plate ignore rule in metrics."""

import pytest
from PIL import Image

from platedet.eval.metrics import GroundTruth, ImageResult, average_precision, recall_at, summarize
from platedet.postprocess import Detection, detect_tiled, merge_detections, tile_windows


def test_tile_windows_cover_the_image_with_overlap():
    windows = tile_windows(1024, 768, 2, overlap=0.2)
    assert windows == [
        (0, 0, 569, 427),
        (455, 0, 1024, 427),
        (0, 341, 569, 768),
        (455, 341, 1024, 768),
    ]
    assert tile_windows(100, 50, 1) == [(0, 0, 100, 50)]
    with pytest.raises(ValueError):
        tile_windows(100, 100, 0)


def test_merge_detections_is_class_aware():
    a = Detection(0, 0, 10, 10, 0.9, 1)
    b = Detection(1, 1, 10, 10, 0.5, 1)  # same plate, lower score
    c = Detection(0, 0, 10, 10, 0.8, 2)  # same box, other class
    assert merge_detections([b, a, c], 0.45) == [a, c]


def test_detect_tiled_shifts_tile_boxes_and_drops_cut_ones():
    image = Image.new("RGB", (200, 100))

    def detect(crop):
        if crop.size == (200, 100):  # whole image: nothing found
            return []
        # Every tile reports one box in its middle and one touching its left edge.
        w, h = crop.size
        return [
            Detection(w / 2 - 5, h / 2 - 2, w / 2 + 5, h / 2 + 2, 0.9, 1),
            Detection(0, 10, 8, 14, 0.8, 1),
        ]

    detections = detect_tiled(image, detect, grid=2, overlap=0.2)
    windows = tile_windows(200, 100, 2, 0.2)
    centres = sorted(((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2) for d in detections if d.score == 0.9)
    assert centres == sorted(((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in windows)
    # Left-edge boxes survive only for tiles that start at the image's left edge.
    edge = [d for d in detections if d.score == 0.8]
    assert len(edge) == 2 and all(d.x1 == 0 for d in edge)


def test_detect_tiled_with_grid_1_is_a_single_pass():
    calls = []

    def detect(crop):
        calls.append(crop.size)
        return [Detection(1, 1, 5, 5, 0.5, 1)]

    assert detect_tiled(Image.new("RGB", (64, 48)), detect, grid=1) == [
        Detection(1, 1, 5, 5, 0.5, 1)
    ]
    assert calls == [(64, 48)]


def _image():
    return ImageResult(
        file_name="a.jpg",
        annotated_classes=frozenset({1, 2}),
        ground_truth=[
            GroundTruth(0, 0, 40, 10, 1),  # readable, found
            GroundTruth(100, 0, 110, 4, 1),  # 10 px wide: unreadable
            GroundTruth(200, 0, 220, 6, 1),  # 20 px: small, missed
        ],
        detections=[
            Detection(0, 0, 40, 10, 0.9, 1),
            Detection(100, 0, 110, 4, 0.8, 1),  # on the unreadable plate
            Detection(300, 0, 340, 10, 0.7, 1),  # false positive
        ],
    )


def test_unreadable_plates_are_neither_misses_nor_false_positives():
    image = _image()
    everything = recall_at([image], 1, 0.2)
    assert everything["ground_truth"] == 3 and everything["missed"] == 1
    assert everything["false_positives_per_image"] == 1.0
    readable = recall_at([image], 1, 0.2, min_plate_width=12)
    assert readable["ground_truth"] == 2 and readable["missed"] == 1
    assert readable["false_positives_per_image"] == 1.0  # only the real false positive
    assert average_precision([image], 1, min_plate_width=12) == pytest.approx(0.5)
    small = summarize([image], min_plate_width=12)["license_plate"]["small_plates"]
    assert small["ground_truth"] == 1 and small["missed"] == 1
