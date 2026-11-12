import pytest

from platedet.eval.metrics import (
    GroundTruth,
    ImageResult,
    average_precision,
    match_image,
    missed_boxes,
    recall_at,
    summarize,
)
from platedet.postprocess import Detection

BOTH = frozenset({1, 2})
PLATE_ONLY = frozenset({1})


def gt(x1, y1, x2, y2, class_id=1):
    return GroundTruth(x1, y1, x2, y2, class_id)


def det(x1, y1, x2, y2, score, class_id=1):
    return Detection(x1, y1, x2, y2, score, class_id)


def test_match_image_greedy_by_score_and_one_to_one():
    truth = [gt(0, 0, 10, 10)]
    detections = [det(0, 0, 10, 10, 0.5), det(0, 0, 10, 9, 0.9)]
    gt_matched, tp = match_image(truth, detections)
    assert gt_matched == [True]
    assert tp == [False, True]  # the higher score claims the box, the duplicate is a FP


def test_recall_ignores_low_scores_and_counts_false_positives():
    image = ImageResult(
        "a.jpg",
        BOTH,
        [gt(0, 0, 10, 10), gt(50, 50, 60, 60)],
        [det(0, 0, 10, 10, 0.9), det(50, 50, 60, 60, 0.1), det(80, 80, 90, 90, 0.5)],
    )
    result = recall_at([image], 1, 0.2)
    assert result["recall"] == 0.5 and result["missed"] == 1
    assert result["false_positives_per_image"] == 1.0


def test_class_only_scored_where_annotated():
    plate_only = ImageResult("p.jpg", PLATE_ONLY, [gt(0, 0, 10, 10)], [det(0, 0, 50, 50, 0.9, 2)])
    full = ImageResult("f.jpg", BOTH, [gt(0, 0, 50, 50, 2)], [det(0, 0, 50, 50, 0.9, 2)])
    result = recall_at([plate_only, full], 2, 0.3)
    assert result["images"] == 1 and result["recall"] == 1.0
    assert result["false_positives_per_image"] == 0.0


def test_small_plate_bucket_uses_ground_truth_width():
    image = ImageResult(
        "a.jpg", BOTH, [gt(0, 0, 20, 8), gt(100, 100, 200, 140)], [det(100, 100, 200, 140, 0.9)]
    )
    small = recall_at([image], 1, 0.2, max_width=32)
    assert small["ground_truth"] == 1 and small["recall"] == 0.0
    assert small["false_positives_per_image"] is None


def test_average_precision_hand_computed():
    image = ImageResult(
        "a.jpg",
        BOTH,
        [gt(0, 0, 10, 10), gt(20, 20, 30, 30)],
        [det(0, 0, 10, 10, 0.9), det(50, 50, 60, 60, 0.8), det(20, 20, 30, 30, 0.7)],
    )
    # PR points: (0.5, 1.0), (0.5, 0.5), (1.0, 2/3) -> AP = 0.5 * 1 + 0.5 * 2/3
    assert average_precision([image], 1) == pytest.approx(0.5 + 1 / 3)
    assert average_precision([ImageResult("b.jpg", BOTH, [], [])], 1) is None


def test_missed_boxes_largest_first_and_summary_marks_unpredicted_classes():
    image = ImageResult(
        "a.jpg", BOTH, [gt(0, 0, 10, 10), gt(0, 0, 100, 100), gt(0, 0, 5, 5, 2)], []
    )
    misses = missed_boxes([image], 1, 0.2)
    assert [box.x2 for _, box in misses] == [100, 10]
    summary = summarize([image], model_classes=frozenset({2}))
    assert summary["license_plate"] is None
    assert summary["vehicle"]["recall"] == 0.0 and summary["vehicle"]["ap50"] == 0.0
