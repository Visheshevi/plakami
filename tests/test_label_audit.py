"""Plate label audit (ADR-0014): sheets, decisions file and ignored labels in metrics."""

import csv
import json

import pytest
from PIL import Image

from platedet.eval import label_audit
from platedet.eval.metrics import GroundTruth, ImageResult, recall_at
from platedet.postprocess import Detection


def _dataset(tmp_path, split="test"):
    root = tmp_path / split
    (root / "annotations").mkdir(parents=True)
    (root / "images").mkdir()
    Image.new("RGB", (200, 100), "grey").save(root / "images" / "a.jpg")
    dataset = {
        "images": [{"id": 1, "file_name": "images/a.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 7, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 12]},
            {"id": 8, "image_id": 1, "category_id": 1, "bbox": [100, 10, 8, 3]},  # unreadable
            {"id": 9, "image_id": 1, "category_id": 2, "bbox": [0, 0, 150, 90]},
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [120, 50, 30, 10]},
        ],
        "categories": [{"id": 1, "name": "license_plate"}, {"id": 2, "name": "vehicle"}],
    }
    (root / "annotations" / f"{split}.json").write_text(json.dumps(dataset))
    return root


def _write_review(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("box_id", "decision", "note"))
        writer.writerows(rows)


def test_prepare_writes_sheets_and_a_template_for_readable_plates_only(tmp_path):
    splits = {"test": _dataset(tmp_path)}
    counts = label_audit.prepare(tmp_path / "audit", splits)
    assert counts == {"test": 2}
    assert (tmp_path / "audit" / "sheets" / "test_001.jpg").is_file()
    with (tmp_path / "audit" / "review_template.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["box_id"], r["sheet"], r["number"]) for r in rows] == [
        ("test/7", "test_001", "1"),
        ("test/10", "test_001", "2"),
    ]


def test_finalize_and_flagged_ids_round_trip(tmp_path):
    splits = {"test": _dataset(tmp_path)}
    review = tmp_path / "review.csv"
    decisions, meta = tmp_path / "plate_labels.csv", tmp_path / "plate_labels.meta.json"
    _write_review(review, [("test/10", "loose", "box 2x plate"), ("test/7", "OK", "")])
    counts = label_audit.finalize(review, decisions, meta, splits)
    assert counts == {"ok": 1, "loose": 1}
    assert decisions.read_text().splitlines()[1:] == ["test/7,ok,", "test/10,loose,box 2x plate"]
    flagged = label_audit.flagged_annotation_ids("test", splits["test"], decisions, meta)
    assert flagged == frozenset({10})
    assert label_audit.flagged_annotation_ids("holdout", splits["test"], decisions, meta) is None


def test_finalize_rejects_incomplete_or_invalid_reviews(tmp_path):
    splits = {"test": _dataset(tmp_path)}
    review = tmp_path / "review.csv"
    _write_review(review, [("test/7", "ok", "")])
    with pytest.raises(ValueError, match="1 missing"):
        label_audit.finalize(review, tmp_path / "d.csv", tmp_path / "m.json", splits)
    _write_review(review, [("test/7", "maybe", ""), ("test/10", "ok", "")])
    with pytest.raises(ValueError, match="Invalid audit decision"):
        label_audit.finalize(review, tmp_path / "d.csv", tmp_path / "m.json", splits)


def test_flagged_ids_refuse_changed_annotations(tmp_path):
    splits = {"test": _dataset(tmp_path)}
    review = tmp_path / "review.csv"
    decisions, meta = tmp_path / "d.csv", tmp_path / "m.json"
    _write_review(review, [("test/7", "ok", ""), ("test/10", "not_plate", "empty holder")])
    label_audit.finalize(review, decisions, meta, splits)
    path = splits["test"] / "annotations" / "test.json"
    path.write_text(path.read_text().replace("120, 50", "121, 50"))
    with pytest.raises(ValueError, match="different test annotations"):
        label_audit.flagged_annotation_ids("test", splits["test"], decisions, meta)


def test_ignored_ids_are_neither_misses_nor_false_positives():
    image = ImageResult(
        file_name="a.jpg",
        annotated_classes=frozenset({1}),
        ground_truth=[GroundTruth(0, 0, 40, 10, 1, 1), GroundTruth(100, 0, 140, 10, 1, 2)],
        detections=[Detection(102, 0, 141, 10, 0.9, 1)],  # on the flagged label only
    )
    plain = recall_at([image], 1, 0.2)
    assert plain["missed"] == 1 and plain["ground_truth"] == 2
    audited = recall_at([image], 1, 0.2, ignored_ids=frozenset({2}))
    assert audited["ground_truth"] == 1 and audited["missed"] == 1
    assert audited["false_positives_per_image"] == 0.0
