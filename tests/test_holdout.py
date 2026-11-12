import json

import pytest
from PIL import Image

from platedet.data.holdout import (
    finalize_holdout,
    holdout_candidates,
    read_review,
    write_review_sheets,
)
from platedet.data.manifest import make_manifest_record, write_manifest


def _plate(xmin=0.1):
    return {"category_id": 1, "xmin": xmin, "xmax": xmin + 0.2, "ymin": 0.4, "ymax": 0.5}


VEHICLE = {"category_id": 2, "xmin": 0.0, "xmax": 0.9, "ymin": 0.1, "ymax": 0.9}


def test_candidates_need_a_plate_skip_excluded_and_are_stable():
    annotations = {f"{i:016x}": [_plate()] for i in range(20)}
    annotations["ffffffffffffffff"] = [VEHICLE]
    picked = holdout_candidates(annotations, {"0000000000000003"}, 10)
    assert len(picked) == 10 and "0000000000000003" not in picked
    assert "ffffffffffffffff" not in picked
    assert picked == holdout_candidates(
        dict(reversed(annotations.items())), {"0000000000000003"}, 10
    )


def test_read_review_rejects_bad_decisions_and_duplicates(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("image_id,decision,reason\na,accept,\nb,REJECT,unlabelled_plate\n")
    assert read_review(path) == {"a": ("accept", ""), "b": ("reject", "unlabelled_plate")}
    path.write_text("image_id,decision,reason\na,maybe,\n")
    with pytest.raises(ValueError, match="Invalid review decision"):
        read_review(path)
    path.write_text("image_id,decision,reason\na,accept,\na,reject,x\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_review(path)


def _candidates(tmp_path, ids):
    holdout_dir = tmp_path / "data" / "processed" / "holdout"
    image_dir = holdout_dir / "images"
    image_dir.mkdir(parents=True)
    records = []
    for index, image_id in enumerate(ids):
        path = image_dir / f"{image_id}.jpg"
        Image.new("RGB", (200, 100), (index * 40, 0, 0)).save(path)
        records.append(
            make_manifest_record(
                image_path=path,
                image_id=f"open_images_v6:{image_id}",
                source_image_id=image_id,
                url=f"https://www.flickr.com/photos/x/{index}",
                author="someone",
                licence_url="https://creativecommons.org/licenses/by/2.0/",
                split="holdout",
                n_plates=1,
                n_vehicles=0,
                annotated_classes=("license_plate",),
            )
        )
    write_manifest(records, holdout_dir / "candidates")
    boxes = {image_id: [_plate()] for image_id in ids}
    (holdout_dir / "candidates" / "plate_boxes.json").write_text(json.dumps(boxes))
    return holdout_dir, records, boxes


def test_finalize_keeps_only_accepted_and_writes_plate_only_coco(tmp_path):
    ids = ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"]
    holdout_dir, _, _ = _candidates(tmp_path, ids)
    review = tmp_path / "review.csv"
    review.write_text(
        "image_id,decision,reason\naaaaaaaaaaaaaaaa,accept,\nbbbbbbbbbbbbbbbb,reject,group_box\n"
    )
    config = {"data_dir": str(tmp_path / "data"), "review_file": str(review)}
    stats = finalize_holdout(config)
    assert stats["accepted_images"] == 1 and stats["reject_reasons"] == {"group_box": 1}
    coco = json.loads((holdout_dir / "annotations" / "holdout.json").read_text())
    (image,) = coco["images"]
    assert image["file_name"] == "images/aaaaaaaaaaaaaaaa.jpg"
    assert image["annotated_classes"] == ["license_plate"]
    assert (holdout_dir / image["file_name"]).is_file()
    assert coco["annotations"][0]["bbox"] == pytest.approx([20, 40, 40, 10])
    assert "holdout" in (holdout_dir / "manifest.csv").read_text()


def test_finalize_requires_a_decision_for_every_candidate(tmp_path):
    _candidates(tmp_path, ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])
    review = tmp_path / "review.csv"
    config = {"data_dir": str(tmp_path / "data"), "review_file": str(review)}
    review.write_text("image_id,decision,reason\naaaaaaaaaaaaaaaa,accept,\n")
    with pytest.raises(ValueError, match="no review decision"):
        finalize_holdout(config)
    review.write_text(
        "image_id,decision,reason\naaaaaaaaaaaaaaaa,accept,\nbbbbbbbbbbbbbbbb,accept,\n"
        "cccccccccccccccc,accept,\n"
    )
    with pytest.raises(ValueError, match="not candidates"):
        finalize_holdout(config)


def test_review_sheets_are_written(tmp_path):
    ids = [f"{i:016x}" for i in range(10)]
    holdout_dir, records, boxes = _candidates(tmp_path, ids)
    write_review_sheets(records, boxes, holdout_dir / "images", tmp_path / "review")
    assert sorted(p.name for p in (tmp_path / "review").iterdir()) == [
        "sheet_01.jpg",
        "sheet_02.jpg",
    ]


def test_prepare_refuses_holdout_splits_that_overlap_training(tmp_path):
    from platedet.data.holdout import prepare_holdout

    config = {
        "data_dir": str(tmp_path),
        "source_splits": ["validation"],
        "training_source_splits": ["train", "validation"],
        "candidates": 5,
    }
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_holdout(config)
