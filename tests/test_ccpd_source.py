import csv
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from platedet.data.build import build_dataset
from platedet.data.ccpd import parse_ccpd_filename
from platedet.data.combine import apply_ccpd_share_cap, build_combined_dataset
from platedet.data.manifest import (
    CCPD_LICENSE_URL,
    KAGGLE_LICENSE_URL,
    ManifestRecord,
    write_manifest,
)


def _ccpd_name() -> str:
    return "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"


def _record(source: str, image_id: str, plates: int) -> ManifestRecord:
    return ManifestRecord(
        image_id=image_id,
        source=source,
        source_image_id=image_id,
        url="https://example.test/dataset",
        author="test author",
        licence="MIT" if source == "ccpd" else "CC0 1.0",
        licence_url=CCPD_LICENSE_URL if source == "ccpd" else KAGGLE_LICENSE_URL,
        sha256="0" * 64,
        phash="0" * 16,
        width=10,
        height=10,
        split="train",
        n_plates=plates,
        n_vehicles=0,
        annotated_classes='["license_plate"]',
    )


def test_parse_ccpd_filename_extracts_bbox_coordinates(tmp_path: Path) -> None:
    path = tmp_path / _ccpd_name()

    item = parse_ccpd_filename(path, tmp_path)

    assert (item.xmin, item.ymin, item.xmax, item.ymax) == (154, 383, 386, 473)
    assert item.relative_path == _ccpd_name()


def test_parse_ccpd_filename_rejects_malformed_and_outside_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_ccpd_filename(tmp_path / "not-ccpd.jpg", tmp_path)

    outside = tmp_path.parent / "outside.jpg"
    with pytest.raises(ValueError, match="outside configured root"):
        parse_ccpd_filename(outside, tmp_path)


def test_ccpd_cap_limits_plate_box_share_deterministically() -> None:
    records = [
        *[_record("kaggle_car_plate_detection", f"kaggle:{index}", 1) for index in range(10)],
        *[_record("ccpd", f"ccpd:{index}", 1) for index in range(10)],
    ]

    included, removed = apply_ccpd_share_cap(records, 0.30)

    included_ccpd = [record for record in included if record.source == "ccpd"]
    total_plates = sum(record.n_plates for record in included)
    assert len(included_ccpd) == 4
    assert removed == 6
    assert sum(record.n_plates for record in included_ccpd) / total_plates <= 0.30
    repeated, _ = apply_ccpd_share_cap(records, 0.30)
    assert {record.image_id for record in included_ccpd} == {
        record.image_id for record in repeated if record.source == "ccpd"
    }


def test_ccpd_cap_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        apply_ccpd_share_cap([], 1.0)


def test_ccpd_builder_reads_local_extracted_images_and_outputs_coco(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "data/raw/ccpd/extracted/base"
    source_dir.mkdir(parents=True)
    image_path = source_dir / _ccpd_name()
    image = Image.new("RGB", (512, 720), "white")
    ImageDraw.Draw(image).rectangle((154, 383, 385, 472), fill="black")
    image.save(image_path)
    config = tmp_path / "ccpd.yaml"
    config.write_text(
        "source: ccpd\n"
        f"source_dir: {source_dir.parent}\n"
        f"data_dir: {tmp_path / 'data'}\n"
        "split_ratios:\n  train: 0.8\n  val: 0.1\n  test: 0.1\n"
        "phash_distance: 0\n",
        encoding="utf-8",
    )

    result = build_dataset(config, max_images=1)

    processed = tmp_path / "data/processed/ccpd"
    manifest_rows = list(csv.DictReader((processed / "manifest.csv").open()))
    stats = json.loads((processed / "stats.json").read_text())
    coco_files = [
        json.loads(path.read_text()) for path in (processed / "annotations").glob("*.json")
    ]
    coco = next(item for item in coco_files if item["images"])
    assert result == {"images": 1, "duplicates": 0, "failures": 0}
    assert manifest_rows[0]["licence"] == "MIT"
    assert manifest_rows[0]["annotated_classes"] == '["license_plate"]'
    assert stats["vehicle_boxes"] == 0
    assert coco["images"][0]["annotated_classes"] == ["license_plate"]
    assert coco["annotations"][0]["bbox"] == [154.0, 383.0, 232.0, 90.0]
    assert coco["images"][0]["file_name"] == f"images/base/{_ccpd_name()}"
    assert (processed / coco["images"][0]["file_name"]).is_file()
    assert (processed / "ATTRIBUTION.md").is_file()


def test_combiner_deduplicates_caps_ccpd_and_keeps_annotation_scope(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    source_names = ["kaggle_car_plate_detection", "ccpd"]
    expected_scope = {
        "kaggle_car_plate_detection": '["license_plate"]',
        "ccpd": '["license_plate"]',
    }
    for source in source_names:
        source_dir = data_dir / "processed" / source
        image_dir = source_dir / "images"
        image_dir.mkdir(parents=True)
        records = []
        images = []
        annotations = []
        for index in range(10):
            name = f"image-{index}.jpg"
            Image.new("RGB", (16, 16), (index * 20, 40, 100)).save(image_dir / name)
            record = _record(source, name, 1)
            record = ManifestRecord(
                **{
                    **record.__dict__,
                    "image_id": f"{source}:{index:02d}",
                    "source_image_id": name,
                    "phash": f"{index + (100 if source == 'ccpd' else 0):016x}",
                    "annotated_classes": expected_scope[source],
                }
            )
            records.append(record)
            images.append(
                {
                    "id": index + 1,
                    "source_image_id": name,
                    "file_name": f"images/{name}",
                    "width": 16,
                    "height": 16,
                    "annotated_classes": ["license_plate"],
                }
            )
            annotations.append(
                {
                    "id": index + 1,
                    "image_id": index + 1,
                    "category_id": 1,
                    "bbox": [1, 1, 5, 5],
                    "area": 25,
                    "iscrowd": 0,
                }
            )
        write_manifest(records, source_dir)
        annotation_dir = source_dir / "annotations"
        annotation_dir.mkdir()
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "license_plate"},
                {"id": 2, "name": "vehicle"},
            ],
        }
        (annotation_dir / "train.json").write_text(json.dumps(coco))
        for split in ("val", "test"):
            empty_coco = {**coco, "images": [], "annotations": []}
            (annotation_dir / f"{split}.json").write_text(json.dumps(empty_coco))

    config = tmp_path / "combined.yaml"
    config.write_text(
        f"data_dir: {data_dir}\n"
        "sources: [kaggle_car_plate_detection, ccpd]\n"
        "max_ccpd_plate_share: 0.30\n"
        "phash_distance: 0\n"
        "split_ratios:\n  train: 0.8\n  val: 0.1\n  test: 0.1\n",
        encoding="utf-8",
    )

    stats = build_combined_dataset(config)

    combined_dir = data_dir / "processed/combined"
    assert stats["images"] == 14
    assert stats["ccpd_images_dropped_by_plate_share_cap"] == 6
    assert stats["actual_ccpd_plate_share"] <= 0.30
    assert stats["annotation_scope_counts"]["plate_only_images"] == 14
    assert sum(path.is_symlink() for path in (combined_dir / "images/ccpd").glob("*.jpg")) == 4
    combined_coco = [
        json.loads(path.read_text()) for path in (combined_dir / "annotations").glob("*.json")
    ]
    assert all(
        "vehicle" not in image["annotated_classes"]
        for dataset in combined_coco
        for image in dataset["images"]
    )
