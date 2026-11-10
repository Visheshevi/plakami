import csv
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from platedet.data.build import _coco_dataset, build_dataset
from platedet.data.dedupe import deduplicate_by_phash
from platedet.data.manifest import (
    OPEN_IMAGES_LICENSE_URL,
    ManifestRecord,
    make_manifest_record,
    validate_license,
    write_manifest,
)
from platedet.data.open_images import load_class_ids, read_target_annotations
from platedet.data.split import assign_split, validate_split_ratios


def test_loads_target_class_ids_and_boxes() -> None:
    descriptions = io.StringIO(
        "/plate,Vehicle registration plate\n"
        "/car,Car\n"
        "/truck,Truck\n"
        "/bus,Bus\n"
        "/van,Van\n"
        "/motorcycle,Motorcycle\n"
        "/land,Land vehicle\n"
    )
    class_ids = load_class_ids(descriptions)
    annotations = io.StringIO(
        "ImageID,LabelName,XMin,XMax,YMin,YMax\n"
        "0123456789abcdef,/plate,0.1,0.5,0.2,0.6\n"
        "0123456789abcdef,/car,0,1,0,1\n"
        "0123456789abcdef,/ignored,0,1,0,1\n"
    )

    result = read_target_annotations(annotations, class_ids)

    assert result["0123456789abcdef"] == [
        {"category_id": 1, "xmin": 0.1, "xmax": 0.5, "ymin": 0.2, "ymax": 0.6},
        {"category_id": 2, "xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0},
    ]


def test_rejects_invalid_boxes_and_missing_classes() -> None:
    with pytest.raises(ValueError, match="missing"):
        load_class_ids(io.StringIO("/plate,Vehicle registration plate\n"))

    class_ids = {"/plate": 1}
    rows = io.StringIO(
        "ImageID,LabelName,XMin,XMax,YMin,YMax\n0123456789abcdef,/plate,0.7,0.2,0.1,0.9\n"
    )
    with pytest.raises(ValueError, match="Out-of-range"):
        read_target_annotations(rows, class_ids)


def test_license_validation_requires_known_license_and_attribution() -> None:
    validate_license(OPEN_IMAGES_LICENSE_URL, "Author", "https://example.test/image")
    with pytest.raises(ValueError, match="Non-allowlisted"):
        validate_license("", "Author", "https://example.test/image")
    with pytest.raises(ValueError, match="author"):
        validate_license(OPEN_IMAGES_LICENSE_URL, " ", "https://example.test/image")


def test_manifest_records_hashes_and_writes_csv_and_parquet(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (12, 8), color=(30, 60, 90)).save(image_path)
    record = make_manifest_record(
        image_path=image_path,
        image_id="open_images_v6:0123456789abcdef",
        source_image_id="0123456789abcdef",
        url="https://example.test/photo/1",
        author="Example Author",
        licence_url=OPEN_IMAGES_LICENSE_URL,
        split="train",
        n_plates=1,
        n_vehicles=2,
        annotated_classes='["license_plate", "vehicle"]',
    )
    output_dir = tmp_path / "processed"

    write_manifest([record], output_dir)

    assert record.width == 12
    assert record.height == 8
    assert len(record.sha256) == 64
    assert len(record.phash) == 16
    with (output_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["licence"] == "CC BY 2.0"
    assert (output_dir / "manifest.parquet").is_file()


def test_split_is_deterministic_and_ratios_are_validated() -> None:
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    assert assign_split("open_images_v6:0123456789abcdef", ratios) == assign_split(
        "open_images_v6:0123456789abcdef", ratios
    )
    with pytest.raises(ValueError, match="sum to 1"):
        validate_split_ratios({"train": 0.7, "val": 0.1, "test": 0.1})


def test_phash_deduplication_keeps_first_match() -> None:
    kept, duplicate_count = deduplicate_by_phash(
        ["0000", "0001", "ffff"], phash_getter=lambda value: value, threshold=1
    )
    assert kept == ["0000", "ffff"]
    assert duplicate_count == 1


def test_coco_conversion_scales_normalized_boxes() -> None:
    record = ManifestRecord(
        image_id="open_images_v6:0123456789abcdef",
        source="open_images_v6",
        source_image_id="0123456789abcdef",
        url="https://example.test/photo/1",
        author="Example Author",
        licence="CC BY 2.0",
        licence_url=OPEN_IMAGES_LICENSE_URL,
        sha256="0" * 64,
        phash="0" * 16,
        width=100,
        height=50,
        split="train",
        n_plates=1,
        n_vehicles=0,
        annotated_classes='["license_plate", "vehicle"]',
    )

    dataset = _coco_dataset(
        [record],
        {
            record.source_image_id: [
                {"category_id": 1, "xmin": 0.1, "xmax": 0.5, "ymin": 0.2, "ymax": 0.6}
            ]
        },
    )

    assert dataset["images"][0]["file_name"] == "images/0123456789abcdef.jpg"
    assert dataset["annotations"][0]["bbox"] == [10.0, 10.0, 40.0, 20.0]
    json.dumps(dataset)


def test_builder_requires_explicit_image_limit_before_download(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(
        "source: open_images_v6\ndata_dir: data\nsource_splits: [train]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unbounded download"):
        build_dataset(config, max_images=None)

    assert not (tmp_path / "data").exists()
