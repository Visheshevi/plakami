import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from platedet.data import kaggle_build
from platedet.data.build import build_dataset
from platedet.data.kaggle import (
    KAGGLE_DATASET_URL,
    KAGGLE_LICENSE_URL,
    parse_voc_xml,
    read_image_from_archive,
    select_annotations,
)
from platedet.data.manifest import validate_license


def _voc_xml(filename: str = "Cars0.jpg") -> bytes:
    return f"""<annotation>
      <filename>{filename}</filename>
      <size><width>32</width><height>24</height><depth>3</depth></size>
      <object><name>license plate</name><bndbox>
        <xmin>5</xmin><ymin>4</ymin><xmax>20</xmax><ymax>16</ymax>
      </bndbox></object>
    </annotation>""".encode()


def _jpg_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), "white")
    ImageDraw.Draw(image).rectangle((4, 3, 19, 15), fill="black")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_parse_voc_converts_inclusive_pixel_box_to_normalized_edges() -> None:
    item = parse_voc_xml(_voc_xml(), "annotations/Cars0.xml")

    assert item.filename == "Cars0.jpg"
    assert item.width == 32
    assert item.height == 24
    assert item.boxes == ((4 / 32, 3 / 24, 20 / 32, 16 / 24),)


def test_parse_voc_rejects_unknown_labels_and_unsafe_filename() -> None:
    with pytest.raises(ValueError, match="Unexpected class"):
        parse_voc_xml(_voc_xml().replace(b"license plate", b"vehicle"), "annotations/Cars0.xml")
    with pytest.raises(ValueError, match="Unsafe or unsupported"):
        parse_voc_xml(_voc_xml("../Cars0.jpg"), "annotations/Cars0.xml")


def test_parse_voc_accepts_british_license_spelling() -> None:
    item = parse_voc_xml(_voc_xml().replace(b"license plate", b"licence"), "annotations/Cars0.xml")

    assert len(item.boxes) == 1


def test_kaggle_cc0_license_is_allowlisted_with_source_metadata() -> None:
    validate_license(
        KAGGLE_LICENSE_URL,
        "andrewmvd (Kaggle uploader)",
        KAGGLE_DATASET_URL,
        source="kaggle_car_plate_detection",
    )
    with pytest.raises(ValueError, match="Non-allowlisted"):
        validate_license(
            "https://example.test/license",
            "andrewmvd",
            KAGGLE_DATASET_URL,
            source="kaggle_car_plate_detection",
        )


def test_archive_selects_annotations_and_finds_matching_image(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("annotations/Cars0.xml", _voc_xml())
        archive.writestr("images/Cars0.jpg", _jpg_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        selected, source_count = select_annotations(archive, max_images=1)
        image_bytes = read_image_from_archive(archive, selected[0].filename)

    assert source_count == 1
    assert selected[0].filename == "Cars0.jpg"
    with Image.open(io.BytesIO(image_bytes)) as image:
        assert image.size == (32, 24)


def test_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../annotations/Cars0.xml", _voc_xml())
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="Unsafe path"):
        select_annotations(archive, max_images=1)


def test_kaggle_builder_creates_manifest_and_coco_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_fixture_archive(destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("annotations/Cars0.xml", _voc_xml())
            archive.writestr("images/Cars0.jpg", _jpg_bytes())

    monkeypatch.setattr(kaggle_build, "download_archive", write_fixture_archive)
    config = tmp_path / "kaggle.yaml"
    config.write_text(
        "source: kaggle_car_plate_detection\n"
        f"data_dir: {tmp_path / 'data'}\n"
        "split_ratios:\n  train: 0.8\n  val: 0.1\n  test: 0.1\n"
        "phash_distance: 0\n",
        encoding="utf-8",
    )

    result = build_dataset(config, max_images=1)

    processed = tmp_path / "data/processed/kaggle_car_plate_detection"
    stats = json.loads((processed / "stats.json").read_text())
    manifest = (processed / "manifest.csv").read_text()
    coco_files = [
        json.loads(path.read_text()) for path in (processed / "annotations").glob("*.json")
    ]
    coco = next(item for item in coco_files if item["images"])
    assert result == {"images": 1, "duplicates": 0, "failures": 0}
    assert stats["vehicle_boxes"] == 0
    assert "CC0 1.0" in manifest
    assert (processed / "images/Cars0.jpg").is_file()
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["licenses"][0]["url"] == KAGGLE_LICENSE_URL
    assert (processed / "ATTRIBUTION.md").is_file()
    assert (processed / "raw").exists() is False
