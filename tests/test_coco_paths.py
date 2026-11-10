"""Every COCO `file_name` must point to a real image, for each source layout and when combined."""

import json
from pathlib import Path

import pytest
from PIL import Image

from platedet.data.build import image_relative_path
from platedet.data.combine import build_combined_dataset
from platedet.data.manifest import (
    CCPD_LICENSE_URL,
    OPEN_IMAGES_LICENSE_URL,
    make_manifest_record,
    write_manifest,
)

OPEN_IMAGES_ID = "0123456789abcdef"
CCPD_PATH = "CCPD2020/ccpd_green/train/image-0.jpg"


def test_image_relative_path_keeps_folders_and_adds_jpg_to_bare_ids() -> None:
    assert image_relative_path(OPEN_IMAGES_ID) == f"{OPEN_IMAGES_ID}.jpg"
    assert image_relative_path(CCPD_PATH) == CCPD_PATH
    assert image_relative_path("Cars0.png") == "Cars0.png"
    for unsafe in ("../escape.jpg", "/abs/path.jpg", ""):
        with pytest.raises(ValueError, match="Unsafe"):
            image_relative_path(unsafe)


def _write_source(
    data_dir: Path, source: str, source_image_id: str, stored_path: str, color: int
) -> None:
    """Write a one-image processed source the way the per-source builders do."""
    source_dir = data_dir / "processed" / source
    image_path = source_dir / "images" / stored_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Distinct patterns so pHash dedupe keeps both images.
    image = Image.new("RGB", (64, 64), (color, color, color))
    for x in range(0, 64, 8 if source == "ccpd" else 16):
        for y in range(64):
            image.putpixel((x, y), (255 - color, 0, 0))
    image.save(image_path)
    is_ccpd = source == "ccpd"
    record = make_manifest_record(
        image_path=image_path,
        image_id=f"{source}:{source_image_id}",
        source_image_id=source_image_id,
        url="https://example.test/item",
        author="Example Author",
        licence_url=CCPD_LICENSE_URL if is_ccpd else OPEN_IMAGES_LICENSE_URL,
        split="train",
        n_plates=1,
        n_vehicles=0 if is_ccpd else 1,
        source=source,
        annotated_classes=("license_plate",) if is_ccpd else ("license_plate", "vehicle"),
    )
    write_manifest([record], source_dir)
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 16, 8]}]
    if not is_ccpd:
        annotations.append({"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 32, 32]})
    coco = {
        "images": [
            {
                "id": 1,
                "source_image_id": source_image_id,
                "file_name": f"images/{stored_path}",
                "width": 64,
                "height": 64,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "license_plate"}, {"id": 2, "name": "vehicle"}],
    }
    annotation_dir = source_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "train.json").write_text(json.dumps(coco), encoding="utf-8")


def test_combined_coco_file_names_resolve_for_bare_and_nested_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_source(data_dir, "open_images_v6", OPEN_IMAGES_ID, f"{OPEN_IMAGES_ID}.jpg", 40)
    _write_source(data_dir, "ccpd", CCPD_PATH, CCPD_PATH, 200)
    config = tmp_path / "combined.yaml"
    config.write_text(
        f"data_dir: {data_dir}\n"
        "sources: [open_images_v6, ccpd]\n"
        "max_ccpd_plate_share: 0.5\n"
        "phash_distance: 0\n"
        "split_ratios:\n  train: 1.0\n  val: 0.0\n  test: 0.0\n",
        encoding="utf-8",
    )

    stats = build_combined_dataset(config)

    combined = data_dir / "processed" / "combined"
    coco = json.loads((combined / "annotations" / "train.json").read_text(encoding="utf-8"))
    names = sorted(image["file_name"] for image in coco["images"])
    assert stats["images"] == 2
    assert names == [
        f"images/ccpd/{CCPD_PATH}",
        f"images/open_images_v6/{OPEN_IMAGES_ID}.jpg",
    ]
    assert all((combined / name).is_file() for name in names)
    assert sorted(a["category_id"] for a in coco["annotations"]) == [1, 1, 2]


def test_plate_size_stats_counts_small_boxes_after_letterbox() -> None:
    from platedet.data.combine import plate_size_stats
    from platedet.data.manifest import ManifestRecord

    record = ManifestRecord(
        "a:1", "open_images_v6", "1", "https://x.test", "a", "CC BY 2.0",
        OPEN_IMAGES_LICENSE_URL, "0", "0", 2000, 1000, "train", 2, 1, '["license_plate"]',
    )  # fmt: skip
    boxes = {
        "1": [
            # 20x10 px -> at 640 (scale 0.32) short side 3.2 px: small; under 32 px wide
            {"category_id": 1, "xmin": 0.0, "xmax": 0.01, "ymin": 0.0, "ymax": 0.01},
            # 200x100 px -> large at both sizes
            {"category_id": 1, "xmin": 0.0, "xmax": 0.1, "ymin": 0.0, "ymax": 0.1},
            {"category_id": 2, "xmin": 0.0, "xmax": 0.001, "ymin": 0.0, "ymax": 0.001},
        ]
    }

    stats = plate_size_stats([record], boxes)

    assert stats["count"] == 2
    assert stats["under_32px_wide_original"] == 1
    assert stats["under_6px_short_side_at_640"] == 1
    assert stats["under_6px_short_side_at_416"] == 1
    assert stats["width_px_p10_p50_p90"] == [20.0, 200.0, 200.0]
