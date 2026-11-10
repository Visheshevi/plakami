"""Build a capped CCPD subset from an already-extracted official archive."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from platedet.data.build import _coco_dataset
from platedet.data.ccpd import (
    CCPD_AUTHOR,
    CCPD_DATASET_URL,
    CCPD_LICENSE_URL,
    CCPD_SOURCE,
    select_ccpd_images,
)
from platedet.data.dedupe import deduplicate_by_phash
from platedet.data.manifest import (
    ManifestRecord,
    make_manifest_record,
    write_attribution,
    write_manifest,
)
from platedet.data.split import DEFAULT_SPLIT_RATIOS, assign_split, validate_split_ratios

LOGGER = logging.getLogger("platedet.data")


def build_ccpd_dataset(config: dict[str, Any], max_images: int) -> dict[str, int]:
    source_dir = Path(config.get("source_dir", "data/raw/ccpd/extracted"))
    data_dir = Path(config.get("data_dir", "data"))
    processed_dir = data_dir / "processed" / CCPD_SOURCE
    ratios = config.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    validate_split_ratios(ratios)
    phash_distance = int(config.get("phash_distance", 4))
    selected, invalid_filenames = select_ccpd_images(source_dir, max_images)
    image_dir = processed_dir / "images"
    annotations: dict[str, list[dict[str, float | int]]] = {}
    records: list[ManifestRecord] = []
    failures = 0

    for item in selected:
        destination = image_dir / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not destination.is_file():
                shutil.copy2(item.path, destination)
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                width, height = image.size
            if item.xmax > width or item.ymax > height:
                raise ValueError(f"CCPD box exceeds image dimensions: {item.relative_path}")
            record = make_manifest_record(
                image_path=destination,
                image_id=f"{CCPD_SOURCE}:{item.relative_path}",
                source_image_id=item.relative_path,
                url=CCPD_DATASET_URL,
                author=CCPD_AUTHOR,
                licence_url=CCPD_LICENSE_URL,
                split="pending",
                n_plates=1,
                n_vehicles=0,
                source=CCPD_SOURCE,
                annotated_classes=("license_plate",),
            )
            records.append(record)
            annotations[item.relative_path] = [
                {
                    "category_id": 1,
                    "xmin": item.xmin / width,
                    "ymin": item.ymin / height,
                    "xmax": item.xmax / width,
                    "ymax": item.ymax / height,
                }
            ]
        except Exception as error:
            failures += 1
            LOGGER.warning("Could not use CCPD image %s: %s", item.relative_path, error)

    records.sort(key=lambda record: record.image_id)
    records, duplicate_count = deduplicate_by_phash(
        records, phash_getter=lambda record: record.phash, threshold=phash_distance
    )
    records = [replace(record, split=assign_split(record.image_id, ratios)) for record in records]
    included = {record.source_image_id for record in records}
    annotations = {key: value for key, value in annotations.items() if key in included}

    write_manifest(records, processed_dir)
    write_attribution(records, processed_dir / "ATTRIBUTION.md")
    split_records = {
        name: [record for record in records if record.split == name]
        for name in ("train", "val", "test")
    }
    annotation_dir = processed_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for split, split_images in split_records.items():
        dataset = _coco_dataset(split_images, annotations)
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
        )

    stats = {
        "source": CCPD_SOURCE,
        "source_dataset_url": CCPD_DATASET_URL,
        "source_image_count": len(list(source_dir.rglob("*.jpg"))),
        "requested_max_images": max_images,
        "images_used": len(records),
        "invalid_filenames_skipped": invalid_filenames,
        "image_failures": failures,
        "near_duplicates_removed": duplicate_count,
        "split_counts": {split: len(items) for split, items in split_records.items()},
        "plate_boxes": len(records),
        "vehicle_boxes": 0,
        "split_ratios": ratios,
        "note": "CCPD is plate-only; do not treat absent vehicle boxes as negatives.",
    }
    (processed_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("CCPD dataset build complete: %s", stats)
    return {"images": len(records), "duplicates": duplicate_count, "failures": failures}
