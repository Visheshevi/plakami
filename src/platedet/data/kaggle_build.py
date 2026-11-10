"""Build the CC0 Kaggle car-plate dataset as COCO plus a licence manifest."""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from platedet.data.build import _coco_dataset
from platedet.data.dedupe import deduplicate_by_phash
from platedet.data.kaggle import (
    KAGGLE_AUTHOR,
    KAGGLE_DATASET_URL,
    KAGGLE_LICENSE_URL,
    KAGGLE_SOURCE,
    download_archive,
    read_image_from_archive,
    select_annotations,
)
from platedet.data.manifest import (
    ManifestRecord,
    make_manifest_record,
    write_attribution,
    write_manifest,
)
from platedet.data.split import DEFAULT_SPLIT_RATIOS, assign_split, validate_split_ratios

LOGGER = logging.getLogger("platedet.data")


def build_kaggle_dataset(config: dict[str, Any], max_images: int) -> dict[str, int]:
    data_dir = Path(config.get("data_dir", "data"))
    raw_dir = data_dir / "raw" / KAGGLE_SOURCE
    processed_dir = data_dir / "processed" / KAGGLE_SOURCE
    archive_path = raw_dir / "dataset-v1.zip"
    ratios = config.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    validate_split_ratios(ratios)
    phash_distance = int(config.get("phash_distance", 4))

    download_archive(archive_path)
    image_dir = processed_dir / "images"
    results: list[ManifestRecord] = []
    annotations: dict[str, list[dict[str, float | int]]] = {}
    failures = 0
    with zipfile.ZipFile(archive_path) as archive:
        selected, source_annotation_count = select_annotations(archive, max_images)
        for item in selected:
            try:
                image_bytes = read_image_from_archive(archive, item.filename)
                source_image_id = item.filename
                image_path = image_dir / source_image_id
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)
                with Image.open(image_path) as image:
                    width, height = image.size
                if (width, height) != (item.width, item.height):
                    raise ValueError(
                        f"VOC dimensions ({item.width}, {item.height}) do not match image "
                        f"dimensions ({width}, {height})"
                    )
                record = make_manifest_record(
                    image_path=image_path,
                    image_id=f"{KAGGLE_SOURCE}:{Path(source_image_id).stem}",
                    source_image_id=source_image_id,
                    url=KAGGLE_DATASET_URL,
                    author=KAGGLE_AUTHOR,
                    licence_url=KAGGLE_LICENSE_URL,
                    split="pending",
                    n_plates=len(item.boxes),
                    n_vehicles=0,
                    source=KAGGLE_SOURCE,
                    annotated_classes=("license_plate",),
                )
                results.append(record)
                annotations[source_image_id] = [
                    {
                        "category_id": 1,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                    for xmin, ymin, xmax, ymax in item.boxes
                ]
            except Exception as error:
                failures += 1
                LOGGER.warning("Could not use Kaggle item %s: %s", item.filename, error)

    results.sort(key=lambda record: record.image_id)
    results, duplicate_count = deduplicate_by_phash(
        results, phash_getter=lambda record: record.phash, threshold=phash_distance
    )
    records = [replace(record, split=assign_split(record.image_id, ratios)) for record in results]
    included_ids = {record.source_image_id for record in records}
    annotations = {key: value for key, value in annotations.items() if key in included_ids}

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
        "source": KAGGLE_SOURCE,
        "source_dataset_url": KAGGLE_DATASET_URL,
        "source_annotation_count": source_annotation_count,
        "requested_max_images": max_images,
        "images_used": len(records),
        "download_or_annotation_failures": failures,
        "near_duplicates_removed": duplicate_count,
        "split_counts": {split: len(items) for split, items in split_records.items()},
        "plate_boxes": sum(record.n_plates for record in records),
        "vehicle_boxes": 0,
        "split_ratios": ratios,
        "note": "This source labels plates only; it provides no vehicle boxes.",
    }
    (processed_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Kaggle dataset build complete: %s", stats)
    return {"images": len(records), "duplicates": duplicate_count, "failures": failures}
