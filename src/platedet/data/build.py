"""Build the Open Images V6 COCO dataset and licence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from platedet.data.dedupe import deduplicate_by_phash
from platedet.data.manifest import (
    ManifestRecord,
    make_manifest_record,
    write_attribution,
    write_manifest,
)
from platedet.data.open_images import (
    ANNOTATION_URLS,
    CLASS_DESCRIPTIONS_URL,
    COCO_CATEGORIES,
    IMAGE_METADATA_URLS,
    download_image,
    download_url,
    load_class_ids,
    read_image_metadata,
    read_target_annotations,
)
from platedet.data.split import DEFAULT_SPLIT_RATIOS, assign_split, validate_split_ratios

LOGGER = logging.getLogger("platedet.data")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset config must be a YAML mapping: {path}")
    return value


def _download_source_files(
    raw_dir: Path, splits: list[str]
) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    class_path = raw_dir / "oidv6-class-descriptions.csv"
    download_url(CLASS_DESCRIPTIONS_URL, class_path)
    annotation_paths: dict[str, Path] = {}
    metadata_paths: dict[str, Path] = {}
    for split in splits:
        if split not in ANNOTATION_URLS:
            raise ValueError(f"Unsupported Open Images source split: {split!r}")
        annotation_path = raw_dir / f"{split}-annotations-bbox.csv"
        metadata_path = raw_dir / f"{split}-images.csv"
        download_url(ANNOTATION_URLS[split], annotation_path)
        download_url(IMAGE_METADATA_URLS[split], metadata_path)
        annotation_paths[split] = annotation_path
        metadata_paths[split] = metadata_path
    return class_path, annotation_paths, metadata_paths


def _load_annotations(
    paths: dict[str, Path], class_path: Path
) -> dict[str, list[dict[str, float | int]]]:
    with class_path.open(encoding="utf-8", newline="") as stream:
        class_ids = load_class_ids(stream)
    annotations: dict[str, list[dict[str, float | int]]] = {}
    for split in sorted(paths):
        LOGGER.info("Reading Open Images %s boxes", split)
        with paths[split].open(encoding="utf-8", newline="") as stream:
            for image_id, boxes in read_target_annotations(stream, class_ids).items():
                annotations.setdefault(image_id, []).extend(boxes)
    return annotations


def _coco_dataset(
    records: list[ManifestRecord], annotations: dict[str, list[dict[str, float | int]]]
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    for image_number, record in enumerate(records, start=1):
        source_filename = Path(record.source_image_id)
        if not source_filename.suffix:
            source_filename = source_filename.with_suffix(".jpg")
        image_path = Path("images") / source_filename.name
        images.append(
            {
                "id": image_number,
                "source_image_id": record.source_image_id,
                "file_name": image_path.as_posix(),
                "width": record.width,
                "height": record.height,
                "annotated_classes": json.loads(record.annotated_classes),
            }
        )
        for box in annotations[record.source_image_id]:
            xmin = float(box["xmin"]) * record.width
            ymin = float(box["ymin"]) * record.height
            xmax = float(box["xmax"]) * record.width
            ymax = float(box["ymax"]) * record.height
            coco_annotations.append(
                {
                    "id": len(coco_annotations) + 1,
                    "image_id": image_number,
                    "category_id": int(box["category_id"]),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0,
                }
            )
    sources = sorted({record.source for record in records})
    licenses = sorted({(record.licence, record.licence_url) for record in records})
    return {
        "info": {"description": f"Plate and vehicle subset from {', '.join(sources)}"},
        "licenses": [
            {"id": index, "name": name, "url": url}
            for index, (name, url) in enumerate(licenses, start=1)
        ],
        "images": images,
        "annotations": coco_annotations,
        "categories": COCO_CATEGORIES,
    }


def build_dataset(config_path: Path, max_images: int | None) -> dict[str, int]:
    config = _read_yaml(config_path)
    if max_images is None:
        raise ValueError(
            "Refusing an unbounded download; pass make data MAX_IMAGES=<positive integer>"
        )
    if max_images <= 0:
        raise ValueError("MAX_IMAGES must be a positive integer")

    source = config.get("source")
    if source == "ccpd":
        from platedet.data.ccpd_build import build_ccpd_dataset

        return build_ccpd_dataset(config, max_images)
    if source == "kaggle_car_plate_detection":
        from platedet.data.kaggle_build import build_kaggle_dataset

        return build_kaggle_dataset(config, max_images)
    if source != "open_images_v6":
        raise ValueError(f"Unsupported data source: {source!r}")

    data_dir = Path(config.get("data_dir", "data"))
    raw_dir = data_dir / "raw" / "open_images_v6"
    processed_dir = data_dir / "processed" / "open_images_v6"
    splits = config.get("source_splits", ["train", "validation"])
    if not isinstance(splits, list) or not splits:
        raise ValueError("source_splits must be a non-empty list")
    ratios = config.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    validate_split_ratios(ratios)
    phash_distance = int(config.get("phash_distance", 4))
    workers = int(config.get("download_workers", 4))
    if workers < 1:
        raise ValueError("download_workers must be positive")

    class_path, annotation_paths, metadata_paths = _download_source_files(raw_dir, splits)
    annotations = _load_annotations(annotation_paths, class_path)
    candidates = sorted(
        annotations,
        key=lambda item: hashlib.sha256(f"sample:{item}".encode()).digest(),
    )[:max_images]
    selected_ids = set(candidates)
    metadata, excluded_licences = read_image_metadata(list(metadata_paths.values()), selected_ids)
    missing_metadata = selected_ids - metadata.keys()
    image_dir = processed_dir / "images"
    results: list[ManifestRecord] = []
    failures = 0

    def process(image_id: str) -> ManifestRecord | None:
        details = metadata.get(image_id)
        if details is None:
            return None
        image_path = download_image(details["image_url"], image_id, image_dir)
        with Image.open(image_path) as image:
            image.verify()
        boxes = annotations[image_id]
        return make_manifest_record(
            image_path=image_path,
            image_id=f"open_images_v6:{image_id}",
            source_image_id=image_id,
            url=details["landing_url"],
            author=details["author"],
            licence_url=details["licence_url"],
            split="pending",
            n_plates=sum(box["category_id"] == 1 for box in boxes),
            n_vehicles=sum(box["category_id"] == 2 for box in boxes),
            annotated_classes=("license_plate", "vehicle"),
        )

    LOGGER.info("Downloading up to %d annotated images with %d workers", max_images, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process, image_id): image_id
            for image_id in candidates
            if image_id in metadata
        }
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as error:
                failures += 1
                LOGGER.warning("Could not use Open Images item %s: %s", futures[future], error)
                continue
            if record is not None:
                results.append(record)

    results.sort(key=lambda record: record.source_image_id)
    results, duplicate_count = deduplicate_by_phash(
        results, phash_getter=lambda record: record.phash, threshold=phash_distance
    )
    records: list[ManifestRecord] = []
    for record in results:
        records.append(
            ManifestRecord(
                **{
                    **record.__dict__,
                    "split": assign_split(record.image_id, ratios),
                }
            )
        )

    write_manifest(records, processed_dir)
    write_attribution(records, processed_dir / "ATTRIBUTION.md")
    split_records = {
        name: [record for record in records if record.split == name]
        for name in ("train", "val", "test")
    }
    for split, split_images in split_records.items():
        images = {record.source_image_id: record for record in split_images}
        dataset = _coco_dataset(split_images, {key: annotations[key] for key in images})
        annotation_dir = processed_dir / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
        )

    stats = {
        "source": "open_images_v6",
        "requested_max_images": max_images,
        "images_used": len(records),
        "excluded_unverified_licence_or_attribution": excluded_licences,
        "missing_metadata": len(missing_metadata),
        "download_or_image_failures": failures,
        "near_duplicates_removed": duplicate_count,
        "split_counts": {split: len(items) for split, items in split_records.items()},
        "plate_boxes": sum(record.n_plates for record in records),
        "vehicle_boxes": sum(record.n_vehicles for record in records),
        "split_ratios": ratios,
    }
    (processed_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Dataset build complete: %s", stats)
    return {"images": len(records), "duplicates": duplicate_count, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset(args.config, args.max_images)


if __name__ == "__main__":
    main()
