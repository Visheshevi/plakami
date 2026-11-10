"""Merge source datasets, deduplicate globally, enforce the CCPD plate cap, and resplit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from platedet.data.build import _coco_dataset, image_relative_path
from platedet.data.dedupe import deduplicate_by_phash
from platedet.data.manifest import (
    SOURCE_LICENSES,
    ManifestRecord,
    write_attribution,
    write_manifest,
)
from platedet.data.split import DEFAULT_SPLIT_RATIOS, assign_split, validate_split_ratios

LOGGER = logging.getLogger("platedet.data")


def apply_ccpd_share_cap(
    records: list[ManifestRecord], max_share: float
) -> tuple[list[ManifestRecord], int]:
    """Limit CCPD plate boxes to a fraction of all included plate boxes."""
    if not 0 <= max_share < 1:
        raise ValueError("max_ccpd_plate_share must be in [0, 1)")
    other_plates = sum(record.n_plates for record in records if record.source != "ccpd")
    allowed_ccpd_plates = math.floor(max_share * other_plates / (1 - max_share))
    ccpd_records = sorted(
        (record for record in records if record.source == "ccpd"),
        key=lambda record: hashlib.sha256(f"ccpd-cap:{record.image_id}".encode()).digest(),
    )
    retained: list[ManifestRecord] = []
    ccpd_count = 0
    dropped = 0
    for record in ccpd_records:
        if ccpd_count + record.n_plates <= allowed_ccpd_plates:
            retained.append(record)
            ccpd_count += record.n_plates
        else:
            dropped += 1
    others = [record for record in records if record.source != "ccpd"]
    return others + retained, dropped


def plate_size_stats(
    records: list[ManifestRecord],
    annotations: dict[str, list[dict[str, float | int]]],
    input_sizes: tuple[int, ...] = (640, 416),
) -> dict[str, Any]:
    """Plate width percentiles in original pixels, plus small-box counts after letterboxing.

    `under_32px_wide_original` matches the SPEC §6 small-plate bucket; `under_6px_short_side_at_N`
    counts boxes SPEC §4.4 step 6 would drop at input size N (reported, not dropped here).
    """
    widths: list[float] = []
    small_at = dict.fromkeys(input_sizes, 0)
    for record in records:
        for box in annotations.get(record.source_image_id, []):
            if box["category_id"] != 1:
                continue
            width = (float(box["xmax"]) - float(box["xmin"])) * record.width
            height = (float(box["ymax"]) - float(box["ymin"])) * record.height
            widths.append(width)
            for size in input_sizes:
                scale = size / max(record.width, record.height)
                small_at[size] += min(width, height) * scale < 6
    widths.sort()

    def percentile(fraction: float) -> float:
        if not widths:
            return 0.0
        return round(widths[min(len(widths) - 1, int(fraction * len(widths)))], 1)

    return {
        "count": len(widths),
        "width_px_p10_p50_p90": [percentile(0.1), percentile(0.5), percentile(0.9)],
        "under_32px_wide_original": sum(width < 32 for width in widths),
        **{f"under_6px_short_side_at_{size}": count for size, count in small_at.items()},
    }


def _read_manifest(path: Path) -> list[ManifestRecord]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            ManifestRecord(
                **{
                    **row,
                    "annotated_classes": row.get(
                        "annotated_classes",
                        '["license_plate"]'
                        if row.get("source") in {"ccpd", "kaggle_car_plate_detection"}
                        else '["license_plate", "vehicle"]',
                    ),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "n_plates": int(row["n_plates"]),
                    "n_vehicles": int(row["n_vehicles"]),
                }
            )
            for row in csv.DictReader(stream)
        ]


def _read_source_annotations(source_dir: Path) -> dict[str, list[dict[str, float | int]]]:
    annotations: dict[str, list[dict[str, float | int]]] = {}
    for path in sorted((source_dir / "annotations").glob("*.json")):
        dataset = json.loads(path.read_text(encoding="utf-8"))
        images = {image["id"]: image for image in dataset.get("images", [])}
        for annotation in dataset.get("annotations", []):
            image = images.get(annotation["image_id"])
            if image is None:
                raise ValueError(f"COCO annotation in {path} references a missing image")
            source_image_id = image.get("source_image_id") or image.get("open_images_id")
            if not source_image_id:
                source_image_id = Path(image["file_name"]).name
            width, height = image["width"], image["height"]
            x, y, box_width, box_height = annotation["bbox"]
            annotations.setdefault(source_image_id, []).append(
                {
                    "category_id": annotation["category_id"],
                    "xmin": x / width,
                    "ymin": y / height,
                    "xmax": (x + box_width) / width,
                    "ymax": (y + box_height) / height,
                }
            )
    return annotations


def build_combined_dataset(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Combined data config must be a YAML mapping")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources):
        raise ValueError("sources must be a non-empty list of unique source IDs")
    unsupported = set(sources) - SOURCE_LICENSES.keys()
    if unsupported:
        raise ValueError(f"Unsupported combined data sources: {sorted(unsupported)}")
    data_dir = Path(config.get("data_dir", "data"))
    output_dir = data_dir / "processed" / "combined"
    ratios = config.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    validate_split_ratios(ratios)
    threshold = int(config.get("phash_distance", 4))
    max_ccpd_share = float(config.get("max_ccpd_plate_share", 0.30))

    records: list[ManifestRecord] = []
    annotation_map: dict[str, list[dict[str, float | int]]] = {}
    record_sources: dict[str, tuple[Path, str]] = {}
    per_source: dict[str, dict[str, int]] = {}
    for source in sources:
        source_dir = data_dir / "processed" / source
        source_records = _read_manifest(source_dir / "manifest.csv")
        annotations = _read_source_annotations(source_dir)
        kept_source_records: list[ManifestRecord] = []
        for record in source_records:
            image_root = source_dir / "images"
            image_path = image_root / image_relative_path(record.source_image_id)
            try:
                image_path.resolve().relative_to(image_root.resolve())
            except ValueError as error:
                raise ValueError(
                    f"Manifest image path escapes source image directory: {record.image_id}"
                ) from error
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image is missing: {image_path}")
            if record.source != source:
                raise ValueError(f"Manifest source mismatch for {record.image_id}: {record.source}")
            records.append(record)
            record_sources[record.image_id] = (source_dir, record.source_image_id)
            annotation_map[record.image_id] = annotations.get(record.source_image_id, [])
            kept_source_records.append(record)
        per_source[source] = {
            "images": len(kept_source_records),
            "plate_boxes": sum(record.n_plates for record in kept_source_records),
            "vehicle_boxes": sum(record.n_vehicles for record in kept_source_records),
        }

    records, duplicate_count = deduplicate_by_phash(
        sorted(records, key=lambda record: record.image_id),
        phash_getter=lambda record: record.phash,
        threshold=threshold,
    )
    records, ccpd_capped = apply_ccpd_share_cap(records, max_ccpd_share)
    records = [replace(record, split=assign_split(record.image_id, ratios)) for record in records]

    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    combined_records: list[ManifestRecord] = []
    combined_annotations: dict[str, list[dict[str, float | int]]] = {}
    for record in records:
        source_dir, source_image_id = record_sources[record.image_id]
        original_path = image_relative_path(source_image_id)
        relative_path = f"{record.source}/{original_path}"
        destination = image_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to((source_dir / "images" / original_path).resolve())
        combined_record = replace(record, source_image_id=relative_path)
        combined_records.append(combined_record)
        combined_annotations[relative_path] = annotation_map[record.image_id]

    write_manifest(combined_records, output_dir)
    write_attribution(combined_records, output_dir / "ATTRIBUTION.md")
    split_records = {
        name: [record for record in combined_records if record.split == name]
        for name in ("train", "val", "test")
    }
    for split, split_images in split_records.items():
        dataset = _coco_dataset(split_images, combined_annotations)
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
        )

    ccpd_boxes = sum(record.n_plates for record in combined_records if record.source == "ccpd")
    all_plates = sum(record.n_plates for record in combined_records)
    stats = {
        "sources": per_source,
        "images": len(combined_records),
        "near_duplicates_removed": duplicate_count,
        "ccpd_images_dropped_by_plate_share_cap": ccpd_capped,
        "max_ccpd_plate_share": max_ccpd_share,
        "actual_ccpd_plate_share": ccpd_boxes / all_plates if all_plates else 0,
        "plate_boxes": all_plates,
        "vehicle_boxes": sum(record.n_vehicles for record in combined_records),
        "annotation_scope_counts": {
            "plate_only_images": sum(
                "vehicle" not in json.loads(record.annotated_classes) for record in combined_records
            ),
            "plate_and_vehicle_images": sum(
                {"license_plate", "vehicle"}.issubset(json.loads(record.annotated_classes))
                for record in combined_records
            ),
        },
        "split_counts": {split: len(items) for split, items in split_records.items()},
        "split_source_counts": {
            split: {
                source: {
                    "images": sum(record.source == source for record in items),
                    "plate_boxes": sum(r.n_plates for r in items if r.source == source),
                    "vehicle_boxes": sum(r.n_vehicles for r in items if r.source == source),
                }
                for source in sources
            }
            for split, items in split_records.items()
        },
        "plate_sizes": plate_size_stats(combined_records, combined_annotations),
        "split_ratios": ratios,
        "note": "Use annotated_classes; plate-only images are not vehicle negatives.",
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Combined dataset build complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_combined_dataset(args.config)


if __name__ == "__main__":
    main()
