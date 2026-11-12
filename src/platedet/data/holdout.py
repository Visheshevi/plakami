"""Curated evaluation-only `holdout` set from unused Open Images plate photos (ADR-0008).

Two steps:

  prepare   pick plate images outside the training sample, download them, drop near-duplicates
            of the combined dataset, and draw numbered review sheets with the boxes.
  finalize  keep only the images accepted in the committed review file and write the holdout
            COCO file, manifest, attribution and stats.

Only plates are scored on the holdout (`annotated_classes = ["license_plate"]`): the review
checks plate labels, not vehicle labels. Never train, tune or calibrate on this set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from platedet.data.build import _coco_dataset, _download_source_files, _load_annotations
from platedet.data.dedupe import _BKTree
from platedet.data.manifest import (
    ManifestRecord,
    make_manifest_record,
    write_attribution,
    write_manifest,
)
from platedet.data.open_images import download_image, read_image_metadata

LOGGER = logging.getLogger("platedet.data")
HOLDOUT_SPLIT = "holdout"
REVIEW_DECISIONS = {"accept", "reject"}
Annotations = dict[str, list[dict[str, float | int]]]


def holdout_candidates(annotations: Annotations, excluded: set[str], count: int) -> list[str]:
    """Plate images not in `excluded`, in a fixed order independent of the training sample."""
    plate_ids = [
        image_id
        for image_id, boxes in annotations.items()
        if image_id not in excluded and any(box["category_id"] == 1 for box in boxes)
    ]
    plate_ids.sort(key=lambda image_id: hashlib.sha256(f"holdout:{image_id}".encode()).digest())
    return plate_ids[:count]


def read_review(path: Path) -> dict[str, tuple[str, str]]:
    """Read `image_id,decision,reason` rows; decisions must be accept or reject."""
    decisions: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = row["image_id"].strip()
            decision = row["decision"].strip().lower()
            if decision not in REVIEW_DECISIONS:
                raise ValueError(f"Invalid review decision for {image_id}: {decision!r}")
            if image_id in decisions:
                raise ValueError(f"Duplicate review row for {image_id}")
            decisions[image_id] = (decision, row.get("reason", "").strip())
    return decisions


def _read_manifest_column(path: Path, column: str) -> list[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [row[column] for row in csv.DictReader(stream)]


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    data_dir = Path(config.get("data_dir", "data"))
    return data_dir, data_dir / "processed" / HOLDOUT_SPLIT


def write_review_sheets(
    records: list[ManifestRecord],
    annotations: Annotations,
    image_dir: Path,
    output_dir: Path,
    per_sheet: int = 9,
) -> None:
    """3x3 numbered sheets with plate boxes in red, for the accept/reject review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height, columns = 520, 400, 3
    for sheet_number, start in enumerate(range(0, len(records), per_sheet), start=1):
        chunk = records[start : start + per_sheet]
        rows = (len(chunk) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(chunk):
            with Image.open(image_dir / f"{record.source_image_id}.jpg") as source:
                image = source.convert("RGB")
            box_draw = ImageDraw.Draw(image)
            line = max(2, image.width // 300)
            for box in annotations[record.source_image_id]:
                if box["category_id"] != 1:
                    continue
                box_draw.rectangle(
                    [
                        float(box["xmin"]) * image.width,
                        float(box["ymin"]) * image.height,
                        float(box["xmax"]) * image.width,
                        float(box["ymax"]) * image.height,
                    ],
                    outline="red",
                    width=line,
                )
            image.thumbnail((tile_width - 4, tile_height - 20))
            x, y = (index % columns) * tile_width, (index // columns) * tile_height
            sheet.paste(image, (x + 2, y + 2))
            draw.text(
                (x + 4, y + tile_height - 16),
                f"{start + index + 1}  {record.source_image_id}",
                fill="black",
            )
        sheet.save(output_dir / f"sheet_{sheet_number:02d}.jpg", quality=90)


def prepare_holdout(config: dict[str, Any]) -> dict[str, int]:
    data_dir, holdout_dir = _paths(config)
    source_splits = list(config["source_splits"])
    training_splits = set(config["training_source_splits"])
    if training_splits & set(source_splits):
        raise ValueError("Holdout source splits must not overlap the training source splits")
    raw_dir = data_dir / "raw" / "open_images_v6"
    class_path, annotation_paths, metadata_paths = _download_source_files(raw_dir, source_splits)
    annotations = _load_annotations(annotation_paths, class_path)
    excluded = set(
        _read_manifest_column(
            data_dir / "processed" / "open_images_v6" / "manifest.csv", "source_image_id"
        )
    )
    candidates = holdout_candidates(annotations, excluded, int(config["candidates"]))
    metadata, _ = read_image_metadata(metadata_paths, set(candidates), "cvdf")
    image_dir = holdout_dir / "images"

    def process(image_id: str) -> ManifestRecord | None:
        details = metadata.get(image_id)
        if details is None:
            return None
        try:
            image_path = download_image(details["image_url"], image_id, image_dir)
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001 - log and skip unusable downloads
            LOGGER.warning("Could not use holdout candidate %s: %s", image_id, error)
            return None
        boxes = annotations[image_id]
        return make_manifest_record(
            image_path=image_path,
            image_id=f"open_images_v6:{image_id}",
            source_image_id=image_id,
            url=details["landing_url"],
            author=details["author"],
            licence_url=details["licence_url"],
            split=HOLDOUT_SPLIT,
            n_plates=sum(box["category_id"] == 1 for box in boxes),
            n_vehicles=0,
            annotated_classes=("license_plate",),
        )

    with ThreadPoolExecutor(max_workers=int(config.get("download_workers", 8))) as executor:
        records = [record for record in executor.map(process, candidates) if record is not None]

    tree = _BKTree()
    for phash in _read_manifest_column(
        data_dir / "processed" / "combined" / "manifest.csv", "phash"
    ):
        tree.add(phash)
    threshold = int(config.get("phash_distance", 4))
    kept: list[ManifestRecord] = []
    for record in records:
        if tree.contains_near(record.phash, threshold):
            continue
        tree.add(record.phash)
        kept.append(record)

    write_manifest(kept, holdout_dir / "candidates")
    plates_only = {
        record.source_image_id: [
            box for box in annotations[record.source_image_id] if box["category_id"] == 1
        ]
        for record in kept
    }
    write_review_sheets(kept, plates_only, image_dir, holdout_dir / "review")
    with (holdout_dir / "review" / "review_template.csv").open("w", encoding="utf-8") as stream:
        stream.write("image_id,decision,reason\n")
        stream.writelines(f"{record.source_image_id},,\n" for record in kept)
    (holdout_dir / "candidates" / "plate_boxes.json").write_text(
        json.dumps(plates_only, indent=1) + "\n", encoding="utf-8"
    )
    stats = {
        "candidates_requested": len(candidates),
        "downloaded": len(records),
        "near_duplicates_of_dataset_removed": len(records) - len(kept),
        "for_review": len(kept),
    }
    LOGGER.info("Holdout candidates ready: %s", stats)
    return stats


def finalize_holdout(config: dict[str, Any]) -> dict[str, Any]:
    _, holdout_dir = _paths(config)
    review = read_review(Path(config["review_file"]))
    candidates_dir = holdout_dir / "candidates"
    with (candidates_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        candidates = {
            row["source_image_id"]: ManifestRecord(
                **{
                    **row,
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "n_plates": int(row["n_plates"]),
                    "n_vehicles": int(row["n_vehicles"]),
                }
            )
            for row in csv.DictReader(stream)
        }
    unknown = set(review) - candidates.keys()
    if unknown:
        raise ValueError(f"Review lists images that are not candidates: {sorted(unknown)[:5]}")
    unreviewed = candidates.keys() - review.keys()
    if unreviewed:
        raise ValueError(f"{len(unreviewed)} candidates have no review decision")
    plate_boxes: Annotations = json.loads(
        (candidates_dir / "plate_boxes.json").read_text(encoding="utf-8")
    )
    accepted = sorted(
        (record for image_id, record in candidates.items() if review[image_id][0] == "accept"),
        key=lambda record: record.source_image_id,
    )
    accepted = [replace(record, split=HOLDOUT_SPLIT) for record in accepted]
    write_manifest(accepted, holdout_dir)
    write_attribution(accepted, holdout_dir / "ATTRIBUTION.md")
    dataset = _coco_dataset(accepted, plate_boxes)
    (holdout_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (holdout_dir / "annotations" / f"{HOLDOUT_SPLIT}.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    reasons: dict[str, int] = {}
    for decision, reason in review.values():
        if decision == "reject":
            reasons[reason or "unspecified"] = reasons.get(reason or "unspecified", 0) + 1
    widths = [
        (float(box["xmax"]) - float(box["xmin"])) * record.width
        for record in accepted
        for box in plate_boxes[record.source_image_id]
    ]
    stats = {
        "reviewed": len(review),
        "accepted_images": len(accepted),
        "plate_boxes": len(widths),
        "plates_under_32px_wide": sum(width < 32 for width in widths),
        "reject_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])),
        "note": "Evaluation only. Plates scored; vehicles not reviewed and not scored.",
    }
    (holdout_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Holdout finalized: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.step == "prepare":
        prepare_holdout(config)
    else:
        finalize_holdout(config)


if __name__ == "__main__":
    main()
