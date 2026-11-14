"""Write YOLOX training annotations from the combined dataset.

For each input size this produces `data/processed/yolox_<size>/` with:
  - annotations/train.json and annotations/val.json (COCO; file names relative to this folder),
  - images -> symlink to the combined dataset's images,
  - stats.json with box counts, dropped small boxes and pseudo-labels added.

Rules (SPEC §4.4, ADR-0011):
  - boxes whose shorter side is under `min_box_px` after letterboxing to the input size are
    dropped from *training* only; evaluation keeps them;
  - if `plate_min_width_px` is set, plates use that rule instead: keep a plate when it is at
    least that wide in the stored image (narrower plates are unreadable and ignored, ADR-0013);
  - plate-only images get vehicle pseudo-labels (train/val training files only);
  - the build refuses images with an EXIF orientation tag, because YOLOX loads images with
    OpenCV (which applies it) while our boxes are in stored-pixel coordinates.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from platedet.eval.baseline import sha256_file

LOGGER = logging.getLogger("platedet.train")
TRAINING_SPLITS = ("train", "val")
EXIF_ORIENTATION = 274
CLASS_NAMES = {1: "license_plate", 2: "vehicle"}


def training_dir(config: dict[str, Any]) -> Path:
    return Path(config["data"]["training_dir"])


def _check_orientation(path: Path) -> None:
    with Image.open(path) as image:
        orientation = image.getexif().get(EXIF_ORIENTATION, 1)
    if orientation != 1:
        raise ValueError(f"EXIF orientation {orientation} would misalign boxes: {path}")


def convert_split(
    coco: dict[str, Any],
    pseudo_labels: dict[str, list[list[float]]],
    input_size: int,
    min_box_px: float,
    plate_min_width_px: float | None = None,
) -> tuple[dict[str, Any], Counter[str]]:
    """Return a training COCO dict plus counters (kept/dropped per class, pseudo added)."""
    counts: Counter[str] = Counter()
    images = {image["id"]: image for image in coco["images"]}

    def keep(box: list[float], image: dict[str, Any], category_id: int) -> bool:
        if category_id == 1 and plate_min_width_px is not None:
            return box[2] >= plate_min_width_px
        scale = input_size / max(image["width"], image["height"])
        return min(box[2], box[3]) * scale >= min_box_px

    annotations: list[dict[str, Any]] = []

    def add(image: dict[str, Any], category_id: int, bbox: list[float], pseudo: bool) -> None:
        name = CLASS_NAMES[category_id]
        if not keep(bbox, image, category_id):
            counts[f"dropped_small_{name}"] += 1
            return
        counts[f"kept_{name}"] += 1
        if pseudo:
            counts["pseudo_vehicle_boxes"] += 1
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image["id"],
                "category_id": category_id,
                "bbox": [round(value, 2) for value in bbox],
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                **({"pseudo": True} if pseudo else {}),
            }
        )

    for annotation in coco["annotations"]:
        add(images[annotation["image_id"]], annotation["category_id"], annotation["bbox"], False)
    for image in coco["images"]:
        if "vehicle" in image.get("annotated_classes", ["vehicle"]):
            continue
        boxes = pseudo_labels.get(image["file_name"])
        if boxes is None:
            raise ValueError(f"No pseudo-labels for plate-only image {image['file_name']}")
        counts["pseudo_labelled_images"] += 1
        for x1, y1, x2, y2, _score in boxes:
            add(image, 2, [x1, y1, x2 - x1, y2 - y1], True)
    boxed = {annotation["image_id"] for annotation in annotations}
    counts["images"] = len(images)
    counts["images_without_boxes"] = len(images.keys() - boxed)
    output = {
        "info": {"description": f"YOLOX training labels at {input_size}px (pseudo-labels added)"},
        "images": [
            {key: image[key] for key in ("id", "file_name", "width", "height")}
            for image in coco["images"]
        ],
        "annotations": annotations,
        "categories": coco["categories"],
    }
    return output, counts


def build_training_data(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    combined = Path(data["combined_dir"])
    output = training_dir(config)
    input_size = int(config["input_size"])
    min_box_px = float(config.get("min_box_px", 6))
    plate_min_width = config.get("plate_min_width_px")
    plate_min_width_px = None if plate_min_width is None else float(plate_min_width)
    pseudo_path = Path(data["pseudo_labels"])
    pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(combined / "manifest.csv")
    if pseudo["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError(
            "Pseudo-labels were built for a different dataset; rerun make pseudo-labels"
        )

    (output / "annotations").mkdir(parents=True, exist_ok=True)
    link = output / "images"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((combined / "images").resolve(), target_is_directory=True)

    stats: dict[str, Any] = {
        "input_size": input_size,
        "min_box_px": min_box_px,
        "plate_min_width_px": plate_min_width_px,
        "combined_manifest_sha256": manifest_hash,
        "pseudo_labels_sha256": sha256_file(pseudo_path),
        "splits": {},
    }
    for split in TRAINING_SPLITS:
        coco = json.loads((combined / "annotations" / f"{split}.json").read_text("utf-8"))
        for image in coco["images"]:
            _check_orientation(combined / image["file_name"])
        converted, counts = convert_split(
            coco, pseudo["labels"], input_size, min_box_px, plate_min_width_px
        )
        path = output / "annotations" / f"{split}.json"
        path.write_text(json.dumps(converted) + "\n", encoding="utf-8")
        stats["splits"][split] = {**dict(sorted(counts.items())), "sha256": sha256_file(path)}
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Training data written to %s: %s", output, stats["splits"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_training_data(yaml.safe_load(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
