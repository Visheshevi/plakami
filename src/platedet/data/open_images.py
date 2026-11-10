"""Open Images V6 CSV parsing and image downloads."""

from __future__ import annotations

import csv
import hashlib
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import TextIO

OPEN_IMAGES_SOURCE = "open_images_v6"
OPEN_IMAGES_LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/"
IMAGE_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$", re.IGNORECASE)

CLASS_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
ANNOTATION_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
}
IMAGE_METADATA_URLS = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images.csv",
}

TARGET_LABELS = {
    "vehicle registration plate": 1,
    "car": 2,
    "truck": 2,
    "bus": 2,
    "van": 2,
    "motorcycle": 2,
    "land vehicle": 2,
}
COCO_CATEGORIES = [
    {"id": 1, "name": "license_plate", "supercategory": "vehicle"},
    {"id": 2, "name": "vehicle", "supercategory": "vehicle"},
]


def load_class_ids(stream: TextIO) -> dict[str, int]:
    descriptions = {
        row[1].strip().casefold(): row[0].strip() for row in csv.reader(stream) if len(row) >= 2
    }
    missing = set(TARGET_LABELS) - descriptions.keys()
    if missing:
        raise ValueError(f"Open Images class descriptions missing: {sorted(missing)}")
    return {descriptions[name]: category_id for name, category_id in TARGET_LABELS.items()}


def read_target_annotations(
    stream: TextIO, class_ids: dict[str, int]
) -> dict[str, list[dict[str, float | int]]]:
    annotations: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for row in csv.DictReader(stream):
        category_id = class_ids.get(row.get("LabelName", ""))
        image_id = row.get("ImageID", "").lower()
        if category_id is None:
            continue
        if not IMAGE_ID_PATTERN.fullmatch(image_id):
            raise ValueError(f"Invalid Open Images image ID: {image_id!r}")
        try:
            xmin, xmax = float(row["XMin"]), float(row["XMax"])
            ymin, ymax = float(row["YMin"]), float(row["YMax"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid box for Open Images image {image_id}") from error
        if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
            raise ValueError(f"Out-of-range box for Open Images image {image_id}")
        annotations[image_id].append(
            {"category_id": category_id, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
        )
    return dict(annotations)


def select_candidates(
    annotations: dict[str, list[dict[str, float | int]]],
    max_images: int,
    plate_image_fraction: float,
) -> list[str]:
    """Pick a deterministic capped sample, reserving a share for images that contain plates.

    Plates are rare in Open Images, so uniform sampling yields almost none. Up to
    `plate_image_fraction` of the cap goes to plate images; vehicle-only images fill
    the rest, and any shortfall on either side is filled from the other.
    """
    if not 0 <= plate_image_fraction <= 1:
        raise ValueError("plate_image_fraction must be in [0, 1]")
    if max_images <= 0:
        raise ValueError("MAX_IMAGES must be positive")

    def order(image_id: str) -> bytes:
        return hashlib.sha256(f"sample:{image_id}".encode()).digest()

    plate_ids = sorted(
        (key for key, boxes in annotations.items() if any(b["category_id"] == 1 for b in boxes)),
        key=order,
    )
    plate_set = set(plate_ids)
    other_ids = sorted((key for key in annotations if key not in plate_set), key=order)
    n_plate = min(len(plate_ids), round(plate_image_fraction * max_images))
    n_other = min(len(other_ids), max_images - n_plate)
    n_plate = min(len(plate_ids), max_images - n_other)
    return plate_ids[:n_plate] + other_ids[:n_other]


def read_image_metadata(
    paths: list[Path], selected_ids: set[str]
) -> tuple[dict[str, dict[str, str]], int]:
    """Return metadata for selected IDs; count rows excluded for unknown licences."""
    selected: dict[str, dict[str, str]] = {}
    unapproved_licences = 0
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                image_id = row.get("ImageID", "").lower()
                if image_id not in selected_ids:
                    continue
                if row.get("License", "").strip() != OPEN_IMAGES_LICENSE_URL:
                    unapproved_licences += 1
                    continue
                author = row.get("Author", "").strip()
                landing_url = row.get("OriginalLandingURL", "").strip()
                image_url = row.get("OriginalURL", "").strip()
                if not author or not landing_url.startswith(("https://", "http://")):
                    unapproved_licences += 1
                    continue
                if not image_url.startswith(("https://", "http://")):
                    continue
                selected[image_id] = {
                    "author": author,
                    "landing_url": landing_url,
                    "image_url": image_url,
                    "licence_url": OPEN_IMAGES_LICENSE_URL,
                }
    return selected, unapproved_licences


def download_url(url: str, destination: Path, *, timeout: int = 60) -> None:
    """Download an HTTPS resource atomically; existing complete files are reused."""
    if not url.startswith("https://"):
        raise ValueError("Open Images downloads must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "platedet-data/0.1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download_image(url: str, image_id: str, image_dir: Path) -> Path:
    if not IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ValueError(f"Invalid Open Images image ID: {image_id!r}")
    image_path = image_dir / f"{image_id}.jpg"
    download_url(url, image_path)
    return image_path
