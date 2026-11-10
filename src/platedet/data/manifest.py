"""Licence-aware image manifest creation and attribution output."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

OPEN_IMAGES_LICENSE = "CC BY 2.0"
OPEN_IMAGES_LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/"
KAGGLE_LICENSE = "CC0 1.0"
KAGGLE_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
CCPD_LICENSE = "MIT"
CCPD_LICENSE_URL = "https://github.com/detectRecog/CCPD/blob/master/LICENSE"
SOURCE_LICENSES = {
    "open_images_v6": (OPEN_IMAGES_LICENSE, OPEN_IMAGES_LICENSE_URL),
    "kaggle_car_plate_detection": (KAGGLE_LICENSE, KAGGLE_LICENSE_URL),
    "ccpd": (CCPD_LICENSE, CCPD_LICENSE_URL),
}
MANIFEST_COLUMNS = (
    "image_id",
    "source",
    "source_image_id",
    "url",
    "author",
    "licence",
    "licence_url",
    "sha256",
    "phash",
    "width",
    "height",
    "split",
    "n_plates",
    "n_vehicles",
    "annotated_classes",
)


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    source: str
    source_image_id: str
    url: str
    author: str
    licence: str
    licence_url: str
    sha256: str
    phash: str
    width: int
    height: int
    split: str
    n_plates: int
    n_vehicles: int
    annotated_classes: str


def validate_license(
    licence_url: str,
    author: str,
    source_url: str,
    *,
    source: str = "open_images_v6",
) -> None:
    expected = SOURCE_LICENSES.get(source)
    if expected is None or licence_url != expected[1]:
        raise ValueError(f"Non-allowlisted licence for {source}: {licence_url!r}")
    if not author.strip():
        raise ValueError("Image attribution author is required")
    if not source_url.startswith(("https://", "http://")):
        raise ValueError("Image landing URL must be an HTTP(S) URL")


def make_manifest_record(
    *,
    image_path: Path,
    image_id: str,
    source_image_id: str,
    url: str,
    author: str,
    licence_url: str,
    split: str,
    n_plates: int,
    n_vehicles: int,
    source: str = "open_images_v6",
    annotated_classes: tuple[str, ...] = ("license_plate", "vehicle"),
) -> ManifestRecord:
    expected_license = SOURCE_LICENSES.get(source)
    if expected_license is None:
        raise ValueError(f"Non-allowlisted data source: {source!r}")
    validate_license(licence_url, author, url, source=source)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        width, height = image.size
        perceptual_hash = str(imagehash.phash(image))
    return ManifestRecord(
        image_id=image_id,
        source=source,
        source_image_id=source_image_id,
        url=url,
        author=author.strip(),
        licence=expected_license[0],
        licence_url=licence_url,
        sha256=digest,
        phash=perceptual_hash,
        width=width,
        height=height,
        split=split,
        n_plates=n_plates,
        n_vehicles=n_vehicles,
        annotated_classes=json.dumps(annotated_classes),
    )


def write_manifest(records: Iterable[ManifestRecord], output_dir: Path) -> None:
    rows = [asdict(record) for record in records]
    for row in rows:
        validate_license(row["licence_url"], row["author"], row["url"], source=row["source"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    schema = pa.schema(
        [
            ("image_id", pa.string()),
            ("source", pa.string()),
            ("source_image_id", pa.string()),
            ("url", pa.string()),
            ("author", pa.string()),
            ("licence", pa.string()),
            ("licence_url", pa.string()),
            ("sha256", pa.string()),
            ("phash", pa.string()),
            ("width", pa.int32()),
            ("height", pa.int32()),
            ("split", pa.string()),
            ("n_plates", pa.int32()),
            ("n_vehicles", pa.int32()),
            ("annotated_classes", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output_dir / "manifest.parquet")


def write_attribution(records: Iterable[ManifestRecord], path: Path) -> None:
    unique = {
        (record.source, record.author, record.url, record.licence, record.licence_url)
        for record in records
    }
    lines = ["# Dataset attribution", ""]
    for source, author, url, licence, licence_url in sorted(unique):
        lines.append(f"- {source}: [{author}]({url}) — [{licence}]({licence_url})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
