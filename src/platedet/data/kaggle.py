"""Public Kaggle CC0 car-plate dataset archive and Pascal VOC parsing."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from PIL import Image

KAGGLE_SOURCE = "kaggle_car_plate_detection"
KAGGLE_OWNER = "andrewmvd"
KAGGLE_DATASET = "car-plate-detection"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/andrewmvd/car-plate-detection"
KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/"
    "car-plate-detection?datasetVersionNumber=1"
)
KAGGLE_AUTHOR = "andrewmvd (Kaggle uploader)"
KAGGLE_LICENSE = "CC0 1.0"
KAGGLE_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
MAX_ARCHIVE_BYTES = 300 * 1024 * 1024
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PLATE_LABELS = {"license", "license plate", "plate"}


@dataclass(frozen=True)
class VocImage:
    annotation_path: str
    filename: str
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]


def download_archive(destination: Path, *, timeout: int = 60) -> None:
    """Download the public dataset without Kaggle credentials, atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        if destination.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Kaggle dataset archive exceeds the configured size limit")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        KAGGLE_DOWNLOAD_URL,
        headers={"User-Agent": "platedet-data/0.1", "Range": f"bytes=0-{MAX_ARCHIVE_BYTES}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            content_type = content_type.casefold()
            if "zip" not in content_type and "octet-stream" not in content_type:
                raise ValueError(
                    "Kaggle returned a non-archive response; this downloader uses no credentials"
                )
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_ARCHIVE_BYTES:
                raise ValueError("Kaggle dataset archive exceeds the configured size limit")
            written = 0
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_ARCHIVE_BYTES:
                        raise ValueError("Kaggle dataset archive exceeds the configured size limit")
                    output.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Kaggle response is not a valid dataset ZIP archive")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _safe_zip_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe path in Kaggle archive: {name!r}")
    return path


def parse_voc_xml(xml_bytes: bytes, annotation_path: str) -> VocImage:
    """Read plate boxes from a VOC XML file, converting 1-based pixels to normalized edges."""
    try:
        root = ElementTree.fromstring(xml_bytes)
        filename = root.findtext("filename", "").strip()
        size = root.find("size")
        if size is None:
            raise ValueError("VOC annotation is missing image size")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError("VOC annotation has invalid image dimensions")
        safe_filename = PurePosixPath(filename)
        if (
            safe_filename.name != filename
            or safe_filename.suffix.casefold() not in ALLOWED_IMAGE_EXTENSIONS
        ):
            raise ValueError(f"Unsafe or unsupported image filename: {filename!r}")
        boxes: list[tuple[float, float, float, float]] = []
        for object_node in root.findall("object"):
            raw_name = object_node.findtext("name", "").strip().casefold().replace("_", " ")
            name = " ".join(raw_name.split())
            name = name.replace("licence", "license")
            if name not in PLATE_LABELS:
                raise ValueError(f"Unexpected class in Kaggle annotation: {name!r}")
            box_node = object_node.find("bndbox")
            if box_node is None:
                raise ValueError("VOC object is missing bndbox")
            xmin = float(box_node.findtext("xmin", "nan"))
            ymin = float(box_node.findtext("ymin", "nan"))
            xmax = float(box_node.findtext("xmax", "nan"))
            ymax = float(box_node.findtext("ymax", "nan"))
            x0, y0 = max(0.0, xmin - 1), max(0.0, ymin - 1)
            x1, y1 = min(float(width), xmax), min(float(height), ymax)
            if not (x0 < x1 <= width and y0 < y1 <= height):
                raise ValueError("VOC annotation contains an invalid bounding box")
            boxes.append((x0 / width, y0 / height, x1 / width, y1 / height))
    except (ElementTree.ParseError, TypeError, OverflowError) as error:
        raise ValueError(f"Malformed Kaggle VOC annotation {annotation_path!r}") from error
    return VocImage(annotation_path, filename, width, height, tuple(boxes))


def select_annotations(archive: zipfile.ZipFile, max_images: int) -> tuple[list[VocImage], int]:
    """Select a reproducible capped subset of safe Pascal VOC annotation entries."""
    if max_images <= 0:
        raise ValueError("MAX_IMAGES must be positive")
    infos = archive.infolist()
    extracted_size = sum(info.file_size for info in infos)
    if extracted_size > MAX_EXTRACTED_BYTES:
        raise ValueError("Kaggle archive exceeds the configured extracted-size limit")
    annotations: list[VocImage] = []
    for info in infos:
        path = _safe_zip_path(info.filename)
        if len(path.parts) != 2 or path.parts[0].casefold() != "annotations":
            continue
        if path.suffix.casefold() != ".xml":
            continue
        annotations.append(parse_voc_xml(archive.read(info), info.filename))
    selected = sorted(
        annotations,
        key=lambda item: hashlib.sha256(f"{KAGGLE_SOURCE}:{item.filename}".encode()).digest(),
    )[:max_images]
    return selected, len(annotations)


def read_image_from_archive(archive: zipfile.ZipFile, filename: str) -> bytes:
    expected = filename.casefold()
    for info in archive.infolist():
        path = _safe_zip_path(info.filename)
        if (
            len(path.parts) == 2
            and path.parts[0].casefold() == "images"
            and path.name.casefold() == expected
        ):
            payload = archive.read(info)
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            return payload
    raise FileNotFoundError(f"No matching image file in Kaggle archive for {filename!r}")
