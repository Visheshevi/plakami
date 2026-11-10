"""Parse plate boxes encoded in CCPD image filenames."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

CCPD_SOURCE = "ccpd"
CCPD_DATASET_URL = "https://github.com/detectRecog/CCPD"
CCPD_AUTHOR = "CCPD authors"
CCPD_LICENSE_URL = "https://github.com/detectRecog/CCPD/blob/master/LICENSE"
_IMAGE_NAME = re.compile(
    r"^(?P<area>\d+)-(?P<tilt>[^-]+)-(?P<box>\d+&\d+_\d+&\d+)-"
    r"(?P<corners>[^-]+)-(?P<plate>[^-]+)-(?P<brightness>\d+)-(?P<blur>\d+)\.jpg$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CcpdImage:
    path: Path
    relative_path: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def parse_ccpd_filename(path: Path, root: Path) -> CcpdImage:
    """Extract CCPD's left-top/right-bottom box coordinates from a filename."""
    try:
        relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"CCPD image is outside configured root: {path}") from error
    match = _IMAGE_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Filename does not match CCPD annotation format: {path.name!r}")
    point_a, point_b = match.group("box").split("_")
    xmin, ymin = (int(value) for value in point_a.split("&"))
    xmax, ymax = (int(value) for value in point_b.split("&"))
    if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid CCPD bounding box in {path.name!r}")
    return CcpdImage(path, relative_path, xmin, ymin, xmax, ymax)


def select_ccpd_images(root: Path, max_images: int) -> tuple[list[CcpdImage], int]:
    """Select a reproducible capped set of CCPD JPEGs from an extracted source tree."""
    if max_images <= 0:
        raise ValueError("MAX_IMAGES must be positive")
    if not root.is_dir():
        raise FileNotFoundError(f"CCPD extracted data directory does not exist: {root}")
    candidates = sorted(root.rglob("*.jpg"))
    valid: list[CcpdImage] = []
    invalid_count = 0
    for path in candidates:
        try:
            valid.append(parse_ccpd_filename(path, root))
        except ValueError:
            invalid_count += 1
    selected = sorted(
        valid,
        key=lambda item: hashlib.sha256(f"{CCPD_SOURCE}:{item.relative_path}".encode()).digest(),
    )[:max_images]
    return selected, invalid_count
