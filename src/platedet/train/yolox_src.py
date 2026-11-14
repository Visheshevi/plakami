"""Fetch the official YOLOX source at a pinned commit and put it on `sys.path` (ADR-0009).

YOLOX can't be a normal locked dependency: its setup.py imports torch at build time. The
pinned commit is the integrity anchor (GitHub serves immutable content for a commit SHA).
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from platedet.data.open_images import download_url

YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
YOLOX_ARCHIVE_URL = f"https://codeload.github.com/Megvii-BaseDetection/YOLOX/tar.gz/{YOLOX_COMMIT}"
DEFAULT_CACHE = Path(".cache")


def yolox_dir(cache_root: Path = DEFAULT_CACHE) -> Path:
    return cache_root / f"YOLOX-{YOLOX_COMMIT}"


def ensure_yolox(cache_root: Path = DEFAULT_CACHE) -> Path:
    """Return the pinned YOLOX source folder, downloading it once, and make it importable."""
    target = yolox_dir(cache_root)
    if not (target / "yolox" / "__init__.py").is_file():
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_root) as temporary:
            archive = Path(temporary) / "yolox.tar.gz"
            download_url(YOLOX_ARCHIVE_URL, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(temporary, filter="data")
            extracted = Path(temporary) / target.name
            if not (extracted / "yolox" / "__init__.py").is_file():
                raise RuntimeError("Unexpected YOLOX archive layout")
            licence = (extracted / "LICENSE").read_text(encoding="utf-8")
            if "Apache License" not in licence:
                raise RuntimeError("YOLOX source is not Apache-2.0 licensed")
            if target.exists():
                shutil.rmtree(target)
            extracted.rename(target)
    path = str(target.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return target


if __name__ == "__main__":
    print(ensure_yolox())
