"""Download a guarded CCPD archive in Colab and extract a deterministic small sample.

Upload this script to a Colab free runtime, then run:
    !python /content/colab_ccpd_sample.py --max-images 100

The full official 12 GB archive is downloaded only after the runtime reports enough
free disk and Google Drive returns a known Content-Length. The archive is never
fully extracted; at most the requested number of JPEG images is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import logging
import shutil
import tarfile
import tempfile
from email.message import Message
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener

CCPD_DOWNLOAD_URL = (
    "https://drive.google.com/uc?export=download&id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc"
)
CCPD_ARCHIVE_PAGE_SIZE_GIB = 12
MAX_SAMPLE_IMAGES = 100
MAX_ARCHIVE_SIZE_BYTES = 16 * 1024**3
MIN_POST_DOWNLOAD_FREE_BYTES = 8 * 1024**3
MAX_JPEG_MEMBER_BYTES = 30 * 1024**2
MAX_CONFIRMATION_HTML_BYTES = 2 * 1024**2
MAX_CONFIRMATION_STEPS = 3
LOG = logging.getLogger("colab_ccpd")


class _DownloadFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.fields: dict[str, str] = {}
        self._inside_download_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self._inside_download_form = True
            self.action = attributes.get("action")
        elif self._inside_download_form and tag == "input":
            name = attributes.get("name")
            value = attributes.get("value")
            if name and value is not None:
                self.fields[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._inside_download_form:
            self._inside_download_form = False


class _VisibleTextParser(HTMLParser):
    """Collect visible HTML text only; never expose hidden confirmation inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def _describe_drive_page(page: bytes) -> str:
    parser = _VisibleTextParser()
    parser.feed(page.decode("utf-8", errors="replace"))
    text = " ".join(parser.parts).casefold()
    if any(phrase in text for phrase in ("download quota", "too many users", "quota exceeded")):
        return (
            "Google Drive reports a download quota/rate limit. Stop retrying for now; "
            "wait and try later, or ask the owner to provide a copy from their Drive."
        )
    if any(phrase in text for phrase in ("sign in", "sign-in", "access denied", "permission")):
        return (
            "Google Drive is asking for sign-in or denying access; the archive link is "
            "not public to this request."
        )
    if any(
        phrase in text
        for phrase in ("virus scan", "can't scan", "cannot scan", "too large for google")
    ):
        return (
            "Google Drive returned another virus-scan confirmation page. The helper could not "
            "follow it safely; download through Drive in a browser and upload it to Colab."
        )
    return (
        "Google Drive returned an HTML page instead of the archive. This may be a temporary "
        "quota, access, or confirmation response; do not repeatedly restart the 12 GB download."
    )


def _confirmation_url(current_url: str, action: str, fields: dict[str, str]) -> str:
    target = urljoin(current_url, action)
    parts = urlsplit(target)
    allowed_hosts = {
        "drive.google.com",
        "drive.usercontent.google.com",
        "drive.googleusercontent.com",
    }
    if parts.scheme != "https" or parts.hostname not in allowed_hosts:
        raise RuntimeError("Google Drive returned an unexpected confirmation destination")
    query = "&".join(part for part in (parts.query, urlencode(fields)) if part)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def parse_known_archive_size(headers: Message) -> int:
    """Require the actual response's byte count; never infer it from a rounded 12 GB label."""
    value = headers.get("Content-Length") or headers.get("x-goog-stored-content-length")
    if value is None:
        raise RuntimeError("Google Drive did not return a usable archive size; refusing download")
    try:
        size = int(value)
    except ValueError as error:
        raise RuntimeError(
            "Google Drive returned an invalid archive size; refusing download"
        ) from error
    if size <= 0 or size > MAX_ARCHIVE_SIZE_BYTES:
        raise RuntimeError(
            f"CCPD archive size {size} bytes is outside the configured safe limit; "
            "refusing download"
        )
    return size


def check_free_space(free_bytes: int, archive_bytes: int) -> None:
    required = archive_bytes + MIN_POST_DOWNLOAD_FREE_BYTES
    if free_bytes < required:
        raise RuntimeError(
            "Not enough free Colab disk space: need archive size plus an 8 GiB reserve "
            f"({required / 1024**3:.1f} GiB), have {free_bytes / 1024**3:.1f} GiB."
        )


def _get_download_response(opener: object) -> BinaryIO:
    url = CCPD_DOWNLOAD_URL
    for _ in range(MAX_CONFIRMATION_STEPS + 1):
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = opener.open(request, timeout=60)  # type: ignore[attr-defined]
        content_type = response.headers.get("Content-Type", "").casefold()
        if "html" not in content_type:
            return response

        page = response.read(MAX_CONFIRMATION_HTML_BYTES + 1)
        response.close()
        if len(page) > MAX_CONFIRMATION_HTML_BYTES:
            raise RuntimeError("Google Drive confirmation page exceeded the safety limit")
        parser = _DownloadFormParser()
        parser.feed(page.decode("utf-8", errors="replace"))
        if not parser.action or not parser.fields.get("id") or not parser.fields.get("confirm"):
            raise RuntimeError(_describe_drive_page(page))
        url = _confirmation_url(url, parser.action, parser.fields)
    raise RuntimeError(
        "Google Drive returned too many confirmation pages. Stop retrying; download the "
        "archive through Drive in a browser and upload it to Colab."
    )


def download_archive(archive_path: Path, *, free_bytes: int | None = None) -> int:
    """Download only after the actual response size and runtime disk space pass checks."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        size = archive_path.stat().st_size
        if size <= 0 or size > MAX_ARCHIVE_SIZE_BYTES:
            raise RuntimeError("Existing CCPD archive has an invalid or oversized length")
        if not tarfile.is_tarfile(archive_path):
            raise RuntimeError("Existing CCPD file is not a readable tar archive")
        current_free = (
            shutil.disk_usage(archive_path.parent).free if free_bytes is None else free_bytes
        )
        check_free_space(current_free, 0)
        return size

    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    response = _get_download_response(opener)
    try:
        size = parse_known_archive_size(response.headers)
        current_free = (
            shutil.disk_usage(archive_path.parent).free if free_bytes is None else free_bytes
        )
        check_free_space(current_free, size)
        content_type = response.headers.get("Content-Type", "").casefold()
        if "html" in content_type or "text/" in content_type:
            raise RuntimeError(
                "Google Drive returned a non-archive response; the download was not saved. "
                "Stop retrying and inspect the Drive link in a browser."
            )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=archive_path.parent, suffix=".part", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            received = 0
            last_reported_gib = 0
            while chunk := response.read(8 * 1024 * 1024):
                received += len(chunk)
                if received > size or received > MAX_ARCHIVE_SIZE_BYTES:
                    raise RuntimeError("CCPD response exceeded its declared safe size")
                temporary.write(chunk)
                downloaded_gib = int(received / 1024**3)
                if downloaded_gib > last_reported_gib:
                    last_reported_gib = downloaded_gib
                    LOG.info("Downloaded %d / %.1f GiB", downloaded_gib, size / 1024**3)
        if received != size:
            raise RuntimeError(
                f"Incomplete CCPD archive: received {received} bytes, expected {size}"
            )
        if not tarfile.is_tarfile(temporary_path):
            raise RuntimeError("Downloaded CCPD file is not a readable tar archive")
        temporary_path.replace(archive_path)
        LOG.info("Downloaded CCPD archive: %.2f GiB", size / 1024**3)
        return size
    except Exception:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        response.close()


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"Unsafe path in CCPD tar archive: {name!r}")
    return path


def extract_deterministic_sample(
    archive_path: Path, sample_dir: Path, max_images: int
) -> tuple[int, int]:
    """Stream a .tar.xz archive and retain only the lowest-hash JPEG paths."""
    if not 0 < max_images <= MAX_SAMPLE_IMAGES:
        raise ValueError(f"max_images must be between 1 and {MAX_SAMPLE_IMAGES}")
    sample_dir.mkdir(parents=True, exist_ok=True)
    heap: list[tuple[int, str, Path]] = []
    total_jpegs = 0
    try:
        with tarfile.open(archive_path, mode="r|xz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.casefold().endswith(".jpg"):
                    continue
                relative = _safe_member_path(member.name)
                if member.size <= 0 or member.size > MAX_JPEG_MEMBER_BYTES:
                    LOG.warning("Skipping CCPD image with unsupported size: %s", relative.name)
                    continue
                total_jpegs += 1
                digest = hashlib.sha256(relative.as_posix().encode()).digest()
                rank = int.from_bytes(digest, "big")
                if len(heap) == max_images and rank >= -heap[0][0]:
                    continue

                source = archive.extractfile(member)
                if source is None:
                    continue
                destination = sample_dir.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                resolved_sample = sample_dir.resolve()
                if not destination.resolve().is_relative_to(resolved_sample):
                    raise RuntimeError(f"CCPD member escapes sample directory: {member.name!r}")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                item = (-rank, relative.as_posix(), destination)
                if len(heap) < max_images:
                    heapq.heappush(heap, item)
                else:
                    _, _, removed_path = heapq.heapreplace(heap, item)
                    removed_path.unlink(missing_ok=True)
        if not heap:
            raise RuntimeError("No JPEG images were found in the CCPD archive")
        return len(heap), total_jpegs
    except Exception:
        shutil.rmtree(sample_dir, ignore_errors=True)
        raise


def create_sample_bundle(sample_dir: Path, bundle_path: Path) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, mode="w:gz") as bundle:
        for image_path in sorted(sample_dir.rglob("*.jpg")):
            if image_path.is_symlink() or not image_path.is_file():
                continue
            bundle.add(image_path, arcname=Path("extracted") / image_path.relative_to(sample_dir))


def run(max_images: int, work_dir: Path) -> Path:
    if not 0 < max_images <= MAX_SAMPLE_IMAGES:
        raise ValueError(f"max_images must be between 1 and {MAX_SAMPLE_IMAGES}")
    archive_path = work_dir / "CCPD2019.tar.xz"
    sample_dir = work_dir / "ccpd_sample"
    bundle_path = work_dir / "ccpd_sample.tar.gz"
    if not archive_path.is_file():
        LOG.info(
            "Google Drive describes CCPD2019.tar.xz as approximately %d GB; "
            "checking actual response size and Colab free space before download.",
            CCPD_ARCHIVE_PAGE_SIZE_GIB,
        )
    download_archive(archive_path)
    extracted, total = extract_deterministic_sample(archive_path, sample_dir, max_images)
    create_sample_bundle(sample_dir, bundle_path)
    shutil.rmtree(sample_dir)
    archive_path.unlink()
    LOG.info("Retained %d deterministic sample images from %d JPEGs", extracted, total)
    LOG.info(
        "Download %s from the Colab file browser and extract under data/raw/ccpd/",
        bundle_path,
    )
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/ccpd_work"))
    parser.add_argument(
        "--download-to-browser",
        action="store_true",
        help="In Colab only, trigger a browser download of the bounded sample bundle.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bundle = run(args.max_images, args.work_dir)
    if args.download_to_browser:
        try:
            files = importlib.import_module("google.colab.files")
        except ImportError as error:
            raise RuntimeError("--download-to-browser is available only in Google Colab") from error
        files.download(str(bundle))


if __name__ == "__main__":
    main()
