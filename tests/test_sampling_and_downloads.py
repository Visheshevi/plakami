"""Sampling must not correlate with the split, and downloads must survive rate limiting."""

import io
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from platedet.data import open_images
from platedet.data.ccpd import select_ccpd_images
from platedet.data.combine import apply_ccpd_share_cap
from platedet.data.manifest import CCPD_LICENSE_URL, ManifestRecord
from platedet.data.split import assign_split


def _ccpd_name(index: int) -> str:
    return f"{index:05d}-95_113-10&20_40&60-40&60_10&50_10&20_40&30-0_0_22_27_27_33_16-37-15.jpg"


def test_ccpd_sample_spans_all_splits(tmp_path: Path) -> None:
    for index in range(400):
        (tmp_path / _ccpd_name(index)).touch()

    selected, _ = select_ccpd_images(tmp_path, max_images=100)

    splits = {assign_split(f"ccpd:{item.relative_path}") for item in selected}
    assert splits == {"train", "val", "test"}


def test_ccpd_cap_keeps_records_from_all_splits() -> None:
    def record(source: str, index: int) -> ManifestRecord:
        return ManifestRecord(
            f"{source}:{index}", source, str(index), "https://x.test", "a", "MIT",
            CCPD_LICENSE_URL, "0", "0", 10, 10, "train", 1, 0, '["license_plate"]',
        )  # fmt: skip

    records = [record("kaggle_car_plate_detection", i) for i in range(233)]
    records += [record("ccpd", i) for i in range(1000)]

    kept, _ = apply_ccpd_share_cap(records, 0.30)

    ccpd = [item for item in kept if item.source == "ccpd"]
    assert len(ccpd) == 99
    assert {assign_split(item.image_id) for item in ccpd} == {"train", "val", "test"}


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x.test/a.jpg", code, "error", headers, None)


def test_download_retries_rate_limits_with_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes: list[object] = [_http_error(429, "7"), _http_error(503), _Response(b"jpeg")]
    monkeypatch.setattr(open_images.urllib.request, "urlopen", lambda *a, **k: _raise_or(outcomes))
    waits: list[float] = []

    open_images.download_url("https://x.test/a.jpg", tmp_path / "a.jpg", sleep=waits.append)

    assert (tmp_path / "a.jpg").read_bytes() == b"jpeg"
    assert waits == [7.0, 10.0]  # Retry-After, then base_delay * 2**1
    assert not (tmp_path / "a.jpg.part").exists()


def test_download_does_not_retry_missing_images_and_gives_up_after_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes: list[object] = [_http_error(404)]
    monkeypatch.setattr(open_images.urllib.request, "urlopen", lambda *a, **k: _raise_or(outcomes))
    waits: list[float] = []
    with pytest.raises(urllib.error.HTTPError, match="404"):
        open_images.download_url("https://x.test/a.jpg", tmp_path / "a.jpg", sleep=waits.append)
    assert waits == []

    outcomes[:] = [_http_error(429) for _ in range(3)]
    with pytest.raises(urllib.error.HTTPError, match="429"):
        open_images.download_url(
            "https://x.test/a.jpg", tmp_path / "a.jpg", max_retries=2, sleep=waits.append
        )
    assert len(waits) == 2
    assert not (tmp_path / "a.jpg").exists()


def _raise_or(outcomes: list[object]) -> _Response:
    outcome = outcomes.pop(0)
    if isinstance(outcome, Exception):
        raise outcome
    assert isinstance(outcome, _Response)
    return outcome
