import io
import tarfile
from email.message import Message
from pathlib import Path

import pytest
from scripts import colab_ccpd_sample
from scripts.colab_ccpd_sample import (
    MAX_ARCHIVE_SIZE_BYTES,
    MIN_POST_DOWNLOAD_FREE_BYTES,
    _DownloadFormParser,
    check_free_space,
    extract_deterministic_sample,
    parse_known_archive_size,
)


def _image_name(index: int) -> str:
    return (
        f"CCPD2019/base/{index:03d}-95_113-10&20_40&60-"
        "40&60_10&50_10&20_40&30-0_0_22_27_27_33_16-37-15.jpg"
    )


def _make_tar_xz(path: Path, count: int) -> None:
    with tarfile.open(path, mode="w:xz") as archive:
        for index in range(count):
            payload = f"image-{index}".encode()
            info = tarfile.TarInfo(_image_name(index))
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_parse_known_archive_size_rejects_unknown_invalid_and_oversized() -> None:
    headers = Message()
    headers["Content-Length"] = "1234"
    assert parse_known_archive_size(headers) == 1234

    with pytest.raises(RuntimeError, match="usable archive size"):
        parse_known_archive_size(Message())

    headers = Message()
    headers["Content-Length"] = "not-a-number"
    with pytest.raises(RuntimeError, match="invalid archive size"):
        parse_known_archive_size(headers)

    headers = Message()
    headers["Content-Length"] = str(MAX_ARCHIVE_SIZE_BYTES + 1)
    with pytest.raises(RuntimeError, match="safe limit"):
        parse_known_archive_size(headers)


def test_disk_preflight_requires_archive_plus_reserve() -> None:
    archive_size = 12 * 1024**3
    check_free_space(archive_size + MIN_POST_DOWNLOAD_FREE_BYTES, archive_size)
    with pytest.raises(RuntimeError, match="Not enough free Colab disk space"):
        check_free_space(archive_size + MIN_POST_DOWNLOAD_FREE_BYTES - 1, archive_size)


def test_download_preflight_aborts_before_reading_archive_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        headers = Message()
        read_called = False

        def read(self, _size: int) -> bytes:
            self.read_called = True
            return b"archive bytes"

        def close(self) -> None:
            pass

    response = FakeResponse()
    response.headers["Content-Length"] = str(12 * 1024**3)
    monkeypatch.setattr(colab_ccpd_sample, "_get_download_response", lambda _opener: response)

    with pytest.raises(RuntimeError, match="Not enough free Colab disk space"):
        colab_ccpd_sample.download_archive(tmp_path / "CCPD2019.tar.xz", free_bytes=8 * 1024**3)

    assert response.read_called is False


def test_drive_confirmation_parser_reads_only_hidden_form_fields() -> None:
    parser = _DownloadFormParser()
    parser.feed(
        '<form id="download-form" action="https://drive.google.com/download" method="get">'
        '<input type="hidden" name="id" value="public-id">'
        '<input type="hidden" name="confirm" value="confirm-token">'
        '<input type="hidden" name="uuid" value="request-id"></form>'
    )
    assert parser.action == "https://drive.google.com/download"
    assert set(parser.fields) == {"id", "confirm", "uuid"}


def test_drive_quota_page_gets_actionable_message_without_exposing_html() -> None:
    message = colab_ccpd_sample._describe_drive_page(
        b"<html><title>Quota</title><p>Too many users have downloaded this file.</p></html>"
    )
    assert "quota/rate limit" in message
    assert "wait" in message
    assert "Too many users" not in message


def test_drive_confirmation_flow_supports_multiple_forms_and_stops_on_quota() -> None:
    class FakeResponse(io.BytesIO):
        def __init__(self, content: bytes, content_type: str) -> None:
            super().__init__(content)
            self.headers = Message()
            self.headers["Content-Type"] = content_type

    class FakeOpener:
        def __init__(self, responses: list[FakeResponse]) -> None:
            self.responses = iter(responses)
            self.urls: list[str] = []

        def open(self, request: object, timeout: int) -> FakeResponse:
            del timeout
            self.urls.append(request.full_url)  # type: ignore[attr-defined]
            return next(self.responses)

    form_one = (
        b'<form id="download-form" action="https://drive.google.com/uc?export=download">'
        b'<input name="id" value="id"><input name="confirm" value="one"></form>'
    )
    form_two = (
        b'<form id="download-form" action="https://drive.usercontent.google.com/download">'
        b'<input name="id" value="id"><input name="confirm" value="two"></form>'
    )
    opener = FakeOpener(
        [
            FakeResponse(form_one, "text/html"),
            FakeResponse(form_two, "text/html"),
            FakeResponse(b"archive", "application/octet-stream"),
        ]
    )

    response = colab_ccpd_sample._get_download_response(opener)
    assert response.read() == b"archive"
    assert len(opener.urls) == 3
    assert "&id=id&confirm=one" in opener.urls[1]

    quota_opener = FakeOpener([FakeResponse(b"<p>Download quota exceeded</p>", "text/html")])
    with pytest.raises(RuntimeError, match="quota/rate limit"):
        colab_ccpd_sample._get_download_response(quota_opener)

    unsafe_form = (
        b'<form id="download-form" action="https://example.test/download">'
        b'<input name="id" value="id"><input name="confirm" value="token"></form>'
    )
    with pytest.raises(RuntimeError, match="unexpected confirmation destination"):
        colab_ccpd_sample._get_download_response(
            FakeOpener([FakeResponse(unsafe_form, "text/html")])
        )


def test_extract_streams_only_deterministic_capped_jpegs(tmp_path: Path) -> None:
    archive = tmp_path / "sample.tar.xz"
    _make_tar_xz(archive, count=20)
    first_dir = tmp_path / "sample-a"
    second_dir = tmp_path / "sample-b"

    selected, total = extract_deterministic_sample(archive, first_dir, max_images=5)
    repeated, repeated_total = extract_deterministic_sample(archive, second_dir, max_images=5)

    first_files = sorted(
        path.relative_to(first_dir).as_posix() for path in first_dir.rglob("*.jpg")
    )
    second_files = sorted(
        path.relative_to(second_dir).as_posix() for path in second_dir.rglob("*.jpg")
    )
    assert (selected, total) == (5, 20)
    assert (repeated, repeated_total) == (5, 20)
    assert first_files == second_files
    assert len(first_files) == 5
    assert all(path.is_file() for path in first_dir.rglob("*.jpg"))


def test_extract_rejects_sample_larger_than_approved_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        extract_deterministic_sample(tmp_path / "missing.tar.xz", tmp_path / "sample", 101)
