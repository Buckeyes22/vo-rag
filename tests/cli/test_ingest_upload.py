"""``vo-rag ingest file`` content-type sniffing (VOR-1434).

Incident: two Automobile Electrical uploads failed server-side with
``content_type_not_allowed: actual content type is missing`` — the extension
-only guess is fragile. ``_sniff_content_type`` reads magic bytes first
(``%PDF-``, the EPUB zip's ``application/epub+zip`` mimetype marker, MOBI's
``BOOKMOBI`` at offset 60), falls back to ``mimetypes.guess_type``, and
finally to ``application/octet-stream`` — never ``None`` sent as a header.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import httpx
import respx
from click.testing import CliRunner

from vo_rag.cli.ingest import _sniff_content_type
from vo_rag.cli.main import cli

PDF_BYTES = (
    b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog>>\nendobj\n"
    b"trailer\n<</Root 1 0 R>>\n%%EOF\n"
)
MOBI_BYTES = b"\x00" * 52 + b"\x00" * 8 + b"BOOKMOBI" + b"EXTH" + b"\xff" * 100


def _epub_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", "<container/>")
    return buf.getvalue()


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# -- _sniff_content_type: magic bytes win over the extension ---------------------


def test_sniff_pdf_by_magic_bytes(tmp_path: Path) -> None:
    # Wrong extension on purpose: magic bytes must win.
    path = _write(tmp_path, "book.bin", PDF_BYTES)
    assert _sniff_content_type(path) == "application/pdf"


def test_sniff_pdf_by_extension(tmp_path: Path) -> None:
    path = _write(tmp_path, "book.pdf", PDF_BYTES)
    assert _sniff_content_type(path) == "application/pdf"


def test_sniff_epub_by_magic_bytes(tmp_path: Path) -> None:
    path = _write(tmp_path, "book.bin", _epub_bytes())
    assert _sniff_content_type(path) == "application/epub+zip"


def test_sniff_mobi_by_magic_bytes(tmp_path: Path) -> None:
    path = _write(tmp_path, "book.bin", MOBI_BYTES)
    assert _sniff_content_type(path) == "application/x-mobipocket-ebook"


def test_sniff_falls_back_to_extension_when_magic_unrecognized(tmp_path: Path) -> None:
    path = _write(tmp_path, "notes.txt", b"plain text, no known magic")
    assert _sniff_content_type(path) == "text/plain"


def test_sniff_falls_back_to_octet_stream_when_nothing_matches(tmp_path: Path) -> None:
    path = _write(tmp_path, "mystery.unknownext", b"\x01\x02\x03 not a recognized format")
    assert _sniff_content_type(path) == "application/octet-stream"


def test_sniff_never_returns_empty_or_none(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty", b"")
    result = _sniff_content_type(path)
    assert result and isinstance(result, str)


# -- end-to-end: ingest file upload carries the sniffed content type -------------


def test_ingest_file_pdf_upload_carries_pdf_content_type(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(PDF_BYTES)
        tmp_file = Path(f.name)

    try:
        with respx.mock:
            respx.post("http://localhost:8000/api/v1/intake/upload-url").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "upload_url": "https://storage.example.com/upload",
                        "garage_key": "gk-pdf-1",
                        "bucket": "test-bucket",
                        "expires_in_s": 3600,
                    },
                )
            )
            put_route = respx.put("https://storage.example.com/upload").mock(
                return_value=httpx.Response(200)
            )
            respx.post("http://localhost:8000/api/v1/intake").mock(
                return_value=httpx.Response(200, json={"envelope_id": "env-pdf"})
            )
            runner = CliRunner()
            result = runner.invoke(cli, ["ingest", "file", str(tmp_file), "--title", "Some PDF"])
            assert result.exit_code == 0
            assert put_route.called
            assert put_route.calls[0].request.headers["content-type"] == "application/pdf"
    finally:
        tmp_file.unlink(missing_ok=True)


def test_ingest_file_epub_upload_carries_epub_content_type(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        f.write(_epub_bytes())
        tmp_file = Path(f.name)

    try:
        with respx.mock:
            respx.post("http://localhost:8000/api/v1/intake/upload-url").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "upload_url": "https://storage.example.com/upload",
                        "garage_key": "gk-epub-1",
                        "bucket": "test-bucket",
                        "expires_in_s": 3600,
                    },
                )
            )
            put_route = respx.put("https://storage.example.com/upload").mock(
                return_value=httpx.Response(200)
            )
            respx.post("http://localhost:8000/api/v1/intake").mock(
                return_value=httpx.Response(200, json={"envelope_id": "env-epub"})
            )
            runner = CliRunner()
            result = runner.invoke(cli, ["ingest", "file", str(tmp_file), "--title", "Some EPUB"])
            assert result.exit_code == 0
            assert put_route.called
            assert put_route.calls[0].request.headers["content-type"] == "application/epub+zip"
    finally:
        tmp_file.unlink(missing_ok=True)


def test_upload_file_sync_always_sends_content_type_header(tmp_path: Path) -> None:
    from vo_rag.cli.ingest import _upload_file_sync

    book = tmp_path / "book.pdf"
    book.write_bytes(PDF_BYTES)
    with respx.mock:
        route = respx.put("http://garage.test/upload").mock(return_value=httpx.Response(200))
        digest = _upload_file_sync(book, "http://garage.test/upload", "application/pdf")
    assert route.calls[0].request.headers["content-type"] == "application/pdf"
    assert digest == hashlib.sha256(book.read_bytes()).hexdigest()
