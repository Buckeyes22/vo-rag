"""``vo_rag.core.parsing.preflight`` (VOR-1434).

One test per per-file check, plus the incident-table regression shapes: the 8
race-schedule PDFs (not_a_book by filename), an image-heavy giant (giant, not
native), an EPUB with an escaping href, a ZIP64 EPUB, an
image-only-with-front-matter EPUB, and a 13,400x9,100pt oversize page.

Checks that need ebooklib/pypandoc (the pandoc dry run + is_image_only) are
guarded with ``pytest.importorskip``, matching
``test_epub_manifest_normalization.py``; everything else (container
integrity, ZIP64 detection, normalization-needed detection, escaping-href
detection, format detection, PDF kind/giant/oversize/not_a_book, duplicate
detection) runs unconditionally.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import fitz  # type: ignore[import-untyped, unused-ignore]
import pytest

from tests.fixtures.epub.build_image_only_epub import build as build_image_only_epub
from tests.fixtures.epub.build_preflight_epub_fixtures import (
    build_corrupt_deflate_epub,
    build_escaping_href_epub,
    build_missing_manifest_item_epub,
    build_zip64_epub,
    build_zip_bomb_epub,
)
from tests.fixtures.epub.build_text_html_manifest_epub import _build_epub
from vo_rag.core.parsing import preflight
from vo_rag.core.parsing.parsers import epub

# ---------------------------------------------------------------------------
# PDF builders (fitz, in-process -- no fixture files needed; matches
# test_pdf_kind.py's convention).
# ---------------------------------------------------------------------------

_BODY = (
    "The boundary layer thickens along the plate as viscous stresses slow the\n"
    "fluid near the wall. The pressure gradient in the free stream controls\n"
    "whether the flow stays attached, and the Reynolds number sets the point\n"
    "where the laminar layer becomes turbulent over the surface of the body.\n"
)


def _png() -> bytes:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 51), False)
    pix.set_rect(pix.irect, (180, 170, 150))
    return bytes(pix.tobytes("png"))


def _build_pdf(
    path: Path,
    pages: list[str],
    *,
    width: float = 400,
    height: float = 500,
) -> Path:
    png = _png()
    doc = fitz.open()
    for spec in pages:
        page = doc.new_page(width=width, height=height)
        if spec == "image":
            page.insert_image(page.rect, stream=png)
        elif spec == "blank":
            pass
        else:
            page.insert_text((20, 30), spec, fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# check_container
# ---------------------------------------------------------------------------


def test_check_container_none_for_valid_epub(tmp_path: Path) -> None:
    epub = _build_epub(tmp_path / "ok.epub")
    assert preflight.check_container(epub) is None


def test_check_container_corrupt_for_garbage_epub(tmp_path: Path) -> None:
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"not a zip at all")
    assert preflight.check_container(bad) == "corrupt_container"


def test_check_container_none_for_valid_pdf(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "ok.pdf", [_BODY])
    assert preflight.check_container(pdf) is None


def test_check_container_corrupt_for_garbage_pdf(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-not-really-a-pdf" + b"\x00" * 32)
    assert preflight.check_container(bad) == "corrupt_container"


def test_check_container_corrupt_for_bad_deflate_stream(tmp_path: Path) -> None:
    """``testzip()`` raises ``zlib.error`` (not
    ``BadZipFile``) on a corrupt deflate stream; that must still come back as
    ``corrupt_container``, never escape as an exception."""
    import zlib

    bad = build_corrupt_deflate_epub(tmp_path / "bad_deflate.epub")
    with zipfile.ZipFile(bad) as archive, pytest.raises(zlib.error):
        archive.testzip()

    assert preflight.check_container(bad) == "corrupt_container"


@pytest.mark.parametrize(
    "exc",
    [EOFError("truncated"), RuntimeError("encrypted"), NotImplementedError("compression 99")],
)
def test_check_container_corrupt_for_any_testzip_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    epub_path = _build_epub(tmp_path / "ok.epub")

    def _raise(_self: zipfile.ZipFile) -> None:
        raise exc

    monkeypatch.setattr(zipfile.ZipFile, "testzip", _raise)
    assert preflight.check_container(epub_path) == "corrupt_container"


def test_check_container_corrupt_for_zero_page_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PDF that opens but has no pages is not a book
    container -- ``corrupt_container``, not a crash later in page stats."""
    pdf = _build_pdf(tmp_path / "ok.pdf", [_BODY])

    class _ZeroPages:
        page_count = 0

        def __enter__(self) -> _ZeroPages:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    monkeypatch.setattr(fitz, "open", lambda *_a, **_k: _ZeroPages())
    assert preflight.check_container(pdf) == "corrupt_container"


def test_check_container_zip_bomb_by_compression_ratio(tmp_path: Path) -> None:
    """A member over 100 MB decompressed at more than 200:1."""
    bomb = build_zip_bomb_epub(tmp_path / "bomb.epub")
    assert preflight.check_container(bomb) == "zip_bomb"


def test_check_container_zip_bomb_by_total_declared_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The total declared size cap (2 GiB in production,
    shrunk here so the fixture stays tiny)."""
    epub_path = _build_epub(tmp_path / "ok.epub")
    monkeypatch.setattr(epub, "ZIP_BOMB_MAX_TOTAL_BYTES", 1024)
    assert preflight.check_container(epub_path) == "zip_bomb"


def test_zip_bomb_epub_is_rejected_before_any_decompression(tmp_path: Path) -> None:
    bomb = build_zip_bomb_epub(tmp_path / "bomb.epub")
    result = preflight.inspect_file(bomb)
    assert result.verdict == "reject"
    assert result.reasons == ("zip_bomb",)


def test_check_container_none_for_unrelated_extension(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert preflight.check_container(other) is None


# ---------------------------------------------------------------------------
# sha256 + fingerprint + duplicate detection
# ---------------------------------------------------------------------------


def test_inspect_file_populates_sha256_and_fingerprint(tmp_path: Path) -> None:
    epub = _build_epub(tmp_path / "book.epub")
    result = preflight.inspect_file(epub)
    assert result.sha256 and len(result.sha256) == 64
    # A tiny 3-chapter fixture is under fingerprint.MIN_TEXT_CHARS, so no
    # fingerprint is expected -- but sha256 must always be present.
    assert result.verdict == "ok"


def test_missing_catalog_skips_duplicate_check(tmp_path: Path) -> None:
    epub = _build_epub(tmp_path / "book.epub")
    result = preflight.inspect_file(epub, catalog=None)
    assert not result.verdict.startswith("duplicate_of=")


def test_catalog_duplicate_by_sha256(tmp_path: Path) -> None:
    from vo_rag.core.parsing.fingerprint import file_sha256

    epub = _build_epub(tmp_path / "book.epub")
    sha = file_sha256(epub)
    catalog = {"indexes": {"by_sha256": {sha: ["vo-rag-corpus-sha256:" + sha]}}}

    result = preflight.inspect_file(epub, catalog=catalog)

    assert result.verdict == f"duplicate_of=vo-rag-corpus-sha256:{sha}"


def test_catalog_duplicate_by_fingerprint(tmp_path: Path) -> None:
    from vo_rag.core.parsing.fingerprint import content_fingerprint

    # A big enough EPUB to clear fingerprint.MIN_TEXT_CHARS (2000 chars).
    epub = _build_epub(tmp_path / "book.epub", n_chapters=3)
    # 3 chapters * ~60 words each is short of 2000 normalized chars, so pad
    # by building with more chapters worth of body text via a bigger fixture.
    fp = content_fingerprint(epub)
    if fp is None:
        pytest.skip("fixture EPUB text is under fingerprint.MIN_TEXT_CHARS")
    catalog = {"indexes": {"by_fingerprint": {fp: ["vo-rag-corpus-sha256:deadbeef"]}}}

    result = preflight.inspect_file(epub, catalog=catalog)

    assert result.verdict == "duplicate_of=vo-rag-corpus-sha256:deadbeef"


def test_inspect_batch_flags_in_batch_exact_duplicate(tmp_path: Path) -> None:
    src = _build_epub(tmp_path / "book.epub")
    dupe = tmp_path / "book_copy.epub"
    dupe.write_bytes(src.read_bytes())

    report = preflight.inspect_batch([src, dupe])

    by_path = {r.path: r for r in report.results}
    assert by_path[str(src)].verdict == "ok"
    assert by_path[str(dupe)].verdict == f"duplicate_of={by_path[str(src)].sha256}"
    assert report.counts["verdict"]["duplicate"] == 1


def test_inspect_batch_flags_in_batch_fingerprint_duplicate(tmp_path: Path) -> None:
    """Same text, different bytes (re-saved with different zip compression)."""
    from vo_rag.core.parsing.fingerprint import content_fingerprint

    src = _build_epub(tmp_path / "book.epub")
    if content_fingerprint(src) is None:
        pytest.skip("fixture EPUB text is under fingerprint.MIN_TEXT_CHARS")

    resaved = tmp_path / "book_resaved.epub"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(resaved, "w") as zout:
        for info in zin.infolist():
            # Force a different compression type -> different bytes, same text.
            compress = (
                zipfile.ZIP_STORED
                if info.compress_type != zipfile.ZIP_STORED
                else zipfile.ZIP_DEFLATED
            )
            zout.writestr(info.filename, zin.read(info.filename), compress_type=compress)
    assert resaved.read_bytes() != src.read_bytes()

    report = preflight.inspect_batch([src, resaved])

    by_path = {r.path: r for r in report.results}
    assert by_path[str(resaved)].verdict.startswith("duplicate_of=")
    assert by_path[str(resaved)].verdict == f"duplicate_of={by_path[str(src)].sha256}"


# ---------------------------------------------------------------------------
# Magic bytes / content type -> unknown_type
# ---------------------------------------------------------------------------


def test_unknown_type_is_rejected(tmp_path: Path) -> None:
    mystery = tmp_path / "mystery.xyz123"
    mystery.write_bytes(b"\x00\x01\x02not a known format\xff")
    result = preflight.inspect_file(mystery)
    assert result.verdict == "reject"
    assert result.reasons == ("unknown_type",)
    assert result.kind is None


# ---------------------------------------------------------------------------
# Container integrity -> reject corrupt_container (end to end via inspect_file)
# ---------------------------------------------------------------------------


def test_corrupt_epub_container_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"PK\x03\x04garbage, not a real zip")
    result = preflight.inspect_file(bad)
    assert result.verdict == "reject"
    assert result.reasons == ("corrupt_container",)


def test_corrupt_pdf_container_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\ngarbage" + b"\x00" * 64)
    result = preflight.inspect_file(bad)
    assert result.verdict == "reject"
    assert result.reasons == ("corrupt_container",)


# ---------------------------------------------------------------------------
# ZIP64 EOCD -> info "zip64" (VOR-1432, ZIP64 shape)
# ---------------------------------------------------------------------------


def test_zip64_epub_is_annotated_and_not_rejected(tmp_path: Path) -> None:
    zip64 = build_zip64_epub(tmp_path / "understanding_f1.epub")
    with zipfile.ZipFile(zip64) as z:
        assert z.testzip() is None  # a valid zip, just ZIP64-terminated

    result = preflight.inspect_file(zip64)

    assert "zip64" in result.reasons
    assert result.verdict != "reject"


# ---------------------------------------------------------------------------
# EPUB normalization needed -> info "normalized"
# ---------------------------------------------------------------------------


def test_missing_manifest_item_is_annotated_normalized(tmp_path: Path) -> None:
    broken = build_missing_manifest_item_epub(tmp_path / "no_nav.epub")

    result = preflight.inspect_file(broken)

    assert "normalized" in result.reasons
    assert result.verdict != "reject"


def test_normal_epub_is_not_annotated_normalized(tmp_path: Path) -> None:
    normal = _build_epub(tmp_path / "ok.epub")

    result = preflight.inspect_file(normal)

    assert "normalized" not in result.reasons
    assert "zip64" not in result.reasons


# ---------------------------------------------------------------------------
# EPUB hrefs escaping the OPF dir -> info escaping_hrefs (an EPUB whose manifest escapes);
# _normalize_epub repairs them (VOR-1434 task 2), so the book is not rejected.
# ---------------------------------------------------------------------------


def test_escaping_href_epub_is_info_not_reject(tmp_path: Path) -> None:
    escaping = build_escaping_href_epub(tmp_path / "suckers_progress.epub")

    result = preflight.inspect_file(escaping)

    assert result.verdict == "ok"
    assert result.route == "epub"
    assert "escaping_hrefs: ../jacket.xhtml" in result.reasons
    assert "normalized" in result.reasons


def test_preflight_uses_the_parsers_escaping_hrefs() -> None:
    assert not hasattr(preflight, "_escaping_hrefs")  # one copy: epub._escaping_hrefs


def test_escaping_hrefs_helper_flags_parent_traversal() -> None:
    opf_bytes = (
        b"<manifest>"
        b'<item id="chap0" href="chap_0.xhtml" media-type="application/xhtml+xml"/>'
        b'<item id="jacket" href="../jacket.xhtml" media-type="application/xhtml+xml"/>'
        b"</manifest>"
    )
    assert epub._escaping_hrefs("OEBPS/content.opf", opf_bytes) == ["../jacket.xhtml"]


def test_escaping_hrefs_helper_allows_normal_relative_paths() -> None:
    opf_bytes = (
        b"<manifest>"
        b'<item id="chap0" href="text/chap_0.xhtml" media-type="application/xhtml+xml"/>'
        b'<item id="img0" href="images/cover.png" media-type="image/png"/>'
        b"</manifest>"
    )
    assert epub._escaping_hrefs("OEBPS/content.opf", opf_bytes) == []


# ---------------------------------------------------------------------------
# Pandoc conversion dry run -> reject pandoc_failed (needs ebooklib/pypandoc)
# ---------------------------------------------------------------------------


def test_pandoc_dry_run_succeeds_on_a_valid_epub(tmp_path: Path) -> None:
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")
    if not _pandoc_binary_available():
        pytest.skip("pandoc binary not installed")
    valid = _build_epub(tmp_path / "ok.epub")

    assert preflight._pandoc_dry_run(str(valid)) is None


def test_pandoc_dry_run_reports_failure_first_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_pandoc_dry_run`` shells out via ``pypandoc.get_pandoc_path()`` +
    ``subprocess.run`` -- exercised here with a
    stand-in "pandoc" that fails, proving only the first stderr line survives
    into the reason."""
    pytest.importorskip("pypandoc")
    import pypandoc  # type: ignore[import-not-found, import-untyped, unused-ignore]

    failing = tmp_path / "failing_pandoc.sh"
    failing.write_text(
        "#!/bin/sh\necho 'pandoc: boom' 1>&2\necho 'some traceback detail' 1>&2\nexit 1\n"
    )
    failing.chmod(0o755)
    monkeypatch.setattr(pypandoc, "get_pandoc_path", lambda: str(failing))

    reason = preflight._pandoc_dry_run("/does/not/matter.epub")

    assert reason is not None
    assert reason.startswith("pandoc_failed:")
    assert "boom" in reason
    assert "traceback" not in reason  # only the first line


def test_pandoc_dry_run_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung pandoc process must be
    genuinely killed, not merely given up on. The stand-in "pandoc" sleeps
    5s; asserting wall-clock time stays well under that proves
    ``subprocess.run(timeout=...)`` actually terminated the child instead of
    a ``ThreadPoolExecutor`` leaving it running in the background."""
    pytest.importorskip("pypandoc")
    import time

    import pypandoc  # type: ignore[import-not-found, import-untyped, unused-ignore]

    slow = tmp_path / "slow_pandoc.sh"
    slow.write_text("#!/bin/sh\nsleep 5\n")
    slow.chmod(0o755)
    monkeypatch.setattr(pypandoc, "get_pandoc_path", lambda: str(slow))

    start = time.monotonic()
    reason = preflight._pandoc_dry_run("/does/not/matter.epub", timeout_s=0.2)
    elapsed = time.monotonic() - start

    assert reason == "pandoc_failed: timed out after 0s"
    assert elapsed < 3.0, f"took {elapsed:.1f}s -- the hung pandoc process was not killed"


def test_pandoc_dry_run_sandboxes_pandoc_and_passes_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pandoc runs with ``--sandbox`` (no reads beyond the
    input, no network) and the input is absolute after ``--``, so a
    filename starting with ``-`` is never read as an option."""
    pytest.importorskip("pypandoc")
    import pypandoc  # type: ignore[import-not-found, import-untyped, unused-ignore]

    argv_file = tmp_path / "argv.txt"
    recorder = tmp_path / "record_pandoc.sh"
    recorder.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {argv_file}\n')
    recorder.chmod(0o755)
    monkeypatch.setattr(pypandoc, "get_pandoc_path", lambda: str(recorder))
    monkeypatch.chdir(tmp_path)

    assert preflight._pandoc_dry_run("-o.epub") is None

    argv = argv_file.read_text().splitlines()
    assert "--sandbox" in argv
    assert argv[-2] == "--"
    assert argv[-1] == str(tmp_path / "-o.epub")


def test_pandoc_dry_run_handles_a_leading_dash_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pandoc invocation, end to end with the real pandoc binary."""
    pytest.importorskip("pypandoc")
    if not _pandoc_binary_available():
        pytest.skip("pandoc binary not installed")
    _build_epub(tmp_path / "-weird.epub")
    monkeypatch.chdir(tmp_path)

    assert preflight._pandoc_dry_run("-weird.epub") is None


def test_epub_with_bad_container_never_reaches_pandoc(tmp_path: Path) -> None:
    """A corrupt container stops the file before the (expensive) pandoc step."""
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"PK\x03\x04garbage")
    result = preflight.inspect_file(bad)
    assert result.verdict == "reject"
    assert result.reasons == ("corrupt_container",)


def test_runtime_extras_missing_short_circuits_to_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ebooklib/pypandoc, a structurally sound EPUB is still ``ok``
    (route ``epub``, no image-only/pandoc verdict possible) -- never a crash."""
    monkeypatch.setattr(preflight, "_runtime_extras_available", lambda: False)
    normal = _build_epub(tmp_path / "ok.epub")

    result = preflight.inspect_file(normal)

    assert result.verdict == "ok"
    assert result.route == "epub"
    assert any(r.startswith("skipped:") for r in result.reasons)


# ---------------------------------------------------------------------------
# EPUB image-only -> route image_epub_ocr (image-only-with-front-matter shape)
# ---------------------------------------------------------------------------


def test_image_only_epub_routes_to_image_epub_ocr(tmp_path: Path) -> None:
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")
    if not _pandoc_binary_available():
        pytest.skip("pandoc binary not installed")
    image_only = build_image_only_epub(tmp_path / "image_only.epub")

    result = preflight.inspect_file(image_only)

    assert result.verdict == "ok"
    assert result.route == "image_epub_ocr"


def test_image_only_epub_with_front_matter_still_routes_to_ocr(tmp_path: Path) -> None:
    """Image-only-with-front-matter shape: real front-matter prose ahead of the scan
    pages must not rescue an otherwise image-only EPUB (VOR-1427). The real
    book was 401 images / 8,086 chars (well under the 50 chars/image budget);
    this fixture keeps the default 25-page scan and scales the front matter
    down to 1 paragraph (under the 25-image budget of 1,250 chars) rather
    than reproducing 401 real pages through pandoc."""
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")
    if not _pandoc_binary_available():
        pytest.skip("pandoc binary not installed")
    from tests.fixtures.epub.build_image_only_with_front_matter_epub import (
        build as build_image_only_with_front_matter,
    )

    fixture = build_image_only_with_front_matter(
        tmp_path / "image_only_front_matter.epub", front_matter_paragraphs=1
    )

    result = preflight.inspect_file(fixture)

    assert result.verdict == "ok"
    assert result.route == "image_epub_ocr"


def test_normal_epub_routes_to_epub(tmp_path: Path) -> None:
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")
    if not _pandoc_binary_available():
        pytest.skip("pandoc binary not installed")
    normal = _build_epub(tmp_path / "ok.epub")

    result = preflight.inspect_file(normal)

    assert result.verdict == "ok"
    assert result.route == "epub"


def _pandoc_binary_available() -> bool:
    import shutil

    return shutil.which("pandoc") is not None


# ---------------------------------------------------------------------------
# PDF kind -> route native / ocr
# ---------------------------------------------------------------------------


def test_native_pdf_routes_to_native(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "native.pdf", [_BODY * 3] * 20)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"
    assert result.route == "native"
    assert result.kind == "pdf"
    assert result.pages == 20


def test_scanned_pdf_routes_to_ocr(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "scanned.pdf", ["image"] * 40)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"
    assert result.route == "ocr"


# ---------------------------------------------------------------------------
# Page count and page size -> route giant; info oversize_pages (the docling
# parser rescales such pages, VOR-1434 task 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pages", "kind", "is_giant"),
    [
        (1001, "native", True),
        (1000, "native", False),
        (700, "scanned", True),
        (699, "scanned", False),
        (700, "native", False),
        (5000, "scanned", True),
    ],
)
def test_giant_route_thresholds(pages: int, kind: str, is_giant: bool) -> None:
    assert preflight._giant_route(pages, kind) == ("giant" if is_giant else None)


def test_giant_pdf_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Giant-textbook shape: image-heavy at 700+ pages routes giant, not ocr."""
    pdf = _build_pdf(tmp_path / "giant_textbook.pdf", ["image"] * 3)
    monkeypatch.setattr(preflight, "_pdf_page_stats", lambda p: (750, 400.0, ""))

    result = preflight.inspect_file(pdf)

    assert result.verdict == "ok"
    assert result.route == "giant"
    assert result.pages == 750


def test_oversize_page_pdf_is_info_not_reject(tmp_path: Path) -> None:
    """The real regression file's own dimensions, 13,400 x 9,100pt. The
    threshold is 5,000pt on the longest side precisely so this real shape trips
    it. It is parsed after a rescale, so the finding is info."""
    pdf = _build_pdf(tmp_path / "oversize.pdf", [_BODY] * 12, width=13_400, height=9_100)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"
    assert result.route == "native"
    assert result.reasons == ("oversize_pages",)


def test_page_at_threshold_is_not_oversize(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "ok_size.pdf", [_BODY], width=5_000, height=500)
    result = preflight.inspect_file(pdf)
    assert "oversize_pages" not in result.reasons


def test_page_just_over_threshold_is_oversize(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "just_over.pdf", [_BODY] * 12, width=5_000.5, height=500)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"
    assert result.reasons == ("oversize_pages",)


# ---------------------------------------------------------------------------
# Not a book -> reject not_a_book
# ---------------------------------------------------------------------------


def test_short_pdf_with_little_text_is_not_a_book(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "flyer.pdf", ["blank", "Hi", "blank"])
    result = preflight.inspect_file(pdf)
    assert result.verdict == "reject"
    assert result.reasons == ("not_a_book",)


def test_short_pdf_with_real_content_is_not_flagged(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "short_book.pdf", [_BODY * 3] * 5)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"


def test_race_schedule_filename_is_not_a_book_regardless_of_content(tmp_path: Path) -> None:
    """The 8 race-schedule PDFs (2026-09-26 failure set): flagged by filename
    even when the content itself looks like a normal multi-page document."""
    pdf = _build_pdf(tmp_path / "f1_2026_race_schedule.pdf", [_BODY * 3] * 20)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "reject"
    assert "not_a_book" in result.reasons


@pytest.mark.parametrize(
    "filename",
    [
        "spring_flyer.pdf",
        "menu_2026.pdf",
        "event-brochure.pdf",
        "concert_poster.pdf",
        "class_schedule.pdf",
    ],
)
def test_filename_patterns_reject_as_not_a_book(tmp_path: Path, filename: str) -> None:
    pdf = _build_pdf(tmp_path / filename, [_BODY * 3] * 20)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "reject"
    assert "not_a_book" in result.reasons


@pytest.mark.parametrize(
    "filename",
    ["A winter program for beginning track and field athletics.pdf", "Race programme history.pdf"],
)
def test_book_length_pdf_with_a_flagged_word_is_a_book(tmp_path: Path, filename: str) -> None:
    """VOR-1434: a 70-page beginners' track and field training program was
    rejected as not_a_book by its filename alone."""
    pdf = _build_pdf(tmp_path / filename, [_BODY * 3] * 70)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"
    assert "not_a_book" not in result.reasons


@pytest.mark.parametrize(
    "filename", ["Scripting Basics.pdf", "Menuhin biography.pdf", "Ticketing systems.pdf"]
)
def test_flagged_words_inside_other_words_do_not_match(tmp_path: Path, filename: str) -> None:
    pdf = _build_pdf(tmp_path / filename, [_BODY * 3] * 20)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"


def test_normal_filename_is_not_flagged_not_a_book(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "a_real_textbook.pdf", [_BODY * 3] * 20)
    result = preflight.inspect_file(pdf)
    assert result.verdict == "ok"


# ---------------------------------------------------------------------------
# inspect_batch / counts
# ---------------------------------------------------------------------------


def test_inspect_batch_counts_by_route_and_verdict(tmp_path: Path) -> None:
    native = _build_pdf(tmp_path / "native.pdf", [_BODY * 3] * 20)
    scanned = _build_pdf(tmp_path / "scanned.pdf", ["image"] * 40)
    schedule = _build_pdf(tmp_path / "race_schedule.pdf", [_BODY * 5] * 21)

    report = preflight.inspect_batch([native, scanned, schedule])

    assert report.counts["route"] == {"native": 1, "ocr": 1, "none": 1}
    assert report.counts["verdict"] == {"ok": 2, "reject": 1}


def test_inspect_batch_process_pool_matches_sequential(tmp_path: Path) -> None:
    """``jobs > 1`` (a real process pool) produces the same verdicts/routes as
    ``jobs=1`` for the same batch."""
    native = _build_pdf(tmp_path / "native.pdf", [_BODY * 3] * 20)
    scanned = _build_pdf(tmp_path / "scanned.pdf", ["image"] * 40)

    sequential = preflight.inspect_batch([native, scanned], jobs=1)
    parallel = preflight.inspect_batch([native, scanned], jobs=2, timeout_s=30)

    seq_by_path = {r.path: (r.verdict, r.route) for r in sequential.results}
    par_by_path = {r.path: (r.verdict, r.route) for r in parallel.results}
    assert seq_by_path == par_by_path


def test_inspect_batch_kills_a_hung_worker_within_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that hangs for 5s must not block
    ``inspect_batch`` anywhere near that long -- the process pool has to
    actually terminate the stuck worker on exit, not just give up waiting on
    its future (``ProcessPoolExecutor``'s context manager blocks forever on
    ``shutdown(wait=True)`` in that case; ``multiprocessing.Pool``'s calls
    ``terminate()`` instead)."""
    import time as time_mod

    pdf = _build_pdf(tmp_path / "slow.pdf", [_BODY])

    def _hang(path: Path) -> tuple[int, float, str]:
        time_mod.sleep(5)
        return (1, 100.0, "irrelevant")  # pragma: no cover - never reached

    monkeypatch.setattr(preflight, "_pdf_page_stats", _hang)

    start = time_mod.monotonic()
    report = preflight.inspect_batch([pdf], jobs=2, timeout_s=0.3)
    elapsed = time_mod.monotonic() - start

    assert elapsed < 3.0, f"took {elapsed:.1f}s -- the hung worker was not killed"
    result = report.results[0]
    assert result.verdict == "reject"
    assert result.reasons == ("preflight_timeout",)


def test_preflight_result_is_frozen(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "native.pdf", [_BODY * 3] * 20)
    result = preflight.inspect_file(pdf)
    with pytest.raises(AttributeError):
        result.verdict = "ok"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# inspect_file never raises
# ---------------------------------------------------------------------------


def test_inspect_file_never_raises_on_unreadable_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    result = preflight.inspect_file(missing)
    assert result.verdict == "reject"
    assert result.sha256 is None


# ---------------------------------------------------------------------------
# CLI: _collect_preflight_paths manifest-header detection (vo_rag.cli.preflight)
#
# A substring match on "path" must not swallow a header-less first row like
# "epath.pdf" as a header and silently drop it. A CLI-layer helper, tested
# here because it has no separate CLI test file.
# ---------------------------------------------------------------------------


def test_manifest_first_row_containing_path_as_substring_is_not_a_header(tmp_path: Path) -> None:
    from vo_rag.cli.preflight import _collect_preflight_paths

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("epath.pdf\nanother.pdf\n")

    paths = _collect_preflight_paths(manifest)

    assert paths == [Path("epath.pdf"), Path("another.pdf")]


def test_manifest_with_exact_path_header_is_recognized(tmp_path: Path) -> None:
    from vo_rag.cli.preflight import _collect_preflight_paths

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("title,path\nSome Book,real/book.pdf\n")

    paths = _collect_preflight_paths(manifest)

    assert paths == [Path("real/book.pdf")]


def test_manifest_with_uppercase_path_header_is_recognized(tmp_path: Path) -> None:
    from vo_rag.cli.preflight import _collect_preflight_paths

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("Path\nreal/book.pdf\n")

    paths = _collect_preflight_paths(manifest)

    assert paths == [Path("real/book.pdf")]


# ---------------------------------------------------------------------------
# select_canary (VOR-1434 task 3)
# ---------------------------------------------------------------------------


def _canary_result(
    path: str,
    *,
    kind: str | None,
    route: str | None,
    pages: int | None = None,
    verdict: str = "ok",
) -> preflight.PreflightResult:
    return preflight.PreflightResult(path, f"sha-{path}", None, kind, pages, route, verdict, ())


def test_select_canary_picks_the_largest_giants_first() -> None:
    giants = [
        _canary_result(f"giant-{pages}.pdf", kind="pdf", route="giant", pages=pages)
        for pages in (1100, 2000, 1500, 1200)
    ]
    report = preflight.PreflightReport(tuple(giants), {})

    sample = preflight.select_canary(report, n=20)

    assert sample.buckets["giants"] == ("giant-2000.pdf", "giant-1500.pdf", "giant-1200.pdf")


def test_select_canary_caps_giants_at_three_even_with_more_available() -> None:
    giants = [
        _canary_result(f"giant-{i}.pdf", kind="pdf", route="giant", pages=1100 + i)
        for i in range(5)
    ]
    report = preflight.PreflightReport(tuple(giants), {})

    sample = preflight.select_canary(report, n=20)

    assert len(sample.buckets["giants"]) == 3


def test_select_canary_image_heavy_epubs_are_the_image_only_routed_ones() -> None:
    results = [
        _canary_result("a.epub", kind="epub", route="image_epub_ocr"),
        _canary_result("b.epub", kind="epub", route="image_epub_ocr"),
        _canary_result("c.epub", kind="epub", route="epub"),  # not image-only -- excluded
    ]
    report = preflight.PreflightReport(tuple(results), {})

    sample = preflight.select_canary(report, n=20)

    assert set(sample.buckets["image_heavy_epubs"]) == {"a.epub", "b.epub"}


def test_select_canary_caps_image_heavy_epubs_at_three() -> None:
    results = [
        _canary_result(f"epub-{i}.epub", kind="epub", route="image_epub_ocr") for i in range(5)
    ]
    report = preflight.PreflightReport(tuple(results), {})

    sample = preflight.select_canary(report, n=20)

    assert len(sample.buckets["image_heavy_epubs"]) == 3


def test_select_canary_scans_are_ocr_routed_pdfs_by_page_count() -> None:
    results = [
        _canary_result(f"scan-{pages}.pdf", kind="pdf", route="ocr", pages=pages)
        for pages in (50, 400, 200, 100, 300)
    ]
    report = preflight.PreflightReport(tuple(results), {})

    sample = preflight.select_canary(report, n=20)

    assert sample.buckets["scans"] == (
        "scan-400.pdf",
        "scan-300.pdf",
        "scan-200.pdf",
        "scan-100.pdf",
    )


def test_select_canary_most_figures_proxies_on_page_count_of_native_pdfs() -> None:
    results = [
        _canary_result(f"native-{pages}.pdf", kind="pdf", route="native", pages=pages)
        for pages in (10, 900, 500, 300, 700)
    ]
    report = preflight.PreflightReport(tuple(results), {})

    sample = preflight.select_canary(report, n=20)

    assert sample.buckets["most_figures"] == (
        "native-900.pdf",
        "native-700.pdf",
        "native-500.pdf",
        "native-300.pdf",
    )


def test_select_canary_fills_the_rest_with_random_native() -> None:
    natives = [
        _canary_result(f"n-{i}.pdf", kind="pdf", route="native", pages=100) for i in range(10)
    ]
    report = preflight.PreflightReport(tuple(natives), {})

    sample = preflight.select_canary(report, n=20, seed=1)

    # 4 of the 10 natives are already claimed by most_figures (sorted by
    # page count -- all tied at 100, so path order breaks the tie), leaving
    # 6 for the random fill: 4 + 6 = 10 total, all of them.
    assert len(sample.results) == 10
    assert len(sample.buckets["most_figures"]) == 4
    assert len(sample.buckets["random_native"]) == 6
    # every native appears exactly once across the two buckets.
    combined = set(sample.buckets["most_figures"]) | set(sample.buckets["random_native"])
    assert combined == {r.path for r in natives}


def test_select_canary_random_fill_is_deterministic_for_a_given_seed() -> None:
    natives = [
        _canary_result(f"n-{i}.pdf", kind="pdf", route="native", pages=100) for i in range(30)
    ]
    report = preflight.PreflightReport(tuple(natives), {})

    first = preflight.select_canary(report, n=20, seed=7)
    second = preflight.select_canary(report, n=20, seed=7)

    assert first.buckets["random_native"] == second.buckets["random_native"]


def test_select_canary_never_exceeds_n_total() -> None:
    natives = [
        _canary_result(f"n-{i}.pdf", kind="pdf", route="native", pages=100) for i in range(50)
    ]
    report = preflight.PreflightReport(tuple(natives), {})

    sample = preflight.select_canary(report, n=20)

    assert len(sample.results) == 20


def test_select_canary_excludes_rejected_and_duplicate_files() -> None:
    results = [
        _canary_result("ok.pdf", kind="pdf", route="native", pages=100, verdict="ok"),
        _canary_result("bad.pdf", kind="pdf", route=None, pages=None, verdict="reject"),
        _canary_result("dup.pdf", kind="pdf", route=None, pages=None, verdict="duplicate_of=abc"),
    ]
    report = preflight.PreflightReport(tuple(results), {})

    sample = preflight.select_canary(report, n=20)

    assert [r.path for r in sample.results] == ["ok.pdf"]


def test_select_canary_handles_a_batch_smaller_than_every_category() -> None:
    """A tiny batch (fewer files than any single category's quota) must not
    error -- every category just takes what's there."""
    results = [
        _canary_result("giant.pdf", kind="pdf", route="giant", pages=1200),
        _canary_result("scan.pdf", kind="pdf", route="ocr", pages=50),
    ]
    report = preflight.PreflightReport(tuple(results), {})

    sample = preflight.select_canary(report, n=20)

    assert len(sample.results) == 2
    assert sample.buckets["giants"] == ("giant.pdf",)
    assert sample.buckets["scans"] == ("scan.pdf",)


def test_select_canary_empty_report_returns_empty_sample() -> None:
    report = preflight.PreflightReport((), {})

    sample = preflight.select_canary(report, n=20)

    assert sample.results == ()
    assert all(bucket == () for bucket in sample.buckets.values())


def test_select_canary_preserves_report_order_in_the_flat_result() -> None:
    """`sample.results` follows the source report's order, not bucket-pick
    order -- a caller writing canary.txt gets a stable, readable list."""
    results = [
        _canary_result("z-native.pdf", kind="pdf", route="native", pages=10),
        _canary_result("a-giant.pdf", kind="pdf", route="giant", pages=1200),
    ]
    report = preflight.PreflightReport(tuple(results), {})

    sample = preflight.select_canary(report, n=20)

    assert [r.path for r in sample.results] == ["z-native.pdf", "a-giant.pdf"]


# ---------------------------------------------------------------------------
# CLI: _write_canary_txt (vo_rag.cli.preflight) -- same "no separate CLI test
# file for this task" precedent as _collect_preflight_paths above.
# ---------------------------------------------------------------------------


def test_write_canary_txt_groups_by_bucket_and_includes_sha_and_route(tmp_path: Path) -> None:
    from vo_rag.cli.preflight import _write_canary_txt

    giant = _canary_result("giant.pdf", kind="pdf", route="giant", pages=1200)
    scan = _canary_result("scan.pdf", kind="pdf", route="ocr", pages=50)
    report = preflight.PreflightReport((giant, scan), {})
    sample = preflight.select_canary(report, n=20)

    out = tmp_path / "canary.txt"
    _write_canary_txt(out, sample)

    text = out.read_text(encoding="utf-8")
    assert "# giants" in text
    assert "# scans" in text
    assert "giant.pdf\tsha-giant.pdf\tgiant" in text
    assert "scan.pdf\tsha-scan.pdf\tocr" in text


def test_write_canary_txt_says_which_buckets_are_proxies(tmp_path: Path) -> None:
    """Preflight never counts images or figures, so the two
    buckets that sound like counts say what they really are."""
    from vo_rag.cli.preflight import _write_canary_txt

    image_epub = _canary_result("scan.epub", kind="epub", route="image_epub_ocr")
    native = _canary_result("long.pdf", kind="pdf", route="native", pages=900)
    sample = preflight.select_canary(preflight.PreflightReport((image_epub, native), {}), n=20)

    out = tmp_path / "canary.txt"
    _write_canary_txt(out, sample)

    lines = out.read_text(encoding="utf-8").splitlines()
    image_note = lines[lines.index("# image_heavy_epubs") + 1]
    figures_note = lines[lines.index("# most_figures") + 1]
    assert image_note.startswith("# note:") and "image-only route flag" in image_note
    assert figures_note.startswith("# note:") and "page count" in figures_note


def test_write_canary_txt_empty_sample_writes_empty_file(tmp_path: Path) -> None:
    from vo_rag.cli.preflight import _write_canary_txt

    report = preflight.PreflightReport((), {})
    sample = preflight.select_canary(report, n=20)

    out = tmp_path / "canary.txt"
    _write_canary_txt(out, sample)

    assert out.read_text(encoding="utf-8") == ""
