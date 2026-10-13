"""EPUBs whose chapter items declare ``media-type="text/html"`` instead of
the EPUB-required ``application/xhtml+xml`` (VOR-1427 task 3b).

Evidence: on EPUBs declaring this media type, ebooklib doesn't treat
``text/html`` spine items as documents and pandoc 3.1.3 skips them, so
``EpubParser._parse_epub_sync`` converted only the cover -- 0 real chunks.
``_normalize_epub`` rewrites the media type in a scratch copy *before*
either ``epub.read_epub`` or ``pypandoc.convert_file`` ever sees the file.

``_normalize_epub`` itself is pure ``zipfile`` + regex -- no ebooklib or
pypandoc needed -- so ``test_normalize_*`` below run unconditionally, even
in the locked CI env (``uv run --locked --extra dev --extra llamaindex``),
which lacks both. Only the full end-to-end parse test
(``test_text_html_chapters_are_parsed``) needs ebooklib + pypandoc + the
``pandoc`` binary, so it's guarded with ``pytest.importorskip`` and skips
there, matching the established pattern in ``test_epub_parser.py`` /
``test_epub_image_only.py``.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import zipfile
from pathlib import Path

import pytest

from tests.fixtures.epub.build_text_html_manifest_epub import _build_epub
from vo_rag.core.parsing.parsers.epub import (
    _epub_needs_normalization,
    _has_zip64_eocd,
    _normalize_epub,
    count_text_html_manifest_items,
)
from vo_rag.core.storage.figures import FigureStore

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "epub" / "text_html_manifest.epub"


# ---------------------------------------------------------------------------
# _normalize_epub: pure zipfile/regex, no ebooklib/pypandoc -- runs in the
# locked CI env unconditionally.
# ---------------------------------------------------------------------------


def test_normalize_rewrites_text_html_manifest_items(tmp_path: Path) -> None:
    out = _normalize_epub(FIXTURE, tmp_path)
    assert out != FIXTURE
    with zipfile.ZipFile(out) as z:
        opf_names = [n for n in z.namelist() if n.endswith(".opf")]
        assert opf_names
        for name in opf_names:
            opf_bytes = z.read(name)
            assert b'media-type="text/html"' not in opf_bytes
            assert opf_bytes.count(b'media-type="application/xhtml+xml"') >= 3


def test_normalize_is_noop_for_valid_epub(tmp_path: Path) -> None:
    valid = _build_epub(tmp_path / "ok.epub", media_type="application/xhtml+xml")
    assert _normalize_epub(valid, tmp_path) == valid


def test_normalized_copy_keeps_mimetype_first_and_stored(tmp_path: Path) -> None:
    out = _normalize_epub(FIXTURE, tmp_path)
    with zipfile.ZipFile(out) as z:
        first = z.infolist()[0]
        assert first.filename == "mimetype"
        assert first.compress_type == zipfile.ZIP_STORED


def test_normalized_copy_is_byte_identical_outside_the_opf(tmp_path: Path) -> None:
    out = _normalize_epub(FIXTURE, tmp_path)
    with zipfile.ZipFile(FIXTURE) as zin, zipfile.ZipFile(out) as zout:
        out_by_name = {info.filename: zout.read(info.filename) for info in zout.infolist()}
        opf_names = {n for n in zin.namelist() if n.endswith(".opf")}
        for info in zin.infolist():
            assert info.filename in out_by_name
            if info.filename in opf_names:
                continue
            assert out_by_name[info.filename] == zin.read(info.filename)


# ---------------------------------------------------------------------------
# Quote-agnostic + case-insensitive detection and rewrite. Single-quoted
# `media-type='text/html'` is legal XML; mixed-case (`Text/HTML`) is
# real-world XML producers being sloppy. Both must be
# normalized, and the rewrite must preserve the original quote character.
# ---------------------------------------------------------------------------


def test_normalize_rewrites_single_quoted_manifest_items(tmp_path: Path) -> None:
    single_quoted = _build_epub(tmp_path / "single_quoted.epub", media_type="text/html", quote="'")

    out = _normalize_epub(single_quoted, tmp_path)

    assert out != single_quoted
    with zipfile.ZipFile(out) as z:
        opf_names = [n for n in z.namelist() if n.endswith(".opf")]
        assert opf_names
        for name in opf_names:
            opf_bytes = z.read(name)
            assert b"text/html" not in opf_bytes.lower()
            assert opf_bytes.count(b"media-type='application/xhtml+xml'") >= 3


def test_normalize_rewrites_mixed_case_manifest_items(tmp_path: Path) -> None:
    mixed_case = _build_epub(tmp_path / "mixed_case.epub", media_type="Text/HTML")

    out = _normalize_epub(mixed_case, tmp_path)

    assert out != mixed_case
    with zipfile.ZipFile(out) as z:
        opf_names = [n for n in z.namelist() if n.endswith(".opf")]
        assert opf_names
        for name in opf_names:
            opf_bytes = z.read(name)
            assert b"text/html" not in opf_bytes.lower()
            assert opf_bytes.count(b'media-type="application/xhtml+xml"') >= 3


def test_needs_normalization_true_for_single_quoted_and_mixed_case(tmp_path: Path) -> None:
    single_quoted = _build_epub(tmp_path / "single_quoted.epub", media_type="text/html", quote="'")
    mixed_case = _build_epub(tmp_path / "mixed_case.epub", media_type="Text/HTML")
    valid = _build_epub(tmp_path / "ok.epub", media_type="application/xhtml+xml")

    assert _epub_needs_normalization(single_quoted) is True
    assert _epub_needs_normalization(mixed_case) is True
    assert _epub_needs_normalization(valid) is False


def test_count_text_html_manifest_items_is_quote_and_case_agnostic(tmp_path: Path) -> None:
    single_quoted = _build_epub(tmp_path / "single_quoted.epub", media_type="text/html", quote="'")
    mixed_case = _build_epub(tmp_path / "mixed_case.epub", media_type="Text/HTML")
    valid = _build_epub(tmp_path / "ok.epub", media_type="application/xhtml+xml")

    assert count_text_html_manifest_items(FIXTURE.read_bytes()) == 3
    assert count_text_html_manifest_items(single_quoted.read_bytes()) == 3
    assert count_text_html_manifest_items(mixed_case.read_bytes()) == 3
    assert count_text_html_manifest_items(valid.read_bytes()) == 0


# ---------------------------------------------------------------------------
# Full end-to-end parse: needs ebooklib + pypandoc + the pandoc binary.
# ---------------------------------------------------------------------------


def test_text_html_chapters_are_parsed(tmp_path: Path) -> None:
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")

    from vo_rag.core.parsing.parsers.epub import EpubParser

    doc = asyncio.run(EpubParser(FigureStore(tmp_path)).parse(str(FIXTURE)))
    text = " ".join(e.text or "" for e in doc.elements)
    assert "Chapter one body" in text and "Chapter three body" in text
    assert doc.total_pages == 3  # one synthetic page per chapter heading


def test_normal_epub_parses_without_creating_a_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal (unaffected) EPUB must parse without
    ``_parse_epub_sync`` ever creating a ``TemporaryDirectory``."""
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")

    from vo_rag.core.parsing.parsers import epub as epub_module

    normal = _build_epub(tmp_path / "normal.epub", media_type="application/xhtml+xml")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("TemporaryDirectory must not be created for a normal EPUB")

    monkeypatch.setattr(epub_module.tempfile, "TemporaryDirectory", _boom)  # type: ignore[attr-defined]

    doc = asyncio.run(epub_module.EpubParser(FigureStore(tmp_path)).parse(str(normal)))

    text = " ".join(e.text or "" for e in doc.elements)
    assert "Chapter one body" in text and "Chapter three body" in text


# ---------------------------------------------------------------------------
# VOR-1431: manifest items whose file is missing from the archive
# ---------------------------------------------------------------------------


def _without_member(src: Path, dest: Path, missing: str) -> Path:
    """Copy ``src`` minus ``missing`` (the OPF still lists it)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dest, "w") as zout:
        for info in zin.infolist():
            if info.filename != missing:
                zout.writestr(info, zin.read(info.filename))
    return dest


def test_missing_nav_item_triggers_normalization_and_is_dropped(tmp_path: Path) -> None:
    valid = _build_epub(tmp_path / "ok.epub")
    broken = _without_member(valid, tmp_path / "no-nav.epub", "OEBPS/nav.xhtml")
    assert not _epub_needs_normalization(valid)
    assert _epub_needs_normalization(broken)

    out = _normalize_epub(broken, tmp_path)

    assert out != broken
    with zipfile.ZipFile(out) as z:
        opf = z.read("OEBPS/content.opf")
        assert b"nav.xhtml" not in opf
        assert opf.count(b"<item ") == 3  # the three chapters survive
        assert opf.count(b"<itemref ") == 3
        assert z.infolist()[0].filename == "mimetype"
    assert not _epub_needs_normalization(out)


def test_missing_spine_chapter_is_dropped_with_its_itemref(tmp_path: Path) -> None:
    valid = _build_epub(tmp_path / "ok.epub")
    broken = _without_member(valid, tmp_path / "no-chap1.epub", "OEBPS/chap_1.xhtml")

    out = _normalize_epub(broken, tmp_path)

    with zipfile.ZipFile(out) as z:
        opf = z.read("OEBPS/content.opf")
        assert b'href="chap_1.xhtml"' not in opf
        assert b'idref="chap1"' not in opf
        assert b'idref="chap0"' in opf and b'idref="chap2"' in opf


def test_missing_manifest_item_book_parses(tmp_path: Path) -> None:
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")
    from vo_rag.core.parsing.parsers.epub import EpubParser

    valid = _build_epub(tmp_path / "ok.epub")
    broken = _without_member(valid, tmp_path / "no-nav.epub", "OEBPS/nav.xhtml")
    parsed = asyncio.run(EpubParser(FigureStore(tmp_path / "figs")).parse(str(broken)))
    assert parsed.elements


# ---------------------------------------------------------------------------
# VOR-1432: archives that end in a ZIP64 end-of-central-directory record
# ---------------------------------------------------------------------------


def _with_zip64_eocd(src: Path, dest: Path) -> Path:
    """Copy ``src`` with its EOCD rewritten the way the real book has it: a
    ZIP64 EOCD record and locator, then a classic EOCD whose fields are all
    0xFFFF / 0xFFFFFFFF."""
    data = src.read_bytes()
    eocd = data.rfind(b"PK\x05\x06")
    _, _, _, n_disk, n_total, cd_size, cd_offset, comment_len = struct.unpack(
        "<4s4H2LH", data[eocd : eocd + 22]
    )
    zip64_eocd = struct.pack(
        "<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, n_disk, n_total, cd_size, cd_offset
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, eocd, 1)
    classic = struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, comment_len
    )
    dest.write_bytes(data[:eocd] + zip64_eocd + locator + classic + data[eocd + 22 :])
    return dest


def test_zip64_eocd_triggers_normalization_and_is_rewritten_plain(tmp_path: Path) -> None:
    valid = _build_epub(tmp_path / "ok.epub", media_type="application/xhtml+xml")
    zip64 = _with_zip64_eocd(valid, tmp_path / "zip64.epub")
    with zipfile.ZipFile(zip64) as z:
        assert z.testzip() is None  # still a valid zip, just ZIP64-terminated
    assert not _has_zip64_eocd(valid)
    assert _has_zip64_eocd(zip64)
    assert _epub_needs_normalization(zip64)

    out = _normalize_epub(zip64, tmp_path)

    assert out != zip64
    assert not _has_zip64_eocd(out)
    assert not _epub_needs_normalization(out)
    with zipfile.ZipFile(zip64) as zin, zipfile.ZipFile(out) as zout:
        assert zout.infolist()[0].filename == "mimetype"
        assert zout.infolist()[0].compress_type == zipfile.ZIP_STORED
        for info in zin.infolist():
            assert zout.read(info.filename) == zin.read(info.filename)


def test_zip64_eocd_book_parses(tmp_path: Path) -> None:
    pytest.importorskip("ebooklib")
    pytest.importorskip("pypandoc")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")
    from vo_rag.core.parsing.parsers.epub import EpubParser

    valid = _build_epub(tmp_path / "ok.epub", media_type="application/xhtml+xml")
    zip64 = _with_zip64_eocd(valid, tmp_path / "zip64.epub")
    doc = asyncio.run(EpubParser(FigureStore(tmp_path / "figs")).parse(str(zip64)))
    text = " ".join(e.text or "" for e in doc.elements)
    assert "Chapter one body" in text and "Chapter three body" in text


# ---------------------------------------------------------------------------
# VOR-1434: manifest hrefs that escape the OPF directory (OPF at
# OEBPS/content.opf, href="../jacket.xhtml" -> pandoc exit 64
# "No entry on path: OEBPS/../jacket.xhtml").
# ---------------------------------------------------------------------------

_JACKET_WITH_LINKS = (
    b'<?xml version="1.0" encoding="utf-8"?>\n'
    b'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Jacket</title>'
    b'<link rel="stylesheet" href="OEBPS/style.css"/></head>'
    b"<body><h1>Jacket</h1><p>Jacket body text, enough to be a real chapter.</p>"
    b'<img src="OEBPS/images/cover%20art.png" alt="cover"/>'
    b"<a href='OEBPS/chap_0.xhtml#start'>go</a>"
    b'<a href="#top">top</a><a href="https://example.com/x">web</a>'
    b"</body></html>\n"
)


def _rewrite_member(src: Path, dest: Path, name: str, data: bytes | None) -> Path:
    """Copy ``src`` to ``dest`` with member ``name`` replaced (or added)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dest, "w") as zout:
        for info in zin.infolist():
            if info.filename == name:
                continue
            zout.writestr(info, zin.read(info.filename))
        if data is not None:
            zout.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    return dest


def _escaping(tmp_path: Path) -> Path:
    from tests.fixtures.epub.build_preflight_epub_fixtures import build_escaping_href_epub

    return build_escaping_href_epub(tmp_path / "suckers_progress.epub")


def _opf(path: Path) -> bytes:
    with zipfile.ZipFile(path) as z:
        return z.read("OEBPS/content.opf")


def test_escaping_hrefs_flags_parent_traversal() -> None:
    from vo_rag.core.parsing.parsers.epub import _escaping_hrefs

    opf_bytes = (
        b"<manifest>"
        b'<item id="chap0" href="chap_0.xhtml" media-type="application/xhtml+xml"/>'
        b'<item id="jacket" href="../jacket.xhtml" media-type="application/xhtml+xml"/>'
        b'<item id="img0" href="images/../images/a.png" media-type="image/png"/>'
        b"</manifest>"
    )
    assert _escaping_hrefs("OEBPS/content.opf", opf_bytes) == ["../jacket.xhtml"]
    root_opf = b'<manifest><item id="a" href="text/../a.xhtml" media-type="x"/></manifest>'
    assert _escaping_hrefs("content.opf", root_opf) == []


def test_escaping_href_triggers_normalization(tmp_path: Path) -> None:
    assert _epub_needs_normalization(_escaping(tmp_path))


def test_escaping_href_member_is_moved_into_the_opf_dir(tmp_path: Path) -> None:
    src = _escaping(tmp_path)
    out = _normalize_epub(src, tmp_path)

    assert out != src
    assert not _epub_needs_normalization(out)
    opf = _opf(out)
    assert b"../jacket.xhtml" not in opf
    assert b'href="jacket.xhtml"' in opf
    assert b'<itemref idref="jacket"/>' in opf
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out) as zout:
        assert zout.infolist()[0].filename == "mimetype"
        assert zout.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert zout.read("OEBPS/jacket.xhtml") == zin.read("jacket.xhtml")


def test_moved_member_gets_a_collision_free_name(tmp_path: Path) -> None:
    src = _rewrite_member(
        _escaping(tmp_path), tmp_path / "collide.epub", "OEBPS/jacket.xhtml", b"<html/>"
    )
    out = _normalize_epub(src, tmp_path)
    opf = _opf(out)
    assert b'href="jacket-1.xhtml"' in opf
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out) as zout:
        assert zout.read("OEBPS/jacket.xhtml") == b"<html/>"  # untouched
        assert zout.read("OEBPS/jacket-1.xhtml") == zin.read("jacket.xhtml")


def test_relative_links_inside_the_moved_document_are_rewritten(tmp_path: Path) -> None:
    src = _rewrite_member(
        _escaping(tmp_path), tmp_path / "links.epub", "jacket.xhtml", _JACKET_WITH_LINKS
    )
    out = _normalize_epub(src, tmp_path)
    with zipfile.ZipFile(out) as z:
        moved = z.read("OEBPS/jacket.xhtml")
    assert b'href="style.css"' in moved
    assert b'src="images/cover%20art.png"' in moved
    assert b"href='chap_0.xhtml#start'" in moved
    assert b'href="#top"' in moved
    assert b'href="https://example.com/x"' in moved


def test_rebased_links_keep_their_original_encoding(tmp_path: Path) -> None:
    jacket = (
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Jacket</h1>'
        b'<a href="OEBPS/a&amp;b.xhtml?x=1#f">amp</a>'
        b'<img src="OEBPS/img/%E2%82%AC%20sign.png?v=2"/>'
        b"</body></html>"
    )
    src = _rewrite_member(_escaping(tmp_path), tmp_path / "enc.epub", "jacket.xhtml", jacket)
    with zipfile.ZipFile(_normalize_epub(src, tmp_path)) as z:
        moved = z.read("OEBPS/jacket.xhtml")
    assert b'href="a&amp;b.xhtml?x=1#f"' in moved
    assert b'src="img/%E2%82%AC%20sign.png?v=2"' in moved


def test_links_elsewhere_in_the_book_follow_the_moved_document(tmp_path: Path) -> None:
    nav = (
        b'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        b'<body><nav epub:type="toc"><ol><li><a href="chap_0.xhtml">One</a></li>'
        b'<li><a href="../jacket.xhtml#j">Jacket</a></li></ol></nav></body></html>'
    )
    ncx = b'<ncx><navMap><navPoint><content src="../jacket.xhtml"/></navPoint></navMap></ncx>'
    chapter = b'<html><body><p>See <a href="../jacket.xhtml">the jacket</a>.</p></body></html>'
    src = _rewrite_member(_escaping(tmp_path), tmp_path / "a.epub", "OEBPS/nav.xhtml", nav)
    src = _rewrite_member(src, tmp_path / "b.epub", "OEBPS/toc.ncx", ncx)
    src = _rewrite_member(src, tmp_path / "c.epub", "OEBPS/chap_0.xhtml", chapter)
    src = _rewrite_member(src, tmp_path / "d.epub", "OEBPS/jacket.xhtml", b"<html/>")  # collide

    with zipfile.ZipFile(_normalize_epub(src, tmp_path)) as z:
        assert b'href="chap_0.xhtml"' in z.read("OEBPS/nav.xhtml")
        assert b'href="jacket-1.xhtml#j"' in z.read("OEBPS/nav.xhtml")
        assert b'src="jacket-1.xhtml"' in z.read("OEBPS/toc.ncx")
        assert b'href="jacket-1.xhtml"' in z.read("OEBPS/chap_0.xhtml")
        assert z.read("OEBPS/jacket.xhtml") == b"<html/>"


def _png() -> bytes:
    import fitz  # type: ignore[import-untyped, unused-ignore]

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 160, 120), False)
    pix.set_rect(fitz.IRect(0, 0, 80, 120), (200, 30, 30))
    pix.set_rect(fitz.IRect(80, 0, 160, 120), (30, 30, 200))
    return bytes(pix.tobytes("png"))


def test_escaping_href_book_converts_with_pandoc(tmp_path: Path) -> None:
    pypandoc = pytest.importorskip("pypandoc")
    pytest.importorskip("ebooklib")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")
    from vo_rag.core.parsing.contract import ElementType
    from vo_rag.core.parsing.parsers.epub import EpubParser

    jacket = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Jacket</title></head>'
        b"<body><h1>Jacket</h1><p>Jacket body text, enough to be a real chapter.</p>"
        b'<img src="OEBPS/images/cover.png" alt="the cover"/></body></html>\n'
    )
    base = _rewrite_member(_escaping(tmp_path), tmp_path / "j.epub", "jacket.xhtml", jacket)
    base = _rewrite_member(base, tmp_path / "i.epub", "OEBPS/images/cover.png", _png())
    opf = _opf(base).replace(
        b"</manifest>",
        b'<item id="cover" href="images/cover.png" media-type="image/png"/></manifest>',
    )
    src = _rewrite_member(base, tmp_path / "suckers_img.epub", "OEBPS/content.opf", opf)

    out = _normalize_epub(src, tmp_path)
    markdown = pypandoc.convert_file(str(out), "gfm", format="epub")
    assert "Jacket body text" in markdown

    doc = asyncio.run(EpubParser(FigureStore(tmp_path / "figs")).parse(str(src)))
    text = " ".join(e.text or "" for e in doc.elements)
    assert "Chapter one body" in text and "Jacket body text" in text
    # The moved document's <img> is still extracted and placed on its IMAGE
    # element (VOR-1435), not lost to the relocation.
    assert len(doc.images) == 1
    (image,) = doc.images
    assert doc.elements[image.element_index].element_type == ElementType.IMAGE
    assert "cover.png" in doc.elements[image.element_index].text


# ---------------------------------------------------------------------------
# Zip-bomb guard (VOR-1434): every ingest reader of the
# archive refuses a bomb before decompressing a member.
# ---------------------------------------------------------------------------


def test_needs_normalization_rejects_a_zip_bomb(tmp_path: Path) -> None:
    from tests.fixtures.epub.build_preflight_epub_fixtures import build_zip_bomb_epub
    from vo_rag.core.parsing.parsers.epub import ZipBombError

    bomb = build_zip_bomb_epub(tmp_path / "bomb.epub")
    with pytest.raises(ZipBombError, match="zip_bomb"):
        _epub_needs_normalization(bomb)


def test_normalize_rejects_a_zip_bomb_even_when_zip64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ZIP64 used to short-circuit ``_epub_needs_normalization`` before the
    archive was opened; the guard must still fire for ``_normalize_epub``."""
    from tests.fixtures.epub.build_preflight_epub_fixtures import build_zip64_epub
    from vo_rag.core.parsing.parsers import epub as epub_module

    zip64 = build_zip64_epub(tmp_path / "zip64.epub")
    monkeypatch.setattr(epub_module, "ZIP_BOMB_MAX_TOTAL_BYTES", 1024)
    with pytest.raises(epub_module.ZipBombError, match="zip_bomb"):
        _normalize_epub(zip64, tmp_path)


def test_epub_parser_fails_a_zip_bomb_book_cleanly(tmp_path: Path) -> None:
    """The ingest path fails the book with a clear ``zip_bomb`` error before
    ebooklib or pandoc decompress anything."""
    from tests.fixtures.epub.build_preflight_epub_fixtures import build_zip_bomb_epub
    from vo_rag.core.parsing.parsers.epub import EpubParser, ZipBombError

    bomb = build_zip_bomb_epub(tmp_path / "bomb.epub")
    parser = EpubParser(FigureStore(tmp_path / "figures"))
    with pytest.raises(ZipBombError, match="zip_bomb"):
        parser._parse_epub_sync(str(bomb))
