"""Tiny synthetic EPUB fixtures for ``vo-rag preflight`` checks that
have no existing builder (VOR-1434):

- a ZIP64-terminated EPUB (VOR-1432, ZIP64 shape)
- a manifest href that resolves outside its OPF directory, e.g.
  ``OEBPS/../jacket.xhtml`` (the escaping-href shape)
- a manifest item whose file is missing from the archive (VOR-1431)

Pure ``zipfile`` + string templates, reusing
``build_text_html_manifest_epub._build_epub`` for the base EPUB -- no
ebooklib/pypandoc, so these run unconditionally in the locked CI env.
Regenerate nothing: these are built fresh by every test via ``tmp_path``.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from tests.fixtures.epub.build_text_html_manifest_epub import _build_epub

_ESCAPING_HREF_OPF = b"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="pub-id">vorag-test-escaping-href</dc:identifier>
    <dc:title>Escaping Href Fixture</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="chap0" href="chap_0.xhtml" media-type="application/xhtml+xml"/>
    <item id="jacket" href="../jacket.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chap0"/>
    <itemref idref="jacket"/>
  </spine>
</package>
"""

_ESCAPING_HREF_NAV = b"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>Nav</title></head>
  <body>
    <nav epub:type="toc" id="toc"><ol><li><a href="chap_0.xhtml">Chapter one</a></li></ol></nav>
  </body>
</html>
"""


def _chapter_xhtml(title: str) -> bytes:
    return (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml">\n'
        f"  <head><title>{title}</title></head>\n"
        f"  <body><h1>{title}</h1><p>{title} body text, enough to be a real chapter.</p></body>\n"
        f"</html>\n"
    ).encode()


def build_escaping_href_epub(path: Path) -> Path:
    """A manifest item whose href, ``../jacket.xhtml``, resolves to
    ``jacket.xhtml`` at the zip root -- outside ``OEBPS/`` (VOR-1434
    incident table)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"), b"application/epub+zip", compress_type=zipfile.ZIP_STORED
        )
        zf.writestr(
            "META-INF/container.xml",
            b"""<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""",
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zf.writestr("OEBPS/content.opf", _ESCAPING_HREF_OPF, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/nav.xhtml", _ESCAPING_HREF_NAV, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr(
            "OEBPS/chap_0.xhtml", _chapter_xhtml("Chapter one"), compress_type=zipfile.ZIP_DEFLATED
        )
        zf.writestr("jacket.xhtml", _chapter_xhtml("Jacket"), compress_type=zipfile.ZIP_DEFLATED)
    return path


def _with_zip64_eocd(src: Path, dest: Path) -> Path:
    """Rewrite ``src``'s EOCD as a ZIP64 end record + locator, then a classic
    EOCD with 0xFFFF/0xFFFFFFFF fields (the real ZIP64 shape,
    VOR-1432). Mirrors ``_with_zip64_eocd`` in
    ``tests/core/parsing/test_epub_manifest_normalization.py`` (that module
    owns the production-facing test; this is preflight's own fixture copy,
    not a shared import, to keep the two test files independent)."""
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


def build_zip64_epub(path: Path) -> Path:
    """A valid EPUB rewritten to end in a ZIP64 EOCD record (VOR-1432)."""
    src = path.with_suffix(".src.epub")
    _build_epub(src, media_type="application/xhtml+xml")
    return _with_zip64_eocd(src, path)


def build_missing_manifest_item_epub(path: Path) -> Path:
    """A manifest item (``OEBPS/nav.xhtml``) declared in the OPF but absent
    from the archive (VOR-1431 shape)."""
    src = path.with_suffix(".src.epub")
    _build_epub(src, media_type="application/xhtml+xml")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(path, "w") as zout:
        for info in zin.infolist():
            if info.filename != "OEBPS/nav.xhtml":
                zout.writestr(info, zin.read(info.filename))
    return path


def build_corrupt_deflate_epub(path: Path) -> Path:
    """A valid EPUB whose first chapter's deflate stream is overwritten with
    junk (VOR-1434). The central directory and EOCD are
    intact, so ``zipfile.ZipFile`` opens it; ``testzip()`` then raises
    ``zlib.error`` (not ``BadZipFile``) while decompressing that member."""
    _build_epub(path, media_type="application/xhtml+xml")
    member = "OEBPS/chap_0.xhtml"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    data = bytearray(path.read_bytes())
    start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    for offset in range(start + 2, start + min(info.compress_size - 2, 42)):
        data[offset] = 0xFF
    path.write_bytes(bytes(data))
    return path


#: One member big enough for the ratio rule (over 100 MiB decompressed).
ZIP_BOMB_MEMBER_BYTES = 101 * 1024 * 1024


def build_zip_bomb_epub(path: Path, *, member_bytes: int = ZIP_BOMB_MEMBER_BYTES) -> Path:
    """A valid EPUB plus one ``member_bytes`` run of zeros, deflated to a
    ~1000:1 ratio (VOR-1434). Streamed through
    ``ZipFile.open(..., "w")`` so the test never holds it all in memory."""
    _build_epub(path, media_type="application/xhtml+xml")
    chunk = b"\x00" * (1024 * 1024)
    with zipfile.ZipFile(path, "a") as archive:
        info = zipfile.ZipInfo("OEBPS/images/bomb.bin")
        info.compress_type = zipfile.ZIP_DEFLATED
        with archive.open(info, "w", force_zip64=member_bytes >= 0xFFFFFFFF) as out:
            remaining = member_bytes
            while remaining > 0:
                piece = chunk[: min(len(chunk), remaining)]
                out.write(piece)
                remaining -= len(piece)
    return path
