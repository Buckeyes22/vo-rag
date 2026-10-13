"""Build the tiny synthetic EPUB fixture for the ``text/html`` chapter
manifest bug (VOR-1427).

Some real-world books declare their chapter items
``media-type="text/html"`` in ``content.opf`` instead of the EPUB-required
``application/xhtml+xml``; ebooklib/pandoc then treat those spine items as
non-documents and skip them, so ``EpubParser`` only ever recovers the cover.

``_build_epub`` writes a minimal, valid EPUB 3 by hand with the stdlib
``zipfile`` module only -- no ebooklib/pypandoc -- so it (and any test that
calls it) runs in the locked CI env with no extra deps. The committed
fixture ``text_html_manifest.epub`` is produced once by
``_build_epub(..., media_type="text/html")``; regenerate with::

    uv run python tests/fixtures/epub/build_text_html_manifest_epub.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

FIXTURE_PATH = Path(__file__).with_name("text_html_manifest.epub")

_CONTAINER_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

_CHAPTER_TITLES = ["Chapter one", "Chapter two", "Chapter three"]


def _content_opf(media_type: str, n_chapters: int, *, quote: str = '"') -> bytes:
    manifest_items = "\n".join(
        f'    <item id="chap{i}" href="chap_{i}.xhtml" media-type={quote}{media_type}{quote}/>'
        for i in range(n_chapters)
    )
    spine_items = "\n".join(f'    <itemref idref="chap{i}"/>' for i in range(n_chapters))
    return f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="pub-id">vorag-test-text-html-manifest</dc:identifier>
    <dc:title>Text Html Manifest Fixture</dc:title>
    <dc:language>en</dc:language>
    <dc:creator>vo-rag test fixtures</dc:creator>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
{manifest_items}
  </manifest>
  <spine>
{spine_items}
  </spine>
</package>
""".encode()


def _nav_xhtml(n_chapters: int) -> bytes:
    entries = "\n".join(
        f'      <li><a href="chap_{i}.xhtml">{_CHAPTER_TITLES[i]}</a></li>'
        for i in range(n_chapters)
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>Nav</title></head>
  <body>
    <nav epub:type="toc" id="toc">
      <ol>
{entries}
      </ol>
    </nav>
  </body>
</html>
""".encode()


def _chapter_xhtml(index: int) -> bytes:
    title = _CHAPTER_TITLES[index]
    body_words = " ".join(f"{title.split()[-1]}word{n}" for n in range(60))
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>{title}</title></head>
  <body>
    <h1>{title}</h1>
    <p>{title} body {body_words}.</p>
  </body>
</html>
""".encode()


def _build_epub(
    path: Path,
    *,
    media_type: str = "application/xhtml+xml",
    n_chapters: int = 3,
    quote: str = '"',
) -> Path:
    """Write a minimal, valid synthetic EPUB 3 to ``path`` and return it.

    ``n_chapters`` chapter items are declared with ``media_type`` in
    ``content.opf`` -- ``"text/html"`` (any case, e.g. ``"Text/HTML"``)
    reproduces the bug, the default ``"application/xhtml+xml"`` is a normal,
    valid EPUB. ``quote`` picks the attribute's quote character (``'"'`` or
    ``"'"`` -- both legal XML; covers single-quoted attributes). Pure
    ``zipfile`` + string templates only (no ebooklib), so this runs with no
    extra deps.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            b"application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", _CONTAINER_XML, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr(
            "OEBPS/content.opf",
            _content_opf(media_type, n_chapters, quote=quote),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zf.writestr("OEBPS/nav.xhtml", _nav_xhtml(n_chapters), compress_type=zipfile.ZIP_DEFLATED)
        for i in range(n_chapters):
            zf.writestr(
                f"OEBPS/chap_{i}.xhtml",
                _chapter_xhtml(i),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    return path


def build(path: Path = FIXTURE_PATH) -> Path:
    return _build_epub(path, media_type="text/html")


if __name__ == "__main__":
    out = build()
    print(f"wrote {out} ({out.stat().st_size} bytes)")
