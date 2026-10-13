"""Image-only EPUB with a few pages of real front-matter text (VOR-1434)
-- the incident-table shape: 401
scanned-page images plus 8,086 characters of real prose (title page,
foreword, table of contents). ``is_image_only`` must not be rescued by that
front matter (VOR-1427: one paragraph of real text must never save 400
scanned pages), so preflight's ``image_epub_ocr`` routing has to hold even
with real prose at the front of the spine.

Extends ``build_image_only_epub.py``'s pattern (ebooklib, one tiny PNG per
page) with a front-matter chapter of real paragraph text ahead of the scan
pages. Needs ebooklib -- guarded with ``pytest.importorskip`` at the call
site, same as ``build_image_only_epub.py``.
"""

from __future__ import annotations

import io
from pathlib import Path

PAGE_COUNT = 25
#: ~280 chars of real prose per paragraph; 30 paragraphs is comfortably over
#: 8,000 characters, matching that incident's front-matter size.
_FRONT_MATTER_PARAGRAPH = (
    "This book collects years of hard-won lessons on reading opponents, "
    "sizing bets, and thinking in ranges rather than hands. It is written "
    "for players who already know the rules and want to close the gap "
    "between a fun weekend game and a disciplined, profitable one over "
    "the long run of many thousands of hands played under pressure."
)


def _tiny_png(index: int) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (8, 8), color=((index * 7) % 256, (index * 13) % 256, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build(
    path: Path,
    *,
    n_pages: int = PAGE_COUNT,
    front_matter_paragraphs: int = 30,
) -> Path:
    from ebooklib import epub  # type: ignore[import-not-found]

    book = epub.EpubBook()
    book.set_identifier("vorag-test-image-only-front-matter")
    book.set_title("Image Only With Front Matter Fixture")
    book.set_language("en")
    book.add_author("vo-rag test fixtures")

    spine: list[object] = ["nav"]
    chapters = []

    front_matter = epub.EpubHtml(title="Front Matter", file_name="front_matter.xhtml", lang="en")
    paragraphs = "".join(
        f"<p>{_FRONT_MATTER_PARAGRAPH}</p>" for _ in range(front_matter_paragraphs)
    )
    front_matter.content = f"<html><body>{paragraphs}</body></html>"
    book.add_item(front_matter)
    chapters.append(front_matter)
    spine.append(front_matter)

    for i in range(n_pages):
        img_name = f"images/page{i:03d}.png"
        img_item = epub.EpubItem(
            uid=f"img{i}",
            file_name=img_name,
            media_type="image/png",
            content=_tiny_png(i),
        )
        book.add_item(img_item)

        chapter = epub.EpubHtml(title=f"Page {i + 1}", file_name=f"page_{i:03d}.xhtml", lang="en")
        chapter.content = (
            f'<html><body><img src="{img_name}" alt="scan page {i + 1}"/></body></html>'
        )
        book.add_item(chapter)
        chapters.append(chapter)
        spine.append(chapter)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine

    epub.write_epub(str(path), book)
    return path
