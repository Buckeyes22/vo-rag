"""Build the tiny synthetic image-only EPUB fixture used by
``tests/core/parsing/test_epub_image_only.py`` (VOR-1427).

25 spine pages, each a single small generated PNG and no real text -- a
stand-in for a page-scan EPUB. No real book content. Regenerate with::

    uv run python tests/fixtures/epub/build_image_only_epub.py
"""

from __future__ import annotations

import io
from pathlib import Path

PAGE_COUNT = 25
FIXTURE_PATH = Path(__file__).with_name("image_only_25_pages.epub")


def _tiny_png(index: int) -> bytes:
    from PIL import Image

    # A distinct-but-trivial 8x8 image per page so each page's picture item
    # is a genuinely different file (helps catch ordering bugs).
    image = Image.new("RGB", (8, 8), color=((index * 7) % 256, (index * 13) % 256, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build(path: Path = FIXTURE_PATH, *, n_pages: int = PAGE_COUNT) -> Path:
    from ebooklib import epub  # type: ignore[import-not-found]

    book = epub.EpubBook()
    book.set_identifier("vorag-test-image-only-epub")
    book.set_title("Image Only Test Fixture")
    book.set_language("en")
    book.add_author("vo-rag test fixtures")

    spine: list[object] = ["nav"]
    chapters = []
    for i in range(n_pages):
        img_name = f"images/page{i:03d}.png"
        img_item = epub.EpubItem(
            uid=f"img{i}",
            file_name=img_name,
            media_type="image/png",
            content=_tiny_png(i),
        )
        book.add_item(img_item)

        chapter = epub.EpubHtml(
            title=f"Page {i + 1}",
            file_name=f"page_{i:03d}.xhtml",
            lang="en",
        )
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


if __name__ == "__main__":
    out = build()
    print(f"wrote {out} ({out.stat().st_size} bytes)")
