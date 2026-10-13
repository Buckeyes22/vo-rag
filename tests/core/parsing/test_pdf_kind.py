"""``ParserRouter._pdf_kind`` on fitz-built PDFs (VOR-1427).

Production case: a scanned 24-page textbook -- 23 image-only scans and one
library collection stamp page with ~880 characters of text. The old probe
averaged characters over the whole sample, so that one page (880 / 12 sampled
pages ~= 73 chars/page >= 50) made the scan "native"; live docling then
failed the coverage gate on 1/24 pages. Native now needs text on a real share
of the spread-sampled pages, not a high average.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vo_rag.core.parsing.router as router_module
from tests.core.parsing._router_fakes import (
    FakeDocling,
    FakeEbookParser,
)
from vo_rag.core.parsing.router import ParserRouter

BODY = (
    "The boundary layer thickens along the plate as viscous stresses slow the\n"
    "fluid near the wall. The pressure gradient in the free stream controls\n"
    "whether the flow stays attached, and the Reynolds number sets the point\n"
    "where the laminar layer becomes turbulent over the surface of the body.\n"
)
# ~880 characters, like the library stamp page: English words, so it passes the
# dictionary check on its own.
STAMP = (
    "This volume belongs to the Northwind sample collection of the Example Library.\n"
    "The pages in this collection were scanned by volunteers at community branches and are\n"
    "shared for preservation and for study by the public.\n"
    "The Example Library is a non-profit project with the goal of preserving and\n"
    "keeping available a wide range of printed works, papers and other documents of\n"
    "our shared culture. We do not host any copies ourselves, and we only point to\n"
    "the copies that are already available from other places on the web.\n"
    "If you notice a problem with this copy, please report it on the project page\n"
    "so that other readers can reach a cleaner version of the same volume there.\n"
    "Thank you for your support of open study and learning for everyone,\n"
    "and please share this collection with anyone who may find it useful.\n"
    "Collection: Northwind. Source: scanned library copy. Pages: image only.\n"
)

# A page spec is "image", "blank", "stamp", "body" or any other literal text.
PageSpec = str


def _png() -> bytes:
    import fitz  # type: ignore[import-untyped]

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 51), False)
    pix.set_rect(pix.irect, (180, 170, 150))
    return bytes(pix.tobytes("png"))


def _build(path: Path, pages: list[PageSpec]) -> str:
    import fitz

    png = _png()
    doc = fitz.open()
    for spec in pages:
        page = doc.new_page(width=607, height=774)  # 2528x3224 px at ~300 dpi
        if spec == "image":
            page.insert_image(page.rect, stream=png)
        elif spec == "blank":
            pass
        else:
            text = {"stamp": STAMP, "body": BODY * 3}.get(spec, spec)
            page.insert_text((36, 48), text, fontsize=9)
    doc.save(str(path))
    doc.close()
    return str(path)


def _router() -> ParserRouter:
    return ParserRouter(
        docling=FakeDocling(),
        epub=FakeEbookParser("epub"),
    )


def _text_chars(path: str, index: int) -> int:
    import fitz

    with fitz.open(path) as pdf:
        return len(pdf[index].get_text("text").strip())


async def test_scan_with_one_stamp_page_is_scanned(tmp_path: Path) -> None:
    """23 image-only pages plus a text stamp page: the scanned-textbook shape."""
    pdf = _build(tmp_path / "scanned_textbook.pdf", ["image"] * 23 + ["stamp"])
    assert 800 <= _text_chars(pdf, 23) <= 950  # the stamp alone clears 50 chars/page on average

    assert await _router()._pdf_kind(pdf) == "scanned"


async def test_regression_one_of_24_pages_with_text_is_scanned(tmp_path: Path) -> None:
    """The exact 1-of-24 shape, wherever the lone text page sits in the sample."""
    router = _router()
    for text_page in (0, 13, 23):  # cover, a sampled mid page, last page
        specs = ["image"] * 24
        specs[text_page] = "stamp"
        pdf = _build(tmp_path / f"one_of_24_{text_page}.pdf", specs)
        assert await router._pdf_kind(pdf) == "scanned", text_page


async def test_scan_with_text_cover_stamp_and_back_page_is_scanned(tmp_path: Path) -> None:
    specs = ["body"] + ["image"] * 60 + ["stamp", "body"]
    pdf = _build(tmp_path / "cover_stamp_back.pdf", specs)

    assert await _router()._pdf_kind(pdf) == "scanned"


async def test_short_scan_with_a_text_cover_is_scanned(tmp_path: Path) -> None:
    """Short books are sampled in full; one text cover of five isn't native."""
    pdf = _build(tmp_path / "short.pdf", ["body"] + ["image"] * 4)

    assert await _router()._pdf_kind(pdf) == "scanned"


async def test_pure_scan_is_scanned(tmp_path: Path) -> None:
    pdf = _build(tmp_path / "pure_scan.pdf", ["image"] * 40)

    assert await _router()._pdf_kind(pdf) == "scanned"


async def test_native_book_with_image_cover_and_blank_plates_is_native(tmp_path: Path) -> None:
    """Cover image, a blank verso, and two blank plates that land on sampled pages."""
    specs = ["body"] * 40
    specs[0] = "image"
    specs[1] = "blank"
    specs[14] = "blank"
    specs[25] = "blank"
    pdf = _build(tmp_path / "native_plates.pdf", specs)

    assert await _router()._pdf_kind(pdf) == "native"


async def test_native_book_with_a_long_gallery_run_is_native(tmp_path: Path) -> None:
    """A 16-page run of text-free plates (a gallery insert) plus an image cover
    and blank verso: a quarter of the book with no text, contiguous."""
    specs = ["image", "blank"] + ["body"] * 18 + ["image"] * 16 + ["body"] * 28
    pdf = _build(tmp_path / "gallery.pdf", specs)

    assert await _router()._pdf_kind(pdf) == "native"


async def test_native_book_with_text_on_every_page_is_native(tmp_path: Path) -> None:
    pdf = _build(tmp_path / "all_text.pdf", ["body"] * 300)

    assert await _router()._pdf_kind(pdf) == "native"


async def test_one_page_prose_pdf_is_native(tmp_path: Path) -> None:
    """A 1-page prose PDF: one page, a few hundred chars of real words."""
    pdf = _build(tmp_path / "one_page.pdf", [BODY])

    assert await _router()._pdf_kind(pdf) == "native"


async def test_multipage_garbage_text_layer_is_scanned(tmp_path: Path) -> None:
    """40 pages whose every sampled page carries a garbled text layer well
    over the character threshold: the page count passes, the dictionary
    check doesn't."""
    garbage = "\n".join(["xqzk wbfp zztq vkxn " * 4] * 8)
    pdf = _build(tmp_path / "garbage.pdf", [garbage] * 40)
    assert _text_chars(pdf, 0) >= 200

    assert await _router()._pdf_kind(pdf) == "scanned"


@pytest.mark.parametrize(("text_pages", "expected"), [(4, "scanned"), (5, "native")])
async def test_text_page_fraction_boundary(tmp_path: Path, text_pages: int, expected: str) -> None:
    """Native needs at least NATIVE_MIN_TEXT_PAGE_FRACTION (0.4) of the
    sampled pages over the per-page character threshold: 4/12 = 0.333 is
    scanned, 5/12 = 0.417 is native. All 12 pages are sampled."""
    assert router_module.NATIVE_MIN_TEXT_PAGE_FRACTION == 0.4
    pdf = _build(tmp_path / "frac.pdf", ["body"] * text_pages + ["image"] * (12 - text_pages))

    assert await _router()._pdf_kind(pdf) == expected
