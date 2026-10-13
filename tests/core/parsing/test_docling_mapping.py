"""Tests for vo_rag.core.parsing.parsers.docling_mapping.

Verifies the
docling → ParsedDocument mapping (heading-stack inference, caption demotion)
without requiring a real docling install — uses a SimpleNamespace stub.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest

from vo_rag.core.parsing.contract import ElementType
from vo_rag.core.parsing.parsers.docling_mapping import (
    _decode_data_uri,
    _extract_image_bytes,
    _infer_heading_level,
    _is_caption_text,
    _pil_to_png_bytes,
    docling_to_parsed_document,
)

_FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


class _FakePil:
    """Duck-typed stand-in for PIL.Image — avoids a Pillow test dependency
    (Pillow ships only with the heavy, optional docling extra)."""

    size = (4, 3)
    mode = "RGB"

    def save(self, buf: io.BytesIO, format: str | None = None) -> None:  # noqa: A002
        buf.write(_FAKE_PNG)


def test_chapter_pattern_is_level_1() -> None:
    assert _infer_heading_level("Chapter 1 - The approach", docling_level=1) == 1
    assert _infer_heading_level("Chapter 2: The Three Basics", docling_level=1) == 1
    assert _infer_heading_level("Part IV", docling_level=1) == 1
    assert _infer_heading_level("Appendix 2", docling_level=0) == 1


def test_other_heading_is_level_2() -> None:
    assert _infer_heading_level("The Friction Circle", docling_level=1) == 2
    assert _infer_heading_level("Brake Bias", docling_level=0) == 2


def test_docling_explicit_level_is_respected() -> None:
    assert _infer_heading_level("Some Sub-sub-heading", docling_level=3) == 3


def test_caption_detection() -> None:
    assert _is_caption_text("Figure 3: airflow over the spoiler")
    assert _is_caption_text("Table 2 - measured coefficients")
    assert _is_caption_text("The home-made pressure measuring puck mounted on the calibration rig.")
    # short period-ending phrase is AMBIGUOUS -> keep as heading
    assert not _is_caption_text("Why Smooth Inputs Matter.")
    assert not _is_caption_text("The home-made pressure measuring puck.")
    assert not _is_caption_text("Brake Bias")
    assert not _is_caption_text("Chapter 1 - The approach")
    assert not _is_caption_text("The Late Apex")


def _build_fake_docling_doc() -> object:
    def text_item(text: str, label: str) -> object:
        return SimpleNamespace(
            text=text,
            label=label,
            prov=[SimpleNamespace(page_no=1)],
        )

    return SimpleNamespace(
        texts=[
            text_item("Chapter 1 Introduction", "section_header"),
            text_item("The Friction Circle", "section_header"),
            text_item("Grip is finite.", "text"),
            text_item("Figure 1: the test rig.", "section_header"),
        ],
        tables=[],
        pictures=[],
        pages=[SimpleNamespace(page_no=1)],
    )


def test_docling_mapping_recovers_structure(tmp_path) -> None:
    from vo_rag.core.storage.figures import FigureStore

    fake_doc = _build_fake_docling_doc()
    parsed = docling_to_parsed_document(
        fake_doc, FigureStore(tmp_path), "book.pdf", "pdf", src_hash="cd" * 32
    )

    assert len(parsed.heading_tree) > 0
    assert any(e.section_depth and e.section_depth >= 1 for e in parsed.elements)

    titles = {e.text: e for e in parsed.elements if e.element_type == ElementType.TITLE}
    assert "Chapter 1 Introduction" in titles
    assert titles["Chapter 1 Introduction"].section_depth == 1
    assert titles["The Friction Circle"].section_depth == 2

    assert "Figure 1: the test rig." not in titles
    captions = [e for e in parsed.elements if e.element_type == ElementType.FIGURE_CAPTION]
    assert any("Figure 1" in (e.text or "") for e in captions)


@pytest.mark.parametrize("empty_text", ["", "   ", None])
def test_empty_formula_text_retains_original_characters_and_location(tmp_path, empty_text) -> None:
    from vo_rag.core.storage.figures import FigureStore

    original = "A a b + ( ) h 2 -------- ="
    item = SimpleNamespace(
        text=empty_text,
        orig=original,
        label="formula",
        self_ref="#/texts/0",
        prov=[
            SimpleNamespace(
                page_no=12,
                bbox=SimpleNamespace(l=126.0, t=441.0, r=177.0, b=420.0, coord_origin="BOTTOMLEFT"),
            )
        ],
    )
    doc = SimpleNamespace(texts=[item], tables=[], pictures=[], pages=[])
    parsed = docling_to_parsed_document(
        doc, FigureStore(tmp_path), "book.pdf", "pdf", src_hash="cd" * 32
    )
    assert len(parsed.elements) == 1
    equation = parsed.elements[0]
    assert equation.element_type is ElementType.EQUATION
    assert equation.text == original
    assert equation.page_number == 12
    assert equation.metadata["docling_orig"] == original
    location = equation.metadata["docling_provenance"]
    assert location["source_self_ref"] == "#/texts/0"
    assert location["geometry_status"] == "available"
    assert location["source_alignment"] == "unverified"
    assert location["occurrences"][0]["page_no"] == 12


def test_enriched_formula_text_takes_precedence_over_original_layout(tmp_path) -> None:
    from vo_rag.core.storage.figures import FigureStore

    item = SimpleNamespace(
        text="A = \\frac{(a+b)h}{2}",
        orig="A a b + h 2 -- =",
        label="formula",
        prov=[SimpleNamespace(page_no=12)],
    )
    doc = SimpleNamespace(texts=[item], tables=[], pictures=[], pages=[])
    parsed = docling_to_parsed_document(
        doc, FigureStore(tmp_path), "book.pdf", "pdf", src_hash="cd" * 32
    )
    assert parsed.elements[0].text == item.text
    assert parsed.elements[0].metadata["docling_orig"] == item.orig


@pytest.mark.parametrize(
    ("marker", "text", "expected"),
    [
        ("1)", "Calculate spindle speeds.", "1) Calculate spindle speeds."),
        ("2)", "2) Calculate cutting speeds.", "2) Calculate cutting speeds."),
        ("2)", "2)Calculate cutting speeds.", "2)Calculate cutting speeds."),
        ("a.", "Use the first row.", "a. Use the first row."),
        ("a.", "a.Use the first row.", "a.Use the first row."),
        ("1.", "1.Calculate speeds.", "1.Calculate speeds."),
        ("•", "Keep the supplied bullet.", "• Keep the supplied bullet."),
        ("•", "•Keep the supplied bullet.", "•Keep the supplied bullet."),
        ("1.", "1.5 inches", "1. 1.5 inches"),
        ("", "No source marker.", "No source marker."),
        (None, "No source marker.", "No source marker."),
        ("x" * 65, "Bounded marker.", "Bounded marker."),
        ("1)\n2)", "Single marker only.", "Single marker only."),
    ],
)
def test_literal_list_markers_are_retained_without_number_inference(
    tmp_path, marker, text, expected
) -> None:
    from vo_rag.core.storage.figures import FigureStore

    item = SimpleNamespace(
        text=text,
        marker=marker,
        enumerated=True,
        label="list_item",
        prov=[SimpleNamespace(page_no=201)],
    )
    parsed = docling_to_parsed_document(
        SimpleNamespace(texts=[item], tables=[], pictures=[], pages=[]),
        FigureStore(tmp_path),
        "book.pdf",
        "pdf",
        src_hash="cd" * 32,
    )
    element = parsed.elements[0]
    assert element.element_type is ElementType.LIST_ITEM
    assert element.text == expected
    assert element.page_number == 201
    assert element.metadata["docling_list_enumerated"] is True
    if marker and len(marker) <= 64 and "\n" not in marker:
        assert element.metadata["docling_list_marker"] == marker
    else:
        assert "docling_list_marker" not in element.metadata


def test_numbered_source_questions_survive_normal_chunking(tmp_path) -> None:
    from vo_rag.core.chunking.hierarchical import HierarchicalChunker
    from vo_rag.core.storage.figures import FigureStore

    questions = [
        "Calculate the spindle speeds for turning bars using the cutting conditions given below.",
        "Calculate the spindle speeds for turning diameter sections "
        "using the materials listed below.",
    ]
    items = [
        SimpleNamespace(
            text=text,
            marker=f"{number})",
            enumerated=True,
            label="list_item",
            prov=[SimpleNamespace(page_no=201)],
        )
        for number, text in enumerate(questions, 1)
    ]
    parsed = docling_to_parsed_document(
        SimpleNamespace(texts=items, tables=[], pictures=[], pages=[]),
        FigureStore(tmp_path),
        "book.pdf",
        "pdf",
        src_hash="cd" * 32,
    )
    chunks = HierarchicalChunker().chunk(parsed, composites=[])
    assert chunks
    for number, text in enumerate(questions, 1):
        assert any(f"{number}) {text}" in chunk.content for chunk in chunks)


# --- modern-docling (2.x) picture image extraction (regression for 0-images bug) ---


def test_pil_to_png_bytes_roundtrips() -> None:
    assert _pil_to_png_bytes(_FakePil()) == _FAKE_PNG
    assert _pil_to_png_bytes("not an image") == b""
    assert _pil_to_png_bytes(None) == b""


def test_decode_data_uri() -> None:
    raw = b"\x89PNG\r\nhello-bytes"
    uri = "data:image/png;base64," + base64.b64encode(raw).decode()
    assert _decode_data_uri(uri) == raw
    assert _decode_data_uri("http://example.com/x.png") == b""
    assert _decode_data_uri(None) == b""


def test_extract_image_bytes_handles_imageref() -> None:
    # docling-core 2.x: PictureItem.image is an ImageRef carrying a PIL image.
    pic = SimpleNamespace(image=SimpleNamespace(pil_image=_FakePil()))
    assert _extract_image_bytes(pic) == _FAKE_PNG
    # ImageRef carrying only a data: URI.
    raw = b"data-uri-image-bytes"
    uri_pic = SimpleNamespace(
        image=SimpleNamespace(
            pil_image=None, uri="data:image/png;base64," + base64.b64encode(raw).decode()
        )
    )
    assert _extract_image_bytes(uri_pic) == raw
    # legacy raw-bytes attr still works.
    assert _extract_image_bytes(SimpleNamespace(image=b"\xff\xd8raw")) == b"\xff\xd8raw"
    # nothing usable -> empty.
    assert _extract_image_bytes(SimpleNamespace(foo="bar")) == b""


def test_docling_mapping_writes_deterministic_figure_refs(tmp_path) -> None:
    """Same source hash + page/index ⇒ identical ref across runs and
    regardless of the (temp) source path — the old temp-path-hash bug."""
    from vo_rag.core.storage.figures import FigureStore

    src_hash = "ab" * 32
    raw = b"second-picture-bytes"
    fake_doc = SimpleNamespace(
        texts=[],
        tables=[],
        pictures=[
            SimpleNamespace(
                image=SimpleNamespace(pil_image=_FakePil()),
                prov=[SimpleNamespace(page_no=2)],
            ),
            SimpleNamespace(
                image=SimpleNamespace(
                    pil_image=None, uri="data:image/png;base64," + base64.b64encode(raw).decode()
                ),
                prov=[SimpleNamespace(page_no=5)],
            ),
        ],
        pages=[SimpleNamespace(page_no=1)],
    )

    store = FigureStore(tmp_path)
    parsed1 = docling_to_parsed_document(
        fake_doc, store, "/tmp/aaa/book.pdf", "pdf", src_hash=src_hash
    )
    parsed2 = docling_to_parsed_document(
        fake_doc, FigureStore(tmp_path), "/tmp/zzz/other-temp-name.pdf", "pdf", src_hash=src_hash
    )

    refs = [img.ref for img in parsed1.images]
    assert refs == [f"{src_hash}/p002_img000.png", f"{src_hash}/p005_img001.png"]
    assert [img.ref for img in parsed2.images] == refs  # temp path is irrelevant now
    for img in parsed1.images:
        assert (tmp_path / img.ref).is_file()  # bytes landed on the filesystem
        assert img.mime_type == "image/png"
    assert parsed1.document_id == src_hash


# --- lane CPU cost (VOR-1427): no whole-document markdown export ---


def test_mapping_never_calls_the_document_markdown_export(tmp_path) -> None:
    """docling's export_to_markdown re-serialises the whole book in Python
    (18 s for a 3,915-page book, about 40% of the mapping); the only readers of
    ParsedDocument.markdown need any text and a 20,000-char sample, which the
    element-based builder provides."""
    from vo_rag.core.storage.figures import FigureStore

    fake_doc = _build_fake_docling_doc()

    def _export() -> str:
        raise AssertionError("export_to_markdown must not be called")

    fake_doc.export_to_markdown = _export  # type: ignore[attr-defined]
    parsed = docling_to_parsed_document(
        fake_doc, FigureStore(tmp_path), "book.pdf", "pdf", src_hash="cd" * 32
    )

    assert "# Chapter 1 Introduction" in parsed.markdown
    assert "Grip is finite." in parsed.markdown


# --- VOR-1452: docling running heads are page furniture, not text ---


def test_docling_page_header_and_footer_labels_are_the_installed_strings() -> None:
    labels = __import__("pytest").importorskip("docling_core.types.doc.labels")
    assert labels.DocItemLabel.PAGE_HEADER.value == "page_header"
    assert labels.DocItemLabel.PAGE_FOOTER.value == "page_footer"


def test_docling_page_header_and_footer_map_to_header_and_footer(tmp_path) -> None:
    from vo_rag.core.storage.figures import FigureStore

    def text_item(text: str, label: str) -> object:
        return SimpleNamespace(text=text, label=label, prov=[SimpleNamespace(page_no=1)])

    fake_doc = SimpleNamespace(
        texts=[
            text_item("SPRINT TRAINING", "page_header"),
            text_item("Chapter 1 Introduction", "section_header"),
            text_item("Grip is finite.", "text"),
            text_item("Track and Field Coaching 17", "page_footer"),
            # Enum-rendered labels ("DocItemLabel.PAGE_HEADER") normalize the same way.
            text_item("The Complete Book", "DocItemLabel.PAGE_HEADER"),
        ],
        tables=[],
        pictures=[],
        pages=[SimpleNamespace(page_no=1)],
    )
    parsed = docling_to_parsed_document(
        fake_doc, FigureStore(tmp_path), "book.pdf", "pdf", src_hash="ef" * 32
    )
    by_text = {e.text: e.element_type for e in parsed.elements}
    assert by_text["SPRINT TRAINING"] is ElementType.HEADER
    assert by_text["Track and Field Coaching 17"] is ElementType.FOOTER
    assert by_text["The Complete Book"] is ElementType.HEADER
    assert by_text["Chapter 1 Introduction"] is ElementType.TITLE
    assert by_text["Grip is finite."] is ElementType.NARRATIVE_TEXT


# --- docling's table of contents ---


def test_docling_document_index_label_is_the_installed_string() -> None:
    labels = __import__("pytest").importorskip("docling_core.types.doc.labels")
    assert labels.DocItemLabel.DOCUMENT_INDEX.value == "document_index"


def test_docling_document_index_maps_to_body_marked_as_contents(tmp_path) -> None:
    """A ``document_index`` item never becomes a TITLE, and keeps its label so
    the chunker files its page as contents (``contents_pages``)."""
    from vo_rag.core.chunking.contents_pages import contents_pages
    from vo_rag.core.storage.figures import FigureStore

    def text_item(text: str, label: str, page: int) -> object:
        return SimpleNamespace(text=text, label=label, prov=[SimpleNamespace(page_no=page)])

    toc_table = SimpleNamespace(
        label="document_index",
        prov=[SimpleNamespace(page_no=3)],
        export_to_markdown=lambda *a, **k: "| Starts | 12 |\n|---|---|\n| Hurdles | 45 |",
        export_to_html=lambda *a, **k: "",
    )
    fake_doc = SimpleNamespace(
        texts=[
            text_item("12 The shot-put 250", "document_index", 2),
            text_item("Chapter 1 Introduction", "section_header", 5),
            text_item("Grip is finite.", "text", 5),
        ],
        tables=[toc_table],
        pictures=[],
        pages=[SimpleNamespace(page_no=p) for p in range(1, 6)],
    )
    parsed = docling_to_parsed_document(
        fake_doc, FigureStore(tmp_path), "book.pdf", "pdf", src_hash="ef" * 32
    )
    entry = next(e for e in parsed.elements if e.text == "12 The shot-put 250")
    assert entry.element_type is ElementType.NARRATIVE_TEXT
    assert entry.metadata["docling_label"] == "document_index"
    table = next(e for e in parsed.elements if e.element_type is ElementType.TABLE)
    assert table.metadata["docling_label"] == "document_index"
    assert contents_pages(parsed) == frozenset({2, 3})
