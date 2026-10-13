"""VOR-1417: docling mapping emits text, tables and pictures in reading order.

Before the fix every table and picture was appended after the document's last
text item, so each one inherited the final heading (one book: all tables filed
under "INDEX"). These tests build DoclingDocument-shaped fakes (no docling
install needed) and prove a table/picture between two sections maps between
them, both via docling's body tree (``iterate_items``) and via the page + bbox
fallback used when no tree is available.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

from vo_rag.core.parsing.contract import ElementType, ParsedDocument
from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document
from vo_rag.core.storage.figures import FigureStore

_PNG = b"\x89PNG\r\n\x1a\nreading-order-picture"


class _FakePil:
    size = (4, 3)
    mode = "RGB"

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, buf: io.BytesIO, format: str | None = None) -> None:  # noqa: A002
        buf.write(self._payload)


def _prov(page: int, top: float) -> list[object]:
    return [
        SimpleNamespace(
            page_no=page,
            bbox=SimpleNamespace(l=10.0, t=top, r=500.0, b=top - 20.0, coord_origin="BOTTOMLEFT"),
        )
    ]


def _text(ref: str, text: str, label: str, page: int, top: float) -> SimpleNamespace:
    return SimpleNamespace(self_ref=ref, text=text, label=label, prov=_prov(page, top))


def _table(ref: str, md: str, page: int, top: float) -> SimpleNamespace:
    return SimpleNamespace(
        self_ref=ref, label="table", export_to_markdown=lambda: md, prov=_prov(page, top)
    )


def _picture(ref: str, payload: bytes, page: int, top: float) -> SimpleNamespace:
    return SimpleNamespace(
        self_ref=ref,
        label="picture",
        image=SimpleNamespace(pil_image=_FakePil(payload)),
        prov=_prov(page, top),
    )


def _items() -> tuple[list[SimpleNamespace], list[SimpleNamespace], list[SimpleNamespace]]:
    texts = [
        _text("#/texts/0", "Chapter 1 Sprinting", "section_header", 1, 700.0),
        _text("#/texts/1", "Sprinters accelerate hard.", "text", 1, 600.0),
        _text("#/texts/2", "Chapter 2 Hurdles", "section_header", 3, 700.0),
        _text("#/texts/3", "Hurdlers keep rhythm.", "text", 3, 600.0),
        _text("#/texts/4", "INDEX", "section_header", 5, 700.0),
        _text("#/texts/5", "acceleration, 12", "text", 5, 600.0),
    ]
    tables = [_table("#/tables/0", "| 100m | 10.0 |", 2, 500.0)]
    pictures = [_picture("#/pictures/0", _PNG, 3, 650.0)]
    return texts, tables, pictures


class _TreeDoc(SimpleNamespace):
    """Docling-2.x shape: body tree order exposed through ``iterate_items``."""

    def iterate_items(self, **_: object) -> Iterator[tuple[object, int]]:
        texts, tables, pictures = self.texts, self.tables, self.pictures
        order = [texts[0], texts[1], tables[0], texts[2], pictures[0], texts[3], texts[4]]
        order.append(texts[5])
        for item in order:
            yield item, 1


def _parse(doc: object, tmp_path) -> ParsedDocument:
    return docling_to_parsed_document(
        doc, FigureStore(tmp_path), "book.pdf", "pdf", src_hash="ef" * 32
    )


def _assert_reading_order(parsed: ParsedDocument) -> None:
    kinds = [(e.element_type, e.text or "<img>") for e in parsed.elements]
    assert kinds == [
        (ElementType.TITLE, "Chapter 1 Sprinting"),
        (ElementType.NARRATIVE_TEXT, "Sprinters accelerate hard."),
        (ElementType.TABLE, "| 100m | 10.0 |"),
        (ElementType.TITLE, "Chapter 2 Hurdles"),
        (ElementType.IMAGE, "<img>"),
        (ElementType.NARRATIVE_TEXT, "Hurdlers keep rhythm."),
        (ElementType.TITLE, "INDEX"),
        (ElementType.NARRATIVE_TEXT, "acceleration, 12"),
    ]
    table = next(e for e in parsed.elements if e.element_type == ElementType.TABLE)
    assert table.section_title == "Chapter 1 Sprinting"
    assert table.page_number == 2
    geometry = table.metadata["docling_provenance"]
    assert geometry["source_self_ref"] == "#/tables/0"
    assert geometry["source_alignment"] == "unverified"
    assert geometry["geometry_status"] == "available"
    assert geometry["occurrences"] == [
        {
            "status": "available",
            "page_number_source_field": "page_no",
            "page_no": 2,
            "page_number_status": "available",
            "coord_origin_status": "known_unverified",
            "bbox": {
                "l": 10.0,
                "t": 500.0,
                "r": 500.0,
                "b": 480.0,
                "coord_origin": "BOTTOMLEFT",
            },
        }
    ]
    assert "coordinate_units" not in geometry
    picture = next(e for e in parsed.elements if e.element_type == ElementType.IMAGE)
    assert picture.section_title == "Chapter 2 Hurdles"

    # Figure linkage stays internally consistent: element_index points at the
    # IMAGE element, image_index / FigureStore ref keep docling's picture index.
    (image,) = parsed.images
    assert parsed.elements[image.element_index].element_type == ElementType.IMAGE
    assert parsed.elements[image.element_index].metadata["image_id"] == image.image_id
    assert image.image_index == 0
    assert image.ref == f"{'ef' * 32}/p003_img000.png"

    # Per-page QA counts are unaffected by the interleaving.
    qa = {q.page_number: q for q in parsed.page_qa}
    assert qa[2].table_count == 1
    assert qa[3].figure_count == 1
    assert qa[1].text_element_count == 2


def test_tree_order_places_table_and_picture_between_sections(tmp_path) -> None:
    texts, tables, pictures = _items()
    doc = _TreeDoc(texts=texts, tables=tables, pictures=pictures, pages=[])
    _assert_reading_order(_parse(doc, tmp_path))


def test_page_bbox_fallback_places_table_and_picture_between_sections(tmp_path) -> None:
    texts, tables, pictures = _items()
    doc = SimpleNamespace(texts=texts, tables=tables, pictures=pictures, pages=[])
    _assert_reading_order(_parse(doc, tmp_path))


def test_items_missing_from_tree_keep_their_list_position(tmp_path) -> None:
    """A text absent from the traversal (e.g. furniture on an older docling)
    stays right after its predecessor in ``doc.texts`` instead of vanishing."""

    texts, tables, pictures = _items()
    footer = _text("#/texts/6", "Page footer", "page_footer", 1, 20.0)
    texts.insert(2, footer)

    class _PartialTree(SimpleNamespace):
        def iterate_items(self) -> Iterator[tuple[object, int]]:  # no kwargs accepted
            t = [x for x in self.texts if x is not footer]
            for item in [t[0], t[1], self.tables[0], t[2], self.pictures[0], *t[3:]]:
                yield item, 1

    doc = _PartialTree(texts=texts, tables=tables, pictures=pictures, pages=[])
    parsed = _parse(doc, tmp_path)
    order = [e.text or "<img>" for e in parsed.elements]
    assert order[:4] == [
        "Chapter 1 Sprinting",
        "Sprinters accelerate hard.",
        "Page footer",
        "| 100m | 10.0 |",
    ]
    assert len(parsed.elements) == 9


def test_table_without_provenance_has_explicit_unavailable_geometry(tmp_path) -> None:
    table = SimpleNamespace(
        self_ref="#/tables/no-prov",
        label="table",
        export_to_markdown=lambda: "| a | b |",
        prov=None,
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)

    geometry = parsed.elements[0].metadata["docling_provenance"]
    assert geometry["geometry_status"] == "unavailable"
    assert geometry["occurrences"] == []
    assert geometry["occurrence_count"] == 0
    assert geometry["source_alignment"] == "unverified"


def test_table_provenance_output_is_bounded_and_tail_is_in_inventory_hash(tmp_path) -> None:
    def _occurrences() -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                page_no=index + 1,
                bbox=SimpleNamespace(
                    l=float(index),
                    t=100.0,
                    r=200.0,
                    b=90.0,
                    coord_origin="TOPLEFT",
                ),
            )
            for index in range(24)
        ]

    prov = _occurrences()

    def _parse_table() -> dict[str, Any]:
        table = SimpleNamespace(
            self_ref="#/tables/many-provs",
            label="table",
            export_to_markdown=lambda: "| row |",
            prov=prov,
        )
        parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)
        return cast(dict[str, Any], parsed.elements[0].metadata["docling_provenance"])

    first = _parse_table()
    assert first["occurrence_count"] == 24
    assert first["total_occurrence_count"] == 24
    assert first["occurrences_truncated"] is True
    assert len(first["occurrences"]) == 16
    assert first["occurrence_inventory_sha256_scope"] == "complete"
    original_hash = first["occurrence_inventory_sha256"]

    prov[-1].bbox.l = 999.0
    second = _parse_table()
    assert second["occurrences"] == first["occurrences"]
    assert second["occurrence_inventory_sha256"] != original_hash
    assert len(str(second["occurrence_inventory_sha256"])) == 64


def test_invalid_table_bbox_is_explicit_and_json_safe(tmp_path) -> None:
    table = SimpleNamespace(
        self_ref="#/tables/invalid-bbox",
        label="table",
        page_number=1,
        export_to_markdown=lambda: "| row |",
        prov=[
            SimpleNamespace(
                page_no=10**100,
                bbox=SimpleNamespace(
                    l=float("inf"),
                    t=10**100,
                    r=3.0,
                    b=4.0,
                    coord_origin="X" * 10_000,
                ),
            )
        ],
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)

    geometry = parsed.elements[0].metadata["docling_provenance"]
    occurrence = geometry["occurrences"][0]
    assert geometry["geometry_status"] == "invalid"
    assert occurrence["status"] == "page_number_invalid"
    assert occurrence["page_number_status"] == "invalid"
    assert "page_no" not in occurrence
    assert occurrence["bbox"]["coord_origin"] is None
    assert occurrence["coord_origin_status"] == "invalid_too_long"
    assert geometry["occurrences"][0]["bbox"]["l"] is None
    assert geometry["occurrences"][0]["bbox"]["t"] is None
    assert "Infinity" not in json.dumps(geometry, allow_nan=False)
    assert len(parsed.model_dump_json()) > 0


def test_provenance_generator_inspection_has_hard_bound(tmp_path) -> None:
    visited = 0

    def _many_provenance() -> Iterator[SimpleNamespace]:
        nonlocal visited
        for index in range(10_000):
            visited += 1
            yield SimpleNamespace(
                page_no=index + 1,
                bbox=SimpleNamespace(l=1.0, t=2.0, r=3.0, b=4.0, coord_origin="TOPLEFT"),
            )

    table = SimpleNamespace(
        self_ref="#/tables/generated",
        label="table",
        export_to_markdown=lambda: "| row |",
        prov=_many_provenance(),
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)

    geometry = parsed.elements[0].metadata["docling_provenance"]
    assert visited == 257  # bounded prefix plus one item to detect truncation
    assert geometry["occurrence_count"] == 256
    assert geometry["total_occurrence_count"] is None
    assert geometry["occurrence_inspection_truncated"] is True
    assert geometry["occurrence_inventory_sha256_scope"] == "inspected_prefix"


def test_mapper_marks_invalid_oversized_provenance_unknown_truncated(tmp_path) -> None:
    class GuardedProvenance(list[SimpleNamespace]):
        def __iter__(self) -> Iterator[SimpleNamespace]:
            source = super().__iter__()
            for index, item in enumerate(source):
                if index >= 257:
                    raise AssertionError("mapper consumed beyond its provenance bound")
                yield item

    table = SimpleNamespace(
        self_ref="#/tables/invalid-many",
        label="table",
        export_to_markdown=lambda: "| row |",
        prov=GuardedProvenance(SimpleNamespace(page_no=-1, bbox=None) for _ in range(10_000)),
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)

    geometry = parsed.elements[0].metadata["docling_provenance"]
    assert geometry["occurrence_count"] == 256
    assert geometry["total_occurrence_count"] is None
    assert geometry["occurrence_inspection_truncated"] is True
    assert geometry["geometry_status"] == "unknown_truncated"
    assert len(parsed.model_dump_json()) > 0


def test_mapper_ignores_huge_boolean_and_negative_page_numbers(tmp_path) -> None:
    for invalid_page in (10**100, True, -10):
        table = SimpleNamespace(
            self_ref="#/tables/bad-page",
            page_number=invalid_page,
            label="table",
            export_to_markdown=lambda: "| row |",
            prov=[SimpleNamespace(page_no=invalid_page, bbox=None)],
        )
        parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)
        element = parsed.elements[0]
        assert element.page_number is None
        assert element.metadata["docling_provenance"]["geometry_status"] == "invalid"
        assert len(parsed.model_dump_json()) > 0


def test_page_idx_is_zero_based_and_other_page_fields_are_positive(tmp_path) -> None:
    for field, value, expected in (("page_idx", 0, 1), ("page_idx", 1, 2), ("page_no", 0, None)):
        table = SimpleNamespace(
            self_ref="#/tables/page-index",
            label="table",
            export_to_markdown=lambda: "| row |",
            prov=None,
        )
        setattr(table, field, value)
        parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)
        assert parsed.elements[0].page_number == expected

    table = SimpleNamespace(
        self_ref="#/tables/provenance-page-index",
        label="table",
        export_to_markdown=lambda: "| row |",
        prov=[SimpleNamespace(page_idx=0, bbox=None)],
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)
    assert parsed.elements[0].page_number == 1


def test_unknown_origin_and_invalid_page_are_unverified(tmp_path) -> None:
    table = SimpleNamespace(
        self_ref="#/tables/unknown-origin",
        page_number=1,
        label="table",
        export_to_markdown=lambda: "| row |",
        prov=[
            SimpleNamespace(
                page_no=-2,
                bbox=SimpleNamespace(l=1, t=2, r=3, b=4, coord_origin="X"),
            )
        ],
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)
    occurrence = parsed.elements[0].metadata["docling_provenance"]["occurrences"][0]
    assert occurrence["page_number_status"] == "invalid"
    assert occurrence["coord_origin_status"] == "unknown_unverified"
    assert occurrence["bbox"]["coord_origin"] == "X"
    assert parsed.elements[0].metadata["docling_provenance"]["source_alignment"] == "unverified"


def test_huge_self_ref_is_bounded_without_full_hash(tmp_path) -> None:
    huge_ref = "#/tables/" + "x" * 1_000_000
    table = SimpleNamespace(
        self_ref=huge_ref,
        label="table",
        export_to_markdown=lambda: "| row |",
        prov=None,
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)
    geometry = parsed.elements[0].metadata["docling_provenance"]
    assert len(geometry["source_self_ref"]) == 256
    assert geometry["source_self_ref_truncated"] is True
    assert geometry["source_self_ref_status"] == "truncated"
    assert geometry["source_self_ref_sha256"] is None
    assert len(parsed.model_dump_json()) > 0


def test_malformed_scalar_provenance_does_not_expand_into_characters(tmp_path) -> None:
    table = SimpleNamespace(
        self_ref="#/tables/scalar-prov",
        label="table",
        export_to_markdown=lambda: "| row |",
        prov="x" * 100_000,
    )
    parsed = _parse(SimpleNamespace(texts=[], tables=[table], pictures=[], pages=[]), tmp_path)

    geometry = parsed.elements[0].metadata["docling_provenance"]
    assert geometry["occurrence_count"] == 1
    assert geometry["occurrences"][0]["status"] == "unavailable"
    assert len(parsed.model_dump_json()) > 0
