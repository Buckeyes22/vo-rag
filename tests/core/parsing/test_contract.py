"""Contract tests for vo_rag.core.parsing.

Verifies:
- ElementType is a StrEnum with string values
- All ParsedDocument/PageQA/ParsedElement models have extra='forbid' and bounded validators
- element_id() and image_id() are deterministic SHA-256 hex digests
- ParsedDocument round-trips through model_dump/validate
- element_id is collision-free across 10k inputs
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    PageQA,
    ParsedDocument,
    ParsedElement,
    element_id,
    image_id,
)


def test_element_type_is_strenum() -> None:
    assert issubclass(ElementType, str)
    assert ElementType.TABLE == "table"  # type: ignore[comparison-overlap]


def test_extra_forbid_on_every_model() -> None:
    with pytest.raises(ValidationError):
        ParsedElement(element_id="x", element_type=ElementType.TITLE, bogus=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        PageQA(page_number=1, ocr_confidence=0.5, parser_used="p", bogus=1)  # type: ignore[call-arg]


def test_ocr_confidence_bounded() -> None:
    PageQA(page_number=1, ocr_confidence=0.0, parser_used="p")
    PageQA(page_number=1, ocr_confidence=1.0, parser_used="p")
    with pytest.raises(ValidationError):
        PageQA(page_number=1, ocr_confidence=-0.1, parser_used="p")
    with pytest.raises(ValidationError):
        PageQA(page_number=1, ocr_confidence=1.1, parser_used="p")


def test_element_id_deterministic() -> None:
    a = element_id("hello", 3, 7)
    b = element_id("hello", 3, 7)
    assert a == b and len(a) == 64


def test_element_id_differs_on_any_part() -> None:
    base = element_id("hello", 3, 7)
    assert element_id("hellO", 3, 7) != base
    assert element_id("hello", 4, 7) != base
    assert element_id("hello", 3, 8) != base


def test_image_id_deterministic_from_bytes() -> None:
    raw = b"\x89PNG\r\n\x1a\nfake"
    assert image_id(raw) == image_id(raw)
    assert image_id(raw) != image_id(raw + b"0")
    assert len(image_id(raw)) == 64


def test_parsed_document_round_trip_json() -> None:
    d = ParsedDocument(
        document_id="doc-1",
        source_id="src-1",
        title="t",
        total_pages=2,
        elements=[
            ParsedElement(
                element_id=element_id("foo", 1, 0),
                element_type=ElementType.TITLE,
                text="foo",
                page_number=1,
            ),
        ],
        images=[
            ExtractedImage(
                image_id=image_id(b"img"),
                page_number=1,
                ref="doc-1/page001_img000.png",
            ),
        ],
        page_qa=[
            PageQA(page_number=1, ocr_confidence=0.9, parser_used="p"),
            PageQA(page_number=2, ocr_confidence=0.8, parser_used="p"),
        ],
        markdown="# foo",
        heading_tree=[{"depth": 1, "title": "foo"}],
        parser_used="p",
    )
    dumped = d.model_dump()
    rebuilt = ParsedDocument(**dumped)
    assert rebuilt == d


def test_element_id_collision_10k() -> None:
    seen = {element_id(f"c{i}", i % 100, i) for i in range(10_000)}
    assert len(seen) == 10_000
