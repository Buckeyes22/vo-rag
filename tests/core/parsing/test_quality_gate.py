"""Tests for vo_rag.core.parsing.quality_gate.

Verifies:
- Clean pages pass
- Low OCR confidence quarantines the page
- Zero text on a non-blank page quarantines
- Detected-but-not-extracted table/figure produces a warning
- evaluate_document populates quarantined_pages with the bad page numbers
"""

from __future__ import annotations

from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement
from vo_rag.core.parsing.quality_gate import QualityGate


def _doc(*, qa: PageQA, elements: list[ParsedElement]) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title="t",
        total_pages=1,
        elements=elements,
        images=[],
        page_qa=[qa],
        markdown="",
        heading_tree=[],
        parser_used="test",
    )


def test_clean_page_passes() -> None:
    g = QualityGate()
    qa = PageQA(page_number=1, ocr_confidence=0.95, parser_used="p", text_element_count=3)
    els = [
        ParsedElement(
            element_id=f"e{i}",
            element_type=ElementType.NARRATIVE_TEXT,
            text="x",
            page_number=1,
        )
        for i in range(3)
    ]
    v = g.evaluate_page(qa, els)
    assert v.severity == "pass" and v.passed


def test_low_ocr_confidence_quarantines() -> None:
    g = QualityGate()
    qa = PageQA(page_number=1, ocr_confidence=0.4, parser_used="p")
    v = g.evaluate_page(qa, [])
    assert v.severity == "quarantine" and not v.passed


def test_zero_text_on_non_blank_quarantines() -> None:
    g = QualityGate()
    qa = PageQA(page_number=1, ocr_confidence=0.9, parser_used="p", figure_count=2)
    v = g.evaluate_page(qa, [])
    assert v.severity == "quarantine"


def test_page_holding_only_a_table_is_not_zero_text() -> None:
    # A records table with its running head and footer (VOR-1453): the
    # table's text is extracted text, so the page must not be quarantined.
    g = QualityGate()
    qa = PageQA(page_number=1, ocr_confidence=1.0, parser_used="p", table_count=1)
    els = [
        ParsedElement(element_id="h", element_type=ElementType.HEADER, text="8", page_number=1),
        ParsedElement(
            element_id="t",
            element_type=ElementType.TABLE,
            text="| Year | Holder | Record |\n|---|---|---|\n| 1896 | Burke | 12.0 |",
            page_number=1,
        ),
        ParsedElement(
            element_id="f",
            element_type=ElementType.FOOTER,
            text="Reproduced with permission of the copyright owner.",
            page_number=1,
        ),
    ]
    v = g.evaluate_page(qa, els)
    assert v.severity == "pass" and v.passed


def test_empty_table_on_a_page_without_text_still_quarantines() -> None:
    g = QualityGate()
    qa = PageQA(page_number=1, ocr_confidence=1.0, parser_used="p", table_count=1)
    els = [ParsedElement(element_id="t", element_type=ElementType.TABLE, text=" ", page_number=1)]
    v = g.evaluate_page(qa, els)
    assert v.severity == "quarantine"


def test_table_detected_not_extracted_warns() -> None:
    g = QualityGate()
    qa = PageQA(page_number=1, ocr_confidence=0.9, parser_used="p", table_count=1)
    els = [
        ParsedElement(
            element_id="e",
            element_type=ElementType.NARRATIVE_TEXT,
            text="x",
            page_number=1,
        )
    ]
    v = g.evaluate_page(qa, els)
    assert v.severity == "warn"


def test_figure_detected_not_extracted_warns() -> None:
    g = QualityGate()
    qa = PageQA(page_number=1, ocr_confidence=0.9, parser_used="p", figure_count=1)
    els = [
        ParsedElement(
            element_id="e",
            element_type=ElementType.NARRATIVE_TEXT,
            text="x",
            page_number=1,
        )
    ]
    v = g.evaluate_page(qa, els)
    assert v.severity == "warn"


def test_evaluate_document_populates_quarantined_pages() -> None:
    g = QualityGate()
    good_qa = PageQA(page_number=1, ocr_confidence=0.9, parser_used="p")
    bad_qa = PageQA(page_number=2, ocr_confidence=0.2, parser_used="p")
    doc = ParsedDocument(
        document_id="d",
        source_id="s",
        title="t",
        total_pages=2,
        elements=[
            ParsedElement(
                element_id="e",
                element_type=ElementType.NARRATIVE_TEXT,
                text="x",
                page_number=1,
            )
        ],
        images=[],
        page_qa=[good_qa, bad_qa],
        markdown="",
        heading_tree=[],
        parser_used="p",
    )
    out = g.evaluate_document(doc)
    assert out.quarantined_pages == [2]
    assert any("ocr_confidence" in warning for warning in bad_qa.warnings)
