"""Tests for vo_rag.core.parsing.reading_order.

The current reading-order heuristic is a documented no-op stub (waiting on a
real column-clustering check). These tests pin the contract so a future
implementation lands visibly.
"""

from __future__ import annotations

from vo_rag.core.parsing.contract import ElementType, ParsedElement
from vo_rag.core.parsing.reading_order import reading_order_broken


def test_reading_order_broken_returns_false_no_op_stub() -> None:
    """Documented: no column-clustering yet; reading-order check is a no-op."""
    elements = [
        ParsedElement(
            element_id="e1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="x",
            page_number=1,
        )
    ]
    assert reading_order_broken(elements, 1) is False
