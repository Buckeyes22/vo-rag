"""Reading-order heuristic for vo-rag page quality gating.

The column-clustering check is a documented no-op stub waiting on a parser-side
coordinate convention. The downstream QualityGate still calls this function
so the gate has a stable interface to land a real implementation against.
"""

from __future__ import annotations

from vo_rag.core.parsing.contract import ParsedElement


def reading_order_broken(elements: list[ParsedElement], page_number: int) -> bool:
    """Heuristic: within elements on `page_number` that have coordinates,
    the y-coordinate should be monotonically non-decreasing within each
    detected column. Returns True when a strong inversion is detected.

    TODO: add a concrete column-clustering check once the parsers share a
    coordinate convention. Until then this returns False (a conservative
    no-op).
    """
    _ = (elements, page_number)
    return False
