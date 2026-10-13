"""Reading order for the chunker (VOR-1452).

A parse stored before VOR-1417 lists every picture, then every table, after
the last text element of the book. Walked in list order, such a table takes
the heading in force at the end of the book: one audited book filed 63
tables from pages 8-205 under "Quick Order Form". Figures were already
placed by page (``FigurePlacement``); tables were not.

``in_reading_order`` returns the elements in page order: an element listed
after text of a later page than its own (out of page order) moves to the end
of its own page's text -- just before the first running-text element of a
later page -- the same anchor ``FigurePlacement`` gives a figure. Moved
elements keep their relative order; everything else keeps its position, so a
parse already in reading order is returned unchanged. Every element then
takes the heading in force on its own page.

Reflowable parses (EPUB/MOBI) are left alone: their "pages" are chapters in
spine order, not print pages.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from typing import Final

from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement

#: Elements whose page order is the book's running order. Tables and pictures
#: are the ones a legacy parse displaces, so they never set the order.
_FLOW_TYPES: Final[frozenset[ElementType]] = frozenset(
    {
        ElementType.NARRATIVE_TEXT,
        ElementType.TITLE,
        ElementType.LIST_ITEM,
        ElementType.FIGURE_CAPTION,
        ElementType.EQUATION,
        ElementType.CODE_BLOCK,
    }
)


def reading_order_permutation(elements: Sequence[ParsedElement]) -> list[int] | None:
    """Old indices in reading order, or ``None`` when the order is already right."""
    # prefix_max[i]: the highest running-text page among elements[:i].
    prefix_max = [0] * (len(elements) + 1)
    running = 0
    for index, element in enumerate(elements):
        if element.element_type in _FLOW_TYPES and element.page_number is not None:
            running = max(running, element.page_number)
        prefix_max[index + 1] = running
    keys: list[tuple[int, int, int]] = []
    moved = False
    for index, element in enumerate(elements):
        page = element.page_number
        if element.element_type in _FLOW_TYPES or page is None or prefix_max[index] <= page:
            keys.append((index, 1, index))
            continue
        # Before element ``anchor``: the first running text of a later page.
        anchor = bisect_right(prefix_max, page) - 1
        keys.append((anchor, 0, index))
        moved = True
    if not moved:
        return None
    return [index for _, _, index in sorted(keys)]


def in_reading_order(
    doc: ParsedDocument, reflowable: frozenset[str]
) -> tuple[ParsedDocument, list[int]]:
    """``(doc in reading order, new position of each old insertion point)``.

    The second value maps an old insertion point ``i`` (before old element
    ``i``; ``len(elements)`` is the end) to the new one, so a composite keyed
    on an element index is emitted where its element now sits.
    """
    count = len(doc.elements)
    identity = list(range(count + 1))
    if doc.parser_used in reflowable:
        return doc, identity
    order = reading_order_permutation(doc.elements)
    if order is None:
        return doc, identity
    new_position = [0] * (count + 1)
    for new_index, old_index in enumerate(order):
        new_position[old_index] = new_index
    new_position[count] = count
    reordered = doc.model_copy(update={"elements": [doc.elements[i] for i in order]})
    return reordered, new_position


__all__ = ["in_reading_order", "reading_order_permutation"]
