"""VOR-1416: a back-matter index must not explode into single-character chunks.

Production (a 216-page track and field handbook) produced
~297k one-character chunks. The index is a long run of short title-case lines
("Aldridge, H. M. 42") interleaved with lower-case entries that the chunker
drops as index fragments. Every title-case line looked like a standalone
heading and was queued as a pending heading; the dropped entries never drained
the queue, so ~170 index entries were glued into one 3,125-char section path.
With the section path alone above the 600-token budget, the token-ceiling
splitter could only fit one character per piece, and ``enforce`` re-chopped
every piece again. The heading-only flush also carried no page number.

This fixture rebuilds that element shape as a ``ParsedDocument`` (no docling
needed) and asserts sane output.
"""

from __future__ import annotations

import itertools
import string

from vo_rag.core.chunking.counters import naive_counter
from vo_rag.core.chunking.hierarchical import HierarchicalChunker
from vo_rag.core.chunking.token_ceiling import TokenCeilingBackstop
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement

TOTAL_PAGES = 216
PROSE = (
    "The sprinter drives out of the blocks with a powerful extension of both legs, "
    "keeping the trunk low and the arms working hard through the first strides of the race."
)
BIO = (
    "Sam Ellis has been personal coach to several regional track and field athletes "
    "and was head coach for the Northwind Athletics Club for many years."
)


def _el(i: int, kind: ElementType, text: str, page: int) -> ParsedElement:
    return ParsedElement(
        element_id=f"e{i}",
        element_type=kind,
        text=text,
        page_number=page,
        section_depth=1,
        section_title="",
    )


def _surnames() -> list[str]:
    letters = string.ascii_lowercase
    return ["A" + "".join(p) for p in itertools.product(letters, letters)]


def _index_book(entries: int = 170) -> ParsedDocument:
    elements: list[ParsedElement] = []
    i = 0
    for page in range(1, 200):
        elements.append(_el(i, ElementType.NARRATIVE_TEXT, f"{PROSE} Page {page} drill.", page))
        i += 1
    elements.append(_el(i, ElementType.TITLE, "INDEX", 210))
    i += 1
    names = _surnames()
    for n in range(entries):
        page = 210 + (n * 5) // entries
        # Title-case "Surname, Forename 42" passes is_standalone_heading.
        elements.append(_el(i, ElementType.NARRATIVE_TEXT, f"{names[n]}, Arne {n % 190 + 1}", page))
        i += 1
        # Lower-case subject entries are dropped as back-matter index fragments.
        elements.append(
            _el(
                i,
                ElementType.NARRATIVE_TEXT,
                f"aerobic topic {names[n].lower()} {n % 90 + 3}",
                page,
            )
        )
        i += 1
    elements.append(_el(i, ElementType.NARRATIVE_TEXT, BIO, 215))
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title="The complete book of track",
        total_pages=TOTAL_PAGES,
        elements=elements,
        images=[],
        page_qa=[
            PageQA(page_number=p, ocr_confidence=1.0, parser_used="docling")
            for p in range(1, TOTAL_PAGES + 1)
        ],
        markdown="",
        heading_tree=[],
        parser_used="docling",
    )


def test_index_heading_run_does_not_explode_into_single_char_chunks() -> None:
    chunks = HierarchicalChunker().chunk(_index_book(), composites=[])

    # Healthy output is a few dozen chunks for ~200 pages of prose; the bug gave >100k.
    assert len(chunks) < 200, len(chunks)
    tiny = [c.content for c in chunks if len(c.content.strip()) < 10]
    assert not tiny, tiny[:10]
    # Section paths stay heading-sized; the bug glued the whole index into one path.
    longest_path = max(len(c.section_path) for c in chunks)
    assert longest_path < 400, longest_path
    # Every chunk of this all-paged document keeps a page number.
    assert all(c.page_number is not None for c in chunks)
    # The bio on the last page is still retrievable as a normal chunk.
    assert any("Sam Ellis has been personal coach" in c.content for c in chunks)


def test_ceiling_never_chops_to_characters_when_section_path_exhausts_budget() -> None:
    ceiling = TokenCeilingBackstop(counter=naive_counter, max_tokens=600)
    giant_path = "[Book > " + " > ".join(f"Heading number {n}" for n in range(300)) + "]"
    body = " ".join(["word"] * 1200)

    parts = ceiling.split_text(body, giant_path)

    assert parts, parts
    assert min(len(p) for p in parts) > 1
    # Pieces are budgeted on the content alone: a handful, not one per character/word.
    assert len(parts) < 20, len(parts)
    assert all(naive_counter(p) <= 600 for p in parts)


def test_index_label_stays_on_the_index_and_not_on_later_body_text() -> None:
    """VOR-1452: the bio after the index is body text, not part of the index."""
    chunks = HierarchicalChunker().chunk(_index_book(), composites=[])

    bio = [c for c in chunks if "Sam Ellis has been personal coach" in c.content]
    assert bio
    assert all("INDEX" not in c.section_path for c in bio), [c.section_path for c in bio]
    # The index entries never carry another label. Inside
    # a live index a title-case line ("Aaa, Arne 1") is index body text, no
    # longer a pending heading flushed as a chunk of its own, so the existing
    # back-matter index filter drops these page-numbered entries like the
    # lower-case ones (recorded as empty_after_normalize); any that survive
    # keep the INDEX label.
    entries = [c for c in chunks if "Aaa, Arne" in c.content or "Aab, Arne" in c.content]
    assert all("INDEX" in c.section_path for c in entries), [c.section_path for c in entries]
    assert not [c for c in chunks if "Arne" in c.section_path]
