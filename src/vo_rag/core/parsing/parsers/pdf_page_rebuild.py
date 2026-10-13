"""Find and splice PDF pages docling's pypdfium backend got wrong (VOR-1453).

Production parses PDFs with docling on the pypdfium2 text backend (VOR-1424:
docling_parse blew up to 20-58 GB RSS on glyph-dense books). On some pages
that backend hands docling overlapping text cells, and the layout-cluster
assembly then emits the page's text twice, interleaved and cropped mid-word
(seen on a biomechanics monograph, page 53). pypdfium2's own page
text for the same page is clean, and so is a docling_parse conversion.

This module is the pure half of the repair -- no docling, no pypdfium2:

- :func:`doubled_pages` finds the doubled pages from the parsed elements,
  measured against pypdfium2's plain text of the same pages;
- :func:`rebuild_ranges` turns the pages to rebuild into small page ranges
  that no multi-page docling item straddles;
- :func:`splice_pages` swaps a page's elements for rebuilt ones, keeping the
  original figures, element order and page numbers;
- :func:`missing_words` is the never-lose-text check on a rebuilt page range;
- :func:`coverage_shortfall` says when a whole parse covers too few pages to
  patch page by page.

``DoclingParser`` (``parsers/docling.py``) runs the conversions.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vo_rag.core.parsing.contract import (
    ElementType,
    PageQA,
    ParsedDocument,
    ParsedElement,
    element_id,
)
from vo_rag.core.parsing.parsers.docling_mapping import _build_heading_tree, _document_markdown

#: Letters per shingle when comparing parsed text with pypdfium2's page text.
#: Comparison is on lowercase letters only (spaces, punctuation, digits and
#: hyphens dropped) so the two texts' different word splitting ("o f" vs "of",
#: "rota- tional" vs "rotational") never reads as a difference.
SHINGLE_LETTERS = 24

#: A page is doubled when at least this many of its shingles occur more often
#: in its parsed text than in pypdfium2's text -- roughly two lines' worth.
#: Tuned on a corpus of sports-science parses (VOR-1453): a page doubled by the
#: pypdfium defect scores from ~50 (one repeated sentence) to 2,000+; the
#: pages below it are legitimate repeats the text layer holds differently
#: (the same caption phrase on two figures, a phrase recurring across a page
#: break with a running head between its halves).
MIN_DOUBLED_SHINGLES = 50

#: ... or when the parse holds at least ``MIN_EXTRA_WORDS`` words (4+
#: letters) more often than the text layer does, making up at least
#: ``MIN_EXTRA_WORD_SHARE`` of the page's words -- or
#: ``SUPPORTED_EXTRA_WORD_SHARE`` when ``SUPPORTING_SHINGLES`` repeated runs
#: back them up. The order-free word check catches doubled fragments shorter
#: than a shingle and pages whose text layer interleaves two columns line by
#: line. Tuned on the same corpus (VOR-1453): the legitimate pages it reaches
#: have 0-3 extra words with repeated runs, or under 10% extra words without.
MIN_EXTRA_WORDS = 5
MIN_EXTRA_WORD_SHARE = 0.10
SUPPORTING_SHINGLES = 15
SUPPORTED_EXTRA_WORD_SHARE = 0.05

#: docling merges a paragraph that runs over a page break into one element
#: filed under its first page, so a page's reference text is its own
#: pypdfium2 text plus the next pages' (a paragraph can skip a full-page
#: figure). Only a shingle already in that reference counts, so text that is
#: not in the text layer at all never scores.
REFERENCE_PAGES_AHEAD = 2

#: Most pages one docling_parse conversion may cover (memory bound: never
#: the whole book through docling_parse at once).
MAX_RANGE_PAGES = 4

#: A pypdfium2 page text with fewer letters than this is not a text page.
MIN_TEXT_PAGE_LETTERS = 40

#: A parse whose content reaches fewer than this share of the pages that
#: carry a text layer is "well below" coverage and is redone whole.
MIN_TEXT_PAGE_COVERAGE = 0.5

#: The parse step's own page-coverage gate (``_enforce_pdf_forward_path``).
GATE_MIN_PAGES = 10
GATE_MIN_COVERAGE = 0.10


_NON_LETTERS = re.compile(r"[^a-z]+")
_WORD = re.compile(r"[a-z]{4,}")
_ANY_WORD = re.compile(r"[a-z]{2,}")
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\s+(?=[a-z])")
# pdfium marks a soft hyphen at a line end with U+FFFE (sometimes U+00AD);
# the word continues on the next line.
_SOFT_HYPHEN = re.compile("[￾­]\\s*")
_NO_DETECT = frozenset({ElementType.TABLE, ElementType.IMAGE})


def letters(text: str) -> str:
    """Lowercase letters of ``text`` only, soft hyphens joined."""
    return _NON_LETTERS.sub("", _SOFT_HYPHEN.sub("", text).lower())


def _shingles(text: str) -> Iterable[str]:
    return (text[i : i + SHINGLE_LETTERS] for i in range(len(text) - SHINGLE_LETTERS + 1))


def _occurrences(text: str, shingle: str) -> int:
    """Overlapping occurrences (``str.count`` skips overlaps, which undercounts
    a periodic text such as a repeated short drill line)."""
    count = 0
    start = text.find(shingle)
    while start != -1:
        count += 1
        start = text.find(shingle, start + 1)
    return count


@dataclass(frozen=True)
class PageDoubling:
    """Why :func:`doubled_pages` flagged a page.

    ``shingles``: letter shingles the parse repeats beyond the text layer.
    ``extra_words``: words (4+ letters) the parse holds more often than the
    text layer. ``words``: the page's parsed words (4+ letters).
    """

    shingles: int
    extra_words: int
    words: int

    @property
    def doubled(self) -> bool:
        if self.shingles >= MIN_DOUBLED_SHINGLES:
            return True
        if self.extra_words < MIN_EXTRA_WORDS:
            return False
        share = (
            SUPPORTED_EXTRA_WORD_SHARE
            if self.shingles >= SUPPORTING_SHINGLES
            else MIN_EXTRA_WORD_SHARE
        )
        return self.extra_words >= share * self.words


def _words(text: str) -> list[str]:
    return _WORD.findall(_SOFT_HYPHEN.sub("", text).lower().replace("-", ""))


def doubled_pages(
    elements: Sequence[ParsedElement], page_texts: Sequence[str]
) -> dict[int, PageDoubling]:
    """The pages whose parsed text repeats pypdfium2's text of them.

    ``page_texts[i]`` is pypdfium2's plain text of page ``i + 1``; a page's
    reference is its own text plus the next ``REFERENCE_PAGES_AHEAD`` pages'.
    Tables and images are not measured. Two measures, either flags a page:

    - repeated runs: 24-letter shingles occurring more often in the parse
      than in the reference (and at least once there);
    - repeated words: words occurring more often in the parse than in the
      reference (and at least once there). Order-free, so it holds when
      pypdfium2 interleaves two columns line by line and breaks the runs,
      and it sees doubled fragments shorter than a shingle.

    Legitimately repetitive pages (a training program repeating "w-up B;
    ext T", a table's repeated cells) repeat in the text layer too and score
    nothing; so does text that is not in the text layer at all.
    """
    by_page: dict[int, list[str]] = defaultdict(list)
    for element in elements:
        page = element.page_number
        if page is None or element.element_type in _NO_DETECT or not element.text:
            continue
        by_page[page].append(element.text)

    reference_letters: dict[int, str] = {}
    reference_words: dict[int, Counter[str]] = {}

    def reference(page: int) -> tuple[str, Counter[str]]:
        if page not in reference_letters:
            reference_letters[page] = letters(page_texts[page - 1])
            reference_words[page] = Counter(_words(page_texts[page - 1]))
        return reference_letters[page], reference_words[page]

    flagged: dict[int, PageDoubling] = {}
    for page, texts in sorted(by_page.items()):
        if not 1 <= page <= len(page_texts):
            continue
        pages = range(page, min(len(page_texts), page + REFERENCE_PAGES_AHEAD) + 1)
        references = [reference(p) for p in pages]

        parsed_letters = [letters(text) for text in texts]
        parsed = Counter(shingle for text in parsed_letters for shingle in _shingles(text))
        shingles = 0
        for shingle, count in parsed.items():
            if count < 2:
                continue
            in_reference = sum(_occurrences(text, shingle) for text, _ in references)
            if 0 < in_reference < count:
                shingles += count - in_reference

        words = Counter(word for text in texts for word in _words(text))
        in_reference_words: Counter[str] = Counter()
        for _, counter in references:
            in_reference_words.update(counter)
        extra_words = sum(
            count - in_reference_words[word]
            for word, count in words.items()
            if 0 < in_reference_words[word] < count
        )

        doubling = PageDoubling(shingles, extra_words, sum(words.values()))
        if doubling.doubled:
            flagged[page] = doubling
    return flagged


def text_layer_pages(page_texts: Sequence[str]) -> set[int]:
    """Pages whose pypdfium2 text has at least ``MIN_TEXT_PAGE_LETTERS`` letters."""
    return {
        index + 1
        for index, text in enumerate(page_texts)
        if len(letters(text)) >= MIN_TEXT_PAGE_LETTERS
    }


def coverage_shortfall(covered: set[int], text_pages: set[int], total_pages: int) -> str | None:
    """Why a parse covering ``covered`` pages is too thin to patch page by
    page, or ``None`` when it is not.

    ``"page_coverage_gate"``: the parse step's own gate would fail it (10+
    pages, content on under 10%). ``"below_text_layer"``: content on under
    half of the pages that carry a text layer.
    """
    if total_pages >= GATE_MIN_PAGES and len(covered) / total_pages < GATE_MIN_COVERAGE:
        return "page_coverage_gate"
    if text_pages and len(covered & text_pages) < MIN_TEXT_PAGE_COVERAGE * len(text_pages):
        return "below_text_layer"
    return None


def rebuild_ranges(
    pages: Iterable[int],
    item_spans: Sequence[tuple[int, int]],
    *,
    max_pages: int = MAX_RANGE_PAGES,
) -> list[tuple[int, int]]:
    """Inclusive page ranges to re-convert for ``pages``.

    ``item_spans`` are the ``(first, last)`` pages of every docling item of
    the original parse that runs over a page break. A range grows until no
    such item straddles its edge: a merged paragraph is filed under its first
    page, so replacing only its second page would lose that page's top lines
    (and replacing only its first would repeat them). Ranges longer than
    ``max_pages`` are split -- both halves are rebuilt, so the split never
    meets original text.
    """
    wanted = set(pages)
    changed = True
    while changed:
        changed = False
        for first, last in item_spans:
            span = set(range(first, last + 1))
            if wanted & span and not span <= wanted:
                wanted |= span
                changed = True
    ranges: list[tuple[int, int]] = []
    for page in sorted(wanted):
        if ranges and page == ranges[-1][1] + 1 and page - ranges[-1][0] < max_pages:
            ranges[-1] = (ranges[-1][0], page)
        else:
            ranges.append((page, page))
    return ranges


def _all_words(text: str) -> list[str]:
    """Words of 2+ letters, soft hyphens and hyphen-plus-break joined."""
    joined = _SOFT_HYPHEN.sub("", text)
    joined = _HYPHEN_BREAK.sub("", joined)
    return _ANY_WORD.findall(joined.lower())


def missing_words(reference_texts: Sequence[str], rebuilt_texts: Sequence[str]) -> Counter[str]:
    """Words (2+ letters) pypdfium2's text of a page range holds more often
    than the rebuilt elements of that range: the never-lose-text check.

    Order-free on purpose: pypdfium2 interleaves two columns line by line and
    runs figure labels into the paragraph beside them, so its lines are not
    the rebuilt text's lines, but every word it holds must still be there.
    Seen on real books: docling_parse dropping the last word of a line
    ("has become a way of / making" -> "a way making").
    """
    reference = Counter(word for text in reference_texts for word in _all_words(text))
    rebuilt = Counter(word for text in rebuilt_texts for word in _all_words(text))
    short = {
        word: count - rebuilt[word] for word, count in reference.items() if count > rebuilt[word]
    }
    if not short:
        return Counter()
    # A word split or joined differently ("intelli- gent" / "intelligent",
    # "self- discipline") is not lost: count it in the letters instead.
    reference_letters = "|".join(letters(text) for text in reference_texts)
    rebuilt_letters = "|".join(letters(text) for text in rebuilt_texts)
    return Counter(
        {
            word: missing
            for word, missing in short.items()
            if _occurrences(rebuilt_letters, word) < _occurrences(reference_letters, word)
        }
    )


def page_words(text: str) -> set[str]:
    return set(_all_words(text))


def plain_text_elements(page: int, text: str) -> list[ParsedElement]:
    """pypdfium2's page text as one NARRATIVE_TEXT element (the fallback when
    docling_parse fails on a page); ``[]`` for a page with no text."""
    joined = " ".join(_SOFT_HYPHEN.sub("", text).split())
    if not letters(joined):
        return []
    return [
        ParsedElement(
            element_id=element_id(joined, page, 0),
            element_type=ElementType.NARRATIVE_TEXT,
            text=joined,
            page_number=page,
            metadata={"docling_label": "text", "source": "pypdfium2_page_text"},
        )
    ]


def splice_pages(
    parsed: ParsedDocument, replacements: dict[int, list[ParsedElement]]
) -> ParsedDocument:
    """``parsed`` with each page in ``replacements`` holding the given elements.

    The replacement elements take the place of the page's text, table and
    caption elements; the page's IMAGE elements (and their figure-store files)
    are kept and spread through the new elements at the same relative
    positions. Element ids, heading context, image element indexes, page QA
    counts, the markdown and the heading tree are then rebuilt for the whole
    document, exactly as ``docling_to_parsed_document`` builds them.
    """
    if not replacements:
        return parsed
    originals: dict[int, list[ParsedElement]] = defaultdict(list)
    for element in parsed.elements:
        if element.page_number in replacements:
            originals[element.page_number].append(element)

    # Each replaced page's block goes where the page's first element was, or,
    # for a page with no element (docling failed on it), before the first
    # element of a later page.
    anchors: dict[int, list[int]] = defaultdict(list)
    for page in sorted(replacements):
        index = next(
            (
                i
                for i, element in enumerate(parsed.elements)
                if element.page_number is not None and element.page_number >= page
            ),
            len(parsed.elements),
        )
        anchors[index].append(page)

    spliced: list[ParsedElement] = []
    for index in range(len(parsed.elements) + 1):
        for page in anchors.get(index, []):
            spliced.extend(_with_original_images(originals[page], replacements[page]))
        if index < len(parsed.elements) and parsed.elements[index].page_number not in replacements:
            spliced.append(parsed.elements[index])

    return _reindexed(parsed, spliced, set(replacements))


def _with_original_images(
    original: list[ParsedElement], rebuilt: list[ParsedElement]
) -> list[ParsedElement]:
    """``rebuilt`` (IMAGE elements dropped) with ``original``'s IMAGE elements
    at the same relative positions, in their original order."""
    text = [element for element in rebuilt if element.element_type != ElementType.IMAGE]
    images = [
        (position / len(original), element)
        for position, element in enumerate(original)
        if element.element_type == ElementType.IMAGE
    ]
    merged: list[ParsedElement] = []
    image_index = 0
    for position, element in enumerate(text):
        while image_index < len(images) and images[image_index][0] * len(text) <= position:
            merged.append(images[image_index][1])
            image_index += 1
        merged.append(element)
    merged.extend(element for _, element in images[image_index:])
    return merged


def _reindexed(
    parsed: ParsedDocument, elements: list[ParsedElement], changed_pages: set[int]
) -> ParsedDocument:
    heading_stack: list[str] = []
    heading_nodes: list[tuple[int, str]] = []
    rebuilt: list[ParsedElement] = []
    for idx, element in enumerate(elements):
        if element.element_type == ElementType.TITLE:
            level = max(1, element.section_depth)
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(element.text)
            heading_nodes.append((level, element.text))
            depth, title = level, element.text
        else:
            depth = len(heading_stack)
            title = heading_stack[-1] if heading_stack else ""
        key = (
            str(element.metadata.get("image_id", ""))
            if element.element_type == ElementType.IMAGE
            else element.text
        )
        rebuilt.append(
            element.model_copy(
                update={
                    "element_id": element_id(key, element.page_number, idx),
                    "section_depth": depth,
                    "section_title": title,
                }
            )
        )

    # ExtractedImage records and IMAGE elements are appended together by the
    # mapping and the splice keeps IMAGE elements in order, so they pair up.
    image_positions = [
        idx for idx, element in enumerate(rebuilt) if element.element_type == ElementType.IMAGE
    ]
    images = [
        image.model_copy(update={"element_index": position})
        for image, position in zip(parsed.images, image_positions, strict=True)
    ]

    text_counts: Counter[int] = Counter()
    table_counts: Counter[int] = Counter()
    for element in rebuilt:
        if element.page_number is None or element.element_type == ElementType.IMAGE:
            continue
        if element.element_type == ElementType.TABLE:
            table_counts[element.page_number] += 1
        else:
            text_counts[element.page_number] += 1
    known = {qa.page_number for qa in parsed.page_qa}
    page_qa = [
        qa.model_copy(
            update={
                "text_element_count": text_counts.get(qa.page_number, 0),
                "table_count": table_counts.get(qa.page_number, 0),
            }
        )
        if qa.page_number in changed_pages
        else qa
        for qa in parsed.page_qa
    ]
    page_qa.extend(
        PageQA(
            page_number=page,
            ocr_confidence=1.0,
            text_element_count=text_counts.get(page, 0),
            table_count=table_counts.get(page, 0),
            parser_used=parsed.parser_used,
        )
        for page in sorted(changed_pages - known)
    )
    page_qa.sort(key=lambda qa: qa.page_number)

    return parsed.model_copy(
        update={
            "elements": rebuilt,
            "images": images,
            "page_qa": page_qa,
            "markdown": _document_markdown(None, rebuilt),
            "heading_tree": _build_heading_tree(heading_nodes),
        }
    )
