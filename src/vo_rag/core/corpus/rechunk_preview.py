"""Pure metric logic for `vo-rag corpus rechunk-preview` (VOR-1452).

Everything here operates on plain ``ChunkView`` dicts -- no Garage, Qdrant or
Postgres import -- so it is unit-testable on hand-built chunk lists. The CLI
(`vo_rag.cli.corpus_rechunk`) does the I/O and calls these functions.

Metric definitions (per chunk list, live or new):
- ``frag``: chunks (any type) with ``len(content) < 300``.
- ``frag_text`` / ``tiny_text``: non-figure ("body") chunks under 300 / 150
  chars, as a share of body chunks.
- ``figures``: chunks flagged as a figure (has_image or a figure chunk_type).
- ``junk``: chunks whose leaf section title matches a back-matter label
  (index, bibliography, ...) *and* whose content is over 300 chars -- a
  short back-matter heading is normal; a long one usually means body text
  mis-labelled as back matter -- *unless* the chunk really is that back
  matter (``back_matter`` below).
- ``back_matter``: over-300-char chunks under a back-matter label whose
  content is that back matter, judged conservatively by content:
  a reference list under References/Bibliography/Sources/Further reading
  (:func:`is_reference_list`) or an index under Index
  (:func:`is_index_text`), or contents under Contents (:func:`is_contents_text`,
  VOR-1454). Every other long chunk under a back-matter label stays ``junk``,
  so body text filed under "Index" is still counted.
- ``garb``: chunks whose letters make up less than half their content
  (OCR/encoding garbage).
- ``fig_bleed``: figure chunks whose content leaks the "Preceding:"/
  "Following:" neighbour-window text that belongs only in ``embedding_text``
  (VOR-1452) -- the window's labels open a line; the book's own words ("Pay
  attention to the following: ...") are not a leak (VOR-1454).
- ``path_quality``: chunks whose section path has a junk component -- a
  library stamp, a byline, a sentence, a measurement or caption label, a
  title- or copyright-page line, OCR garble
  (``heading_filter.path_component_junk_reason``; VOR-1454). The share of
  chunks with a junk path: lower is better.

``frag``, ``junk``, ``back_matter``, ``garb`` and ``path_quality`` are rates
over every chunk; ``frag_text`` and ``tiny_text`` are rates over body
(non-figure) chunks only; ``figures`` and ``fig_bleed`` are raw counts.
:func:`total_counts` (via :func:`add_counts`) sums the underlying numerators
and denominators across books before taking the rate, so a multi-book total
is an exact micro-average, not an average of per-book rates.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any, TypedDict

from vo_rag.core.chunking.heading_filter import path_component_junk_reason
from vo_rag.core.corpus.tf_schema import content_sha256

#: Leaf section titles that mark back-matter (index, bibliography, ...).
#: Verbatim from the /tmp/tf/rechunk_preview.py prototype.
JUNK = re.compile(
    r"^(index|bibliography|references|about the (author|contributors)|titles in this series"
    r"|contents|table of contents|acknowledg|cip|copyright|quick order|dedication)",
    re.I,
)
#: Back-matter labels whose content can be checked (``back_matter``): reference
#: lists and indexes. Other back-matter labels (contents, acknowledgements, about
#: the author ...) are never exempt from ``junk``.
_REFERENCE_LABEL = re.compile(
    r"^(bibliography|references|reference list|sources|further reading|notes and references)",
    re.I,
)
_INDEX_LABEL = re.compile(r"^(subject |author |name )?index\b", re.I)
_CONTENTS_LABEL = re.compile(r"^(table of )?contents\b", re.I)
#: A contents entry's page: a dot leader (OCR often reads it as letters:
#: ".....c.ccscss") or a number closing an entry.
_DOT_LEADER = re.compile(r"(?:\.\s?){4,}|\u2026{2,}")
_NUMBER_TOKEN = re.compile(r"^\(?\d{1,4}[a-z]?[.,;:)]*$")
#: A contents page holds at least one page number per this many tokens.
_CONTENTS_TOKENS_PER_NUMBER = 12
#: A neighbour-window label opening a line of a figure chunk's content.
_WINDOW_LABEL = re.compile(r"(?m)^(?:Preceding|Following):")
#: A page reference in an index: "34", "22-23", "104-5", "329f", "xii".
_PAGE_REF = re.compile(r"(?<![\w.])(?:\d{1,4}[a-z]?(?:\s*[-–]\s*\d{1,4})?|[ivxl]{1,6})(?![\w.]\w)")
#: A prose sentence: 8+ words ending in terminal punctuation. Index and
#: reference entries are not sentences in this sense (a citation's title may
#: end in a period, so references use their own markers below).
_PROSE_SENTENCE = re.compile(r"(?:[A-Za-z'’-]+[,;:]?\s+){7,}[A-Za-z'’-]+[.!?](?:\s|$)")
#: Citation markers: a year (1900-2099, optionally "1998a", "(1998)"), "et al.",
#: page spans, volume/issue, a DOI/URL, a publisher/press, an ISBN.
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_CITATION_MARKER = re.compile(
    r"\bet al\b|\bpp?\.\s*\d|\bvol\.|\bno\.\s*\d|\bdoi\b|https?://|\bisbn\b|\bpress\b"
    r"|\bpublish(?:ers?|ing)\b|\bjournal\b|\bed(?:s|itor|itors)?\.|\d+\s*\(\d+\)\s*[:,]",
    re.I,
)


def is_index_text(content: str) -> bool:
    """Whether ``content`` reads as a book index ("finishing / 32", "abrasion 45").

    Conservative: page references make up at least one token in seven, and
    the text holds no prose sentence (8+ words ending in ``.!?``). Body text
    with numbers in it ("3 sets of 100 m at 90 percent") fails one or both.
    """
    tokens = content.split()
    if len(tokens) < 4:
        return False
    refs = sum(1 for token in tokens if _PAGE_REF.fullmatch(token.strip(",;.:()")))
    return refs * 7 >= len(tokens) and not _PROSE_SENTENCE.search(content)


def is_reference_list(content: str) -> bool:
    """Whether ``content`` reads as a reference list (citations).

    Conservative: at least three citation signals (years, "et al.", page
    spans, volumes, journals, publishers, DOIs/URLs) and at least one per 40
    words, so an essay paragraph mentioning one study is not exempt.
    """
    words = len(content.split())
    if words == 0:
        return False
    signals = len(_YEAR.findall(content)) + len(_CITATION_MARKER.findall(content))
    return signals >= 3 and signals * 40 >= words


def is_contents_text(content: str) -> bool:
    """Whether ``content`` reads as a table of contents (VOR-1454).

    Conservative: dot leaders on at least three entries, or a page number at
    least every twelfth token; and no prose sentence. A foreword filed under
    "Contents" fails.
    """
    tokens = content.split()
    if len(tokens) < 4 or _PROSE_SENTENCE.search(content):
        return False
    if len(_DOT_LEADER.findall(content)) >= 3:
        return True
    numbers = sum(1 for token in tokens if _NUMBER_TOKEN.match(token))
    return numbers * _CONTENTS_TOKENS_PER_NUMBER >= len(tokens)


def back_matter_kind(label: str, content: str) -> str | None:
    """``"references"`` / ``"index"`` / ``"contents"`` when a chunk under
    ``label`` really is that back matter by its content, else ``None``."""
    if _REFERENCE_LABEL.match(label) and is_reference_list(content):
        return "references"
    if _INDEX_LABEL.match(label) and is_index_text(content):
        return "index"
    if _CONTENTS_LABEL.match(label) and is_contents_text(content):
        return "contents"
    return None


#: A "real word": 3+ lowercase letters. Used for the coverage word count.
WORD = re.compile(r"[a-z]{3,}")
#: Sentence boundary: whitespace after a `.`/`!`/`?`.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: A live sentence must have at least this many words to count toward
#: sentence coverage -- short fragments ("See Fig. 3.") are noise.
_MIN_SENTENCE_WORDS = 6
#: A chunk is "fragmentary" under this many characters.
_FRAG_CHARS = 300
#: A body chunk is "tiny" under this many characters.
_TINY_CHARS = 150
#: A chunk is "garbled" when its letter share falls below this.
_GARB_LETTER_RATIO = 0.5


class ChunkView(TypedDict):
    """The slice of a chunk the metrics need: content, section and figure-ness."""

    content: str
    section_path: str
    figure: bool


def junk_path(path: str) -> bool:
    """Whether a section path has a junk component below the document root
    (``path_quality``)."""
    # One bracket each side: a heading may end in its own "[74]".
    inner = path[1:-1] if path.startswith("[") and path.endswith("]") else path
    components = inner.split(" > ")[1:] if inner else []
    return any(path_component_junk_reason(component) is not None for component in components)


def leaf(path: str) -> str:
    """The last segment of a ``" > "``-joined section path, brackets stripped."""
    return path.strip("[]").split(" > ")[-1].strip() if path else ""


def chunk_view_from_retrieval(chunk: Any) -> ChunkView:
    """``ChunkView`` from a freshly-built ``RetrievalChunk`` (the "new" side).

    Duck-typed (``getattr``, not an import of ``RetrievalChunk``) so this
    module stays free of the chunking package's heavier import chain; the
    CLI passes real ``RetrievalChunk`` instances.
    """
    chunk_type = str(getattr(chunk, "chunk_type", "") or "")
    has_image = bool(getattr(chunk, "has_image", False))
    return ChunkView(
        content=str(getattr(chunk, "content", "") or ""),
        section_path=str(getattr(chunk, "section_path", "") or ""),
        figure=has_image or chunk_type in ("figure_composite", "figure"),
    )


def chunk_view_from_payload(payload: dict[str, Any]) -> ChunkView:
    """``ChunkView`` from a live Qdrant point's payload (the "live" side)."""
    return ChunkView(
        content=str(payload.get("content") or ""),
        section_path=str(payload.get("section_path") or ""),
        figure=bool(payload.get("has_image")),
    )


@dataclass(frozen=True)
class RawCounts:
    """Numerators and denominators behind one side's (live or new) metrics.

    All fields default to 0 so ``RawCounts()`` is the identity element for
    :func:`add_counts` -- the accumulator a multi-book total starts from.
    """

    chunks: int = 0
    body: int = 0
    figures: int = 0
    frag: int = 0
    frag_text: int = 0
    tiny_text: int = 0
    junk: int = 0
    back_matter: int = 0
    garb: int = 0
    fig_bleed: int = 0
    junk_path: int = 0


def raw_counts(chunks: Sequence[ChunkView]) -> RawCounts:
    """Raw numerators/denominators for one chunk list (live or new, one book)."""
    body_chunks = [c for c in chunks if not c["figure"]]
    frag = sum(1 for c in chunks if len(c["content"]) < _FRAG_CHARS)
    frag_text = sum(1 for c in body_chunks if len(c["content"]) < _FRAG_CHARS)
    tiny_text = sum(1 for c in body_chunks if len(c["content"]) < _TINY_CHARS)
    labelled = [
        c for c in chunks if JUNK.match(leaf(c["section_path"])) and len(c["content"]) > _FRAG_CHARS
    ]
    back_matter = sum(
        1 for c in labelled if back_matter_kind(leaf(c["section_path"]), c["content"]) is not None
    )
    junk = len(labelled) - back_matter
    garb = sum(1 for c in chunks if _is_garbled(c["content"]))
    fig_bleed = sum(1 for c in chunks if c["figure"] and _WINDOW_LABEL.search(c["content"]))
    junk_paths = sum(1 for c in chunks if junk_path(c["section_path"]))
    return RawCounts(
        chunks=len(chunks),
        body=len(body_chunks),
        figures=len(chunks) - len(body_chunks),
        frag=frag,
        frag_text=frag_text,
        tiny_text=tiny_text,
        junk=junk,
        back_matter=back_matter,
        garb=garb,
        fig_bleed=fig_bleed,
        junk_path=junk_paths,
    )


def _is_garbled(content: str) -> bool:
    if not content:
        return False
    letters = sum(1 for ch in content if ch.isalpha())
    return (letters / len(content)) < _GARB_LETTER_RATIO


def add_counts(a: RawCounts, b: RawCounts) -> RawCounts:
    """Field-wise sum of two :class:`RawCounts` (order-independent)."""
    return RawCounts(**{f.name: getattr(a, f.name) + getattr(b, f.name) for f in fields(RawCounts)})


def rates(counts: RawCounts) -> dict[str, int | float]:
    """The printable metric dict (rates + raw counts) for one ``RawCounts``.

    ``frag``/``junk``/``back_matter``/``garb`` are shares of ``chunks``; ``frag_text``/
    ``tiny_text`` are shares of ``body`` (non-figure chunks); ``figures``
    and ``fig_bleed`` are left as raw counts, matching the prototype's
    output shape.
    """
    n = counts.chunks or 1
    body = counts.body or 1
    return {
        "chunks": counts.chunks,
        "frag": round(counts.frag / n, 3),
        "frag_text": round(counts.frag_text / body, 3),
        "tiny_text": round(counts.tiny_text / body, 3),
        "figures": counts.figures,
        "junk": round(counts.junk / n, 3),
        "back_matter": round(counts.back_matter / n, 3),
        "garb": round(counts.garb / n, 3),
        "fig_bleed": counts.fig_bleed,
        "path_quality": round(counts.junk_path / n, 4),
    }


def metrics(chunks: Sequence[ChunkView]) -> dict[str, int | float]:
    """Convenience: ``rates(raw_counts(chunks))``, for tests and one-off use."""
    return rates(raw_counts(chunks))


def words(chunks: Sequence[ChunkView]) -> Counter[str]:
    """Word counts over non-figure chunk content, lower-cased."""
    counter: Counter[str] = Counter()
    for chunk in chunks:
        if not chunk["figure"]:
            counter.update(WORD.findall(chunk["content"].lower()))
    return counter


def word_coverage(live: Sequence[ChunkView], new: Sequence[ChunkView]) -> float:
    """Share of live (non-figure) word occurrences still present in ``new``."""
    live_words, new_words = words(live), words(new)
    total = sum(live_words.values()) or 1
    covered = sum(min(count, new_words[word]) for word, count in live_words.items())
    return round(covered / total, 4)


def _fold(text: str) -> str:
    return " ".join(text.split())


def _sentences(chunks: Sequence[ChunkView]) -> set[str]:
    """Distinct live sentences (>= 6 words, whitespace-folded, lower-cased).

    A set, not a list: robust to a live sentence appearing in more than one
    chunk (duplication the new chunker may have collapsed).
    """
    out: set[str] = set()
    for chunk in chunks:
        if chunk["figure"]:
            continue
        for sentence in _SENTENCE_SPLIT.split(_fold(chunk["content"])):
            if len(sentence.split()) >= _MIN_SENTENCE_WORDS:
                out.add(sentence.lower())
    return out


def sentence_coverage(live: Sequence[ChunkView], new: Sequence[ChunkView]) -> float:
    """Share of distinct live sentences (>= 6 words) still found in ``new``'s text.

    Substring match against the whole (whitespace-folded, lower-cased) new
    text, not per-chunk -- a sentence the new chunker split across a chunk
    boundary differently than live still counts as covered.
    """
    new_text = " ".join(_fold(c["content"]).lower() for c in new if not c["figure"])
    live_sentences = _sentences(live)
    covered = sum(1 for sentence in live_sentences if sentence in new_text)
    return round(covered / (len(live_sentences) or 1), 4)


def _input_sentences(texts: Sequence[str]) -> set[str]:
    """Distinct sentences (>= 6 words) of the chunker's input, one element at a time.

    Split per element, so a sentence never spans two elements; a trailing
    word broken by a line-end hyphen ("golf-") is left off, since the chunker
    joins it to the next element's first word.
    """
    out: set[str] = set()
    for text in texts:
        for sentence in _SENTENCE_SPLIT.split(_fold(text)):
            words = sentence.split()
            if words and words[-1].endswith("-"):
                words = words[:-1]
            # A body sentence has letters: a run of chart ticks or week numbers
            # ("9 10 11 12 13 14") is page furniture the chunker drops.
            if len(words) >= _MIN_SENTENCE_WORDS and any(ch.isalpha() for ch in sentence):
                out.add(" ".join(words).lower())
    return out


#: How far back a chunk's overlap prefix (a copy of the previous chunk's tail)
#: is looked for when output chunks are joined.
_OVERLAP_SEARCH_CHARS = 4000
#: A seam repeat shorter than this is not taken for an overlap prefix.
_OVERLAP_PROBE_CHARS = 20


def _joined_output(new: Sequence[ChunkView]) -> str:
    """The non-figure output text in order, each chunk's overlap prefix removed.

    A chunk the chunker split mid-sentence starts with a copy of the previous
    chunk's tail; plain joining would repeat it and break the sentence across
    the seam. Only text that repeats the previous chunk's end is removed.
    """
    parts: list[str] = []
    for chunk in new:
        if chunk["figure"]:
            continue
        text = _fold(chunk["content"]).lower()
        if parts and len(text) >= _OVERLAP_PROBE_CHARS:
            tail = parts[-1][-_OVERLAP_SEARCH_CHARS:]
            probe = text[:_OVERLAP_PROBE_CHARS]
            start = tail.find(probe)
            while start >= 0:
                overlap = len(tail) - start
                if text.startswith(tail[start:]):
                    text = text[overlap:].lstrip()
                    break
                start = tail.find(probe, start + 1)
        parts.append(text)
    return " ".join(part for part in parts if part)


def input_coverage(
    input_texts: Sequence[str], new: Sequence[ChunkView], ocr_dropped: Sequence[str] = ()
) -> float:
    """Share of the chunker's input sentences found in its output.

    ``input_texts`` is the body text the chunker was given
    (``HierarchicalChunker.body_input_texts``: page furniture it skips on
    purpose is already left out). A sentence counts as covered when it is a
    substring of the new non-figure text, joined in order with each chunk's
    overlap prefix removed (``_joined_output``) -- figure chunks are left out
    so a figure's page context cannot stand in for lost body text. Sentences the
    OCR gate dropped (found in ``ocr_dropped``) are not counted either way.
    Independent of the live chunks, so a re-parse's text changes do not
    show here: only what the chunker lost of what it was given.
    """
    new_text = _joined_output(new)
    dropped_text = " ".join(_fold(text).lower() for text in ocr_dropped)
    sentences = [
        sentence
        for sentence in _input_sentences(input_texts)
        if not (dropped_text and sentence in dropped_text)
    ]
    if not sentences:
        return 1.0  # nothing given, nothing to lose
    covered = sum(1 for sentence in sentences if sentence in new_text)
    return round(covered / len(sentences), 4)


def coverage_baseline(
    live: Sequence[ChunkView], is_noise: Callable[[str], bool]
) -> list[ChunkView]:
    """Live chunks that coverage is measured against: body chunks judged noise are left out.

    The new chunk step drops OCR noise on purpose, so a garbled live chunk's
    "sentences" missing from the new text is the gate working, not text lost.
    Figure chunks are kept (coverage ignores them anyway).
    """
    return [c for c in live if c["figure"] or not is_noise(c["content"])]


def carry_over(live: Sequence[ChunkView], new: Sequence[ChunkView]) -> float:
    """Share of ``new`` chunks whose ``content_sha256`` key already exists in ``live``.

    ``content_sha256`` (VOR-1450) folds whitespace/Unicode, so this survives
    a re-parse's whitespace reflow -- it is the same key TF tags are looked
    up by.
    """
    live_keys = {content_sha256(c["content"]) for c in live}
    total = len(new) or 1
    reused = sum(1 for c in new if content_sha256(c["content"]) in live_keys)
    return round(reused / total, 3)


__all__ = [
    "JUNK",
    "WORD",
    "ChunkView",
    "RawCounts",
    "add_counts",
    "back_matter_kind",
    "carry_over",
    "coverage_baseline",
    "input_coverage",
    "is_contents_text",
    "is_index_text",
    "junk_path",
    "is_reference_list",
    "chunk_view_from_payload",
    "chunk_view_from_retrieval",
    "leaf",
    "metrics",
    "rates",
    "raw_counts",
    "sentence_coverage",
    "word_coverage",
    "words",
]
