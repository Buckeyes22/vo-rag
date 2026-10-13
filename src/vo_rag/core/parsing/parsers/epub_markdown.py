"""EPUB + pandoc-rendered Markdown -> ParsedElement[].

Two real defects this function fixes:

1. pandoc renders GFM emphasis markers around styled chapter titles
   ("## **ADJUSTABLE DAMPERS**"); the markers poison downstream
   ``is_degenerate_heading`` filters (math_operator / ocr_garbage_punct),
   which drop every heading and let the chunker page-bucket the book.
2. EPUBs whose chapter titles are <p>/<div> (not <h1>-<h6>) produce ZERO
   ``#`` lines from pandoc. The author's hierarchy survives only in the
   EPUB ``toc_tree``. We recover it by matching plain body lines against
   TOC entries (consume-once, in document order) and emitting TITLE
   elements at the TOC's own depth.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict, deque
from collections.abc import Iterator
from urllib.parse import unquote

import structlog

from vo_rag.core.parsing.contract import ElementType, ParsedElement, element_id

log = structlog.get_logger(__name__)

# GFM inline emphasis/code markers pandoc wraps around styled chapter titles
# ("## **ADJUSTABLE DAMPERS**", "### *The System Fluid*", "# `Glossary`"). Left
# in the heading text they trip `is_degenerate_heading` (math_operator /
# ocr_garbage_punct), so every heading gets dropped and the chunker page-buckets
# the book. Underscores are intentionally NOT stripped: pandoc gfm uses '*' for
# emphasis, and '_' can be a literal in identifier-style headings.
_EMPHASIS_RE = re.compile(r"\*{1,3}|~~|`")


def _strip_inline_emphasis(text: str) -> str:
    return _EMPHASIS_RE.sub("", text).strip()


# A line that is one internal markdown link: "[**1 Introduction**](#toc.xhtml#chap-1)".
_WHOLE_LINE_LINK_RE = re.compile(r"^\[(?P<label>.+)\]\((?P<target>#[^)\s]*)\)$")


# A TOC-page line: a whole-line link, or the tail of one pandoc wrapped
# ("things](#chapter001.xhtml#Sec_1.2.1)"). Links inside body sentences are not.
_TOC_PAGE_LINE_RE = re.compile(r"^(?:\[.*|[^\[]*)\]\((?P<target>#[^)\s]*)\)$")

# Raw HTML/SVG pandoc passes through with no text of its own, e.g. the
# ``<svg ...>`` / ``</svg>`` wrapper of a fixed-layout page image (pandoc drops
# the ``<image>`` inside). Emitting these as narrative text is noise in any
# book, and on a page-scan EPUB it hides the book from ``is_image_only``
# (VOR-1436: The Science of Car Racing, 52 markup "paragraphs").
_MARKUP_ONLY_RE = re.compile(r"(?:\s*(?:</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>|<!--.*?-->))+")
_UNTERMINATED_TAG_RE = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>]*)?")


def _is_markup_only(stripped: str) -> bool:
    return bool(_MARKUP_ONLY_RE.fullmatch(stripped) or _UNTERMINATED_TAG_RE.fullmatch(stripped))


# Pandoc's gfm writer emits this line for a table it cannot express (VOR-1427).
_TABLE_PLACEHOLDER = "[TABLE]"

# Pipe-table structure with no cell text: `|`, the `|---|:--:|` delimiter row.
_TABLE_SKELETON_RE = re.compile(r"[|:\-\s]+")


# Plain lines matching a TOC entry within this many non-empty lines of TOC-page
# links on both sides are unlinked entries of the TOC page itself, not headings.
# The widest gap between links inside a real TOC page seen was 44 lines.
TOC_PAGE_WINDOW_LINES = 60


def _link_key(target: str) -> str:
    """``file-basename#fragment`` of a pandoc link target or a TOC href."""
    file_part, _, fragment = target.strip().lstrip("#").partition("#")
    base = file_part.rsplit("/", 1)[-1]
    return f"{base}#{fragment}" if fragment else base


def _norm_title(text: str) -> str:
    """Normalize a heading/title to word tokens for robust TOC matching.

    Letters of every script count: ASCII-only tokens reduced CJK titles to
    their digits, so "项目2 汽车分类与代码" and "任务2 发电机" both became "2"
    (VOR-1427, de9420be5179)."""
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


# One EPUB anchor id, written by epub_anchors.lua as its own paragraph (ids are
# percent-encoded). GFM has no identifier syntax, so without the filter pandoc
# drops the span at every spine-file start and every source id (VOR-1427).
_ANCHOR_RE = re.compile(r"⟦vorag-anchor:(?P<id>[^⟧\s]*)⟧")
_ANCHOR_LINE_RE = re.compile(r"^⟦vorag-anchor:[^⟧\s]*⟧[ \t]*(?:\n(?:[ \t]*\n)?|\Z)", re.MULTILINE)


def strip_anchor_sentinels(markdown: str) -> str:
    """Drop the anchor lines epub_anchors.lua adds; the rest is untouched."""
    if "⟦vorag-anchor:" not in markdown:
        return markdown
    return _ANCHOR_LINE_RE.sub("", markdown).rstrip("\n")


def _split_anchor(line: str) -> tuple[str | None, str]:
    """(decoded anchor id, "") for a line that is exactly one anchor, else
    (None, the stripped line). The filter always writes an anchor as its own
    paragraph, so sentinel text inside a line is book text, not an anchor. A
    byte-order mark (U+FEFF, left at the top of some spine files) is not text:
    as a line of its own it sat between a chapter's anchor and its heading."""
    stripped = line.replace("\ufeff", "").strip()
    if not stripped.startswith("⟦vorag-anchor:"):
        return None, stripped
    anchor = _ANCHOR_RE.fullmatch(stripped)
    if anchor is None:
        return None, stripped
    return unquote(anchor.group("id")), ""


# GFM blockquote markers ("> ", nested "> > "). pandoc escapes a literal ">"
# that starts a paragraph, so outside code a leading ">" is always a marker.
# Four leading spaces make an indented code block (pandoc's gfm writer indents
# class-less <pre>), where ">>> " and "> " are prompts, not markers.
_BLOCKQUOTE_RE = re.compile(r"^ {0,3}(?:>[ \t]?)+")
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*$")


def _fence_open(stripped: str) -> tuple[str, int] | None:
    """Return the delimiter of a valid CommonMark fenced-code opener."""
    match = _FENCE_OPEN_RE.fullmatch(stripped)
    if match is None:
        return None
    fence = match.group("fence")
    if fence[0] == "`" and "`" in match.group("info"):
        return None
    return fence[0], len(fence)


def _fence_close(stripped: str, fence: tuple[str, int]) -> bool:
    """Whether a line closes the matching fenced-code delimiter."""
    match = _FENCE_CLOSE_RE.fullmatch(stripped)
    return (
        match is not None
        and match.group("fence")[0] == fence[0]
        and len(match.group("fence")) >= fence[1]
    )


def _logical_lines(markdown: str) -> Iterator[tuple[str, str | None, str]]:
    """(line text, anchor id or None, stripped text) per markdown line, with
    blockquote markers removed so quoted content reads like any other line
    (VOR-1427, d076890761f1). Inside a code block only the opening fence's
    own markers are removed: ">" there is code. Every pass that numbers lines
    reads them through here, so their ordinals agree."""
    fence: tuple[str, int] | None = None
    code_prefix = ""
    for raw_line in markdown.splitlines():
        if fence is not None:
            if raw_line.startswith(code_prefix):
                text = raw_line[len(code_prefix) :]
            elif code_prefix and raw_line.strip() == code_prefix.strip():
                text = ""
            else:
                text = raw_line
            anchor_id, stripped = _split_anchor(text)
            if anchor_id is not None:
                anchor_id, stripped = None, text.strip()
            if _fence_close(text, fence):
                fence = None
        else:
            quote = _BLOCKQUOTE_RE.match(raw_line)
            prefix = quote.group(0) if quote else ""
            text = raw_line[len(prefix) :]
            anchor_id, stripped = _split_anchor(text)
            opening = _fence_open(text)
            if opening is not None:
                fence, code_prefix = opening, prefix
        yield text, anchor_id, stripped


def _anchor_key(anchor_id: str, toc_files: set[str]) -> str:
    """``_link_key`` of a pandoc id. pandoc 3.1 writes file-scoped ids as
    ``c1.xhtml#s1``; later versions write ``c1.xhtml_s1``."""
    key = _link_key(anchor_id)
    if "#" in key or key in toc_files:
        return key
    for file_key in toc_files:
        if key.startswith(file_key + "_"):
            return f"{file_key}#{key[len(file_key) + 1 :]}"
    return key


# ---------------------------------------------------------------------------
# Printed contents lists (VOR-1427): books whose navigation TOC has at most one
# entry (calibre's "Start -> titlepage.xhtml") but which print a contents list.
# ---------------------------------------------------------------------------

# A line that opens a printed contents list, after markdown/emphasis stripping.
_CONTENTS_HEADING_RE = re.compile(
    r"(?:table\s+of\s+contents?|contents|índice|indice|inhalt|inhaltsverzeichnis|sommaire)\s*:?",
    re.IGNORECASE,
)
# Markdown decoration around a contents entry or a body heading line.
_LINE_DECORATION_RE = re.compile(r"^(?:#{1,6}\s+|\\?[-*+•]\s+)")
_BULLET_RE = re.compile(r"^\\?[-*+•]\s+")
_LABEL_WORD_RE = re.compile(r"^(?:chapter|capítulo|capitulo|part|parte|kapitel)\s+", re.IGNORECASE)
# Leading numbering: "1", "1.", "1.-", "1 –", "1.1", "1\.", a roman numeral
# with a separator and a space ("I. ", "IV — ", "VII: "), or a capital letter
# with "." or ")" ("A.", "B)"). "X-Men" and "E-commerce" keep their start.
_LEADING_NUMBER_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s*[.):\-–—]*\s*"
    r"|[IVXLCDM]+(?:[.)]|\s*[:\-–—])\s+"
    r"|[A-Z][.)]\s*)"
)
# Dot leaders before a trailing page number ("Title ........ 17").
_LEADER_CHARS = " \t.·…_"
# A numbered line in a leading run: "1. Title", "1.  Title", "1\. Title", "1) Title".
_NUMBERED_LINE_RE = re.compile(r"^(?P<n>\d{1,3})\s*(?:\\?[.)])?\s+\S")
_RULE_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")

# Entries are short lines; a longer line is prose and ends the block.
CONTENTS_ENTRY_MAX_CHARS = 100
# contents_key ignores lines longer than this: no heading is that long.
CONTENTS_KEY_MAX_CHARS = 4 * CONTENTS_ENTRY_MAX_CHARS
CONTENTS_BLOCK_MAX_LINES = 200
# The block ends at the first non-entry line once it has this many entries.
CONTENTS_MIN_ENTRIES = 3
# At least this many distinct entries must match body lines after the block.
CONTENTS_MIN_BODY_MATCHES = 3
# A key found this often after the block is a running header, not a chapter.
CONTENTS_RUNNING_HEADER_MIN = 3
# A leading numbered run: at least this many numbered lines 1, 2, 3, ...
# starting within the first LEADING_RUN_START_LINES non-empty lines and before
# the first prose line longer than LEADING_RUN_PROSE_CHARS; its entries must
# match the body in document order, with at most LEADING_RUN_MAX_SKIPS
# entries out of order. A "Key takeaways: 1. ... 5." list inside chapter one
# fails both (VOR-1427).
LEADING_RUN_MIN_NUMBERED = 5
LEADING_RUN_START_LINES = 30
LEADING_RUN_PROSE_CHARS = 200
LEADING_RUN_MAX_SKIPS = 1


def _clean_contents_text(line: str) -> str:
    """A contents line or body line without markdown decoration, emphasis and
    backslash escapes, whitespace collapsed."""
    text = _LINE_DECORATION_RE.sub("", line.strip())
    text = _strip_inline_emphasis(text).replace("\\", "")
    return " ".join(text.split())


def _strip_trailing_page(text: str) -> str:
    """``text`` without a trailing page number and the dot leaders before it.
    Plain rstrip calls, no regex: a leader run can be tens of thousands long."""
    body = text.rstrip("0123456789")
    return text if body == text else body.rstrip(_LEADER_CHARS)


def contents_key(line: str) -> str:
    """Matching key of a printed contents entry or a body heading line: no
    leading "Chapter"/"Part" word, numbering, trailing page number, dot
    leaders, emphasis or punctuation. "Chapter 1 – X", "1 – X", "*1* X" and
    "1.X." share one key. Lines longer than CONTENTS_KEY_MAX_CHARS have none."""
    if len(line) > CONTENTS_KEY_MAX_CHARS:
        return ""
    text = _strip_trailing_page(_clean_contents_text(line))
    text = _LABEL_WORD_RE.sub("", text)
    text = _LEADING_NUMBER_RE.sub("", text, count=1)
    key = _norm_title(text)
    return key if any(ch.isalpha() for ch in key) else ""


def is_contents_heading(line: str) -> bool:
    return bool(_CONTENTS_HEADING_RE.fullmatch(_clean_contents_text(line)))


def _is_neutral_line(line: str) -> bool:
    """Lines that neither extend nor end a contents block: rules, images,
    markup, and lines with no letters (a bare bullet, a page number)."""
    return bool(
        _RULE_RE.match(line)
        or line.startswith("![")
        or _is_markup_only(line)
        or not any(ch.isalpha() for ch in line)
    )


def _is_entry_line(line: str) -> bool:
    return (
        len(line) <= CONTENTS_KEY_MAX_CHARS
        and len(_clean_contents_text(line)) <= CONTENTS_ENTRY_MAX_CHARS
        and bool(contents_key(line))
    )


# (first ordinal, last ordinal, [(depth, title, key)]) of a printed contents block.
_ContentsBlock = tuple[int, int, list[tuple[int, str, str]]]


def _entry(line: str) -> tuple[int, str, str]:
    depth = 2 if _BULLET_RE.match(line) else 1
    return depth, _clean_contents_text(line), contents_key(line)


def _heading_block(lines: list[tuple[int, str]]) -> _ContentsBlock | None:
    """Entries after the first "Contents" line: up to the first non-entry line
    once there are CONTENTS_MIN_ENTRIES entries, a repeat of the first entry
    (the body starting), or CONTENTS_BLOCK_MAX_LINES lines."""
    start = next((i for i, (_, line) in enumerate(lines) if is_contents_heading(line)), None)
    if start is None:
        return None
    entries: list[tuple[int, str, str]] = []
    last = lines[start][0]
    for ordinal, line in lines[start + 1 : start + 1 + CONTENTS_BLOCK_MAX_LINES]:
        if _is_neutral_line(line):
            continue
        if not _is_entry_line(line):
            if len(entries) >= CONTENTS_MIN_ENTRIES:
                break
            continue
        if entries and contents_key(line) == entries[0][2]:
            break
        entries.append(_entry(line))
        last = ordinal
    return (lines[start][0], last, entries) if entries else None


def _leading_run_block(lines: list[tuple[int, str]]) -> _ContentsBlock | None:
    """A run of numbered lines 1, 2, 3, ... near the start of the book, before
    any long prose line, with bullet sub-entries, rules and at most one other
    short line between them."""
    for start, (first_ordinal, first_line) in enumerate(lines[:LEADING_RUN_START_LINES]):
        if len(first_line) > LEADING_RUN_PROSE_CHARS and not _is_neutral_line(first_line):
            return None
        numbered = _NUMBERED_LINE_RE.match(first_line)
        if numbered is None or numbered.group("n") != "1" or not _is_entry_line(first_line):
            continue
        entries = [_entry(first_line)]
        expected, count, others, last = 2, 1, 0, first_ordinal
        for ordinal, line in lines[start + 1 : start + CONTENTS_BLOCK_MAX_LINES]:
            if _is_neutral_line(line):
                continue
            if not _is_entry_line(line) or contents_key(line) == entries[0][2]:
                break
            numbered = _NUMBERED_LINE_RE.match(line)
            if numbered is not None:
                if int(numbered.group("n")) != expected:
                    break
                expected, count, others = expected + 1, count + 1, 0
            elif not _BULLET_RE.match(line):
                others += 1
                if others > 1:
                    break
            entries.append(_entry(line))
            last = ordinal
        if count >= LEADING_RUN_MIN_NUMBERED:
            return first_ordinal, last, entries
    return None


def _printed_contents_entries(
    lines: list[tuple[int, str]],
) -> tuple[int, list[tuple[int, str, str]]]:
    """(last ordinal of the block, entries worth matching) of a book's printed
    contents list, or (-1, []) when there is none or too few entries match body
    lines after it. Entries whose key recurs as a running header are dropped."""
    for find, ordered in ((_heading_block, False), (_leading_run_block, True)):
        block = find(lines)
        if block is None:
            continue
        _, block_end, entries = block
        wanted = {key for _, _, key in entries}
        # Only lines markdown_to_elements can turn into a TITLE count.
        body_keys = [
            (ordinal, key)
            for ordinal, line in lines
            if ordinal > block_end and not line.startswith(("- ", "* ", "!", "|"))
            for key in (contents_key(line),)
            if key in wanted
        ]
        seen: dict[str, int] = defaultdict(int)
        for _, key in body_keys:
            seen[key] += 1
        usable = [
            entry for entry in entries if 0 < seen.get(entry[2], 0) < CONTENTS_RUNNING_HEADER_MIN
        ]
        if len({key for _, _, key in usable}) < CONTENTS_MIN_BODY_MATCHES:
            continue
        if ordered and _out_of_order(usable, body_keys) > LEADING_RUN_MAX_SKIPS:
            continue
        return block_end, usable
    return -1, []


def _out_of_order(entries: list[tuple[int, str, str]], body_keys: list[tuple[int, str]]) -> int:
    """How many ``entries`` match the body (consume-once, in document order)
    earlier than an entry listed before them."""
    queues: dict[str, deque[int]] = defaultdict(deque)
    for index, (_, _, key) in enumerate(entries):
        queues[key].append(index)
    matched_at: dict[int, int] = {}
    for ordinal, key in body_keys:
        if queues.get(key):
            matched_at[queues[key].popleft()] = ordinal
    skips, latest = 0, -1
    for index in range(len(entries)):
        at = matched_at.get(index)
        if at is None:
            continue
        if at < latest:
            skips += 1
        latest = max(latest, at)
    return skips


def _body_lines(markdown: str) -> list[tuple[int, str]]:
    """(non-empty-line ordinal, stripped line) outside code blocks, numbered
    exactly as ``markdown_to_elements`` numbers them."""
    out: list[tuple[int, str]] = []
    ordinal = -1
    fence: tuple[str, int] | None = None
    for line, _, stripped in _logical_lines(markdown):
        if not stripped:
            continue
        ordinal += 1
        if fence is None:
            fence = _fence_open(line)
            if fence is not None:
                continue
        elif _fence_close(line, fence):
            fence = None
            continue
        if fence is None:
            out.append((ordinal, stripped))
    return out


def markdown_to_elements(markdown: str, toc_tree: list[dict[str, object]]) -> list[ParsedElement]:
    """Convert pandoc Markdown output into normalized parser elements.

    Headings come from two sources: pandoc '#' lines AND the EPUB table of
    contents. Many EPUBs style chapter titles as <p>/<div> rather than
    <h1>-<h6>, so pandoc emits NO '#' headings — the author's hierarchy survives
    only in ``toc_tree``. We recover it by matching plain body lines against TOC
    entries (consume-once, in document order) and emitting TITLE elements at the
    TOC's own depth, so the chunker builds real chapters instead of fabricating
    ``pp. X-Y`` page buckets.
    """

    # TOC entries by index; every consumer (title line, back-link, anchor) takes
    # an entry at most once through ``consumed``.
    entries: list[tuple[int, str]] = []
    consumed: list[bool] = []
    # Consume-once queue of entry indices keyed by normalized title, so a
    # recurring running-header line can't spam the heading tree and matches land
    # in document order (the first occurrence is the real chapter break).
    toc_lookup: dict[str, deque[int]] = defaultdict(deque)
    # Some EPUBs link every body heading back to the TOC page
    # ("[**1 Introduction**](#toc.xhtml#chap-1)") while the TOC page links
    # forward to the entries' own hrefs. A whole-line link to anything other
    # than a TOC href is such a heading; without this, only the TOC page's
    # plain lines matched and the book's last one titled every chunk
    # (VOR-1427). Back-links are matched without consuming, so a TOC-page line
    # that took the entry first cannot hide the real heading.
    toc_hrefs: set[str] = set()
    toc_first: dict[str, tuple[int, str]] = {}
    # Files of the TOC entries with each title. A contents page may link each
    # entry to a spot inside the entry's own file ("[1 Introduction]
    # (#index_split_005.html#filepos42442)" for the file-level href
    # "index_split_005.html"): a forward link, not a back-link (VOR-1427).
    title_files: dict[str, set[str]] = defaultdict(set)
    # Anchor matching: exact ``file#fragment`` key, and per file the entries in
    # TOC order for the file-start fallback.
    href_entries: dict[str, list[int]] = defaultdict(list)
    file_entries: dict[str, list[int]] = defaultdict(list)
    entry_keys: list[str] = []
    for node in toc_tree or []:
        title = str(node.get("title") or "").strip()
        key = _norm_title(_strip_inline_emphasis(title))
        href = str(node.get("href") or "").strip()
        href_key = _link_key(href) if href else ""
        if href_key:
            toc_hrefs.add(href_key)
        if not title:
            continue
        raw_depth = node.get("depth", 1)
        depth = raw_depth if isinstance(raw_depth, int) else 1
        index = len(entries)
        entries.append((max(1, depth), title))
        consumed.append(False)
        entry_keys.append(href_key)
        if key:
            toc_lookup[key].append(index)
            toc_first.setdefault(key, (max(1, depth), title))
        if href_key:
            title_files[key].add(href_key.partition("#")[0])
            href_entries[href_key].append(index)
            file_entries[href_key.partition("#")[0]].append(index)
    toc_files = {href_key.partition("#")[0] for href_key in toc_hrefs}

    # A printed contents list stands in for a missing TOC: its entries match
    # body lines after the block by ``contents_key``, consume-once, in order.
    contents_end = -1
    contents_lookup: dict[str, deque[int]] = defaultdict(deque)
    # Gated like epub.py's contents-page fallback: at most one TOC href.
    if sum(1 for node in toc_tree or [] if str(node.get("href") or "").strip()) <= 1:
        contents_end, printed = _printed_contents_entries(_body_lines(markdown))
        for depth, title, key in printed:
            contents_lookup[key].append(len(entries))
            entries.append((depth, title))
            consumed.append(False)
            entry_keys.append("")

    def take(title_key: str) -> tuple[int, str] | None:
        """Consume the next not-yet-consumed TOC entry with this title."""
        queue = toc_lookup.get(title_key)
        while queue:
            index = queue.popleft()
            if not consumed[index]:
                consumed[index] = True
                return entries[index]
        return None

    def take_printed(line: str, line_ordinal: int) -> tuple[int, str] | None:
        """Consume the next printed-contents entry this body line matches."""
        if not contents_lookup or line_ordinal <= contents_end:
            return None
        queue = contents_lookup.get(contents_key(line))
        while queue:
            index = queue.popleft()
            if not consumed[index]:
                consumed[index] = True
                return entries[index]
        return None

    def has_unconsumed(title_key: str) -> bool:
        queue = toc_lookup.get(title_key)
        while queue and consumed[queue[0]]:
            queue.popleft()
        return bool(queue)

    # Non-empty-line ordinals of TOC-page link lines, and every anchor key in
    # the book. Anchor-only lines do not count as lines, so they cannot widen
    # the gaps between TOC-page links. ``gap_breaks[j]`` marks the gap before
    # ``toc_page_lines[j]`` as body: it holds an anchor that opens a TOC entry
    # and a line of text, so a chapter started between two contents pages
    # (a stray id on the page itself has no text of its own).
    toc_page_lines: list[int] = []
    gap_breaks: list[bool] = []
    present_anchor_keys: set[str] = set()

    def into_entry_file(target: str, label: str) -> bool:
        """A link to a spot (not a TOC href) inside the file of the TOC entry
        named by its label: "[1 Introduction](#index_split_005.html#filepos42442)"
        for the file-level href "index_split_005.html"."""
        key = _anchor_key(target, toc_files)
        if key in toc_hrefs:
            return False
        label_key = _norm_title(_strip_inline_emphasis(label))
        return key.partition("#")[0] in title_files.get(label_key, ())

    if toc_hrefs:
        ordinal = 0
        gap_anchor = gap_text = False
        for _, anchor_id, candidate in _logical_lines(markdown):
            if anchor_id is not None:
                anchor_key = _anchor_key(anchor_id, toc_files)
                present_anchor_keys.add(anchor_key)
                gap_anchor = gap_anchor or anchor_key in toc_hrefs or anchor_key in toc_files
            if not candidate:
                continue
            toc_link = _TOC_PAGE_LINE_RE.match(candidate)
            whole_link = _WHOLE_LINE_LINK_RE.match(candidate)
            if (toc_link and _anchor_key(toc_link.group("target"), toc_files) in toc_hrefs) or (
                whole_link
                and into_entry_file(whole_link.group("target"), whole_link.group("label"))
            ):
                toc_page_lines.append(ordinal)
                gap_breaks.append(gap_anchor and gap_text)
                gap_anchor = gap_text = False
            else:
                gap_text = True
            ordinal += 1

    def toc_neighbours(line_ordinal: int) -> tuple[bool, bool]:
        """Whether a TOC-page line lies within TOC_PAGE_WINDOW_LINES before and
        after ``line_ordinal`` with no body gap (``gap_breaks``) in between."""
        k = bisect.bisect_left(toc_page_lines, line_ordinal)
        own = k < len(toc_page_lines) and toc_page_lines[k] == line_ordinal
        after_k = k + 1 if own else k
        before = (
            k > 0
            and line_ordinal - toc_page_lines[k - 1] <= TOC_PAGE_WINDOW_LINES
            # Past the last TOC-page line there is no "after" to pair with.
            and not (k < len(gap_breaks) and gap_breaks[k])
        )
        after = (
            after_k < len(toc_page_lines)
            and toc_page_lines[after_k] - line_ordinal <= TOC_PAGE_WINDOW_LINES
            and not gap_breaks[after_k]
        )
        return before, after

    def in_contents_cluster(line_ordinal: int) -> bool:
        """Any line between TOC-page lines of one cluster, or a TOC-page line
        at either end of a cluster of two or more."""
        before, after = toc_neighbours(line_ordinal)
        if before and after:
            return True
        k = bisect.bisect_left(toc_page_lines, line_ordinal)
        own = k < len(toc_page_lines) and toc_page_lines[k] == line_ordinal
        return own and (before or after)

    def links_back(target: str, label: str, line_ordinal: int) -> bool:
        """A whole-line link to a spot that is not a TOC href: a body heading
        linking back to the TOC page or to itself. A link into the file of the
        TOC entry with its label, among other TOC-page lines, is a contents
        line instead (d076890761f1)."""
        if _anchor_key(target, toc_files) in toc_hrefs:
            return False
        return not (into_entry_file(target, label) and in_contents_cluster(line_ordinal))

    def inside_toc_page(line_ordinal: int) -> bool:
        before, after = toc_neighbours(line_ordinal)
        return before and after

    def anchor_entry(key: str) -> int | None:
        """The TOC entry an anchor opens: the exact ``file#fragment`` entry
        first. A file-start anchor of a file with no file-level TOC entry falls
        back to the file's first TOC entry when that entry's own fragment id
        never appears (pandoc drops ids on ``<p>``), so it cannot pre-empt an
        anchor further down the file."""
        exact = href_entries.get(key)
        if exact:
            return next((i for i in exact if not consumed[i]), None)
        if "#" in key or not file_entries.get(key):
            return None
        first = file_entries[key][0]
        if consumed[first] or entry_keys[first] in present_anchor_keys:
            return None
        return first

    elements: list[ParsedElement] = []
    current_depth = 0
    current_title = ""
    code_fence: tuple[str, int] | None = None
    code_lines: list[str] = []

    # Normalized title of the TOC entry the last anchor opened, while that
    # anchor's TITLE is still the latest one. A heading, back-link or plain line
    # repeating it is the chapter's own heading, however many label, epigraph or
    # image lines sit in between ("CHAPTER 1" / "# Introduction"): dropped once.
    open_anchor_title: str | None = None
    # Titles of entries an anchor consumed: a later back-link to one of them
    # must not fall back to the first TOC occurrence and open a second section.
    anchor_taken: set[str] = set()

    def emit_title(text: str, depth: int, *, from_anchor: bool = False) -> None:
        nonlocal current_depth, current_title, open_anchor_title
        # TOC titles keep the EPUB's number/title padding ("1.2.1     Classification").
        text = " ".join(text.split())
        current_depth = depth
        current_title = text
        open_anchor_title = _norm_title(text) if from_anchor else None
        metadata: dict[str, object] = {"markdown_depth": depth}
        if from_anchor:
            # is_image_only ignores these: a page-scan TOC lists every page.
            metadata["toc_anchor"] = True
        elements.append(
            ParsedElement(
                element_id=element_id(text, None, len(elements)),
                element_type=ElementType.TITLE,
                text=text,
                section_depth=depth,
                section_title=text,
                metadata=metadata,
            )
        )

    # Rows of the pipe table being read. A GFM table is consecutive pipe lines;
    # emitting each row as its own TABLE shattered tables into one-row chunks
    # and turned calibre spacer tables (`<td/>`) into blank ones (VOR-1438).
    table_rows: list[str] = []

    def flush_table() -> None:
        table_text = "\n".join(table_rows)
        table_rows.clear()
        if not _TABLE_SKELETON_RE.sub("", table_text).strip():
            return
        elements.append(
            ParsedElement(
                element_id=element_id(table_text, None, len(elements)),
                element_type=ElementType.TABLE,
                text=table_text,
                section_depth=current_depth,
                section_title=current_title,
            )
        )

    def anchor_duplicate(title_key: str) -> bool:
        nonlocal open_anchor_title
        if not title_key or title_key != open_anchor_title:
            return False
        open_anchor_title = None
        return True

    def open_anchor(anchor_id: str, next_ordinal: int) -> None:
        if not toc_hrefs or inside_toc_page(next_ordinal):
            return
        key = _anchor_key(anchor_id, toc_files)
        index = anchor_entry(key)
        if index is None:
            return
        # Entries sharing one href (3cedb67ba6d0 lists chapters 43 and 44 at
        # "#p432") all open here, in TOC order: the last one owns the text.
        # A file-level key opens only its first unused entry: the others are
        # placed by their own title lines further down the file.
        shared = [i for i in href_entries.get(key, []) if not consumed[i]] if "#" in key else []
        for opened in shared or [index]:
            consumed[opened] = True
            depth, clean_title = entries[opened]
            title = _strip_inline_emphasis(clean_title) or clean_title
            anchor_taken.add(_norm_title(title))
            emit_title(title, depth, from_anchor=True)

    line_ordinal = -1
    placeholders_dropped = 0
    for line, anchor_id, stripped in _logical_lines(markdown):
        if stripped:
            line_ordinal += 1

        is_table_row = code_fence is None and stripped.startswith("|") and stripped.endswith("|")
        if table_rows and not is_table_row:
            flush_table()

        # An anchor line (after the flush: it also ends an open table).
        if anchor_id is not None:
            open_anchor(anchor_id, line_ordinal + 1)
            continue

        if code_fence is not None and _fence_close(line, code_fence):
            code_text = "\n".join(code_lines).strip()
            if code_text:
                elements.append(
                    ParsedElement(
                        element_id=element_id(code_text, None, len(elements)),
                        element_type=ElementType.CODE_BLOCK,
                        text=code_text,
                        section_depth=current_depth,
                        section_title=current_title,
                    )
                )
            code_lines = []
            code_fence = None
            continue
        elif code_fence is None and (opening := _fence_open(line)) is not None:
            code_fence = opening
            code_lines = []
            continue

        if code_fence is not None:
            code_lines.append(line)
            continue

        if not stripped or _is_markup_only(stripped):
            continue

        # "[TABLE]" is pandoc's gfm placeholder for a table it could not write;
        # epub_anchors.lua rewrites those tables, so any left carry no text and
        # mean a table shape the filter does not handle yet (logged below).
        if stripped == _TABLE_PLACEHOLDER:
            placeholders_dropped += 1
            continue

        if stripped.startswith("#"):
            hashes = len(stripped) - len(stripped.lstrip("#"))
            heading_text = _strip_inline_emphasis(stripped[hashes:].strip())
            # "# [Title](#content.xhtml#toc5)": a heading that links back to the
            # TOC page (a forward link to a TOC href stays as it was).
            heading_link = _WHOLE_LINE_LINK_RE.match(heading_text)
            if (
                heading_link
                and toc_hrefs
                and _anchor_key(heading_link.group("target"), toc_files) not in toc_hrefs
            ):
                heading_text = _strip_inline_emphasis(heading_link.group("label"))
            if heading_text and not anchor_duplicate(_norm_title(heading_text)):
                entry = take(_norm_title(heading_text)) or take_printed(heading_text, line_ordinal)
                if entry is not None:
                    depth, clean_title = entry
                    emit_title(_strip_inline_emphasis(clean_title) or heading_text, depth)
                else:
                    emit_title(heading_text, hashes)
            continue

        if stripped.startswith(("- ", "* ")):
            text = stripped[2:].strip()
            elements.append(
                ParsedElement(
                    element_id=element_id(text, None, len(elements)),
                    element_type=ElementType.LIST_ITEM,
                    text=text,
                    section_depth=current_depth,
                    section_title=current_title,
                )
            )
            continue

        if stripped.startswith("!") and "](" in stripped:
            elements.append(
                ParsedElement(
                    element_id=element_id(stripped, None, len(elements)),
                    element_type=ElementType.IMAGE,
                    text=stripped,
                    section_depth=current_depth,
                    section_title=current_title,
                )
            )
            continue

        if is_table_row:
            table_rows.append(stripped)
            continue

        link = _WHOLE_LINE_LINK_RE.match(stripped)
        if (
            link
            and toc_hrefs
            and links_back(link.group("target"), link.group("label"), line_ordinal)
        ):
            key = _norm_title(_strip_inline_emphasis(link.group("label")))
            known = toc_first.get(key)
            if known is not None:
                if anchor_duplicate(key):
                    continue
                # Prefer the entry this heading consumes (repeated titles such as
                # "Exercises" sit at different depths); once they are all taken,
                # fall back to the first occurrence -- unless an anchor took one,
                # in which case the chapter already has its TITLE.
                entry = take(key)
                if entry is None and key in anchor_taken:
                    continue
                depth, clean_title = entry or known
                emit_title(_strip_inline_emphasis(clean_title) or clean_title, depth)
                continue

        # Plain text line: recover a heading if it matches a not-yet-consumed TOC
        # entry (the EPUB authored its chapter title as a styled paragraph, so no
        # '#' survived). Otherwise it is ordinary narrative text.
        plain_key = _norm_title(stripped)
        if anchor_duplicate(plain_key):
            continue
        if has_unconsumed(plain_key) and not inside_toc_page(line_ordinal):
            entry = take(plain_key)
            if entry is not None:
                depth, clean_title = entry
                emit_title(_strip_inline_emphasis(clean_title) or clean_title, depth)
                continue
        printed_entry = take_printed(stripped, line_ordinal)
        if printed_entry is not None:
            emit_title(printed_entry[1], printed_entry[0])
            continue

        elements.append(
            ParsedElement(
                element_id=element_id(stripped, None, len(elements)),
                element_type=ElementType.NARRATIVE_TEXT,
                text=stripped,
                section_depth=current_depth,
                section_title=current_title,
            )
        )

    if table_rows:
        flush_table()

    if placeholders_dropped:
        log.warning("epub_table_placeholder_dropped", count=placeholders_dropped)

    if code_fence is not None and code_lines:
        code_text = "\n".join(code_lines).strip()
        if code_text:
            elements.append(
                ParsedElement(
                    element_id=element_id(code_text, None, len(elements)),
                    element_type=ElementType.CODE_BLOCK,
                    text=code_text,
                    section_depth=current_depth,
                    section_title=current_title,
                )
            )

    return elements
