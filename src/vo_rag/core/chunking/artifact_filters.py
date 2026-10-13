from __future__ import annotations

import math
import re
import unicodedata

_DISTRIBUTION_PATTERNS = (re.compile(r"\bthis file was downloaded from\b", re.I),)

_RIGHTS_PATTERNS = (
    re.compile(r"\ball rights reserved\b", re.I),
    re.compile(r"\bcopyright\b", re.I),
    re.compile(r"\bunauthori[sz]ed reprint\b", re.I),
    re.compile(r"\bno part of (?:this|the) (?:book|booklet|work|publication)\b", re.I),
    re.compile(r"\bmay be reproduced or transmitted\b", re.I),
)

_AUTHOR_BIO_PATTERNS = (
    re.compile(r"\bwho has written\b", re.I),
    re.compile(r"\bstarted his working life\b", re.I),
    re.compile(r"\bstarted her working life\b", re.I),
    re.compile(r"\bworked as a .*teacher\b", re.I),
    re.compile(r"\btechnical writer\b", re.I),
    re.compile(r"\bedited a .*magazine\b", re.I),
    re.compile(r"\bhas written more than\b", re.I),
)

_TECHNICAL_CONTEXT_PATTERNS = (
    re.compile(r"\baero(?:dynamic|dynamics)?\b", re.I),
    re.compile(r"\bair\s*flow\b", re.I),
    re.compile(r"\bairflow\b", re.I),
    re.compile(r"\bdrag\b", re.I),
    re.compile(r"\blift\b", re.I),
    re.compile(r"\bdownforce\b", re.I),
    re.compile(r"\btuft(?:s|ing)?\b", re.I),
    re.compile(r"\bpressure\b", re.I),
    re.compile(r"\bpitot\b", re.I),
    re.compile(r"\bspoiler\b", re.I),
    re.compile(r"\bwing\b", re.I),
    re.compile(r"\bundertray\b", re.I),
    re.compile(r"\bdiffuser\b", re.I),
    re.compile(r"\bride height\b", re.I),
    re.compile(r"\bthrottle[- ]stop\b", re.I),
    re.compile(r"\bcoefficient\b", re.I),
    re.compile(r"\bdiagram\b", re.I),
    re.compile(r"\bgraph\b", re.I),
    re.compile(r"\bfigure\b", re.I),
    re.compile(r"\btesting\b", re.I),
    re.compile(r"\bsensor\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|mph|km/h|kph|deg|degrees|psi|pa|%)\b", re.I),
)

_PORTRAIT_PATTERNS = (
    re.compile(r"\bportrait\b", re.I),
    re.compile(r"\bauthor\b", re.I),
    re.compile(r"\bbiograph(?:y|ical)\b", re.I),
    re.compile(r"\bsmiling\b", re.I),
)

_NONTECHNICAL_VISUAL_KINDS = frozenset({"cover", "logo", "portrait", "decorative", "blank"})

_NONTECHNICAL_VISUAL_PATTERNS = (
    re.compile(r"\bbook\s+cover\b", re.I),
    re.compile(r"\bfront\s+cover\b", re.I),
    re.compile(r"\bcover\s+(?:page|image|art)\b", re.I),
    re.compile(r"\b(?:publisher|company|brand|sponsor|team)\s+logo\b", re.I),
    re.compile(r"\blogo\b", re.I),
    re.compile(r"\b(?:author|person|profile)\s+(?:photo|portrait|headshot)\b", re.I),
    re.compile(r"\bdecorative\b", re.I),
    re.compile(r"\b(?:ornament|border|divider|flourish)\b", re.I),
    re.compile(r"\b(?:blank|mostly\s+blank)\s+page\b", re.I),
    re.compile(r"\b(?:barcode|qr\s+code)\b", re.I),
)

_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*(\d{1,4})\s*$")
_RUNNING_LINE_DIGITS_RE = re.compile(r"\d+")
_RUNNING_LINE_SPACE_RE = re.compile(r"\s+")
_RUNNING_LINE_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|#")
_RUNNING_LINE_RAW_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_RUNNING_LINE_MAX_CHARS = 160
_DIGIT_PLACEHOLDER = "#"
_RUNNING_LINE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}

# F13/F17/F26 — publisher-distributor download watermarks that contaminate the
# PDF text layer on figure pages. When this kind of text becomes preceding_text
# for a VLM call, the VLM has no real prose context and may anchor on the
# watermark itself. Phrase set derived from the 2026-05-25 batch-2 audit.
_PUBLISHER_WATERMARK_PATTERNS = (
    re.compile(r"\bdownloaded from sae\b", re.I),
    re.compile(r"\bdownloaded from jstor\b", re.I),
    re.compile(r"\bthis content downloaded from\b", re.I),
    re.compile(r"\bcopyright\s+\d{4}\s+cengage\b", re.I),
    re.compile(r"\bcengage learning\.\s*all rights reserved\b", re.I),
    re.compile(r"\bmay not be copied,?\s*scanned,?\s*or duplicated\b", re.I),
    re.compile(r"\bthis page intentionally left blank\b", re.I),
)


# F16 — lead-byte signatures for UTF-8-bytes-as-Latin-1 mojibake. When a UTF-8
# string is misinterpreted as Latin-1 (or CP1252), the C2 / C3 lead bytes appear
# as "Â" / "Ã" in the broken text. These almost never occur in clean prose
# (they would only appear in author names from a tiny set of orthographies), so
# their presence is a strong signal that the string is double-encoded.
_MOJIBAKE_SIGNATURES = ("Ã", "Â", "â\x80", "â€")


def fix_mojibake_text(text: str) -> str:
    """F16 — fix UTF-8/CP1252 double-encoded text (e.g. "HeiÃŸing" → "Heißing").

    The classic double-encoding chain is UTF-8 bytes interpreted as CP1252 (or
    Latin-1 for the lower 128 bytes). CP1252 covers more cases than Latin-1
    because high bytes 0x80–0x9F map to printable smart quotes / dashes in
    CP1252 (e.g., 0x9F = "Ÿ") but to control codes in Latin-1. Both round-trips
    are tried; the better-looking result wins.

    Returns the original string when no mojibake signature is present or every
    round-trip fails — never makes the input worse.
    """
    if not text:
        return text
    if not any(sig in text for sig in _MOJIBAKE_SIGNATURES):
        return text
    for codec in ("cp1252", "latin-1"):
        try:
            candidate = text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not any(sig in candidate for sig in _MOJIBAKE_SIGNATURES):
            return candidate
    return text


def is_publisher_download_watermark(text: str) -> bool:
    """F13/F17/F26 — true when text is dominated by a publisher download stamp.

    A single match against the watermark phrase set is enough; these phrases do
    not occur in body prose.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in _PUBLISHER_WATERMARK_PATTERNS)


def contains_distribution_artifact(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _DISTRIBUTION_PATTERNS)


def strip_distribution_artifact_tail(text: str) -> str:
    """``text`` without the sentences that carry a distribution watermark.

    Only the watermark's own sentence goes ("This file was
    downloaded from <site>."); the book's text before and after it
    is kept (cutting everything after the watermark would drop book text).
    """
    if not text:
        return ""
    if not contains_distribution_artifact(text):
        return text.strip()
    spans: list[tuple[int, int]] = []
    for pattern in _DISTRIBUTION_PATTERNS:
        for match in pattern.finditer(text):
            starts = [m.end() for m in _WATERMARK_SENTENCE_START.finditer(text, 0, match.start())]
            start = starts[-1] if starts else 0
            end_match = _WATERMARK_SENTENCE_END.search(text, match.end())
            spans.append((start, end_match.end() if end_match else len(text)))
    kept: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            kept.append(text[cursor:start])
        cursor = max(cursor, end)
    kept.append(text[cursor:])
    cleaned = re.sub(r"[ \t]+", " ", "".join(kept))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


#: A watermark's sentence starts after the previous sentence end or line break
#: and ends at the next one.
_WATERMARK_SENTENCE_START = re.compile(r"[.!?](?=\s)|\n")
_WATERMARK_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")


def strip_front_matter_rights_tail(text: str, pages: list[int] | None) -> str:
    if not text or not _is_early_pages(pages):
        return (text or "").strip()
    starts = [
        match.start()
        for pattern in (
            re.compile(r"\bthe right of .{0,160}? to be identified\b", re.I | re.S),
            re.compile(r"\ball rights reserved\b", re.I),
            re.compile(r"\bcopyright\b", re.I),
            re.compile(r"\bno part of (?:this|the) (?:book|booklet|work|publication)\b", re.I),
            re.compile(r"\btrademark notice\b", re.I),
            re.compile(r"\blibrary of congress cataloging\b", re.I),
            re.compile(r"\bnames:\s+[A-Z][A-Za-z'-]+", re.I),
        )
        for match in [pattern.search(text)]
        if match is not None
    ]
    if not starts:
        return text.strip()
    return _clean_fragment(text[: min(starts)])


def strip_page_furniture_text(
    text: str,
    pages: list[int | None] | None = None,
    *,
    strip_numeric_page_lines: bool = True,
) -> str:
    """Remove standalone page-number furniture without disturbing equations.

    Native PDFs often expose printed page numbers as ordinary narrative text.
    A page number line can bisect a sentence when page text is joined, which
    creates retrieval chunks ending in fragments such as "has 4". Keep the rule
    conservative: remove standalone numeric lines only when they plausibly
    match the current parsed page's printed-page offset. A numeric run beside
    mathematical notation is data, including a wrapped fraction denominator.
    """

    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return text.strip()

    nonblank_indexes = [idx for idx, line in enumerate(lines) if line.strip()]
    if not nonblank_indexes:
        return ""
    protected_numeric = _mathematical_numeric_lines(lines)
    cleaned: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            strip_numeric_page_lines
            and _PAGE_NUMBER_LINE_RE.match(stripped)
            and _is_probable_printed_page_number(stripped, pages)
            and idx not in protected_numeric
        ):
            continue
        cleaned.append(line)

    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _mathematical_numeric_lines(lines: list[str]) -> set[int]:
    """Protect numeric runs beside math notation in one bounded linear pass."""
    protected: set[int] = set()
    index = 0
    while index < len(lines):
        if not _PAGE_NUMBER_LINE_RE.fullmatch(lines[index].strip()):
            index += 1
            continue
        first = index
        while index < len(lines) and _PAGE_NUMBER_LINE_RE.fullmatch(lines[index].strip()):
            index += 1
        neighbors = lines[max(0, first - 1) : first] + lines[index : index + 1]
        if any(
            ch in "/=+*^" or unicodedata.category(ch) == "Sm" for line in neighbors for ch in line
        ):
            protected.update(range(first, index))
    return protected


def detect_running_lines(
    pages_text: list[str],
    *,
    min_fraction: float = 0.5,
    max_words: int = 8,
) -> set[str]:
    """Return normalized page-edge lines that recur across many document pages.

    Running headers/footers are positional and distinctive: only the first and
    last non-empty line of each page can qualify, bare stopword lines are kept,
    and candidates must contain digits or look like a title/header.
    """

    if not pages_text or max_words <= 0:
        return set()

    pages_by_line: dict[str, set[int]] = {}
    for page_index, page_text in enumerate(pages_text):
        page_candidates: set[str] = set()
        for raw_line in _page_edge_lines(page_text):
            normalized = _normalize_running_line(raw_line)
            if _is_running_line_candidate(raw_line, normalized, max_words=max_words):
                page_candidates.add(normalized)
        for candidate in page_candidates:
            pages_by_line.setdefault(candidate, set()).add(page_index)

    min_pages = max(2, math.ceil(len(pages_text) * min_fraction))
    return {normalized for normalized, pages in pages_by_line.items() if len(pages) >= min_pages}


def strip_running_lines(text: str, running_lines: set[str]) -> str:
    """Remove standalone lines whose normalized form matches a running-line set."""

    if not text:
        return ""
    if not running_lines:
        return text.strip()

    cleaned: list[str] = []
    for line in text.splitlines():
        normalized = _normalize_running_line(line)
        if normalized and normalized in running_lines:
            continue
        cleaned.append(line)

    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def has_book_technical_context(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _TECHNICAL_CONTEXT_PATTERNS)


def is_front_matter_rights_text(text: str, pages: list[int] | None) -> bool:
    if not _is_early_pages(pages):
        return False
    return any(pattern.search(text or "") for pattern in _RIGHTS_PATTERNS)


def is_front_matter_author_bio(text: str, pages: list[int] | None) -> bool:
    if not _is_early_pages(pages):
        return False
    return _matches_author_bio_signals(text)


def is_back_matter_author_bio(
    text: str,
    page_number: int | None,
    total_pages: int | None,
) -> bool:
    """F11 — author/back-matter portraits anchored in late-book bio sections.

    Triggers when the page is within the last 5 pages AND the surrounding text
    matches the author-bio signature set OR an explicit "About the Author" /
    "Acknowledgements" header. A 5-page tail is conservative on purpose —
    rejecting real figures in the back matter has a higher cost than missing a
    portrait, and back matter is almost universally the final few pages.
    """
    if not page_number or not total_pages or total_pages <= 0:
        return False
    if page_number <= total_pages - 5:
        return False
    return _matches_author_bio_signals(text) or _matches_explicit_back_matter_header(text)


def _matches_author_bio_signals(text: str) -> bool:
    bio_hits = sum(1 for pattern in _AUTHOR_BIO_PATTERNS if pattern.search(text or ""))
    if bio_hits >= 1:
        return True
    lowered = (text or "").lower()
    return "author" in lowered and "technical writer" in lowered


def _matches_explicit_back_matter_header(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(header in lowered for header in _BACK_MATTER_HEADERS)


_BACK_MATTER_HEADERS = (
    "about the author",
    "about the authors",
    "about the contributor",
    "acknowledgements",
    "acknowledgments",
)


def image_rejection_reason_from_context(
    *,
    page_number: int,
    nearby_text: str,
    description: str = "",
) -> str | None:
    combined = "\n".join(part for part in (nearby_text, description) if part)
    if contains_distribution_artifact(combined):
        return "distribution_artifact"
    if page_number <= 4:
        if is_front_matter_rights_text(combined, [page_number]) and not has_book_technical_context(
            combined
        ):
            return "front_matter_nontechnical_image"
        if is_front_matter_author_bio(combined, [page_number]) and not has_book_technical_context(
            combined
        ):
            return "front_matter_nontechnical_image"
        if _looks_like_author_portrait(description, combined):
            return "front_matter_nontechnical_image"
    return None


def visual_composite_skip_reason(
    *,
    page_number: int,
    description: str,
    context_text: str,
    total_pages: int | None = None,
    figure_kind: str | None = None,
    admitted: bool | None = None,
    rejection_reason: str | None = None,
) -> str | None:
    if admitted is False:
        return rejection_reason or "visual_admission_rejected"

    normalized_kind = (figure_kind or "").strip().lower()
    if normalized_kind in _NONTECHNICAL_VISUAL_KINDS:
        return f"nontechnical_visual_{normalized_kind}"

    combined = "\n".join(part for part in (context_text, description) if part)
    if contains_distribution_artifact(combined):
        return "distribution_artifact"
    if _looks_like_nontechnical_visual(description):
        return "nontechnical_visual_description"
    if page_number <= 4 and (
        _looks_like_author_portrait(description, combined)
        or (
            is_front_matter_rights_text(combined, [page_number])
            and not has_book_technical_context(combined)
        )
        or (
            is_front_matter_author_bio(combined, [page_number])
            and not has_book_technical_context(combined)
        )
    ):
        return "front_matter_nontechnical_image"
    if is_back_matter_author_bio(combined, page_number, total_pages) and not (
        has_book_technical_context(combined)
    ):
        return "back_matter_author_portrait"
    return None


def _looks_like_nontechnical_visual(description: str) -> bool:
    if not description:
        return False
    return any(pattern.search(description) for pattern in _NONTECHNICAL_VISUAL_PATTERNS)


def _is_early_pages(pages: list[int] | None) -> bool:
    real_pages = [page for page in (pages or []) if page is not None]
    return bool(real_pages) and min(real_pages) <= 6 and max(real_pages) <= 6


def _is_probable_printed_page_number(
    line: str,
    pages: list[int | None] | None,
) -> bool:
    match = _PAGE_NUMBER_LINE_RE.match(line)
    if match is None:
        return False
    real_pages = [page for page in (pages or []) if page is not None and page > 0]
    if not real_pages:
        return False
    value = int(match.group(1))
    # Books with front matter commonly have printed page numbers offset
    # 5-20 pages behind the PDF page number.
    return any(max(1, page - 30) <= value <= page + 2 for page in real_pages)


def normalize_running_line(line: str) -> str:
    """The key a running line is matched by (digits folded, case folded)."""
    return _normalize_running_line(line)


def _normalize_running_line(line: str) -> str:
    cleaned = line.strip().casefold()
    cleaned = _RUNNING_LINE_DIGITS_RE.sub("#", cleaned)
    return _RUNNING_LINE_SPACE_RE.sub(" ", cleaned).strip()


def _page_edge_lines(page_text: str) -> tuple[str, ...]:
    lines = [line for line in page_text.splitlines() if line.strip()]
    if not lines:
        return ()
    if len(lines) == 1:
        return (lines[0],)
    return (lines[0], lines[-1])


def _is_running_line_candidate(raw_line: str, normalized: str, *, max_words: int) -> bool:
    if not normalized or len(normalized) > _RUNNING_LINE_MAX_CHARS:
        return False
    if not any(ch.isalpha() for ch in normalized):
        return False
    tokens = _RUNNING_LINE_TOKEN_RE.findall(normalized)
    if not tokens or len(tokens) > max_words:
        return False
    if normalized.endswith(".") and len(tokens) > 3:
        return False
    if _is_stopword_only_running_line(tokens):
        return False
    return _is_distinctive_running_line(raw_line)


def _is_stopword_only_running_line(tokens: list[str]) -> bool:
    alpha_tokens = [part for part in tokens if part != _DIGIT_PLACEHOLDER]
    return bool(alpha_tokens) and all(token in _RUNNING_LINE_STOPWORDS for token in alpha_tokens)


def _is_distinctive_running_line(raw_line: str) -> bool:
    if any(ch.isdigit() for ch in raw_line):
        return True
    raw_tokens = _RUNNING_LINE_RAW_ALPHA_TOKEN_RE.findall(raw_line.strip())
    if len(raw_tokens) < 2:
        return False
    title_like = sum(1 for token in raw_tokens if token.istitle() or token.isupper())
    return title_like / len(raw_tokens) >= 0.8


def _looks_like_author_portrait(description: str, combined: str) -> bool:
    if not description:
        return False
    portrait_hits = sum(1 for pattern in _PORTRAIT_PATTERNS if pattern.search(description))
    lowered = combined.lower()
    has_author_context = "author" in lowered or "written by" in lowered
    return portrait_hits >= 1 and has_author_context and not has_book_technical_context(description)


def _clean_fragment(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(" \t\r\n-:;,.")
