"""Degenerate-heading filter for HierarchicalChunker.

The parser classifies elements as `ElementType.TITLE` based on visual cues
(font size, position) — but it doesn't validate that the TEXT content is a
meaningful heading. Result: in an audit of sampled books, 96% had
degenerate `section_title` values like:

- Bare numerics ("1", "2") — chapter/page numbers misread as headings
- Copyright disclaimers
- Publisher download watermarks (SAE, JSTOR, Cengage)
- "This page intentionally left blank" boilerplate
- OCR garbage ("Q \\norm", "Ti 2 mn", "ee \\nae WORK \\nS HOP \\nBR")
- Spaced-out display-font book titles ("T H E  H A N D B O O K  O F")

This module classifies heading-candidate text and returns a `reject_reason`
slug for downstream observability when the text isn't a real heading. The
chunker uses it at two sites:

1. AT STORAGE: skip adding degenerate text to the heading-tree dict
2. AT LOOKUP: when picking the deepest heading, skip degenerate entries and
   fall back to the next-best (defense in depth)
"""

from __future__ import annotations

import re
from typing import Final

# Phrases that indicate the text is a publisher watermark, DRM notice, or
# distributor download stamp — never a real section heading.
# Pattern: case-insensitive substring match.
_WATERMARK_PHRASES: Final[tuple[str, ...]] = (
    "downloaded from sae",
    "downloaded from jstor",
    "this content downloaded from",
    "copyright ©",
    "copyright cengage",
    "all rights reserved",
    "may not be copied, scanned",
    "this page intentionally left blank",
    "intentionally left blank",
    "for promotional use only",
    "not for distribution",
    "preview only",
)

# Publisher marketing slogans that some ebooks tag as headings (e.g. the
# O'Reilly back-matter ad page). A heading is a slogan when it IS the phrase
# (case/whitespace/trailing punctuation aside) or starts with it and names the
# publisher -- "How to Get the Most Out of Your GPU" is a real heading.
_SLOGAN_PHRASES: Final[tuple[str, ...]] = (
    "learn from experts. become one yourself",
    "o'reilly online learning",
    "get the most out of",
    "visit us at",
)

# Regex patterns for degenerate heading text shapes.
_DEGENERATE_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # Bare numeric ("1", "23", "5.")
    (re.compile(r"^\s*\d+\.?\s*$"), "numeric_only"),
    # Roman numeral page markers
    (re.compile(r"^\s*[ivxlcdm]+\s*$", re.I), "roman_only"),
    # Pure punctuation / OCR garbage (no word characters at all)
    (re.compile(r"^[^\w]*$"), "no_word_chars"),
    # Very short (<= 2 chars) — too short to be a real heading
    (re.compile(r"^.{0,2}$"), "too_short"),
    # Page-number-like prefix with no content after ("Page 5", "p. 12")
    (re.compile(r"^\s*(page|p\.?)\s*\d+\s*$", re.I), "page_marker_only"),
    # Markdown/EPUB TOC link rendered as a heading: "[7.9 Exercises](#chapter007.xhtml#Sec_7.9)"
    (re.compile(r"^\s*\[[^\]]+\]\([^)]*\)\s*$"), "markdown_link"),
    # ...or one wrapped over two lines: "[23.3.2 Distance-based: Euclidean" / "Problem](#...)"
    (re.compile(r"^\s*(?:[^\[\]]*\]\([^)]*\)|\[[^\]]*)\s*$"), "markdown_link"),
    # Spaced-out display-font book titles ("T H E  B O O K") — single chars
    # separated by spaces with at least 4 such tokens
    (re.compile(r"^(?:\s*[A-Za-z]\s+){4,}[A-Za-z]\s*$"), "spaced_display_font"),
    # OCR-garbage heuristic: contains 3+ consecutive non-alphanumeric chars
    # in the middle (excluding common punctuation like comma/period/dash/quotes)
    (re.compile(r"[^\w\s,.\-\"'()&/]{3,}"), "ocr_garbage_punct"),
)

# Maximum allowed character ratio of digits — section headings rarely consist
# mostly of digits. Tuned to permit "Chapter 12: Foo" but reject "1.2.3.4.5".
_MAX_DIGIT_RATIO: Final[float] = 0.5

# Rule: math/relational operators — real headings don't contain bare operators.
# Catches: "= 1.2 x 10^-3", "+ M.,rGxYgr", "+--- VR/CI,f/10", "SPOILER > O.J"
_MATH_OPERATOR_RE: Final[re.Pattern[str]] = re.compile(r"[=<>±×÷∑∫√]|(?<!\w)[+*^](?!\w)|/\d|\d/")

# Rule: alpha-ratio floor — headings must be mostly alphabetic characters.
# Threshold tuned to permit "15.3.4.2 The Marlowe…" but tighten filtering.
_MIN_ALPHA_RATIO: Final[float] = 0.55

# Rule: word-like fraction — headings must contain a minimum density of real
# words.  A token is "word-like" if it has ≥3 alphabetic chars AND contains at
# least one vowel (aeiouy, case-insensitive).
_VOWEL_RE: Final[re.Pattern[str]] = re.compile(r"[aeiouy]", re.IGNORECASE)
# Last code point of the Latin Extended-B block: letters above it are another
# script (Greek, Cyrillic, CJK, ...), where the vowel rule does not apply.
_LATIN_MAX_CODEPOINT: Final[int] = 0x024F
_MIN_WORD_FRACTION: Final[float] = 0.5

# Numbered section headings ("3.2.1.3 Prior", "5.1.3.1 Units of KL"): the ratio
# and word-density rules judge the text after the section number, which is
# structure rather than content -- counting it rejected ~180 real headings of
# one book and left the previous heading on their chunks (VOR-1427). The prefix
# may carry margin line numbers the parser merged in ("41 31.4.3 GP-SSM"). In
# such a heading, function words and acronyms count as words, and one-character
# math symbols ("Derivation of q ( θ )") are not counted at all.
_SECTION_NUMBER_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)+(?=\S)")
_ACRONYM_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9-]*[A-Z0-9]s?$")
_FUNCTION_WORDS: Final[frozenset[str]] = frozenset(
    {"a", "an", "as", "at", "by", "in", "is", "of", "on", "or", "to", "vs", "vs."}
)


_PUBLISHER_MARKERS: Final[tuple[str, ...]] = ("o'reilly", "oreilly")


def _is_publisher_slogan(lowered: str) -> bool:
    normalized = " ".join(lowered.replace("\u2019", "'").split()).rstrip(".!")
    for phrase in _SLOGAN_PHRASES:
        if normalized == phrase:
            return True
        if normalized.startswith(phrase) and any(m in normalized for m in _PUBLISHER_MARKERS):
            return True
    return False


def is_section_numbered(text: str) -> bool:
    """True for a heading opening with a section number ("9.2.1 BP for
    polytrees", "20 37.4.3.2 GP-MPC" with a margin line number)."""
    return _split_section_number(" ".join(text.split()))[1]


def _split_section_number(text: str) -> tuple[str, bool]:
    """``(text after a leading section number, True)`` or ``(text, False)``."""
    match = _SECTION_NUMBER_PREFIX_RE.match(text)
    if match is None:
        return text, False
    rest = text[match.end() :]
    if not any(ch.isalpha() for ch in rest):
        return text, False
    return rest, True


def _is_strict_word(token: str) -> bool:
    # The vowel rule only means something for Latin script: CJK, Cyrillic and
    # Greek headings have no [aeiouy] and were all rejected, collapsing whole
    # books into the one heading with Latin letters in it (VOR-1442). Two
    # letters from another script make a word (most CJK words are two
    # characters).
    non_latin = sum(c.isalpha() and ord(c) > _LATIN_MAX_CODEPOINT for c in token)
    if non_latin:
        return non_latin >= 2
    return sum(c.isalpha() for c in token) >= 3 and bool(_VOWEL_RE.search(token))


def _is_long_acronym(token: str) -> bool:
    return bool(_ACRONYM_RE.match(token)) and sum(c.isalpha() for c in token) >= 3


def _is_wordlike(token: str, *, numbered: bool) -> bool:
    if _is_strict_word(token):
        return True
    if not numbered:
        return False
    return token.lower() in _FUNCTION_WORDS or bool(_ACRONYM_RE.match(token))


def is_degenerate_heading(text: str) -> str | None:
    """Return a `reason_slug` if `text` is NOT a meaningful section heading.

    Returns None if `text` looks like a legitimate heading.

    Used at both storage time (don't add to heading-tree dict) and lookup time
    (skip when picking deepest heading). The caller treats the return value as
    "reject if not None."
    """
    if text is None:
        return "empty"
    # A soft hyphen is invisible: "100 \u00ad and 110 \u00ad Meter Hurdles" (VOR-1454).
    stripped = " ".join(text.replace("\u00ad", " ").split())
    if not stripped:
        return "empty"

    # Substring check against watermark phrases
    lowered = stripped.lower()
    for phrase in _WATERMARK_PHRASES:
        if phrase in lowered:
            return "watermark_or_drm"
    if _is_publisher_slogan(lowered):
        return "publisher_slogan"

    # A chapter's number line with its title ("8 -- 3000 to 10,000 Meters") is a
    # chapter, however many digits its title has (VOR-1454).
    if is_chapter_heading(stripped):
        return None

    # Regex patterns
    for pattern, reason in _DEGENERATE_PATTERNS:
        if pattern.search(stripped):
            return reason

    # The ratio/word rules below judge the heading text, not its section number.
    content, numbered = _split_section_number(stripped)

    # Digit-ratio check
    digits = sum(1 for c in content if c.isdigit())
    total = sum(1 for c in content if not c.isspace())
    if total > 0 and digits / total > _MAX_DIGIT_RATIO:
        return "mostly_digits"

    # Rule 1: math/relational operators — bare operators never appear in real
    # section headings.
    if _MATH_OPERATOR_RE.search(stripped):
        return "math_operator"

    # Rule 2: alpha-ratio floor — heading text must be at least 55% alphabetic
    # characters (out of all non-whitespace characters).
    alpha = sum(1 for c in content if c.isalpha())
    nonspace = sum(1 for c in content if not c.isspace())
    if nonspace > 0 and alpha / nonspace < _MIN_ALPHA_RATIO:
        return "low_alpha_ratio"

    # Rule 3: word-like fraction — a meaningful heading must contain a
    # sufficient density of real words (≥3 alpha chars + at least one vowel).
    tokens = content.split()
    if numbered:
        tokens = [t for t in tokens if len(t) > 1] or tokens
    if numbered and not any(_is_strict_word(t) or _is_long_acronym(t) for t in tokens):
        # Function words and short acronyms only fill the ratio of a numbered
        # heading; alone ("3 of 5", "12 OR", "7 II") they are not one.
        return "insufficient_words"
    if tokens:
        wordlike = [t for t in tokens if _is_wordlike(t, numbered=numbered)]
        if len(wordlike) / len(tokens) < _MIN_WORD_FRACTION:
            return "insufficient_words"

    junk = junk_heading_reason(stripped)
    if junk is not None:
        return junk
    # Judge the heading as a path shows it too: "'by Waldemar ..." with its
    # stray quote cut is a byline, "~ NOTE:" a callout (VOR-1454).
    # (From the text as parsed: its line breaks tell which lines are debris.)
    cleaned = " ".join(clean_heading_text(text.replace("\u00ad", " ")).split())
    if cleaned != stripped:
        return is_degenerate_heading(cleaned) if cleaned else "empty"
    return None


def is_meaningful_heading(text: str) -> bool:
    """Convenience inverse of `is_degenerate_heading`."""
    return is_degenerate_heading(text) is None


# --- VOR-1454: junk headings -------------------------------------------------
#
# Section paths are shown to the track-and-field tagger ancestor by ancestor, so
# a line the parser typed as a heading must be one. A spot check of 40 live
# chunks after the VOR-1452/1453 re-ingest found ~14 paths with a junk
# ancestor: library stamps ("ACPL ITEM DISCARDED", "WEEK LOAN", "PROPERTY OF"),
# author bylines ("DR SAM ELLIS", "K. VELDRIMOR"), sentences typed as
# headings, axis and measurement labels ("MASS IN POUNDS", "6-7 meters"). A
# depth-1 junk line parents every mixed-case heading after it (the parser
# ranks an all-caps line above a mixed-case one), so one stamp on the title
# page mislabelled a whole book. These text-only classes are rejected here;
# the chunker rejects the document-dependent ones (garbled OCR, clipped
# running heads, running heads of the book's title) and keeps front matter
# from parenting the body (``hierarchical``, ``heading_tracker``).

#: Library ownership stamps, loan labels and scan-provenance lines: page
#: furniture of the physical copy that was scanned, never text of the book.
_STAMP_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:acpl|item discarded|discarded|withdrawn from|i?withdra[wv]\s?w?n|property of"
    r"|(?:week|day|hour|short|long|standard|overnight|one|two|three|seven)[ -]loan|loan copy"
    r"|date due|due date|ex libris|not to be (?:taken|removed)|reference only"
    r"|digiti[sz]ed by|internet archive"
    r"|(?:public|university|college|county|city|state|school|free|memorial) library"
    r")\b",
    re.IGNORECASE,
)
#: Honorifics that open an author's byline ("DR SAM ELLIS", "Professor X Y").
_HONORIFIC: Final[str] = r"(?:dr|prof|professor|mr|mrs|ms|sir)\.?"
#: A name token: capitalised word or initial(s) ("K.", "B.I.", "O'Neil").
_NAME_TOKEN: Final[str] = r"(?:[A-Z]\.(?:\s?[A-Z]\.)*|[A-Z][A-Za-z'’-]+)"
_BYLINE_RES: Final[tuple[re.Pattern[str], ...]] = (
    # "by Professor Lena Marlow Whitfield", "by Dr Casey Marlow, ..."
    re.compile(rf"^(?i:by)\s+(?:{_HONORIFIC}\s+)?[A-Z][A-Za-z'’.-]+(?:[\s,]+\S+){{0,12}}$"),
    # "DISCUS by Dr. Sam Ellis": a byline inside a title-page line.
    re.compile(rf"\bby\s+{_HONORIFIC}\s+[A-Za-z]", re.IGNORECASE),
    # "DR SAM ELLIS", "Dr Alex K Rivera"
    re.compile(rf"^{_HONORIFIC}\s+(?:[A-Za-z][A-Za-z'’.-]*\s*){{1,4}}$", re.IGNORECASE),
    # "K. VELDRIMOR", "B.I. QUILLSKY", "J. G. Ellis"
    re.compile(r"^(?:[A-Z]\.\s?){1,3}\s?[A-Z][A-Za-z'’-]+$"),
    # "WILLIAM J. ALDRIDGE", "Samuel K. Ellis"
    re.compile(r"^[A-Z][A-Za-z'’-]+\s+(?:[A-Z]\.\s?){1,2}\s?[A-Z][A-Za-z'’-]+$"),
)
#: Index in ``_BYLINE_RES`` of the first initials-and-surname pattern.
_INITIALS_PATTERNS: Final[int] = 3
#: A byline starting "by" must continue with a name, not "By the Numbers".
_BY_NAME_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?i:by)\s+(?:(?i:{_HONORIFIC})\s+)?{_NAME_TOKEN}\s+{_NAME_TOKEN}"
)
#: Words that end a line mid-clause: a heading does not end on them.
_MID_CLAUSE_END: Final[frozenset[str]] = frozenset(
    {
        "and", "or", "the", "a", "an", "of", "that", "which", "but", "whose", "than",
        "was", "were", "is", "are", "be", "by", "with", "from", "into", "have", "has", "had",
    }
)  # fmt: skip
#: Prepositions a longer line does not end on as a heading ("... likely related
#: to increases in").
_PREPOSITION_END: Final[frozenset[str]] = frozenset(
    {"in", "to", "for", "on", "at", "as", "if", "when", "while", "because", "such"}
)
_PREPOSITION_END_MIN_WORDS: Final[int] = 6
#: A citation list closing a sentence: "(29, 148).", "147). Improvements".
_CITATION_TAIL_RE: Final[re.Pattern[str]] = re.compile(r"\d\)[.,;]|\(\d+(?:,\s*\d+)+\)")
#: The tail of a citation: "Kinetics), 258.".
_CITATION_END_RE: Final[re.Pattern[str]] = re.compile(r"\),\s*\d{1,4}\.?$")
#: A modal verb phrase: a sentence's predicate ("can be classified").
_MODAL_PHRASE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:can|could|should|must|will|would|may|might)\s+(?:be|not|also|often|only)\b"
)
_MODAL_MIN_WORDS: Final[int] = 7
#: Openers only a sentence starts with ("Even though Sir Isaac Newton").
_SENTENCE_OPENER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:although|even though|however|because|whereas|therefore|thus|moreover|furthermore"
    r"|nevertheless|consequently|hence|in addition to this)\b",
    re.IGNORECASE,
)
#: A sentence stop inside the line followed by more words: two sentences.
_INNER_SENTENCE_STOP_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z]{2})[.!?]\s+[A-Za-z]")
#: A clause with its own subject after a comma: ", she competed", ", we have".
_CLAUSE_SUBJECT_RE: Final[re.Pattern[str]] = re.compile(
    r",\s+(?:she|he|they|we|you|it|i|this|these|there)\s+[a-z]"
)
#: Finite verbs and subject pronouns: a long line with one reads as prose.
_FINITE_WORDS: Final[frozenset[str]] = frozenset(
    {
        "is", "are", "was", "were", "has", "have", "had", "can", "will", "should", "must",
        "may", "might", "would", "could", "does", "did", "he", "she", "they", "we", "you",
        "it", "his", "her", "their", "our",
    }
)  # fmt: skip
#: A heading longer than this many words that holds a finite verb or pronoun
#: is a sentence; past _MAX_HEADING_WORDS words any line is.
_SENTENCE_MIN_WORDS: Final[int] = 15
_MAX_HEADING_WORDS: Final[int] = 28
#: A heading in sentence case runs to this many words at most.
_SENTENCE_CASE_MAX_WORDS: Final[int] = 10
#: A line opening in lower case this long is a sentence fragment, whatever the
#: book's heading style (lower-case styled headings are short: "lower-body plyos").
_LOWER_FRAGMENT_MIN_WORDS: Final[int] = 7
_UNITS: Final[str] = (
    r"(?:mm|cm|m|km|meters?|metres?|ft|feet|foot|in|inch(?:es)?|yards?|yds?|miles?"
    r"|kg|g|lbs?|pounds?|oz|s|sec|secs|seconds?|min|mins|minutes?|hrs?|hours?"
    r"|mph|km/h|m/s|%|percent|per cent|degrees?|°|reps?|sets?)"
)
#: A bare measurement: a range or a decimal with a unit ("6-7 meters", "2.5
#: m"), or a whole number of a unit no event is named by ("40%", "30 sec"). A
#: whole distance is an event: "100 m", "800 metres".
_MEASUREMENT_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?:\d[\d,]*\s*(?:-|–|to|x)\s*[\d.,]+|\d*[.,/]\d+)\s*{_UNITS}\.?$"
    r"|^\d+\s*(?:%|percent|per cent|degrees?|°|s|sec|secs|seconds?|reps?|sets?)\.?$",
    re.IGNORECASE,
)
#: A table row opening with a measured range and holding another number:
#: "40-60 seconds anaerobic 10-15% grade".
_MEASURED_ROW_RE: Final[re.Pattern[str]] = re.compile(
    rf"^\d[\d.,]*\s*(?:-|\u2013|to)\s*\d[\d.,]*\s*{_UNITS}\b.*\d", re.IGNORECASE
)
#: An axis or table-column label: "MASS IN POUNDS", "VELOCITY IN FEET PER SECOND".
_AXIS_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z .-]{0,40}\b(?:in|per)\s+(?:pounds|kilograms|kg|lbs?|meters|metres|feet|yards"
    r"|seconds|degrees|mph|newtons?|joules|watts|percent|centimet(?:er|re)s|inches"
    r"|(?:feet|meters|metres) per second)$",
    re.IGNORECASE,
)
#: A figure or table caption label: "Illustration y", "Figure 6-22", "TABLE 10.13",
#: "(Figure 16)".
_CAPTION_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\(?(?:illustration|ilustration|fig\.?|f[i!1l]g\.|figure|table|plate|photo|picture|diagram"
    r"|chart|graph|exhibit)"
    r"\s*(?:\d+(?:[.\-–]\d+)*[a-z]?|[ivxlc]+|[a-z])\b(?![-'])",
    re.IGNORECASE,
)
#: A blank-form field: "DATE", "NAME", "DATE MM,", "DATE February" -- the
#: label alone, or in capitals with what was written in beside it.
_FORM_FIELD_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?i:date|name|signature)[\s:_.,-]*"
    r"|(?:DATE|NAME|SIGNATURE)\b[\s:_.,-]*(?:\S+[\s:_.,-]*){0,2})$"
)
#: A training sheet's column label set alone in capitals: "ATHLETE", "EVENT".
_FORM_LABEL_ALONE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:ATHLETE|EVENT|COACH|SCHOOL|TEAM|GRADE|AGE GROUP)[\s:_.]*$"
)
#: Verbs that make a colon-ended line a definition's lead-in ("Strength
#: Endurance is usually defined as:").
_LEAD_IN_VERBS: Final[frozenset[str]] = frozenset(
    {"is", "are", "was", "were", "include", "includes", "means", "defined"}
)
#: A dateline or quoted-source line: "Kentish Post 14 August, 1754".
_DATELINE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august"
    r"|september|october|november|december),?\s+1[5-9]\d{2}\b",
    re.IGNORECASE,
)


#: Title-page and copyright-page lines: cataloging data, copyright and printing
#: history, ISBNs. They label no section of the book.
_FRONT_MATTER_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"catalog(?:u)?ing[- ]?in[- ]?publication|\bcip data\b"
    r"|^(?:the |u\.?s\.? )?library of congress\b"
    r"|^british library\b|^copyright\s*(?:©|\(c\)|\d{4}|by\b)|^©|\bisbn\b"
    r"|^first published\b|^published (?:by|in)\b|^printed (?:and bound )?in\b|^orcid\b"
    r"|^an? dk publishing book$",
    re.IGNORECASE,
)
#: A cataloging-data author line: "Ellis, Samuel, 1930–".
_CIP_AUTHOR_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z][A-Za-z'’-]+,\s+[A-Z][A-Za-z.'’ -]+,\s*(?:\d{4})?\s*[-–]\s*(?:\d{4})?\.?$"
)


def is_front_matter_line(text: str, page: int | None, front_pages: int = 0) -> bool:
    """True for a title-page or copyright-page line typed as a heading (VOR-1454).

    Cataloging-in-publication data, copyright and printing-history lines,
    ISBNs, a CIP author line, and a publisher imprint on the book's first
    pages (``IMPRINT_MAX_PAGE``, or its ``front_pages`` when more). They are
    never headings: the chunker keeps them as text. A publisher imprint later
    in the book (an ad page's "Human Kinetics") stays a terminal back-matter
    heading of its own page.
    """
    stripped = " ".join(text.split())
    if not stripped:
        return False
    if _FRONT_MATTER_LINE_RE.search(stripped) or _CIP_AUTHOR_RE.match(stripped):
        return True
    if page is None or page > max(IMPRINT_MAX_PAGE, front_pages):
        return False
    normalized = _normalize_heading_for_back_matter(stripped)
    return bool(
        _IMPRINT_SUFFIX_RE.match(normalized)
        or _IMPRINT_NAME_RE.match(normalized)
        or _AMBIGUOUS_IMPRINT_SUFFIX_RE.match(normalized)
    )


#: A structural label in any case: "part one: - jumping out of the fire".
_PART_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:part|chapter|section|appendix|book|unit|lesson)\s+(?:\d+|[ivxlc]+|[a-z]"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.IGNORECASE,
)
#: Words before an inner "by Dr ..." that still make a title-page line
#: ("DISCUS by Dr. Sam Ellis"), not a heading that names its source
#: ("PRACTICAL QUESTIONS RELATING TO TRAINING ANSWERED BY DR. ELLIS").
_MAX_WORDS_BEFORE_BYLINE: Final[int] = 3


#: A bullet inside the line: list or chart text run together ("Multilateral
#: Development \u00b7 Slower performance improvement").
_INNER_BULLET_RE: Final[re.Pattern[str]] = re.compile(r"\S\s+[\u00b7\u2022\u25aa\u25cf]\s+\S")
#: A picture credit: "AP Photo/Matt Dunham", "Photo courtesy of ...".
_CREDIT_RE: Final[re.Pattern[str]] = re.compile(
    r"\bphotos?\s*/|\bphoto(?:graph)?s?\s+(?:by|courtesy)\b|^courtesy of\b|\bgetty images\b"
    r"|^(?:photo|picture|illustration) credits?\b",
    re.IGNORECASE,
)
#: Words of a training sheet's header ("440-Yard Dash NAME").
_FORM_TITLE_MAX_WORDS: Final[int] = 6
#: A running head with its folio: "242 Basketball", "188 Fluid Mechanics" -- a
#: three-digit page number, not an event distance ("400 Meters", "110 m Hurdles").
_FOLIO_HEAD_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?!(?:100|110|200|300|400|500|600|800)\b)\d{3}\s+[A-Z][a-z]+:?"
    r"(?:\s+(?:[A-Z][a-z]+:?|and|of|the|&)){0,4}$"
)
#: A contents line: an entry and its page ("Inertia, 60", "Relays ........ 45").
_CONTENTS_LINE_TAIL: Final[int] = 200
_CONTENTS_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z)],\s\d{1,3}$|(?:\.\s?){3,}\s*\d{1,3}$"
)
#: A dedication: "To my parents, ROSA and Sam Ellis".
_DEDICATION_RE: Final[re.Pattern[str]] = re.compile(r"^(?:to|for) my\s", re.IGNORECASE)
#: A cataloging-style inverted name: "Aldridge, William J.", "Ellis, Samuel".
_INVERTED_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z][a-z'\u2019-]+,\s[A-Z][a-z]+(?:\s[A-Z]\.)?\.?$"
)
#: An institution line under a contributor: "York University", "University of Illinois".
_INSTITUTION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Z][A-Za-z.'\u2019-]+\s){1,3}(?:University|College|Institute)$"
    r"|^(?:University|College|Institute) of (?:[A-Z][A-Za-z.'\u2019-]+\s?){1,3}$"
)
#: An enumerated list run into one line: "1. Introduction 2. Retrospective analysis".
_INLINE_ENUMERATION_RE: Final[re.Pattern[str]] = re.compile(r"(?:^|\s)\d{1,2}[.)]\s+[A-Z]")
#: A citation bracket left open at the line end.
_OPEN_CITATION_RE: Final[re.Pattern[str]] = re.compile(r"\[\d{1,3}$")
#: A table or figure carried over from the page before: "... (continued)".
_CONTINUED_RE: Final[re.Pattern[str]] = re.compile(r"\((?:continued|cont(?:\.|'d))\)\s*$", re.I)
#: A clock time: a timetable's row ("Time Activity 10:00 a.m. to ..."), not a heading.
_CLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)(?!\w)|\b\d{1,2}:\d{2}\b.*\b(?:noon|midnight)\b",
    re.I,
)


#: A contributor with a degree ("Sam Ellis, PhD", "Jane Doe, M.S., ATC").
_DEGREE_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z][A-Za-z.'\u2019-]+(?:\s+[A-Z][A-Za-z.'\u2019-]+){0,3},\s*"
    r"(?:ph\.?\s?d|m\.?\s?d|ed\.?\s?d|m\.?\s?[sa]|b\.?\s?[sa]|atc|cscs|dpe|mbe|obe)\.?\b",
    re.IGNORECASE,
)
#: A contributor's post: "Ted McLaughlin Head Track Coach", "..., Assistant Coach".
_AFFILIATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?!(?:the|a|an|your|our|his|her|their|my)\b)"
    r"[A-Z][A-Za-z.'\u2019-]+(?:\s+[A-Z][A-Za-z.'\u2019-]+){0,3},?\s+"
    r"(?:head|assistant|national|chief)\s+(?:track\s+(?:and\s+field\s+)?|athletics\s+)?coach\b",
    re.IGNORECASE,
)
#: A post line under a contributor's name: "Coach Northfield College",
#: "Head Coach, University of Oregon".
_POST_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:head|assistant)\s+)?coach,?\s+(?:[A-Z][a-z]+\s+){0,3}"
    r"(?:college|university|high school|club)\b"
    r"|^(?:head|assistant)\s+(?:\w+\s+){0,4}coach,?\s",
    re.IGNORECASE,
)
#: A contributor's place line: "Alameda, CA", "Sarasota, FL".
_PLACE_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2},\s(?:[A-Z]{2}|[A-Z][a-z]+\.?)$"
)


def _inverted_name(text: str) -> bool:
    """ "Aldridge, William J.": a name, not a pair of topics ("Relays, Hurdles")."""
    from vo_rag.pipelines.book_rag.ocr_quality import is_word

    if not _INVERTED_NAME_RE.match(text):
        return False
    surname = text.split(",", 1)[0].strip().lower()
    return not is_word(surname)


def _form_field_with_entry(text: str) -> bool:
    """ "Date Pa)", "Name ___": a form label with what was filled in beside it --
    at most two short tokens that are no words."""
    from vo_rag.pipelines.book_rag.ocr_quality import is_word

    words = text.split()
    if not words or words[0].lower().strip(":") not in {"date", "name"} or len(words) > 3:
        return False
    rest = [_WORD_TOKEN_RE.findall(word.casefold()) for word in words[1:]]
    return all(not tokens or not any(is_word(t) for t in tokens) for tokens in rest)


def _is_byline(text: str) -> bool:
    from vo_rag.pipelines.book_rag.ocr_quality import is_word

    if text[:3].lower() == "by ":
        # "By Jordan Avery, Ukraine" names; "By the Numbers", "BY THE NUMBERS" do not.
        honorific = re.match(rf"^by\s+{_HONORIFIC}\s", text, re.IGNORECASE)
        return bool(_BY_NAME_RE.match(text)) and (not text.isupper() or bool(honorific))
    inner = _BYLINE_RES[1].search(text)
    if inner is not None and inner.start() > 0:
        return len(text[: inner.start()].split()) <= _MAX_WORDS_BEFORE_BYLINE
    for index, pattern in enumerate(_BYLINE_RES):
        if index == 1 or not pattern.search(text):
            continue
        if index >= _INITIALS_PATTERNS:
            # "K. BARTONIETZ" is a name; "E. CONDITIONING", "I. INTRODUCTION"
            # are outline letters before a heading word.
            surname = text.split()[-1].casefold().strip(".,'’")
            return not is_word(surname)
        return True
    return False


def _reads_as_sentence(text: str) -> bool:
    """A sentence or a piece of one typed as a heading (VOR-1454)."""
    words = text.split()
    count = len(words)
    first_alpha = next((ch for ch in text if ch.isalpha()), "")
    if count > _MAX_HEADING_WORDS:
        return True
    if count > _SENTENCE_CASE_MAX_WORDS and all(word[:1].islower() for word in words[1:]):
        return True  # "Aside from the traditional method of classifying sport activities ..."
    if first_alpha.islower() and count >= _LOWER_FRAGMENT_MIN_WORDS:
        return True
    # In a capitalised line the case says nothing: "JOINING THE" ends mid-clause.
    last = words[-1].lower() if text.isupper() else words[-1]
    if count <= 2 and text.endswith(","):
        return True  # "Further,": the start of a sentence
    if count >= 2 and (last.strip(",;") in _MID_CLAUSE_END or last.endswith(",")):
        return True
    if count >= 3 and re.search(r"[a-z]-$", last):
        return True  # a word broken at the line end: "The dynam-"
    if count >= 4 and _SENTENCE_OPENER_RE.match(text):
        return True
    if count >= 6 and (_INNER_SENTENCE_STOP_RE.search(text) or _CLAUSE_SUBJECT_RE.search(text)):
        return True
    if count >= _PREPOSITION_END_MIN_WORDS and last in _PREPOSITION_END:
        return True
    if count >= 5 and text.endswith("!"):
        return True  # an advertisement's call: "Maximize performance in every ... event!"
    if count >= 3 and _CITATION_TAIL_RE.search(text):
        return True  # "99, 125, 147). Improvements in ..."
    if count >= _MODAL_MIN_WORDS and _MODAL_PHRASE_RE.search(text):
        return True  # "Competitions can be classified into two broad categories"
    lowered = {word.lower().strip(",.;:()\"'") for word in words}
    if count >= 4 and text.endswith(":") and lowered & _LEAD_IN_VERBS:
        return True
    # A question is a heading's form ("What Exercises Are Included?").
    return count > _SENTENCE_MIN_WORDS and not text.endswith("?") and bool(lowered & _FINITE_WORDS)


#: A structural label with no number or title: "CHAPTER", "Part", "CHAPTER £3"
#: (the title is the next line).
_BARE_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:chapter|part|section|unit|lesson)\b[^A-Za-z0-9]*(?:[^A-Za-z0-9\s]\S{0,2})?$",
    re.IGNORECASE,
)
#: A callout label that introduces a line of the text, not a section.
_CALLOUT_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:remember|note|notes|example|examples|hint|important|caution|warning|n\.?\s?b\.?)"
    r"\s*[:!.-]$",
    re.IGNORECASE,
)
#: A line opening with a conjunction or "of": the tail of a heading split over
#: lines ("OF THE JAVELIN", "AND WEB SITES").
_CONTINUATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:of|and|or|nor|but|than|whose|which|whether)\s+\S", re.IGNORECASE
)
#: Mastheads of the coaching journals whose articles these booklets reprint:
#: page furniture, never a section ("ATHLETICS COACH" over an article).
_MASTHEAD_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:the\s+)?(?:athletics coach|track technique|track and field quarterly review"
    r"|track (?:&|and) field news|modern athlete and coach|new studies in athletics"
    r"|athletics weekly|track coach|coaching review|athletics journal)"
    r"(?:[\s,.:;-]+(?:vol(?:ume)?\.?|no\.?|issue)?\s*\d*)*[\s.]*$",
    re.IGNORECASE,
)


#: A masthead opening a line, however the rest is garbled ("ATHLETICS COACH
#: ATHLETI COACH").
_MASTHEAD_START_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:athletics coach|track technique|modern athlete and coach|new studies in athletics)\b"
    r"(?!\s*[:\-\u2014]\s*\w)",
    re.IGNORECASE,
)


def junk_heading_reason(text: str) -> str | None:
    """A ``reason_slug`` when ``text`` is a junk line typed as a heading (VOR-1454).

    Text-only classes (see the section comment): ``library_stamp``,
    ``bare_label``, ``masthead``, ``callout``, ``continuation``, ``byline``,
    ``sentence``, ``measurement``, ``axis_label``, ``caption_label``,
    ``form_field``, ``dateline``. ``None`` for anything else.
    """
    stripped = " ".join(text.split())
    if not stripped:
        return None
    if _STAMP_RE.search(stripped):
        return "library_stamp"
    if _BARE_LABEL_RE.match(stripped):
        return "bare_label"
    if _MASTHEAD_RE.match(stripped.strip("\u00b7\u2022 ")) or _MASTHEAD_START_RE.match(stripped):
        return "masthead"
    if _CALLOUT_RE.match(stripped):
        return "callout"
    if _CONTINUATION_RE.match(stripped) or re.match(r"^[-\u2013][^\W\d_]", stripped):
        return "continuation"  # "OF THE JAVELIN", "-Meter Hurdles (Off-Season)"
    if (
        _is_byline(stripped)
        or _DEGREE_NAME_RE.match(stripped)
        or _AFFILIATION_RE.search(stripped)
        or _POST_LINE_RE.match(stripped)
    ):
        return "byline"
    if _PLACE_LINE_RE.match(stripped) and re.search(r",\s[A-Z]{2}$", stripped):
        return "place_line"
    if _MEASUREMENT_RE.match(stripped) or _MEASURED_ROW_RE.match(stripped):
        return "measurement"
    if _AXIS_LABEL_RE.match(stripped):
        return "axis_label"
    if _CAPTION_LABEL_RE.match(stripped):
        return "caption_label"
    if (
        _FORM_FIELD_RE.match(stripped)
        or _form_field_with_entry(stripped)
        or _FORM_LABEL_ALONE_RE.match(stripped)
    ):
        return "form_field"
    if _DATELINE_RE.search(stripped):
        return "dateline"
    if _CONTINUED_RE.search(stripped):
        return "continued"
    if _CREDIT_RE.search(stripped):
        return "credit"
    if _INNER_BULLET_RE.search(stripped) or len(_INLINE_ENUMERATION_RE.findall(stripped)) >= 2:
        return "bullet_list"
    if _OPEN_CITATION_RE.search(stripped) or _CITATION_END_RE.search(stripped):
        return "sentence"  # cut inside a citation: "... G. Velmorin [74"
    if (re.search(r"\bDATE\b", stripped) and re.search(r"\bNAME\b", stripped)) or (
        len(stripped.split()) <= _FORM_TITLE_MAX_WORDS
        and not stripped.isupper()
        and re.search(r"\b(?:NAME|DATE)\b", stripped)
    ):
        return "form_field"  # "440-Yard Dash NAME": a training sheet's header
    if _FOLIO_HEAD_RE.match(stripped):
        return "running_head"
    # End-anchored: only the tail can match (VOR-1454: bounded on untrusted text).
    if _CONTENTS_LINE_RE.search(stripped[-_CONTENTS_LINE_TAIL:]):
        return "contents_line"
    if _DEDICATION_RE.match(stripped) or _inverted_name(stripped):
        return "byline"
    if _INSTITUTION_RE.match(stripped):
        return "byline"
    if _CLOCK_RE.search(stripped):
        return "clock_time"
    # A section-numbered heading ("35.4.2 User Study-based Evaluation: Does
    # the Method Help a User Perform a Task?") is structure, however worded.
    content, numbered = _split_section_number(stripped)
    if is_chapter_heading(stripped):
        return None  # "5 -- 100 and 110 Meter Hurdles"
    if numbered and content[:1].isupper():
        # Structure, however worded -- unless it stops mid-clause: "1
        # Bounding over balls or other" is a list line cut at the line end.
        words = content.split()
        if len(words) >= 3 and words[-1] in _MID_CLAUSE_END | {"other", "the"}:
            return "sentence"
        return None
    if (
        not is_structural_heading(stripped)
        and not _PART_LABEL_RE.match(stripped)
        and _reads_as_sentence(content if numbered else stripped)
    ):
        return "sentence"
    return None


# --- VOR-1452: back-matter headings --------------------------------------------
#
# Back-matter headings (References, Index, About the contributors, imprint
# lines, ...) are real headings of their own section -- they are NOT degenerate
# -- but they are *terminal*: nothing that follows them in the book belongs
# under them except the entries of that back matter itself. Scanned back matter
# rarely produces the next TITLE, so ``HeadingTracker`` used to keep the label
# for every later page (an audit of a track-and-field corpus found body
# chunks filed under "References" and "Index"). ``HeadingTracker`` expires a
# terminal heading; these predicates say what one is and what its entries look
# like.

# The whole heading (case-folded, whitespace collapsed, trailing punctuation
# removed) must be one of these phrases, so "Notes on the Pole Vault" or "The
# Reference Point" are never terminal. A section-numbered heading ("38.9 Further
# Reading", "12.5 References") is an in-chapter section of a structured book
# whose next numbered heading the parser does find; it is never terminal.
_BACK_MATTER_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"references?(?: (?:list|cited))?|references and (?:further|suggested) readings?"
    r"|(?:selected |select |annotated )?bibliography|works cited|sources"
    r"|(?:subject |author |name |general )?index|index of (?:names|subjects|authors)"
    r"|about the (?:authors?|contributors|editors?|series(?: editors?)?)"
    r"|(?:list of )?contributors|notes on (?:the )?contributors"
    r"|(?:other )?(?:titles|books) in (?:this|the) series|also in (?:this|the) series"
    r"|(?:also|other (?:books|titles)) (?:available )?(?:by|from) .{1,60}"
    r"|acknowledge?ments?"
    r"|notes|endnotes|chapter notes|notes and references"
    r"|(?:further|suggested|recommended|additional) readings?"
    r"|glossary(?: of terms)?"
    r"|copyright(?: page)?|cip|(?:library of congress|british library) catalog(?:u)?ing.*"
    r")$"
)
# Imprint lines the parser promotes to headings on the title and copyright
# pages ("Prentice Hall International", "Human Kinetics"). A publisher-suffix
# word only counts at the end of a short line; a bare imprint name must be the
# whole heading (optionally with an imprint suffix).
#
# "press" and "limited" also end exercise and training headings ("Bench Press",
# "Push Press", "Full-court press", "Speed Limited"), so on
# their own they are not a publisher suffix: only the unambiguous words are,
# plus "... University Press" and a company form ("Tafnews Press, Inc.").
# A bare "X Press" / "X Limited" is an imprint only on the first pages of the
# book (``is_front_page_imprint``, applied by ``HeadingTracker``).
_IMPRINT_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[a-z&.'-]+ ){0,4}(?:publishing|publishers|publications|inc|ltd|llc|verlag)$"
    r"|^(?:[a-z&.'-]+ ){0,4}(?:university|college) press$"
    r"|^(?:[a-z&.'-]+ ){0,4}(?:press|publishing|books),? (?:inc|ltd|llc|limited|co)$"
)
_AMBIGUOUS_IMPRINT_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[a-z&.'-]+ ){1,4}(?:press|limited)$"
)
#: The title, half-title and copyright pages: a bare "X Press" there is an imprint.
IMPRINT_MAX_PAGE: Final[int] = 4
_IMPRINT_NAMES: Final[tuple[str, ...]] = (
    "prentice hall",
    "prentice-hall",
    "human kinetics",
    "mcgraw-hill",
    "mcgraw hill",
    "routledge",
    "elsevier",
    "springer",
    "pearson",
    "wiley",
    "crowood",
    "a & c black",
    "a&c black",
    "chelsea house",
    "random house",
)
_IMPRINT_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    + "|".join(re.escape(name) for name in _IMPRINT_NAMES)
    + r")(?: (?:international|press|publishing|publishers|inc|ltd|books|education|sport))*$"
)


_TRAILING_FOLIO_RE: Final[re.Pattern[str]] = re.compile(r"\s*\|?\s+\d{1,4}$")


def _normalize_heading_for_back_matter(text: str) -> str:
    lowered = " ".join(text.replace("\u2019", "'").lower().split())
    lowered = lowered.strip(" .:;,-\u2013\u2014")
    # A trailing page number is a running head's folio ("INDEX 999").
    return _TRAILING_FOLIO_RE.sub("", lowered).strip(" .:;,-|\u2013\u2014")


def is_back_matter_heading(text: str) -> bool:
    """True when ``text`` is a back-matter heading (terminal, see above)."""
    if not text or not text.strip():
        return False
    if _SECTION_NUMBER_PREFIX_RE.match(text.strip()):
        return False
    normalized = _normalize_heading_for_back_matter(text)
    if not normalized:
        return False
    return bool(
        _BACK_MATTER_HEADING_RE.match(normalized)
        or _IMPRINT_SUFFIX_RE.match(normalized)
        or _IMPRINT_NAME_RE.match(normalized)
    )


#: Back-matter headings whose section is a list of entries (citations, index
#: lines, notes, terms): its entries can resume after another section's short
#: note laid out between them.
_ENTRY_LIST_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"references?(?: (?:list|cited))?|references and (?:further|suggested) readings?"
    r"|(?:selected |select |annotated )?bibliography|works cited|sources"
    r"|(?:subject |author |name |general )?index|index of (?:names|subjects|authors)"
    r"|notes|endnotes|chapter notes|notes and references"
    r"|(?:further|suggested|recommended|additional) readings?"
    r"|glossary(?: of terms)?"
    r")$"
)


def is_entry_list_heading(text: str) -> bool:
    """True for a back-matter heading whose section lists entries ("References",
    "Index", "Notes"), not prose ("Acknowledgements", "About the author")."""
    if not text or not text.strip() or _SECTION_NUMBER_PREFIX_RE.match(text.strip()):
        return False
    return bool(_ENTRY_LIST_HEADING_RE.match(_normalize_heading_for_back_matter(text)))


def has_trailing_folio(text: str) -> bool:
    """True when ``text`` ends in a page number ("Index 245", "INDEX | 999")."""
    lowered = " ".join(text.lower().split()).strip(" .:;,-\u2013\u2014")
    return bool(_TRAILING_FOLIO_RE.search(lowered))


def is_imprint_heading(text: str) -> bool:
    """True for a publisher imprint line ("Human Kinetics", "Tafnews Press, Inc.")."""
    if not text or not text.strip() or _SECTION_NUMBER_PREFIX_RE.match(text.strip()):
        return False
    normalized = _normalize_heading_for_back_matter(text)
    return bool(_IMPRINT_SUFFIX_RE.match(normalized) or _IMPRINT_NAME_RE.match(normalized))


def is_front_page_imprint(text: str, page: int | None) -> bool:
    """A bare "X Press" / "X Limited" line on the book's first pages.

    The suffix alone is ambiguous ("Bench Press"); on the title and copyright
    pages (``page <= IMPRINT_MAX_PAGE``) a short line ending in it is the imprint.
    """
    if page is None or page > IMPRINT_MAX_PAGE or not text or not text.strip():
        return False
    if _SECTION_NUMBER_PREFIX_RE.match(text.strip()):
        return False
    return bool(_AMBIGUOUS_IMPRINT_SUFFIX_RE.match(_normalize_heading_for_back_matter(text)))


_INDEX_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:subject |author |name |general )?index|index of (?:names|subjects|authors))$"
)
# A heading that stays a heading even inside an index (VOR-1452).
_STRUCTURAL_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:chapter|part|appendix|section|book)\b", re.I
)
#: Index entries the parser typed as TITLE are short terms ("Ankle Hops").
MAX_INDEX_ENTRY_TITLE_WORDS: Final[int] = 6


def is_index_heading(text: str) -> bool:
    """True when ``text`` is an index heading ("Index", "Subject index", "INDEX 999")."""
    if not text or not text.strip() or _SECTION_NUMBER_PREFIX_RE.match(text.strip()):
        return False
    return bool(_INDEX_HEADING_RE.match(_normalize_heading_for_back_matter(text)))


_DOTTED_SECTION_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"^\d+(?:\.\d+)+\.?\s+(?=\S)")
_NUMBERED_TITLE_RE: Final[re.Pattern[str]] = re.compile(r"^\d{1,2}\.?\s+(?P<title>[^\W\d_].*)$")


def is_labelled_heading(text: str) -> bool:
    """True for a heading that is a unit of the book's structure by its label:

    - a labelled unit: "Chapter 3", "Part II", "Appendix A", "Section 4";
    - a multi-level section number: "6.12 Derivative free optimization".
    """
    stripped = " ".join(text.split())
    return bool(_STRUCTURAL_HEADING_RE.match(stripped) or _DOTTED_SECTION_NUMBER_RE.match(stripped))


def is_numbered_title(text: str) -> bool:
    """True for a chapter number (1-2 digits) and a capitalised, mixed-case
    title: "3 The Technique of Hurdling", "7 Inference algorithms: an overview".

    Not every title that opens with a number is one: the numbered
    steps of a session are all-caps run-in heads ("1 WARM UP", "2 MOBILITY
    EXERCISES"), a clock time is four digits ("1030 PVCC: w-up A"), and a
    distance opens a phrase ("300 and 400 METRES HURDLING"). A sentence-case
    step ("1 Warm up") reads as a chapter here; whether it is one is decided
    by where it sits (``short_text``: a chapter is not nested under an
    ordinary section, a step is).
    """
    numbered = _NUMBERED_TITLE_RE.match(" ".join(text.split()))
    if numbered is None:
        return False
    title = numbered.group("title")
    return title[0].isupper() and not title.isupper()


#: A chapter by its label: "Chapter 5", "CHAPTER XII", "Part II", "2 — Predictive
#: Testing of Athletes", "12 — High Jump".
_CHAPTER_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:chapter|part|book)\s+(?:\d+|[ivxlc]+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\d{1,2}\s*[\u2014\u2013]\s*\w)",
    re.IGNORECASE,
)


_PART_LABEL_START_RE: Final[re.Pattern[str]] = re.compile(r"^(?:part|book)\b", re.IGNORECASE)


def is_part_heading(text: str) -> bool:
    """True for a part or book heading ("Part II", "BOOK ONE"): the one unit
    a chapter sits inside."""
    return bool(_PART_LABEL_START_RE.match(" ".join(text.split())))


def is_chapter_heading(text: str) -> bool:
    """True for a chapter or part heading by its label (VOR-1454): never a
    section inside an ordinary heading."""
    return bool(_CHAPTER_LABEL_RE.match(" ".join(text.split())))


def is_structural_heading(text: str) -> bool:
    """True for a labelled unit or a numbered, mixed-case title (see both)."""
    return is_labelled_heading(text) or is_numbered_title(text)


#: Words that make a colon-ended line a sentence ("The constants are:", "Details
#: from:", "The steps include:"), not a label ("Coaching points:"). "as" is not
#: one ("Sprinting as a skill:" is a label).
_LEAD_IN_WORDS: Final[frozenset[str]] = frozenset(
    {
        "are", "is", "was", "were", "be", "been", "include", "includes", "including",
        "follow", "follows", "following", "from", "shown", "below", "above",
        "have", "has", "can", "should", "must", "will", "would", "may", "might",
        "see", "note", "say", "says", "means", "mean", "given", "described", "listed",
    }
)  # fmt: skip
#: Lead-in words that are a label's infinitive after "to" ("Points to note:",
#: "Drills to include:"), and a sentence's verb otherwise.
_INFINITIVE_LEAD_IN_WORDS: Final[frozenset[str]] = frozenset({"note", "include", "see", "say"})
#: A line opened by a connective is a sentence's lead-in whatever follows:
#: "For example:", "In summary:", "In other words:", "To summarise:".
_CONNECTIVE_OPENERS: Final[frozenset[str]] = frozenset({"for", "in", "to", "as", "e.g.", "i.e."})


def is_lead_in_line(text: str) -> bool:
    """True for a sentence that introduces what follows, typed as a TITLE.

    "The constants are:", "Details from:": it ends
    in a colon, every word after the first is lower-case, and it reads as a
    sentence -- it opens with a connective ("For example:", "In summary:";
    ``_CONNECTIVE_OPENERS``) or one of its later words is a verb or connective
    (``_LEAD_IN_WORDS``; "note"/"include"/"see" only when not after "to":
    "Key points to note:" is a label). A title-case or all-caps heading ("Key
    Points:", "INT III:") and a sentence-case label ("Coaching points:",
    "Warm-up routine:", "Sprinting as a skill:") are
    not. A lead-in is body text that belongs with the sentence before it and
    the list after it, never a section of its own.
    """
    stripped = " ".join(text.split())
    if not stripped.endswith(":") or stripped.isupper() or is_structural_heading(stripped):
        return False
    words = stripped[:-1].split()
    rest = [word for word in words[1:] if word[:1].isalpha()]
    if len(words) < 2 or not rest or not all(word[0].islower() for word in rest):
        return False
    if words[0].lower() in _CONNECTIVE_OPENERS:
        return True
    lowered = [word.lower().strip(",;") for word in words]
    return any(
        word in _LEAD_IN_WORDS
        and not (word in _INFINITIVE_LEAD_IN_WORDS and lowered[index - 1] == "to")
        for index, word in enumerate(lowered)
        if index >= 1
    )


def is_index_entry_title(text: str) -> bool:
    """True when a TITLE inside an index region reads as one of its entries.

    Parsers type index lines as headings ("Ankle Hops", "Back Toss"); inside an
    index they are entries, not sections. A back-matter heading (the next
    "About the Author") or a structural heading ("Chapter 3", "Part II",
    "Appendix A", a section-numbered title) is still a heading, as is anything
    longer than ``MAX_INDEX_ENTRY_TITLE_WORDS`` words.
    """
    stripped = " ".join(text.split())
    if not stripped or is_back_matter_heading(stripped):
        return False
    if _SECTION_NUMBER_PREFIX_RE.match(stripped) or _STRUCTURAL_HEADING_RE.match(stripped):
        return False
    return len(stripped.split()) <= MAX_INDEX_ENTRY_TITLE_WORDS


_CITATION_FRAGMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"[.,;]$|\b(?:1[89]|20)\d{2}\b|\b\d{1,4}\s*[-\u2013]\s*\d{1,4}\b|^[a-z]"
)


_YEAR_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:1[89]|20)\d{2}[a-z]?\b")


def reads_as_citations(text: str) -> bool:
    """True when ``text`` reads as reference-list entries: two or more years,
    at least one per 40 words, or entry-shaped lines."""
    words = len(text.split())
    years = len(_YEAR_RE.findall(text))
    return (years >= 2 and years * 40 >= words) or looks_like_back_matter_entry(text)


def is_entry_fragment_title(text: str) -> bool:
    """True when a TITLE inside a reference list reads as a piece of an entry.

    "Brook & Harbor, Madison.", "Society," or "(1985) Effect of
    training" typed as TITLE between citations. A heading does not end in a
    period, comma or semicolon, carry a year or a page range, or start in
    lower case; a back-matter or structural heading ("Chapter 18", "Index")
    stays a heading.
    """
    stripped = " ".join(text.split())
    if not stripped or is_back_matter_heading(stripped):
        return False
    if _SECTION_NUMBER_PREFIX_RE.match(stripped) or _STRUCTURAL_HEADING_RE.match(stripped):
        return False
    return bool(_CITATION_FRAGMENT_RE.search(stripped))


# Entry shapes that keep a terminal heading alive on a later page. Each is
# judged per line; an element is an entry when most of its lines are.
# - index entry: a short term (at most _MAX_INDEX_TERM_WORDS words, no
#   sentence stop) then 1-3 digit page numbers ("aerobic capacity, 12, 45-47",
#   "Aldridge, H. M. 42", "hurdles 88, 102"). Prose ending in a number ("... in
#   1968.", "... against the neck, page 10.") has a longer lead-in;
_INDEX_ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<term>[^\d]{1,80}?)[\s,]\s*\d{1,3}(?:\s*[-\u2013]\s*\d{1,3})?"
    r"(?:\s*,\s*\d{1,3}(?:\s*[-\u2013]\s*\d{1,3})?)*\.?$"
)
_MAX_INDEX_TERM_WORDS: Final[int] = 8
_SENTENCE_STOP_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z]{3})[.!?]\s")
# - glossary entry: a short term, then a colon or dash and its definition
#   ("Aerobic: requiring oxygen", "Isometrics - hold training that ...");
_GLOSSARY_ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<term>[^\s:.;!?][^:.;!?]{0,48}?)\s*(?::|\u2014|\u2013|\s-\s)\s*\S"
)
_MAX_GLOSSARY_TERM_WORDS: Final[int] = 5
# A training program reads like a glossary ("Monday: acceleration work ...",
# "Squat: 3x5 at 80%", "Day 1: 4x60m @95%"). A glossary
# defines a term in words: a definition with a digit, or a term that is a
# schedule label (a weekday, "Day/Week/Session N"), is not an entry.
_SCHEDULE_TERM_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day"
    r"|(?:day|week|session|workout|phase|stage|block|set|rep|round) \w+)$",
    re.I,
)
_DIGIT_RE: Final[re.Pattern[str]] = re.compile(r"\d")
# - citation opening with an author ("Avery, J.", "Ellis, J. G.", "Calder C.");
_AUTHOR_LEAD_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z][A-Za-z'\u2019-]+,?\s+(?:[A-Z]\.\s*){1,3}(?:,|\(|\s|$)"
)
# - numbered or keyed note/reference ("[12] ...", "[Aba+16] ...", "3. See ...");
_NUMBERED_NOTE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\[\d{1,3}\]|\[[A-Za-z][A-Za-z0-9+.\- ]{1,11}\]|\d{1,3}[.)]?\s)"
)
# - bibliographic signal anywhere: a year in brackets or after a comma, "pp.",
#   "et al.", "ibid", a journal/volume marker, a DOI or ISBN.
_BIBLIO_SIGNAL_RE: Final[re.Pattern[str]] = re.compile(
    r"\((?:19|20)\d{2}[a-z]?\)|,\s*(?:19|20)\d{2}[a-z]?[.;,)]|\bpp?\.\s*\d|\bet al\b"
    r"|\bibid\b|\bjournal\b|\bvol\.\s*\d|\bdoi\b|\bisbn\b",
    re.I,
)
# A citation is a single bibliographic item: past this length a line is prose
# that happens to cite something ("Smith (1998) found that ...").
_MAX_ENTRY_LINE_CHARS: Final[int] = 400


def _is_entry_line(line: str) -> bool:
    if len(line) > _MAX_ENTRY_LINE_CHARS:
        return False
    index_entry = _INDEX_ENTRY_RE.match(line)
    if index_entry is not None:
        term = index_entry.group("term")
        if len(term.split()) <= _MAX_INDEX_TERM_WORDS and not _SENTENCE_STOP_RE.search(term):
            return True
    glossary_entry = _GLOSSARY_ENTRY_RE.match(line)
    if glossary_entry is not None:
        term = glossary_entry.group("term").strip()
        definition = line[glossary_entry.end() - 1 :]
        if (
            len(term.split()) <= _MAX_GLOSSARY_TERM_WORDS
            and not _SCHEDULE_TERM_RE.match(term)
            and not _DIGIT_RE.search(definition)
        ):
            return True
    has_signal = bool(_BIBLIO_SIGNAL_RE.search(line))
    if _AUTHOR_LEAD_RE.match(line):
        return has_signal or len(line) < 200
    if _NUMBERED_NOTE_RE.match(line):
        return has_signal
    return False


def looks_like_back_matter_entry(text: str) -> bool:
    """True when ``text`` reads as index entries, citations or numbered notes.

    Used to decide whether content on a later page still belongs to a terminal
    back-matter heading. Prose paragraphs, short title-case headings and
    technique text are not entries.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    entries = sum(1 for line in lines if _is_entry_line(line))
    return entries * 2 > len(lines)


# --- VOR-1454: heading text as shown in a section path -------------------------

#: Characters that decorate an OCR'd heading's ends: rules, bullets, stray
#: underscores and pipes ("~ BIOMECHANICAL ANALYSIS", "BASIC CONSIDERATIONS |").
_EDGE_JUNK: Final[str] = " _~|»«•·*^\\/#=+<>—–-"
_UNDERSCORE_RUN_RE: Final[re.Pattern[str]] = re.compile(r"_{2,}")
#: An OCR smear: one letter four or more times in a row ("—mmecaimenmssssssesssnesem").
_SMEAR_RE: Final[re.Pattern[str]] = re.compile(r"([a-z])\1{3,}", re.IGNORECASE)
_WORD_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^\W\d_]+")


def _line_has_word(line: str) -> bool:
    from vo_rag.pipelines.book_rag.ocr_quality import is_word

    return any(
        len(token) >= 3 and is_word(token) for token in _WORD_TOKEN_RE.findall(line.casefold())
    ) or any(ord(ch) > _LATIN_MAX_CODEPOINT and ch.isalpha() for ch in line)


def clean_heading_text(text: str) -> str:
    """``text`` as a section-path component (VOR-1454): one line, OCR debris cut.

    A multi-line heading keeps the lines that hold a word ("COMMUNICATING /
    YO G / YOUR APPROACH" -> "COMMUNICATING YOUR APPROACH"); underscores and
    rule characters at its ends, an unmatched leading quote and a trailing
    letter smear are cut. Only the path label changes: the chunker's heading
    logic still works on the parsed text.
    """
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        kept = [line for line in lines if _line_has_word(line)]
        lines = kept or lines
    cleaned = " ".join(lines)
    cleaned = " ".join(_UNDERSCORE_RUN_RE.sub(" ", cleaned).split())
    halves = cleaned.split()
    half = len(halves) // 2
    if half and len(halves) % 2 == 0 and halves[:half] == halves[half:]:
        # A heading set twice over ("COACHING PRINCIPLES COACHING PRINCIPLES").
        cleaned = " ".join(halves[:half])
    words = cleaned.split()
    while len(words) > 1 and _SMEAR_RE.search(words[-1]):
        words.pop()
    cleaned = " ".join(words).strip(_EDGE_JUNK)
    for quote, closing in (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
        if cleaned.startswith(quote) and closing not in cleaned[1:]:
            cleaned = cleaned[1:].lstrip()
        if cleaned.endswith(closing) and quote not in cleaned[:-1]:
            cleaned = cleaned[:-1].rstrip()
    cleaned = cleaned.strip(_EDGE_JUNK)
    # Nothing but spacing changed: the heading as parsed (its label stays the
    # text the chunker's other rules and merged-section lists use).
    if cleaned == " ".join(text.split()):
        return re.sub(r"\s*[\r\n]+\s*", " ", text.strip())
    return cleaned


# --- VOR-1454: back matter the parser left without its heading ------------------

#: A page reference in an index: "34", "22-23", "104-5", "329f", "xii".
_PAGE_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.])(?:\d{1,4}[a-z]?(?:\s*[-–]\s*\d{1,4})?|[ivxl]{1,6})(?![\w.]\w)"
)
#: A prose sentence: 8+ words ending in terminal punctuation.
_CITATION_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"\bet al\b|\bpp?\.\s*\d|\bvol\.|\bno\.\s*\d|\bdoi\b|https?://|\bisbn\b|\bpress\b"
    r"|\bpublish(?:ers?|ing)\b|\bjournal\b|\bed(?:s|itor|itors)?\.|\b\d++\s*\(\d++\)\s*[:,]",
    re.IGNORECASE,
)
#: An author as a reference list names one: "Smith, J.", "Ellis, J. G.", "et al",
#: "Marlow PJ".
_CITED_AUTHOR_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Z][A-Za-z'\u2019-]+,?\s+(?:[A-Z]\.\s*){1,3}|\bet al\b"
    # Vancouver style: "Marlow PJ, Quill RN," / "Northrop JM 3rd,"
    r"|\b[A-Z][a-z'\u2019-]{2,}\s[A-Z]{1,3},"
)
_INDEX_MIN_TOKENS: Final[int] = 12
_PAGE_REFS_PER_TOKENS: Final[int] = 7
_INDEX_MIN_TERMS: Final[int] = 8
_INDEX_MIN_DISTINCT_TERMS: Final[int] = 5
_INDEX_TERM_PREFIX: Final[int] = 3
_INDEX_SORTED_SHARE: Final[float] = 0.8
_INDEX_INTERLEAVED_SHARE: Final[float] = 0.55
_DENSE_REFS_PER_TOKENS: Final[int] = 5
_TOKENS_PER_REF_LIST: Final[int] = 40
#: A list of page references: "126, 128, 130", "551-3, 558-64".
_PAGE_REF_LIST_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\d{1,4}(?:\s*[-\u2013]\s*\d{1,4})?,\s*\d{1,4}\b"
)
_SPACED_RANGE_RE: Final[re.Pattern[str]] = re.compile(r"(\d)\s+([-\u2013])\s*(\d)")
_SPACED_COMMA_RE: Final[re.Pattern[str]] = re.compile(r"\s+,")
#: Cell joints of a linearised table ("551=spinal", ": spinal").
_TABLE_JOIN_RE: Final[re.Pattern[str]] = re.compile(r"[=|]")
_REFERENCE_MIN_SIGNALS: Final[int] = 3
_WORDS_PER_REFERENCE_SIGNAL: Final[int] = 40
_REFERENCE_MIN_AUTHORS: Final[int] = 2


def reads_as_index(text: str, *, strict: bool = False) -> bool:
    """A run of index entries ("abrasion 45-47, finishing 32"): page
    references at least every seventh token, no prose sentence, and entry
    terms in alphabetical order -- a training table is as full of numbers, but
    its rows are not sorted. An index read across two columns interleaves two
    sorted runs; it passes on a denser run of page-reference lists ("126, 128,
    130") with the order only mostly kept."""
    # OCR spaces a range's dash and a list's comma: "40 -42", "85 , 96".
    joined = _SPACED_RANGE_RE.sub(r"\1\2\3", _SPACED_COMMA_RE.sub(",", text))
    tokens = _TABLE_JOIN_RE.sub(" ", joined).split()
    if len(tokens) < _INDEX_MIN_TOKENS:
        return False
    is_ref = [bool(_PAGE_REF_RE.fullmatch(token.strip(",;.:()"))) for token in tokens]
    refs = sum(is_ref)
    if refs * _PAGE_REFS_PER_TOKENS < len(tokens) or _has_prose_sentence(text):
        return False
    # An entry's term is the first word after a run of page references; its
    # first letters give the order ("abrasion" < "acceleration").
    initials = [
        token[:_INDEX_TERM_PREFIX].lower()
        for index, token in enumerate(tokens)
        if token[:1].isalpha() and not is_ref[index] and (index == 0 or is_ref[index - 1])
    ]
    if len(initials) < _INDEX_MIN_TERMS or len(set(initials)) < _INDEX_MIN_DISTINCT_TERMS:
        return False  # "x 10 each side / x 10 each way": one term over and over
    ordered = sum(1 for a, b in zip(initials, initials[1:], strict=False) if a <= b)
    share = ordered / (len(initials) - 1)
    if share >= _INDEX_SORTED_SHARE:
        return True
    if strict:
        return False  # a table must keep the order: a results grid interleaves too
    ref_lists = len(_PAGE_REF_LIST_RE.findall(text))
    return (
        share >= _INDEX_INTERLEAVED_SHARE
        and refs * _DENSE_REFS_PER_TOKENS >= len(tokens)
        and ref_lists * _TOKENS_PER_REF_LIST >= len(tokens)
    )


#: A prose sentence: this many words in a row (a word may carry ",;:"), the last
#: ending in ".", "!" or "?". A token scan, not a regex: nested quantifiers over
#: untrusted text are quadratic (VOR-1454).
_PROSE_SENTENCE_MIN_WORDS: Final[int] = 8
_PROSE_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z'\u2019-]+[,;:]?")
_PROSE_END_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z'\u2019-]+[.!?]")


def _has_prose_sentence(text: str) -> bool:
    run = 0
    for token in text.split():
        if _PROSE_END_RE.fullmatch(token) and run + 1 >= _PROSE_SENTENCE_MIN_WORDS:
            return True
        run = run + 1 if _PROSE_WORD_RE.fullmatch(token) else 0
    return False


def reads_as_reference_list(text: str) -> bool:
    """A reference list: citation signals (years, "et al.", pages, volumes,
    journals, publishers) at least every 40 words and cited authors."""
    words = len(text.split())
    if words == 0:
        return False
    signals = len(_YEAR_RE.findall(text)) + len(_CITATION_MARKER_RE.findall(text))
    return (
        signals >= _REFERENCE_MIN_SIGNALS
        and signals * _WORDS_PER_REFERENCE_SIGNAL >= words
        and len(_CITED_AUTHOR_RE.findall(text)) >= _REFERENCE_MIN_AUTHORS
    )


def _dictionary_garbled(text: str) -> bool:
    """No word of ``text`` is a dictionary word and one has no vowel at all
    ("RW ALVG", "SL/PERTRAINfNC"): OCR garble, not a term, a name or another
    language's heading ("DOMS", "Erste Doppelstunde" have vowels)."""
    from vo_rag.pipelines.book_rag.ocr_quality import is_word

    if any(ord(ch) > _LATIN_MAX_CODEPOINT for ch in text):
        return False  # another script: the English dictionary cannot judge
    tokens = [t for t in _WORD_TOKEN_RE.findall(text.casefold()) if len(t) >= 2]
    if not tokens or any(is_word(token) for token in tokens):
        return False
    return any(not _VOWEL_RE.search(token) for token in tokens)


#: The page-range label the chunker gives a book with no usable headings
#: (``TitleOnlyFallback``): a label of its own making, not a parsed heading.
_PAGE_BUCKET_RE: Final[re.Pattern[str]] = re.compile(r"^pp\. \d+-\d+$")


def path_component_junk_reason(component: str) -> str | None:
    """Why a section-path component is junk (VOR-1454), or ``None``.

    The ``path_quality`` check of ``rechunk-preview``: the heading classes the
    chunker rejects by text (``is_degenerate_heading``: stamps, bylines,
    sentences, measurements, captions ...), title- and copyright-page lines,
    and a component none of whose words is a dictionary word (OCR garble;
    judged in Latin script only). The document-dependent rules (running heads,
    clipped heads, front matter) need the parse, so a path alone cannot show
    them.
    """
    if _PAGE_BUCKET_RE.match(component.strip()):
        return None
    reason = is_degenerate_heading(component)
    if reason is not None:
        return reason
    if is_front_matter_line(component, None):
        return "front_matter_line"
    if _dictionary_garbled(component):
        return "garbled"
    return None
