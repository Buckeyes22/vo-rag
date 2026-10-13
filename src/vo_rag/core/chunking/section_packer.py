from __future__ import annotations

import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from typing import Protocol

from vo_rag.core.chunking.artifact_filters import (
    contains_distribution_artifact,
    detect_running_lines,
    strip_distribution_artifact_tail,
    strip_page_furniture_text,
    strip_running_lines,
)
from vo_rag.core.chunking.counters import TokenCounterFn
from vo_rag.core.chunking.models import DropReason
from vo_rag.core.chunking.token_ceiling import TokenCeilingBackstop
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.parsing.text_normalization import normalize_extracted_text

_OUTLINE_MARKERS = {
    "agenda",
    "examples",
    "overview",
    "process",
    "q&a",
    "resources",
    "session",
    "what we'll cover",
    "what well cover",
}

_LOW_VALUE_STANDALONE_HEADINGS = {
    "acknowledgments",
    "agenda",
    "contents",
    "copyright",
    "index",
    "isbn",
    "process",
    "publisher",
    "q&a",
    "question",
    "resources",
    "session",
    "what we'll cover",
    "what well cover",
}

_TECHNICAL_MARKERS = {
    "accel",
    "acceleration",
    "aero",
    "aerodynamic",
    "airflow",
    "apex",
    "brake bias",
    "brake pressure",
    "brake",
    "braking",
    "camber",
    "channel",
    "channels",
    "coaching",
    "corner",
    "corners",
    "damper",
    "data logger",
    "delta",
    "diffuser",
    "downforce",
    "drag",
    "driver",
    "drivers",
    "g-load",
    "g load",
    "hardware",
    "lap",
    "laps",
    "lift",
    "logger",
    "longitudinal",
    "pitot",
    "pressure",
    "rpm",
    "sensor",
    "sensors",
    "software",
    "spoiler",
    "spring",
    "speed",
    "steering",
    "table",
    "throttle",
    "tire",
    "tyre",
    "trace",
    "traces",
    "turn-in",
    "tuft",
    "tufts",
    "tufting",
    "undertray",
    "vehicle",
    "wing",
}

_COPYRIGHT_MARKERS = {
    "all rights reserved",
    "copyright",
    "edition",
    "isbn",
    "printed in",
    "publisher",
    "publishing",
}

_TOC_MARKERS = {
    "chapter",
    "contents",
    "index",
    "page",
    "part",
    "table of contents",
}

TEXT_BEARING_ELEMENT_TYPES = {
    ElementType.NARRATIVE_TEXT,
    ElementType.TITLE,
    ElementType.LIST_ITEM,
    ElementType.TABLE,
    ElementType.FIGURE_CAPTION,
    ElementType.EQUATION,
    ElementType.CODE_BLOCK,
}

APPENDED_TEXT_ELEMENT_TYPES = {
    ElementType.NARRATIVE_TEXT,
    ElementType.FIGURE_CAPTION,
    ElementType.EQUATION,
    ElementType.CODE_BLOCK,
}

_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mph|kph|km/h|psi|bar|rpm|g|deg|degree|degrees|mm|cm|kg|"
    r"lb|lbs|sec|secs|second|seconds|ms|hz|%|ft|m)\b",
    re.I,
)
_EQUATION_RE = re.compile(r"(?:\b[A-Za-z][A-Za-z0-9_]*\s*=|[=<>]\s*[-+]?\d|\b\d+\s*[*/+-]\s*\d+)")


_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_#]+")
#: A heading-only chunk (VOR-1452) is at most this many heading lines ...
_MAX_HEADING_ONLY_LINES = 3
#: ... of at most this many words each, and this many in all.
_MAX_HEADING_LINE_WORDS = 10
_MAX_HEADING_ONLY_WORDS = 16


class ProseSplitter(Protocol):
    def split_text(self, content: str, section_path: str) -> list[str]: ...


class SectionPacker:
    def __init__(
        self,
        *,
        counter: TokenCounterFn,
        target: int,
        max_tokens: int,
        overlap: int,
        table_linearizer: object | Callable[[ParsedElement], str] | None = None,
        min_text_chunk_words: int | None = None,
        prose_splitter: ProseSplitter | None = None,
    ) -> None:
        self.count = counter
        self.target = target
        self.max = max_tokens
        self.overlap = overlap
        self.linearize_table = table_linearizer
        self.ceiling = TokenCeilingBackstop(counter=counter, max_tokens=max_tokens, overlap=overlap)
        self.prose_splitter = prose_splitter
        self.min_text_chunk_words = (
            min_text_chunk_words
            if min_text_chunk_words is not None
            else int(os.environ.get("VORAG_MIN_TEXT_CHUNK_WORDS", "20"))
        )

    @staticmethod
    def should_flush_for_title(depth: int) -> bool:
        return depth in (1, 2)

    def clean_text_for_append(
        self,
        text: str,
        page_number: int | None,
        running_lines: set[str],
        *,
        preserve_numeric_data: bool = False,
    ) -> str:
        cleaned = normalize_extracted_text(text)
        cleaned = strip_page_furniture_text(
            cleaned,
            [page_number] if page_number is not None else None,
            strip_numeric_page_lines=not preserve_numeric_data,
        )
        return strip_running_lines(cleaned, running_lines)

    def clean_chunk_body(
        self,
        text: str,
        pages: list[int],
        running_lines: set[str],
    ) -> str:
        cleaned = normalize_extracted_text(text)
        # Element cleaning already removes evidenced folios. A bare operand
        # in the assembled body has lost its element/page-edge provenance.
        cleaned = strip_page_furniture_text(cleaned, list(pages), strip_numeric_page_lines=False)
        cleaned = strip_running_lines(cleaned, running_lines)
        # Watermark lines go; a rights notice stays (nothing the chunker is
        # given is dropped for what it says).
        return strip_distribution_artifact_tail(cleaned)

    def table_text(self, element: ParsedElement) -> str:
        if self.linearize_table is None:
            return normalize_extracted_text(element.text)
        if callable(self.linearize_table):
            return normalize_extracted_text(self.linearize_table(element))
        linearize = getattr(self.linearize_table, "linearize", None)
        if callable(linearize):
            return normalize_extracted_text(linearize(element))
        return normalize_extracted_text(element.text)

    def split_text(self, content: str, section_path: str) -> list[str]:
        if self.prose_splitter is not None:
            return self.prose_splitter.split_text(content, section_path)
        return self.ceiling.split_text(content, section_path)

    def split_table(self, content: str, section_path: str) -> list[str]:
        return self.ceiling.split_table(content, section_path)

    def hard_token_ceiling(self) -> int:
        return self.ceiling.hard_ceiling()

    def embedding_token_count(self, section_path: str, content: str) -> int:
        return self.ceiling.embedding_token_count(section_path, content)

    def tail_for_overlap(self, text: str) -> str:
        return self.ceiling.tail_for_overlap(text)

    @staticmethod
    def ends_at_retrieval_boundary(text: str) -> bool:
        return TokenCeilingBackstop.ends_at_retrieval_boundary(text)

    def rendered_pending_heading(self, heading: str) -> str:
        """Keep source numerals and mathematical text promoted to a heading."""
        collapsed = " ".join(heading.split())
        if any(ch.isdigit() for ch in collapsed) or self.has_mathematical_value(collapsed):
            return collapsed
        return self.normalize_line_for_signal(heading)

    @staticmethod
    def has_mathematical_value(text: str) -> bool:
        """Recognize source operators, angular units, delimiters and Greek variables."""
        return (
            any(
                unicodedata.category(ch) == "Sm"
                or ch in "°′″*/^"
                or (ord(ch) > 127 and ch.isalpha() and "GREEK" in unicodedata.name(ch, ""))
                for ch in text
            )
            or text.strip() in {"-", "–", "(", ")"}
            or ("()" in text and any(ch.isalpha() for ch in text))
        )

    def filtered_pending_headings(self, headings: list[str]) -> list[str]:
        if not headings:
            return []
        normalized = [self.rendered_pending_heading(heading) for heading in headings]
        normalized = [heading for heading in normalized if heading]
        if not normalized:
            return []

        lowered = " ".join(normalized).lower()
        outline_hits = sum(marker in lowered for marker in _OUTLINE_MARKERS)
        technical_hits = sum(marker in lowered for marker in _TECHNICAL_MARKERS)
        if len(normalized) >= 3 and outline_hits >= 2 and technical_hits == 0:
            return [
                heading
                for heading in normalized
                if heading.lower() not in _OUTLINE_MARKERS
                and heading.lower() not in _LOW_VALUE_STANDALONE_HEADINGS
                and not heading.lower().startswith("more in ")
            ]

        kept: list[str] = []
        for heading in headings:
            cleaned = self.rendered_pending_heading(heading)
            lowered_heading = cleaned.lower()
            if not cleaned:
                continue
            if lowered_heading in _LOW_VALUE_STANDALONE_HEADINGS:
                continue
            if lowered_heading.startswith("more in "):
                continue
            kept.append(cleaned)
        return kept

    @staticmethod
    def is_standalone_heading(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.startswith("•"):
            return False
        if "\n" in stripped:
            return False
        words = stripped.replace("…", "").replace("!", "").replace(":", "").split()
        if not words or len(words) > 4:
            return False
        if stripped.endswith("?"):
            return False
        alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
        if not alpha_words:
            return False
        titleish = sum(1 for word in alpha_words if word[0].isupper()) / len(alpha_words)
        return titleish >= 0.8

    @staticmethod
    def is_heading_only_text(text: str) -> bool:
        """True when ``text`` is nothing but a heading ("**part one:**" /
        "**jumping out of the fire**", "TRAILING", "Box Height").

        Every line (at most three) must look like a heading -- wrapped in
        markdown bold, a markdown ``#`` heading, all capitals, or a standalone
        title-case heading -- with no sentence ending (". , ; ?"), at most
        ten words per line and sixteen in all. Used to fold a heading that has
        no body text of its own into the text it heads (VOR-1452).
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or len(lines) > _MAX_HEADING_ONLY_LINES:
            return False
        total_words = 0
        for line in lines:
            plain = " ".join(_MARKDOWN_EMPHASIS_RE.sub(" ", line).split())
            words = plain.split()
            alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
            if not alpha_words or len(words) > _MAX_HEADING_LINE_WORDS:
                return False
            if plain.endswith((".", ",", ";", "?")):
                return False
            total_words += len(words)
            bold = line.startswith(("**", "__")) and line.endswith(("**", "__"))
            capitals = all(word.upper() == word for word in alpha_words)
            if not (
                bold
                or line.startswith("#")
                or capitals
                or SectionPacker.is_standalone_heading(plain)
            ):
                return False
        return total_words <= _MAX_HEADING_ONLY_WORDS

    @staticmethod
    def normalize_line_for_signal(text: str) -> str:
        cleaned = re.sub(r"^[\s•\-–\d.()]+", "", text).strip()
        cleaned = cleaned.replace("…", "").strip().strip(":").strip()
        return re.sub(r"\s+", " ", cleaned)

    @staticmethod
    def pages_with_text_elements(elements: list[ParsedElement]) -> set[int]:
        return {
            element.page_number
            for element in elements
            if element.page_number is not None
            and element.element_type in TEXT_BEARING_ELEMENT_TYPES
        }

    @staticmethod
    def page_text_for_running_line_detection(doc: ParsedDocument) -> list[str]:
        return SectionPacker.page_text_from_elements_for_running_line_detection(
            doc.elements,
            doc.total_pages,
        )

    @staticmethod
    def page_text_from_elements_for_running_line_detection(
        elements: list[ParsedElement],
        page_count: int,
    ) -> list[str]:
        page_text = [""] * max(0, page_count)
        running_line_element_types = {
            ElementType.TITLE,
            ElementType.NARRATIVE_TEXT,
            ElementType.LIST_ITEM,
            ElementType.FIGURE_CAPTION,
            ElementType.HEADER,
            ElementType.FOOTER,
        }
        for element in elements:
            if (
                element.page_number is None
                or element.element_type not in running_line_element_types
            ):
                continue
            if element.page_number < 1 or element.page_number > len(page_text):
                continue
            text = normalize_extracted_text(element.text).strip()
            if not text:
                continue
            page_index = element.page_number - 1
            page_text[page_index] = f"{page_text[page_index]}\n{text}".strip()
        return page_text

    @staticmethod
    def repeated_short_page_text(elements: list[ParsedElement], page_count: int) -> set[str]:
        page_text = SectionPacker.page_text_from_elements_for_running_line_detection(
            elements,
            page_count,
        )
        running_keys = detect_running_lines(page_text, min_fraction=0.0, max_words=14)
        if not running_keys:
            return set()

        pages_by_text: dict[str, set[int]] = defaultdict(set)
        counts: dict[str, int] = defaultdict(int)
        for page_index, page in enumerate(page_text, start=1):
            for line in SectionPacker.page_edge_lines_for_running_detection(page):
                if SectionPacker.running_line_detection_key(line) not in running_keys:
                    continue
                normalized = SectionPacker.normalize_line_for_signal(
                    normalize_extracted_text(line)
                ).lower()
                if not normalized or len(normalized) > 160 or len(normalized.split()) > 14:
                    continue
                pages_by_text[normalized].add(page_index)
                counts[normalized] += 1
        return {
            text
            for text, pages in pages_by_text.items()
            if len(pages) >= 3 and (len(pages) / max(1, page_count) >= 0.20 or counts[text] >= 5)
        }

    @staticmethod
    def page_edge_lines_for_running_detection(page_text: str) -> tuple[str, ...]:
        lines = [line for line in page_text.splitlines() if line.strip()]
        if not lines:
            return ()
        if len(lines) == 1:
            return (lines[0],)
        return (lines[0], lines[-1])

    @staticmethod
    def running_line_detection_key(text: str) -> str:
        normalized = re.sub(r"\d+", "#", text.strip().casefold())
        return re.sub(r"\s+", " ", normalized).strip()

    def low_signal_text_reason(
        self,
        text: str,
        *,
        pages: list[int] | None = None,
        doc_title: str = "",
        doc_total_pages: int | None = None,
    ) -> DropReason | None:
        """Why gathered text is not a chunk of its own, or ``None`` when it is.

        Nothing the chunker is given is dropped for what it says. Text is
        dropped only when it holds neither letters nor numeric source values,
        is a distribution watermark, or is nothing but low-value words ("Contents", "Index");
        text under ``min_text_chunk_words`` is ``below_min_words`` and merged
        into a neighbour, never dropped. Index lines, contents lines, imprint
        and rights notices, title pages, contact details and digit-heavy lines
        ("2 X 220, 550, 2 X 220 or 2 X 660") are all kept: the old content
        rules dropped real text with them (a javelin drill under an index-like
        page reference, "CHAPTER 8" and its prose as "contents", a preface
        mentioning an "edition", sets and reps as "digits").
        ``pages``/``doc_title``/``doc_total_pages`` are kept for callers.
        """
        del pages, doc_title, doc_total_pages
        # A watermark sentence goes, the book's text around it stays.
        stripped = strip_distribution_artifact_tail(text)
        has_numeric_value = any(ch.isdigit() for ch in stripped)
        has_mathematical_value = self.has_mathematical_value(stripped)
        if not stripped or not (
            any(ch.isalpha() for ch in stripped) or has_numeric_value or has_mathematical_value
        ):
            return "empty_after_normalize"
        normalized_lines = [
            cleaned
            for line in stripped.splitlines()
            if (cleaned := SectionPacker.normalize_line_for_signal(line.strip()))
        ]
        if not normalized_lines and (has_numeric_value or has_mathematical_value):
            normalized_lines = [" ".join(stripped.split())]
        if not normalized_lines:
            return "empty_after_normalize"
        low_value_lines = sum(
            1
            for line in normalized_lines
            if line.lower() in _LOW_VALUE_STANDALONE_HEADINGS or line.lower().startswith("more in ")
        )
        if low_value_lines == len(normalized_lines):
            return "empty_after_normalize"
        word_count = len(re.findall(r"\b[\w'-]+\b", stripped))
        if word_count < self.min_text_chunk_words and not self.has_short_technical_value(stripped):
            return "below_min_words"
        return None

    @staticmethod
    def has_short_technical_value(text: str) -> bool:
        if contains_distribution_artifact(text):
            return False
        lowered = text.lower()
        if _EQUATION_RE.search(text):
            return True
        if _UNIT_RE.search(text):
            return True
        if any(marker in lowered for marker in _TECHNICAL_MARKERS):
            return True
        return any(
            marker in lowered for marker in ("figure", "fig.", "image", "chart", "graph", "diagram")
        )

    def low_signal_element_text_reason(
        self,
        text: str,
        *,
        pages: list[int] | None,
        doc_title: str,
        doc_total_pages: int | None,
    ) -> DropReason | None:
        """``low_signal_text_reason`` for one element's text (same rules)."""
        return self.low_signal_text_reason(
            text, pages=pages, doc_title=doc_title, doc_total_pages=doc_total_pages
        )

    @staticmethod
    def is_title_page_fragment(
        text: str,
        *,
        pages: list[int] | None,
        doc_title: str,
    ) -> bool:
        unique_pages = set(pages or [])
        if unique_pages and unique_pages != {1}:
            return False
        normalized = SectionPacker.normalize_line_for_signal(text).lower()
        if not normalized:
            return True
        title = SectionPacker.normalize_line_for_signal(doc_title).lower()
        words = normalized.split()
        if title and normalized in {title, f"{title} by"}:
            return True
        if len(words) <= 12 and not SectionPacker.has_short_technical_value(text):
            titleish_words = list(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text))
            if titleish_words:
                titleish = sum(1 for word in titleish_words if word[0].isupper()) / len(
                    titleish_words
                )
                return titleish >= 0.70
        return False

    @staticmethod
    def looks_like_table_of_contents(text: str) -> bool:
        lowered = text.lower()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        if "table of contents" in lowered or lowered.strip() == "contents":
            return True
        dotted = sum(1 for line in lines if re.search(r"\.{3,}\s*\d+\s*$", line))
        chapter_page = sum(
            1
            for line in lines
            if re.search(r"\b(?:chapter|part|section)\b.+\b\d{1,4}\s*$", line, re.I)
        )
        marker_hits = sum(marker in lowered for marker in _TOC_MARKERS)
        return (dotted + chapter_page) >= 2 or (
            len(lines) >= 4 and marker_hits >= 2 and chapter_page >= 1
        )

    @staticmethod
    def looks_like_front_matter_toc_fragment(
        text: str,
        *,
        pages: list[int] | None,
    ) -> bool:
        real_pages = [page for page in (pages or []) if page is not None]
        if real_pages and max(real_pages) > 2:
            return False

        lowered = " ".join(text.lower().split())
        words = re.findall(r"\b[\w'-]+\b", text)
        if not words or len(words) > 120:
            return False

        entry_markers = re.findall(r"\b(?:chapter|appendix|part)\s+[a-z0-9]+\b", lowered)
        if len(entry_markers) >= 2:
            return True
        if not entry_markers:
            return False
        if re.search(r"\b(?:chapter|appendix|part)\s+[a-z0-9]+\b.{0,90}\b\d{1,3}\b", lowered):
            return True
        return bool(re.search(r"\b\d{1,3}\s*$", lowered))

    @staticmethod
    def looks_like_acknowledgements_fragment(
        text: str,
        *,
        pages: list[int] | None,
    ) -> bool:
        real_pages = [page for page in (pages or []) if page is not None]
        if real_pages and max(real_pages) > 3:
            return False
        lowered = text.lower()
        if not re.search(r"\backnowledg(?:e)?ments?\b", lowered):
            return False
        technical_hits = sum(marker in lowered for marker in _TECHNICAL_MARKERS)
        return technical_hits <= 1

    @staticmethod
    def looks_like_front_matter_contact_fragment(
        text: str,
        *,
        pages: list[int] | None,
    ) -> bool:
        real_pages = [page for page in (pages or []) if page is not None]
        if real_pages and max(real_pages) > 4:
            return False
        lowered = text.lower()
        if not re.search(r"\b(?:e-mail|email|phone|fax|customer\s*service)\b|@", lowered):
            return False
        technical_hits = sum(marker in lowered for marker in _TECHNICAL_MARKERS)
        return technical_hits == 0

    @staticmethod
    def looks_like_back_matter_index_fragment(
        text: str,
        *,
        pages: list[int] | None,
        doc_total_pages: int | None,
    ) -> bool:
        if not doc_total_pages:
            return False
        real_pages = [page for page in (pages or []) if page is not None]
        back_matter_window = max(3, int(doc_total_pages * 0.08))
        if doc_total_pages >= 100:
            back_matter_window = max(30, back_matter_window)
        if not real_pages or min(real_pages) < max(1, doc_total_pages - back_matter_window):
            return False

        lowered = text.lower()
        words = re.findall(r"\b[\w'-]+\b", text)
        if not words:
            return False
        if "index" in lowered and len(words) <= 80:
            return True
        no_cont_text = re.sub(r"\(\s*cont\.?\s*\)", "", text, flags=re.I)
        if (
            doc_total_pages >= 100
            and len(words) <= 14
            and not re.search(r"[.!?=]", no_cont_text)
            and not _UNIT_RE.search(text)
            and not _EQUATION_RE.search(text)
        ):
            return True

        page_ref_runs = re.findall(
            r"\b\d{1,3}(?:[-–]\d{1,3})?(?:,\s*\d{1,3}(?:[-–]\d{1,3})?){1,}",
            text,
        )
        page_refs = re.findall(r"\b\d{1,3}(?:[-–]\d{1,3})?\b", text)
        term_page_pairs = re.findall(
            r"\b[A-Za-z][A-Za-z0-9/&().'-]*(?:\s+[A-Za-z][A-Za-z0-9/&().'-]*){0,4}"
            r"\s+\d{1,3}(?:[-–]\d{1,3})?\b",
            text,
        )
        term_page_pairs = [
            pair for pair in term_page_pairs if not re.search(r"\bpage\s+\d", pair, re.I)
        ]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_only_lines = [
            line
            for line in lines
            if re.fullmatch(
                r"\d{1,3}(?:[-–]\d{1,3})?(?:,\s*\d{1,3}(?:[-–]\d{1,3})?)*",
                line,
            )
        ]
        term_lines = [
            line
            for line in lines
            if line not in page_only_lines and any(ch.isalpha() for ch in line)
        ]
        if len(page_only_lines) >= 2 and len(term_lines) >= 2:
            return True
        if len(page_ref_runs) >= 2:
            return True
        if len(words) <= 40 and len(page_refs) >= 3 and "," in text:
            return True
        if len(words) <= 40 and len(term_page_pairs) >= 2 and not re.search(r"[.!?]", text):
            return True
        if len(words) <= 18 and page_refs and not re.search(r"[.!?]", text):
            return True
        return len(words) <= 120 and len(page_refs) >= 8 and "," in text

    @staticmethod
    def is_copyright_or_publisher_fragment(text: str) -> bool:
        lowered = text.lower()
        if not any(marker in lowered for marker in _COPYRIGHT_MARKERS):
            return False
        words = re.findall(r"\b[A-Za-z0-9'-]+\b", text)
        if len(words) <= 40:
            return True
        marker_lines = sum(
            1
            for line in text.splitlines()
            if any(marker in line.lower() for marker in _COPYRIGHT_MARKERS)
        )
        return (
            marker_lines / max(1, len([line for line in text.splitlines() if line.strip()])) >= 0.6
        )
