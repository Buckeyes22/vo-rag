"""Tests for the degenerate-heading filter.

Patterns derived from a corpus audit, which catalogued 8+ distinct
degenerate `section_title` shapes appearing across 96% of audited books.
"""

from __future__ import annotations

import pytest

from vo_rag.core.chunking.heading_filter import is_degenerate_heading, is_meaningful_heading


@pytest.mark.parametrize(
    "text,expected_reason",
    [
        # Bare numerics (chapter/page numbers misread as headings).
        # `numeric_only` fires before `too_short` because patterns are ordered
        # by specificity in _DEGENERATE_PATTERNS.
        ("1", "numeric_only"),
        ("23", "numeric_only"),
        ("1.", "numeric_only"),
        ("234", "numeric_only"),
        # Roman numerals (same pattern-order rule: roman_only beats too_short)
        ("iv", "roman_only"),
        ("xii", "roman_only"),
        # Watermarks
        (
            "Downloaded from SAE International by Example University, Monday",
            "watermark_or_drm",
        ),
        ("This content downloaded from JSTOR on 2018-09-18", "watermark_or_drm"),
        ("Copyright © 2024 Cengage Learning. All Rights Reserved.", "watermark_or_drm"),
        ("This page intentionally left blank", "watermark_or_drm"),
        # Spaced-out display-font book titles
        ("T H E  B O O K", "spaced_display_font"),
        ("R E A D I N G", "spaced_display_font"),
        # Pure punctuation / no word chars
        ("***", "no_word_chars"),
        ("---", "no_word_chars"),
        # Length-based: short non-meaningful strings.
        # Note: "X" / "I" / "V" would hit roman_only first; "12" hits
        # numeric_only first; so true too_short hits are short
        # non-roman-numeral non-numeric strings.
        ("ab", "too_short"),
        ("Zz", "too_short"),
        # Empty / whitespace
        ("", "empty"),
        ("   ", "empty"),
        # Page markers
        ("Page 5", "page_marker_only"),
        ("p. 12", "page_marker_only"),
        # Mostly digits
        ("1.2.3.4.5", "mostly_digits"),
    ],
)
def test_rejects_degenerate_patterns(text: str, expected_reason: str) -> None:
    actual = is_degenerate_heading(text)
    assert actual == expected_reason, f"text={text!r}: expected {expected_reason!r}, got {actual!r}"


@pytest.mark.parametrize(
    "text",
    [
        # The word-like-fraction rule (Rule 3) catches these three
        # ("MCRL Gs", "Ti 2 mn", "if*"): none of their tokens satisfy the
        # ≥3-alpha-+vowel word-like criterion, so the word-like fraction is
        # 0/N < 0.5.
        # Keeping as parametrize so new regressions are reported individually.
        "if*",
        "MCRL Gs",
        "Ti 2 mn",
    ],
)
def test_previously_slipping_garbage_now_caught(text: str) -> None:
    # Rule 3 (insufficient_words) catches these — no longer silently passing.
    assert is_degenerate_heading(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "Introduction",
        "Chapter 1: Suspension Geometry",
        "1.2 Aerodynamic Drag",
        "Methods",
        "Results and Discussion",
        "Hurricane Track Forecasting",
        "Appendix A: Symbols and Notation",
        # Edge case: legitimate short heading
        "Foo",
        # Short heading with section number (not mostly digits — has real text)
        "1.2.3 Vector Calculus",  # 5 of 21 chars are digits = 24%, under threshold
        # Legitimate title with year (year is not mostly digits)
        "Mercedes-Benz in Motorsport 1934-2024",
    ],
)
def test_does_not_reject_real_headings(text: str) -> None:
    reason = is_degenerate_heading(text)
    assert reason is None, f"text={text!r} should be meaningful but was rejected as {reason!r}"


def test_is_meaningful_heading_inverts_is_degenerate() -> None:
    assert is_meaningful_heading("Introduction") is True
    assert is_meaningful_heading("1") is False  # too_short
    assert is_meaningful_heading("Copyright © 2024") is False  # watermark


def test_handles_none_input() -> None:
    assert is_degenerate_heading(None) == "empty"  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    [
        "+ M.,rGxYgr + MurGxY,1r",
        "+--- VR/CI,f/10",
        "D.l",
        " SPOILER > O.J > H",
        "= 1.2 x 10^-3",
        "Gx Y gr Mur",
    ],
)
def test_equation_and_gibberish_fragments_rejected(text: str) -> None:
    assert is_degenerate_heading(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "ROD END BEARINGS",
        "RAISING RATE SUSPENSION",
        "15.3.4.2 The Marlowe Lateral Tire Force Model",
        "7.5 The Understeer Angle",
        "Chapter 12: Aerodynamic Downforce",
        "Optimizing the Brake System",
        "Aerodynamics",
    ],
)
def test_real_headings_still_accepted(text: str) -> None:
    assert is_meaningful_heading(text), is_degenerate_heading(text)


@pytest.mark.parametrize(
    "text",
    [
        # VOR-1442: the word check demanded a Latin vowel, so CJK chapter
        # titles were dropped and whole books collapsed into the one heading
        # that happened to contain Latin letters ("任务2 汽车VIN码的认知").
        "项目4 发动机工作原理及总体结构认知",
        "项目12 电控汽油机供给系统的结构认知与拆装",
        "5.2.2 主减速器形式",
        "第三章 数据分析",
        "3.5.2 Ansys Workbench静力学仿真",
        "Глава 3. Аэродинамика крыла",
        "Κεφάλαιο 2 Δυναμική",
        "第2章 機械学習の基礎",
    ],
)
def test_non_latin_script_headings_are_accepted(text: str) -> None:
    assert is_degenerate_heading(text) is None


@pytest.mark.parametrize("text", ["项", "3 发", "4.1 ×"])
def test_single_character_non_latin_fragments_still_rejected(text: str) -> None:
    assert is_degenerate_heading(text) is not None
