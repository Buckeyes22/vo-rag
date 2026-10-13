"""Tests for vo_rag.core.parsing.text_normalization.

Verifies:
- Private-use PDF ligature glyphs are mapped to ASCII
- Standard Unicode ligatures are NFKC-normalized
- Hard end-of-line hyphens are removed (de-hyphenation)
- Mid-line spaced hyphens (ambiguous) are preserved
- Bidi-override codepoints (F-099) are stripped (Trojan-source defense)
"""

from __future__ import annotations

from vo_rag.core.parsing.text_normalization import normalize_extracted_text


def test_normalize_private_use_pdf_ligatures() -> None:
    raw = (
        "e driver had a la of grip, then researed the "
        "le-front tire aer a wrien briefing from een's."
    )

    assert normalize_extracted_text(raw) == (
        "The driver had a lack of grip, then researched the "
        "left-front tire after a written briefing from Queen's."
    )


def test_normalize_standard_unicode_ligatures() -> None:
    assert normalize_extracted_text("scientiﬁc staﬀ") == "scientific staff"


def test_dehyphenates_hard_line_wraps() -> None:
    assert normalize_extracted_text("co-\noperate") == "cooperate"
    assert normalize_extracted_text("frac-\nture") == "fracture"


def test_keeps_undecidable_mid_line_spaced_hyphens() -> None:
    assert normalize_extracted_text("boot- mounted") == "boot- mounted"
    assert normalize_extracted_text("under- or") == "under- or"
    assert normalize_extracted_text("reason- ably") == "reason- ably"
    assert normalize_extracted_text("state-of-the-art") == "state-of-the-art"


def test_strips_bidi_override_codepoints() -> None:
    """F-099: U+202E and friends never survive normalization."""
    hostile = "admin‮.txt"
    cleaned = normalize_extracted_text(hostile)
    assert "‮" not in cleaned
    assert cleaned == "admin.txt"


def test_strips_all_bidi_directional_codepoints() -> None:
    codepoints = [
        "‪",
        "‫",
        "‬",
        "‭",
        "‮",
        "⁦",
        "⁧",
        "⁨",
        "⁩",
    ]
    for cp in codepoints:
        result = normalize_extracted_text(f"prefix{cp}suffix")
        assert cp not in result, f"codepoint U+{ord(cp):04X} survived"
        assert result == "prefixsuffix"


def test_empty_input_returns_empty() -> None:
    assert normalize_extracted_text("") == ""
