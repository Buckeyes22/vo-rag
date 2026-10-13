"""Tests for vo_rag.core.parsing.corruption_gate.

Verifies the
pure fail-closed content-corruption detector flags:
- CID-font glyph-soup (e.g. char+3 caesar-shift mojibake)
- control-character contamination
- non-dictionary garbage (low dict hit + low space ratio)

and lets clean English prose through.
"""

from __future__ import annotations

from vo_rag.core.parsing.corruption_gate import assess_text_corruption

CID_SOUP = "YHUWLFDO\n\nFF]\n\nWDQJHQWLDO\n\nVXEVFULSW\x03V\n\x11\x03)LJXUH"
# Pure CID caesar-shift (no control chars): "the tire data of the vehicle model
# is shown in the figure and the equation of the chapter" with every letter +3.
# The existing CID_SOUP fixture embeds \x03/\x11 control chars that short-circuit
# to the control_char branch, so it never exercises cid_caesar_shift. This
# fixture does.
CID_SHIFTED = (
    "wkh wluh gdwd ri wkh yhklfoh prgho lv vkrzq lq wkh iljxuh dqg wkh htxdwlrq ri wkh fkdswhu"
)
CLEAN = (
    "The lateral force and SAT were calculated by inverting equations 1 "
    "and 4 to obtain the magic formula parameters for the tire model. "
    "This is a normal paragraph of clean English prose about vehicle dynamics."
)
NUMERIC = (
    "0.125,0.0312,0.250,0.0488,0.0625,0.0788,0.0963,0.1150,0.1350,0.1563,"
    "0.1788,0.2025,0.2275,0.2538,0.2813,0.3100,0.3400,0.3713,0.4038,0.4375,"
    "0.4725,0.5088,0.5463,0.5850,0.6250,0.6663,0.7088,0.7525,0.7975,0.8438,"
    "0.8913,0.9400,0.9900,1.0413,1.0938,1.1475,1.2025,1.2588,1.3163,1.3750"
)
CJK = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏" * 15
GLYPH_SOUP = "qzxkjhwpnv" * 30


def test_cid_glyph_soup_flagged_corrupt() -> None:
    v = assess_text_corruption(CID_SOUP)
    assert v.is_corrupt
    assert v.reason in {"control_char", "cid_caesar_shift", "non_dictionary"}


def test_cid_caesar_shift_reason_locked() -> None:
    """VOR-297 regression: the cid_caesar_shift branch must still fire for real CID mojibake."""
    v = assess_text_corruption(CID_SHIFTED)
    assert v.is_corrupt
    assert v.reason == "cid_caesar_shift"


def test_numeric_tabular_not_flagged() -> None:
    """VOR-297: numeric/tabular content (airfoil coords, odds) is not corruption."""
    assert len(NUMERIC) > 200
    v = assess_text_corruption(NUMERIC)
    assert not v.is_corrupt
    assert v.reason == "ok"


def test_cjk_text_not_flagged() -> None:
    """VOR-297: non-Latin (CJK) text is not corruption — dictionary is English-only."""
    assert len(CJK) > 200
    v = assess_text_corruption(CJK)
    assert not v.is_corrupt
    assert v.reason == "ok"


def test_latin_glyph_soup_still_flagged() -> None:
    """VOR-297 guard must not over-suppress: genuine Latin glyph-soup still flags."""
    assert len(GLYPH_SOUP) > 200
    v = assess_text_corruption(GLYPH_SOUP)
    assert v.is_corrupt
    assert v.reason == "non_dictionary"


def test_clean_prose_passes() -> None:
    v = assess_text_corruption(CLEAN)
    assert not v.is_corrupt
    assert v.reason == "ok"


def test_empty_is_not_corrupt() -> None:
    assert not assess_text_corruption("").is_corrupt
    assert not assess_text_corruption("   \n  ").is_corrupt
