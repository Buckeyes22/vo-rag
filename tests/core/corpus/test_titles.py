"""Book-title normalization and "already replaced" matching (VOR-1427)."""

from __future__ import annotations

import pytest

from vo_rag.core.corpus.titles import (
    is_already_replaced,
    is_truncated_title,
    normalize_title,
    title_year,
)


def test_normalize_title_folds_reencode_suffix_and_id_tag() -> None:
    assert normalize_title("101 garage performance projec_.reencoded") == normalize_title(
        "101 garage performance projec_"
    )
    assert normalize_title("Strengthcraft [12345678]") == normalize_title("Strengthcraft")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Strengthcraft", "strengthcraft"),
        ("Strengthcraft.pdf", "strengthcraft"),
        ("Strengthcraft.EPUB", "strengthcraft"),
        ("Strengthcraft (2009).pdf", "strengthcraft"),
        ("Strengthcraft [12345678]", "strengthcraft"),
        ("Strengthcraft (1234567)", "strengthcraft"),
        ("Strengthcraft_12345678", "strengthcraft"),
        ("Strengthcraft___12345678", "strengthcraft"),
        ("Strengthcraft__", "strengthcraft"),
        ("Strengthcraft.reencoded", "strengthcraft"),
        ("Strengthcraft.reencoded.pdf", "strengthcraft"),
        (
            "Practical_Engines_Applied_Dynam_",
            "practical engines applied dynam",
        ),
        (
            "Practical Engines: Applied Dynamics",
            "practical engines applied dynamics",
        ),
        ("  The   Art of   War!  ", "the art of war"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


def test_normalize_title_keeps_short_digit_runs_and_non_year_parens() -> None:
    # Edition numbers and 4-digit non-parenthesized numbers are part of the title.
    assert normalize_title("Chess 1234 (2nd ed)") == "chess 1234 2nd ed"
    assert normalize_title("Vol. 2") == "vol 2"


@pytest.mark.parametrize(
    ("raw", "truncated"),
    [
        ("Practical_Engines_Applied_Dynam_", True),
        ("Some Long Title_12345678", True),
        ("Some Long Title__12345678.pdf", True),
        ("Some Long Title_.reencoded", True),
        ("Some Long Title_.pdf", True),
        ("Practical Engines: Applied Dynamics", False),
        ("Strengthcraft [12345678]", False),
        ("", False),
    ],
)
def test_is_truncated_title(raw: str, truncated: bool) -> None:
    assert is_truncated_title(raw) is truncated


def test_exact_normalized_match_is_replaced() -> None:
    completed = {normalize_title("Strengthcraft")}
    assert is_already_replaced("Strengthcraft [12345678].pdf", completed) is True


def test_truncated_failed_title_prefix_of_completed_is_replaced() -> None:
    completed = {normalize_title("Practical Engines: Applied Dynamics")}
    assert is_already_replaced("Practical_Engines_Applied_Dynam_", completed) is True


def test_planning_false_match_stays_unmatched() -> None:
    """The reverse direction (failed starts with a shorter completed title) is never used."""
    completed = {normalize_title("Practical Engine")}
    assert normalize_title("Practical_Engines_Applied_Dynam_") == (
        "practical engines applied dynam"
    )
    assert is_already_replaced("Practical_Engines_Applied_Dynam_", completed) is False
    assert is_already_replaced("Practical Engines: Applied Dynamics", completed) is False


def test_prefix_match_requires_truncated_raw_title() -> None:
    completed = {normalize_title("Practical Rowing Skills Second Edition")}
    assert is_already_replaced("Practical Rowing Skills", completed) is False
    assert is_already_replaced("Practical Rowing Skills_", completed) is True


def test_prefix_match_requires_twelve_normalized_chars() -> None:
    completed = {normalize_title("Chess Openings for Beginners")}
    # "chess openi" is 11 chars -- too short to trust a prefix match.
    assert is_already_replaced("Chess Openi_", completed) is False
    assert is_already_replaced("Chess Openin_", completed) is True


def test_empty_title_never_matches() -> None:
    assert is_already_replaced("", {"", "anything"}) is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Applied Strength Methods (2013)", "2013"),
        ("Applied Strength Methods (2019).pdf", "2019"),
        ("Applied Strength Methods", None),
        ("Applied Strength Methods (2nd ed)", None),
        ("Strengthcraft (2009).pdf", "2009"),
        ("Chess 1234 (2nd ed)", None),
        ("", None),
    ],
)
def test_title_year(raw: str, expected: str | None) -> None:
    """`title_year` reports the parenthesized
    publication year `normalize_title` strips, WITHOUT itself changing
    `normalize_title` -- `plan_backlog`'s title matching depends on years
    staying folded away there."""
    assert title_year(raw) == expected


def test_title_year_with_two_parenthesized_years_returns_the_first() -> None:
    """Pins the documented "first match wins"
    behaviour for a title carrying two `(YYYY)` markers."""
    assert title_year("Applied Strength Methods (2013) (2019)") == "2013"


def test_normalize_title_keeps_generic_numeric_suffix_rules() -> None:
    from vo_rag.core.corpus.titles import normalize_title

    assert normalize_title("Strength Training [12345678].pdf") == normalize_title(
        "Strength Training.pdf"
    )
