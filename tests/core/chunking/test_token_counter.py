from __future__ import annotations

import pytest

from vo_rag.core.chunking.token_counter import TokenCounter, default_token_counter


@pytest.fixture(scope="module")
def tc() -> TokenCounter:
    return default_token_counter()


def test_empty_string_zero(tc: TokenCounter) -> None:
    assert tc.count("") == 0
    assert tc.count("   ") >= 0  # whitespace may or may not produce tokens


def test_short_string_small_count(tc: TokenCounter) -> None:
    n = tc.count("hello world")
    assert 1 <= n <= 10


def test_stable_across_calls(tc: TokenCounter) -> None:
    s = "This is a sentence. " * 40
    assert tc.count(s) == tc.count(s)


def test_default_token_counter_is_singleton() -> None:
    assert default_token_counter() is default_token_counter()


def test_callable_matches_count(tc: TokenCounter) -> None:
    s = "A different sentence."
    assert tc(s) == tc.count(s)
