"""Tests for the code-aware BM25 tokenizer.

Shares the same blake2b-4 persistence contract as the base ``bm25_tokenizer``
(via ``tokens_to_bm25_sparse``); only the tokenization rule differs.
"""

from __future__ import annotations

from vo_rag.core.retrieval.bm25_tokenizer import _stable_token_id
from vo_rag.core.retrieval.code_bm25_tokenizer import to_sparse, tokenize


def test_camelCase_splits() -> None:
    out = tokenize("useHourlyForecast")
    assert "use" in out
    assert "hourly" in out
    assert "forecast" in out


def test_PascalCase_splits() -> None:
    out = tokenize("HourlyForecastStrip")
    assert "hourly" in out
    assert "forecast" in out
    assert "strip" in out


def test_snake_case_splits() -> None:
    out = tokenize("hourly_forecast_strip")
    assert "hourly" in out
    assert "forecast" in out
    assert "strip" in out


def test_path_segments_split() -> None:
    out = tokenize("src/components/weather/HourlyForecastStrip.tsx")
    assert "src" in out
    assert "components" in out
    assert "weather" in out
    assert "hourly" in out
    assert "forecast" in out
    assert "strip" in out
    assert "tsx" in out


def test_dotted_identifiers_preserved_plus_split() -> None:
    out = tokenize("module.Class.method")
    # Both forms present
    assert "module" in out
    assert "class" in out
    assert "method" in out
    assert "module.class.method" in out


def test_stopwords_filtered() -> None:
    out = tokenize("the quick brown fox jumps over the lazy dog")
    assert "the" not in out
    # Content words remain
    assert "quick" in out
    assert "fox" in out


def test_to_sparse_returns_dict() -> None:
    tokens = ["foo", "bar", "foo"]
    sparse = to_sparse(tokens)
    assert isinstance(sparse, dict)
    # TF of "foo" should be 2; "bar" should be 1 - but the IDs are hashed
    values = sorted(sparse.values())
    assert values == [1.0, 2.0] or values == sorted([2.0, 1.0])


def test_max_tokens_cap() -> None:
    long = "word " * 20_000  # 20K words > 16384 cap
    out = tokenize(long)
    assert len(out) <= 16_384


def test_to_sparse_uses_base_persistence_contract() -> None:
    """Code tokenizer's sparse output must use the same blake2b-4 ids."""
    toks = ["foo", "bar", "foo"]
    sparse = to_sparse(toks)
    # Same id scheme as the base tokenizer
    assert _stable_token_id("foo") in sparse
    assert _stable_token_id("bar") in sparse
    assert sparse[_stable_token_id("foo")] == 2.0
    assert sparse[_stable_token_id("bar")] == 1.0
