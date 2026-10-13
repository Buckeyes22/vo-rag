"""Tests for the Layer-1A summary JSON-response parser + truncation salvage.

The parser must extract clean ``summary_text`` from the LLM
JSON, stash the structured extract under ``metadata.summary_extract``, and
gracefully fall back to a raw string when the response is unparseable.
"""

from __future__ import annotations

from vo_rag.core.summary.builder import _parse_summary_response


def test_extracts_summary_text_and_stashes_structured_fields() -> None:
    raw = (
        '{"summary_text": "Clean faithful prose.", '
        '"key_points": ["a", "b"], "technical_terms": ["slip angle"], '
        '"figure_dependent_points": [], "caveats": []}'
    )
    text, meta = _parse_summary_response(raw)
    assert text == "Clean faithful prose."
    assert meta == {
        "summary_extract": {
            "key_points": ["a", "b"],
            "technical_terms": ["slip angle"],
            "figure_dependent_points": [],
            "caveats": [],
        }
    }
    assert "summary_text" not in text and "{" not in text


def test_strips_markdown_json_fence() -> None:
    raw = '```json\n{"summary_text": "Fenced summary."}\n```'
    text, meta = _parse_summary_response(raw)
    assert text == "Fenced summary."
    assert meta == {}


def test_salvages_summary_text_from_truncated_json() -> None:
    # The real failure mode: trailing arrays cut at max_tokens => invalid JSON,
    # but summary_text (first field) is intact and must be recovered as clean prose.
    raw = (
        '{\n  "summary_text": "The traction circle plots lateral vs longitudinal '
        'acceleration to bound grip.",\n  "key_points": [\n    "Combined accel is the '
        'vector sum",\n    "Used to find unu'  # <-- truncated mid-array
    )
    text, meta = _parse_summary_response(raw)
    assert text == ("The traction circle plots lateral vs longitudinal acceleration to bound grip.")
    assert meta == {"summary_parse_fallback": "salvaged_summary_text"}
    assert not text.startswith("{")


def test_salvage_unescapes_quotes_and_newlines() -> None:
    raw = '{"summary_text": "He said \\"go\\".\\nLine two.", "key_points": ['  # truncated
    text, meta = _parse_summary_response(raw)
    assert text == 'He said "go".\nLine two.'
    assert meta == {"summary_parse_fallback": "salvaged_summary_text"}


def test_unparseable_non_json_falls_back_to_raw() -> None:
    raw = "This is plain prose, not JSON at all."
    text, meta = _parse_summary_response(raw)
    assert text == raw
    assert meta == {"summary_parse_fallback": "unparseable"}


def test_missing_summary_text_key_is_unparseable() -> None:
    raw = '{"key_points": ["x"]}'
    text, meta = _parse_summary_response(raw)
    assert text == raw
    assert meta == {"summary_parse_fallback": "unparseable"}


def test_empty_summary_text_is_unparseable() -> None:
    raw = '{"summary_text": "   "}'
    _text, meta = _parse_summary_response(raw)
    assert meta == {"summary_parse_fallback": "unparseable"}
