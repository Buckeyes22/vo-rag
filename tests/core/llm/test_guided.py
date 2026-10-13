"""Tests for the shared structured-output ``extra`` builder.

``structured_output_extra`` centralizes how a JSON Schema is expressed on the
wire for advanced-layer extraction, selected by ``VORAG_GUIDED_DECODING_MODE``.
Both the CLI build path and the LlamaIndex AdvancedBuildWorkflow delegate here.
"""

from __future__ import annotations

import pytest

from vo_rag.core.llm.guided import structured_output_extra

_SCHEMA = {"type": "object", "required": ["kind"]}


def test_default_mode_uses_response_format_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VORAG_GUIDED_DECODING_MODE", raising=False)

    extra = structured_output_extra(_SCHEMA)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": _SCHEMA},
    }
    assert "guided_json" not in extra
    assert "json_schema" not in extra


def test_guided_json_mode_uses_legacy_top_level_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "guided_json")

    extra = structured_output_extra(_SCHEMA)

    assert extra["guided_json"] == _SCHEMA
    assert "response_format" not in extra


def test_native_json_schema_mode_uses_top_level_json_schema_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp native mode: schema goes in a top-level ``json_schema`` field.

    Build b8234's OpenAI ``response_format: json_schema`` shim does not constrain
    decoding (it 500s on the strict parse of free text), but its native top-level
    ``json_schema`` request field enforces the grammar (VOR-302).
    """
    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "native_json_schema")

    extra = structured_output_extra(_SCHEMA)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["json_schema"] == _SCHEMA
    assert "response_format" not in extra
    assert "guided_json" not in extra


def test_json_object_mode_drops_schema_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "json_object")

    extra = structured_output_extra(_SCHEMA)

    assert extra["response_format"] == {"type": "json_object"}
    assert "guided_json" not in extra
    assert "json_schema" not in extra


def test_unknown_mode_falls_back_to_response_format_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "totally-bogus")

    extra = structured_output_extra(_SCHEMA)

    assert extra["response_format"]["type"] == "json_schema"


def test_no_schema_always_uses_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "native_json_schema")

    extra = structured_output_extra(None)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["response_format"] == {"type": "json_object"}
    assert "json_schema" not in extra
    assert "guided_json" not in extra
