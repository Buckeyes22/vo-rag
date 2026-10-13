from __future__ import annotations

from hashlib import sha256

from vo_rag.core.dsi.prompts import (
    DSI_EXTRACTION_PROMPT_HASH,
    DSI_EXTRACTION_SYSTEM_PROMPT,
    DSI_EXTRACTION_USER_TEMPLATE,
    DSI_PROMPT_VERSION,
)


def test_dsi_prompt_version_is_v13_for_plain_string_list_guidance() -> None:
    assert DSI_PROMPT_VERSION == "v1.3"


def test_dsi_prompt_hash_is_derived_from_system_and_user_template() -> None:
    expected = sha256((DSI_EXTRACTION_SYSTEM_PROMPT + DSI_EXTRACTION_USER_TEMPLATE).encode())

    assert expected.hexdigest() == DSI_EXTRACTION_PROMPT_HASH


def test_user_template_defines_key_components_as_physical_only() -> None:
    assert "key_components (list of strings)" in DSI_EXTRACTION_USER_TEMPLATE
    assert "PHYSICAL parts" in DSI_EXTRACTION_USER_TEMPLATE
    assert "NOT concepts" in DSI_EXTRACTION_USER_TEMPLATE
    assert (
        "`key_components`, `technical_domains`, and `prerequisite_topics` are PLAIN STRINGS"
        in DSI_EXTRACTION_USER_TEMPLATE
    )
    assert "NOT objects" in DSI_EXTRACTION_USER_TEMPLATE
    assert "do NOT attach chunk_id" in DSI_EXTRACTION_USER_TEMPLATE


def test_user_template_defines_key_concepts_as_abstract_knowledge() -> None:
    assert "key_concepts (provenance-linked" in DSI_EXTRACTION_USER_TEMPLATE
    assert "ABSTRACT knowledge" in DSI_EXTRACTION_USER_TEMPLATE
    assert "not physical objects" in DSI_EXTRACTION_USER_TEMPLATE


def test_user_template_instructs_complete_list_extraction_without_padding() -> None:
    assert (
        "For each field, extract ALL distinct, supportable items present in the chunks"
        in DSI_EXTRACTION_USER_TEMPLATE
    )
    assert "aim for completeness, not just a few" in DSI_EXTRACTION_USER_TEMPLATE
    assert "Do not pad with duplicates or vague entries" in DSI_EXTRACTION_USER_TEMPLATE
    assert "For provenance-linked fields only, every item must cite a supporting chunk" in (
        DSI_EXTRACTION_USER_TEMPLATE
    )
