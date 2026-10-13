from __future__ import annotations

from hashlib import sha256

from vo_rag.core.graph.prompts import (
    GRAPH_EXTRACTION_PROMPT_HASH,
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
    GRAPH_EXTRACTION_USER_TEMPLATE,
    GRAPH_PROMPT_VERSION,
)


def test_graph_prompt_version_is_bumped_for_directed_edge_semantics() -> None:
    assert GRAPH_PROMPT_VERSION == "v2.4"


def test_graph_prompt_hash_is_derived_from_system_and_user_template() -> None:
    expected = sha256((GRAPH_EXTRACTION_SYSTEM_PROMPT + GRAPH_EXTRACTION_USER_TEMPLATE).encode())

    assert expected.hexdigest() == GRAPH_EXTRACTION_PROMPT_HASH


def test_user_template_documents_directed_relationship_semantics() -> None:
    assert "## Relationship types" in GRAPH_EXTRACTION_USER_TEMPLATE
    assert "SOURCE -> edge_type -> TARGET" in GRAPH_EXTRACTION_USER_TEMPLATE
    assert "Direction check" in GRAPH_EXTRACTION_USER_TEMPLATE


def test_relationship_types_section_names_every_edge_type() -> None:
    section = GRAPH_EXTRACTION_USER_TEMPLATE.split("## Relationship types", maxsplit=1)[1].split(
        "Rules:", maxsplit=1
    )[0]

    assert "- is_a:" in section
    assert "- part_of:" in section
    assert "- used_by:" in section
    assert "- introduces:" in section
    assert "- authored_by:" in section
    assert "- mentions:" in section
    assert "- relates_to:" in section


def test_user_template_documents_taxonomy_and_used_by_direction() -> None:
    assert (
        "- is_a: SOURCE is a specific type, instance, or member of the general category TARGET."
        in GRAPH_EXTRACTION_USER_TEMPLATE
    )
    assert "Use is_a (NOT part_of) for type/category/instance/membership relationships." in (
        GRAPH_EXTRACTION_USER_TEMPLATE
    )
    assert 'If "SOURCE is a kind of TARGET" reads true, use is_a instead.' in (
        GRAPH_EXTRACTION_USER_TEMPLATE
    )
    assert "The TOOL is always the SOURCE." in GRAPH_EXTRACTION_USER_TEMPLATE
    assert "Never put the process/technique as source." in GRAPH_EXTRACTION_USER_TEMPLATE


def test_user_template_blocks_physical_parts_as_used_by() -> None:
    assert "tool/component/material" not in GRAPH_EXTRACTION_USER_TEMPLATE
    assert "installed on, mounted to, bolted to, fitted to, or equipped on" in (
        GRAPH_EXTRACTION_USER_TEMPLATE
    )
    assert "XIDA Dampers -> part_of -> SuperMiata" in GRAPH_EXTRACTION_USER_TEMPLATE
    assert "NA8 SM Engine -> used_by -> Intake" in GRAPH_EXTRACTION_USER_TEMPLATE


def test_user_template_biases_toward_canonical_directed_relations() -> None:
    assert "Prefer part_of, is_a, or used_by when a directed relationship is plausible." in (
        GRAPH_EXTRACTION_USER_TEMPLATE
    )
    assert "Reserve relates_to for genuinely symmetric or unknown relationships." in (
        GRAPH_EXTRACTION_USER_TEMPLATE
    )
