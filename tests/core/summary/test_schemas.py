"""Tests for vo_rag.core.summary.schemas — deterministic summary node IDs.

Covers four properties:
- ``node_id_for`` is deterministic across repeated calls.
- ``node_id_for`` is input-order independent over ``source_chunk_ids``.
- ``node_id_for`` changes when ``prompt_hash`` changes (prompt-versioned).
- Empty ``source_chunk_ids`` raises ``ValueError``.
"""

from __future__ import annotations

import pytest

from vo_rag.core.summary.schemas import NodeLevel, node_id_for


def test_same_inputs_produce_byte_identical_node_ids() -> None:
    book_id = "book-test-001"
    level = NodeLevel.SECTION
    source_chunk_ids = ["chunk-0001", "chunk-0002", "chunk-0003"]
    prompt_hash = "a" * 64

    node_id1 = node_id_for(book_id, level, source_chunk_ids, prompt_hash)
    node_id2 = node_id_for(book_id, level, source_chunk_ids, prompt_hash)

    assert node_id1 == node_id2


def test_input_order_independence() -> None:
    book_id = "book-test-001"
    level = NodeLevel.SECTION
    prompt_hash = "a" * 64

    node_id_ab = node_id_for(
        book_id, level, ["chunk-0001", "chunk-0002", "chunk-0003"], prompt_hash
    )
    node_id_ba = node_id_for(
        book_id, level, ["chunk-0003", "chunk-0001", "chunk-0002"], prompt_hash
    )

    assert node_id_ab == node_id_ba


def test_empty_source_chunk_ids_raises_value_error() -> None:
    book_id = "book-test-001"
    level = NodeLevel.SECTION
    prompt_hash = "a" * 64

    with pytest.raises(ValueError):
        node_id_for(book_id, level, [], prompt_hash)


def test_different_prompt_hash_produces_distinct_ids() -> None:
    book_id = "book-test-001"
    level = NodeLevel.SECTION
    source_chunk_ids = ["chunk-0001", "chunk-0002", "chunk-0003"]

    node_id_a = node_id_for(book_id, level, source_chunk_ids, "a" * 64)
    node_id_b = node_id_for(book_id, level, source_chunk_ids, "b" * 64)

    assert node_id_a != node_id_b


def test_different_level_produces_distinct_ids() -> None:
    book_id = "book-test-001"
    source_chunk_ids = ["chunk-0001", "chunk-0002", "chunk-0003"]
    prompt_hash = "a" * 64

    node_id_section = node_id_for(book_id, NodeLevel.SECTION, source_chunk_ids, prompt_hash)
    node_id_chapter = node_id_for(book_id, NodeLevel.CHAPTER, source_chunk_ids, prompt_hash)

    assert node_id_section != node_id_chapter


def test_different_book_id_produces_distinct_ids() -> None:
    level = NodeLevel.SECTION
    source_chunk_ids = ["chunk-0001", "chunk-0002", "chunk-0003"]
    prompt_hash = "a" * 64

    node_id_book_a = node_id_for("book-test-001", level, source_chunk_ids, prompt_hash)
    node_id_book_b = node_id_for("book-test-002", level, source_chunk_ids, prompt_hash)

    assert node_id_book_a != node_id_book_b


# ---------------------------------------------------------------------------
# Node-id determinism and prompt versioning
# ---------------------------------------------------------------------------


def test_node_id_for_is_deterministic_and_prompt_hash_versioned() -> None:
    """Ordering independence and prompt-hash versioning.

    Reversed input order must produce the same id (sorted hashing invariant).
    A different ``prompt_hash`` MUST produce a different id — that is the
    prompt-versioning leg of the deterministic id contract.
    """

    ids = ["chunk-b", "chunk-a"]
    first = node_id_for("book-1", NodeLevel.SECTION, ids, "prompt-a")
    second = node_id_for("book-1", NodeLevel.SECTION, reversed(ids), "prompt-a")
    changed = node_id_for("book-1", NodeLevel.SECTION, ids, "prompt-b")

    assert first == second
    assert first != changed
