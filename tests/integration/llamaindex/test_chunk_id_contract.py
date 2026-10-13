"""Chunk-id contract: SearchResult.id is not the logical chunk id.

Decision (VOR-237):
- Keep ``SearchResult`` narrow: ``id``, ``score``, ``collection``, ``payload``.
- The logical LlamaIndex node id is read from ``payload["chunk_id"]``.
- ``SearchResult.id`` is the Qdrant point id and is preserved only as
  ``metadata["qdrant_point_id"]``.
- A missing ``payload["chunk_id"]`` fails closed with ``ChunkIdMissingError``;
  the converter must never fall back to ``SearchResult.id``.

Source seams pinned here:
- ``core/retrieval/search.py``: ``SearchResult.id`` is the Qdrant point id and
  ``payload`` is the unparsed Qdrant payload.
- ``core/storage/qdrant.py``: ``chunk_id`` is an indexed payload field on the
  contextualized persisted-hit schema.
- ``core/storage/qdrant.py``: ``deterministic_point_id`` is index-addressed by
  ``source_id`` + ``chunk_index`` + ``stage``.
- ``core/chunking/ids.py``: ``_chunk_id`` is content-addressed by
  ``source_id`` + content digest + ``stage``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import pytest
from qdrant_client.models import PayloadSchemaType

from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.retrieval.search import SearchResult
from vo_rag.core.storage.qdrant import (
    CONTEXTUALIZED_EXTRA_INDEXES,
    deterministic_point_id,
)
from vo_rag.integrations.llamaindex.convert import (
    ChunkIdMissingError,
    search_result_to_node_with_score,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_SOURCE_ID = "book:k1-contract"
_CHUNK_INDEX = 7
_STAGE = "raw"
_CONTENT = "The same logical chunk can move index slots without changing content identity."
_EXPECTED_POINT_ID = "a8f7dd76-22b8-e025-5c0c-fb6525c15597"
_EXPECTED_CHUNK_ID = "da2b75d3-03e3-31c0-989c-0700c6ec9a63"


def test_search_result_stays_narrow_without_chunk_id_field() -> None:
    """The chunk id lives at the payload/converter seam; SearchResult stays narrow."""
    assert {field.name for field in fields(SearchResult)} == {
        "id",
        "score",
        "collection",
        "payload",
    }


def test_chunk_id_is_indexed_payload_field() -> None:
    """A missing payload chunk_id is malformed/legacy data, not a fallback case."""
    assert CONTEXTUALIZED_EXTRA_INDEXES["chunk_id"] == PayloadSchemaType.KEYWORD


def test_point_id_and_chunk_id_are_distinct_deterministic_spaces() -> None:
    """Fixed bytes prove SearchResult.id and payload["chunk_id"] are different keys."""
    point_id = deterministic_point_id(_SOURCE_ID, _CHUNK_INDEX, _STAGE)
    chunk_id = _chunk_id(_SOURCE_ID, _CONTENT, _STAGE)

    assert point_id == _EXPECTED_POINT_ID
    assert chunk_id == _EXPECTED_CHUNK_ID
    assert point_id != chunk_id
    assert deterministic_point_id(_SOURCE_ID, _CHUNK_INDEX + 1, _STAGE) != point_id


def test_converter_reads_payload_chunk_id_and_preserves_point_id(
    make_search_result: Callable[..., SearchResult],
) -> None:
    """The converter keys node.id_ on payload["chunk_id"], never SearchResult.id."""
    point_id = deterministic_point_id(_SOURCE_ID, _CHUNK_INDEX, _STAGE)
    chunk_id = _chunk_id(_SOURCE_ID, _CONTENT, _STAGE)
    result = make_search_result(
        id=point_id,
        payload={
            "chunk_id": chunk_id,
            "content": _CONTENT,
            "section_path": "K1 > Identity",
        },
    )

    assert result.id != result.payload["chunk_id"]

    node_with_score = search_result_to_node_with_score(result)

    assert node_with_score.node.id_ == chunk_id
    assert node_with_score.node.id_ != result.id
    assert node_with_score.node.metadata["qdrant_point_id"] == point_id


def test_converter_missing_payload_chunk_id_fails_closed(
    make_search_result: Callable[..., SearchResult],
) -> None:
    """A malformed hit raises instead of silently re-keying identity to result.id."""
    result = make_search_result(
        id=deterministic_point_id(_SOURCE_ID, _CHUNK_INDEX, _STAGE),
        payload={"content": _CONTENT, "section_path": "K1 > Identity"},
    )

    with pytest.raises(ChunkIdMissingError, match=r"payload\['chunk_id'\]"):
        search_result_to_node_with_score(result)
