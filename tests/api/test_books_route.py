"""Book-reader API route coverage."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from tests.api.conftest import FakeQdrantStore, auth_header, make_client, make_services

SOURCE_ID = "vo-rag-corpus-sha256:" + "a" * 64


class FakeBooksQdrantClient:
    def __init__(self, points_by_collection: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.points_by_collection = points_by_collection or {}

    async def count(
        self,
        collection_name: str,
        count_filter: Any | None = None,
        exact: bool = True,
    ) -> SimpleNamespace:
        points = self._matching_payloads(collection_name, count_filter)
        return SimpleNamespace(count=len(points))

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any | None = None,
        limit: int = 100,
        offset: Any | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
    ) -> tuple[list[SimpleNamespace], int | None]:
        del with_vectors
        points = self._matching_payloads(collection_name, scroll_filter)
        start = int(offset or 0)
        page = points[start : start + limit]
        next_offset = start + limit if start + limit < len(points) else None
        return [
            SimpleNamespace(id=payload.get("chunk_id", str(i)), payload=payload)
            for i, payload in enumerate(page)
        ], next_offset

    def _matching_payloads(
        self, collection_name: str, qdrant_filter: Any | None
    ) -> list[dict[str, Any]]:
        return [
            payload
            for payload in self.points_by_collection.get(collection_name, [])
            if _matches_filter(payload, qdrant_filter)
        ]


class FakeBookPool:
    def __init__(
        self,
        *,
        source_row: dict[str, Any] | None = None,
        artifact_row: dict[str, Any] | None = None,
    ) -> None:
        self.source_row = source_row
        self.artifact_row = artifact_row
        self.queries: list[str] = []

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        del args
        self.queries.append(query)
        if "FROM sources" in query:
            return self.source_row
        if "FROM knowledge_artifacts" in query:
            return self.artifact_row
        raise AssertionError(f"unexpected query: {query}")


def _matches_filter(payload: dict[str, Any], qdrant_filter: Any | None) -> bool:
    conditions: Iterable[Any] = getattr(qdrant_filter, "must", None) or []
    for condition in conditions:
        expected = getattr(getattr(condition, "match", None), "value", None)
        actual = payload.get(condition.key)
        if isinstance(actual, list):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    should_conditions: Iterable[Any] = getattr(qdrant_filter, "should", None) or []
    if should_conditions:
        matched_should = False
        for condition in should_conditions:
            expected = getattr(getattr(condition, "match", None), "value", None)
            if expected is not None and payload.get(condition.key) == expected:
                matched_should = True
            empty_field = getattr(getattr(condition, "is_empty", None), "key", None)
            if empty_field is not None and payload.get(empty_field) is None:
                matched_should = True
        if not matched_should:
            return False
    return True


def _source_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": SOURCE_ID,
        "title": "Applied Chassis Dynamics",
        "author": "Marlow",
        "license_class": "cleared",
    }
    row.update(overrides)
    return row


def _artifact_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "artifact-1",
        "content": {
            "source_id": SOURCE_ID,
            "stats": {
                "text_chunks": 3,
                "figure_chunks": 1,
                "pages": 12,
                "domain_tags": ["motorsport", "vehicle-dynamics"],
            },
        },
    }
    row.update(overrides)
    return row


def _client(monkeypatch: Any, qdrant: FakeBooksQdrantClient, pool: FakeBookPool):
    return make_client(
        make_services(db_pool=pool, qdrant=FakeQdrantStore(_client=qdrant)),
        monkeypatch=monkeypatch,
    )


def test_book_overview_happy_path(monkeypatch) -> None:
    qdrant = FakeBooksQdrantClient(
        {
            "vo_rag_books_raw_v1": [
                {"source_id": SOURCE_ID, "chunk_id": "c-1"},
                {"source_id": SOURCE_ID, "chunk_id": "c-2"},
                {"source_id": "other", "chunk_id": "c-3"},
            ],
            "vo_rag_books_summary_tree_v1": [
                {"book_id": SOURCE_ID},
                {"book_id": SOURCE_ID},
                {"book_id": SOURCE_ID},
            ],
            "vo_rag_books_raptor_v1": [{"book_id": SOURCE_ID}, {"book_id": SOURCE_ID}],
            "vo_rag_books_dsi_v1": [{"book_id": SOURCE_ID}],
            "vo_rag_books_graph_nodes_v1": [
                {"book_ids": [SOURCE_ID]},
                {"book_ids": ["other", SOURCE_ID]},
                {"book_ids": ["other"]},
            ],
        }
    )
    client = _client(
        monkeypatch, qdrant, FakeBookPool(source_row=_source_row(), artifact_row=_artifact_row())
    )

    response = client.get(f"/api/v1/books/{SOURCE_ID}", headers=auth_header())

    assert response.status_code == 200
    assert response.json() == {
        "source_id": SOURCE_ID,
        "title": "Applied Chassis Dynamics",
        "author": "Marlow",
        "license_class": "cleared",
        "artifact_id": "artifact-1",
        "stats": {
            "text_chunks": 3,
            "figure_chunks": 1,
            "pages": 12,
            "domain_tags": ["motorsport", "vehicle-dynamics"],
        },
        "layers": {"summary_tree": 3, "raptor": 2, "dsi": 1, "graph_nodes": 2},
    }


def test_book_overview_404_when_no_source_row_or_raw_chunks(monkeypatch) -> None:
    client = _client(monkeypatch, FakeBooksQdrantClient(), FakeBookPool())

    response = client.get(f"/api/v1/books/{SOURCE_ID}", headers=auth_header())

    assert response.status_code == 404
    assert response.json()["detail"] == "book_not_found"


def test_book_overview_rejects_nonactive_generation_points(monkeypatch) -> None:
    qdrant = FakeBooksQdrantClient(
        {
            "vo_rag_books_raw_v1": [
                {"source_id": SOURCE_ID, "chunk_id": state, "ingest_state": state}
                for state in ("staging", "failed", "quarantined", "superseded")
            ]
        }
    )
    client = _client(monkeypatch, qdrant, FakeBookPool())

    response = client.get(f"/api/v1/books/{SOURCE_ID}", headers=auth_header())

    assert response.status_code == 404


def test_book_page_returns_sorted_chunks_and_max_page(monkeypatch) -> None:
    qdrant = FakeBooksQdrantClient(
        {
            "vo_rag_books_raw_v1": [
                {
                    "source_id": SOURCE_ID,
                    "page_number": 7,
                    "chunk_id": "chunk-b",
                    "section_title": "Late section",
                    "content": "Second chunk",
                    "has_image": True,
                    "image_id": "img-2",
                    "vlm_description": "damper plot",
                    "token_count": 11,
                },
                {
                    "source_id": SOURCE_ID,
                    "page_number": 7,
                    "chunk_id": "chunk-a",
                    "section_title": None,
                    "content": "First chunk",
                    "has_image": False,
                    "image_id": None,
                    "vlm_description": None,
                    "token_count": 7,
                },
                {"source_id": SOURCE_ID, "page_number": 8, "chunk_id": "chunk-c"},
            ]
        }
    )
    client = _client(monkeypatch, qdrant, FakeBookPool(artifact_row=_artifact_row()))

    response = client.get(f"/api/v1/books/{SOURCE_ID}/pages/7", headers=auth_header())

    assert response.status_code == 200
    assert response.json() == {
        "page": 7,
        "max_page": 12,
        "chunks": [
            {
                "chunk_id": "chunk-a",
                "section_title": None,
                "text": "First chunk",
                "has_image": False,
                "image_id": None,
                "vlm_description": None,
                "token_count": 7,
            },
            {
                "chunk_id": "chunk-b",
                "section_title": "Late section",
                "text": "Second chunk",
                "has_image": True,
                "image_id": "img-2",
                "vlm_description": "damper plot",
                "token_count": 11,
            },
        ],
    }


def test_book_page_allows_empty_blank_page(monkeypatch) -> None:
    client = _client(
        monkeypatch, FakeBooksQdrantClient(), FakeBookPool(artifact_row=_artifact_row())
    )

    response = client.get(f"/api/v1/books/{SOURCE_ID}/pages/9", headers=auth_header())

    assert response.status_code == 200
    assert response.json() == {"page": 9, "max_page": 12, "chunks": []}


def test_book_page_returns_only_active_or_legacy_generation_chunks(monkeypatch) -> None:
    points = [
        {
            "source_id": SOURCE_ID,
            "page_number": 7,
            "chunk_id": state,
            "content": state,
            "ingest_state": state,
        }
        for state in ("active", "staging", "failed", "quarantined", "superseded")
    ]
    points.append(
        {
            "source_id": SOURCE_ID,
            "page_number": 7,
            "chunk_id": "legacy",
            "content": "legacy",
        }
    )
    client = _client(
        monkeypatch,
        FakeBooksQdrantClient({"vo_rag_books_raw_v1": points}),
        FakeBookPool(),
    )

    response = client.get(f"/api/v1/books/{SOURCE_ID}/pages/7", headers=auth_header())

    assert response.status_code == 200
    assert [chunk["chunk_id"] for chunk in response.json()["chunks"]] == [
        "active",
        "legacy",
    ]


def test_book_artifact_lookup_requires_active_canonical_artifact(monkeypatch) -> None:
    pool = FakeBookPool(source_row=_source_row(), artifact_row=_artifact_row())
    client = _client(monkeypatch, FakeBooksQdrantClient(), pool)

    response = client.get(f"/api/v1/books/{SOURCE_ID}", headers=auth_header())

    assert response.status_code == 200
    artifact_query = next(query for query in pool.queries if "knowledge_artifacts" in query)
    assert "status = 'active'" in artifact_query
    assert "COALESCE(content->>'raw_collection', 'vo_rag_books_raw_v1')" in artifact_query


def test_book_page_rejects_less_than_one(monkeypatch) -> None:
    client = _client(monkeypatch, FakeBooksQdrantClient(), FakeBookPool())

    response = client.get(f"/api/v1/books/{SOURCE_ID}/pages/0", headers=auth_header())

    assert response.status_code == 422


def test_book_summaries_returns_all_layers(monkeypatch) -> None:
    qdrant = FakeBooksQdrantClient(
        {
            "vo_rag_books_dsi_v1": [
                {
                    "book_id": SOURCE_ID,
                    "chapter_outline": "Chapter 1: tires",
                    "key_concepts": [{"value": "slip angle"}, "load transfer", {"bad": "skip"}],
                    "key_methods": [{"value": "free-body diagram"}],
                    "key_equations_or_models": [{"value": "F = ma"}],
                    "prerequisite_topics": ["vectors"],
                    "figure_heavy_topics": ["suspension geometry"],
                }
            ],
            "vo_rag_books_summary_tree_v1": [
                {
                    "book_id": SOURCE_ID,
                    "level": "chapter",
                    "display_path": "02 Aero",
                    "metadata": {"summary_extract": {"key_points": ["downforce"]}},
                },
                {
                    "book_id": SOURCE_ID,
                    "level": "book",
                    "display_path": None,
                    "metadata": {"summary_extract": {"key_points": ["whole book"]}},
                },
                {
                    "book_id": SOURCE_ID,
                    "level": "section",
                    "display_path": "01 Tires / Slip",
                    "metadata": {"summary_extract": {"key_points": ["ignored in response"]}},
                },
                {
                    "book_id": SOURCE_ID,
                    "level": "chapter",
                    "display_path": "01 Tires",
                    "metadata": {"summary_extract": {"key_points": ["contact patch"]}},
                },
            ],
            "vo_rag_books_graph_nodes_v1": [
                {
                    "book_ids": [SOURCE_ID],
                    "label": "Slip Angle",
                    "node_type": "concept",
                    "chunk_ids": ["c1", "c2", "c3"],
                },
                {
                    "book_ids": [SOURCE_ID],
                    "label": "Camber",
                    "node_type": "concept",
                    "chunk_ids": ["c4"],
                },
            ],
            "vo_rag_books_raptor_v1": [{"book_id": SOURCE_ID}, {"book_id": SOURCE_ID}],
        }
    )
    client = _client(monkeypatch, qdrant, FakeBookPool())

    response = client.get(f"/api/v1/books/{SOURCE_ID}/summaries", headers=auth_header())

    assert response.status_code == 200
    assert response.json() == {
        "dsi": {
            "chapter_outline": "Chapter 1: tires",
            "key_concepts": ["slip angle", "load transfer"],
            "key_methods": ["free-body diagram"],
            "key_equations_or_models": ["F = ma"],
            "prerequisite_topics": ["vectors"],
            "figure_heavy_topics": ["suspension geometry"],
        },
        "summary_tree": {
            "book": [{"display_path": None, "key_points": ["whole book"]}],
            "chapter": [
                {"display_path": "01 Tires", "key_points": ["contact patch"]},
                {"display_path": "02 Aero", "key_points": ["downforce"]},
            ],
            "section_count": 1,
        },
        "graph": {
            "node_count": 2,
            "top_nodes": [
                {"label": "Slip Angle", "node_type": "concept", "mentions": 3},
                {"label": "Camber", "node_type": "concept", "mentions": 1},
            ],
        },
        "raptor": {"node_count": 2},
    }


def test_book_summaries_returns_none_for_absent_layers(monkeypatch) -> None:
    client = _client(monkeypatch, FakeBooksQdrantClient(), FakeBookPool())

    response = client.get(f"/api/v1/books/{SOURCE_ID}/summaries", headers=auth_header())

    assert response.status_code == 200
    assert response.json() == {"dsi": None, "summary_tree": None, "graph": None, "raptor": None}
