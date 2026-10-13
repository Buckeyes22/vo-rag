"""Integration tests for `vo_rag.core.corpus.purge.purge_source` (VOR-1427).

Gated on live testinfra (``VORAG_TEST_DATABASE_URL`` / ``VORAG_TEST_QDRANT_URL``
-- ``make up`` starts throwaway local containers, see the Makefile
``test-int`` target). Never points at production.

Graph nodes/edges are seeded through the *real* ``upsert_node``/``upsert_edge``
(``core/graph/persistence.py``) with real ``node_id_for``/``edge_id_for`` ids
(``core/graph/schemas.py``) -- i.e. production-shaped, unhyphenated 32-hex
node ids -- not a hand-rolled id scheme. A hand-rolled scheme that kept ids
self-consistent internally does not reproduce Qdrant's hyphenation-on-scroll
behaviour and would not catch a mismatch between the two id forms.

``book_categories`` isn't created by any migration in this tree.
``purge_source``/``_purge_postgres`` guard against that with
``to_regclass``, so most tests here create a minimal stand-in table
per-test via ``_create_book_categories_table`` (never automatically --
one test specifically exercises the table being absent).

``postgres["book_categories"]`` is always present and one of
three shapes: an ``int`` row count when ``keep_categories=False`` and the
table exists, the string ``"kept"`` when ``keep_categories=True`` (no
existence check needed -- the table is never touched either way), or
``"absent"`` when ``keep_categories=False`` and the table doesn't exist.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pytest

from vo_rag.core.corpus import purge as purge_module
from vo_rag.core.corpus.purge import purge_source
from vo_rag.core.db.postgres import run_migrations
from vo_rag.core.graph.persistence import upsert_edge, upsert_node
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    edge_id_for,
    node_id_for,
)
from vo_rag.core.storage import VoPoint
from vo_rag.core.storage.qdrant import QdrantStore, setup_collections

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"
)

_GRAPH_NODES = "vo_rag_books_graph_nodes_v1"
_GRAPH_EDGES = "vo_rag_books_graph_edges_v1"
_DENSE = [0.01] * 1024
_COLBERT = [[0.01] * 1024]


def _require_pg_dsn() -> str:
    dsn = os.environ.get("VORAG_TEST_DATABASE_URL")
    if not dsn:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_DATABASE_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_DATABASE_URL not set -- skipping integration test")
    return dsn


def _require_qdrant_url() -> str:
    url = os.environ.get("VORAG_TEST_QDRANT_URL")
    if not url:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_QDRANT_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_QDRANT_URL not set -- skipping integration test")
    return url


@pytest.fixture
async def pg_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Fresh, freshly-migrated Postgres for every test. No ``book_categories``
    stand-in here -- call ``_create_book_categories_table`` in the
    tests that need it to exist."""
    dsn = _require_pg_dsn()
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, statement_cache_size=0)
    try:
        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.execute("CREATE SCHEMA public")
        await run_migrations(pool, _MIGRATIONS_DIR)
        yield pool
    finally:
        await pool.close()


@pytest.fixture(scope="module")
def qdrant_url() -> str:
    return _require_qdrant_url()


@pytest.fixture(scope="module", autouse=True)
async def _prepare_qdrant(qdrant_url: str) -> AsyncGenerator[None, None]:
    """Wipe and recreate vo-rag's declared collections for this module."""
    from qdrant_client import AsyncQdrantClient

    from vo_rag.core.storage import COLLECTION_NAMES

    client = AsyncQdrantClient(url=qdrant_url, api_key=None, prefer_grpc=False, timeout=120)

    async def _clean() -> None:
        existing = {item.name for item in (await client.get_collections()).collections}
        for name in COLLECTION_NAMES:
            if name in existing:
                await client.delete_collection(collection_name=name)

    try:
        await _clean()
        await setup_collections(client)
        yield
    finally:
        await _clean()
        await client.close()


@pytest.fixture
async def store(qdrant_url: str) -> AsyncGenerator[QdrantStore, None]:
    s = QdrantStore(url=qdrant_url, api_key=None, prefer_grpc=False)
    try:
        yield s
    finally:
        await s._client.close()  # noqa: SLF001


async def _upsert_by(store: QdrantStore, collection: str, key: str, value: str, n: int) -> None:
    points = [
        VoPoint(
            source_id=f"{value}-{i}",
            chunk_index=i,
            stage="itest",
            dense=_DENSE,
            sparse={1: 0.5},
            bm25={2: 1.0},
            colbert=_COLBERT,
            payload={"project_id": "itest", key: value},
        )
        for i in range(n)
    ]
    await store.upsert_vectors(collection, points)


async def _seed_node(store: QdrantStore, label: str, book_ids: list[str]) -> str:
    """Seed a graph node exactly as production does: real `GraphNode` +
    `node_id_for` (unhyphenated 32-hex) + the real `upsert_node`. Returns
    the node's `node_id` -- the string edges reference, NOT the Qdrant point
    id `scroll`/`retrieve` return (see module docstring)."""
    node = GraphNode(
        node_id=node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label=label),
        node_type=GraphNodeType.CONCEPT,
        label=label,
        aliases=[],
        book_ids=book_ids,
        chunk_ids=["c1"],
        attributes={},
        created_at=datetime.now(UTC),
    )
    await upsert_node(
        node=node,
        dense_embedding=_DENSE,
        sparse_embedding={1: 0.5},
        colbert_embedding=_COLBERT,
        store=store,
    )
    return node.node_id


async def _seed_edge(store: QdrantStore, source_node_id: str, target_node_id: str) -> str:
    """Seed a graph edge exactly as production does: real `GraphEdge` +
    `edge_id_for` + the real `upsert_edge`. Returns the edge_id."""
    edge = GraphEdge(
        edge_id=edge_id_for(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=GraphEdgeType.RELATES_TO,
        ),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=GraphEdgeType.RELATES_TO,
        book_ids=[],
        evidence_chunk_ids=["c1"],
        created_at=datetime.now(UTC),
    )
    await upsert_edge(edge=edge, dense_embedding=_DENSE, store=store)
    return edge.edge_id


async def _create_book_categories_table(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS book_categories (
                source_id  TEXT PRIMARY KEY,
                categories TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[]
            )
            """
        )


async def _seed_postgres(
    pool: asyncpg.Pool, *, source_a: str, source_b: str, drop_categories_row: bool = False
) -> str:
    """Seed `sources` (A+B), one `knowledge_artifacts` row for A, one
    `source_ingest_generations` row for A. Returns A's artifact_id.
    ``drop_categories_row`` requires the caller to have already created the
    `book_categories` table (`_create_book_categories_table`)."""
    async with pool.acquire() as conn:
        envelope_id = f"env-{uuid.uuid4().hex[:12]}"
        await conn.execute(
            """
            INSERT INTO intake_envelopes (id, source_type, raw_content)
            VALUES ($1, 'file_drop', '{}'::jsonb)
            """,
            envelope_id,
        )
        for sid, title in ((source_a, "Book A"), (source_b, "Book B")):
            await conn.execute(
                "INSERT INTO sources (id, title, source_type) VALUES ($1, $2, 'book')",
                sid,
                title,
            )
        artifact_id = f"artifact-{uuid.uuid4().hex[:12]}"
        await conn.execute(
            """
            INSERT INTO knowledge_artifacts
                (id, envelope_id, pipeline, artifact_type, title, content)
            VALUES ($1, $2, 'book_rag', 'summary', 'Book A summary', $3::jsonb)
            """,
            artifact_id,
            envelope_id,
            f'{{"source_id": "{source_a}"}}',
        )
        generation_id = f"gen-{uuid.uuid4().hex[:12]}"
        await conn.execute(
            """
            INSERT INTO source_ingest_generations
                (generation_id, source_id, raw_collection, envelope_id,
                 schema_version, chunking_version)
            VALUES ($1, $2, 'vo_rag_books_raw_v1', $3, 'v1', 'v1')
            """,
            generation_id,
            source_a,
            envelope_id,
        )
        if drop_categories_row:
            await conn.execute(
                "INSERT INTO book_categories (source_id, categories) VALUES ($1, $2)",
                source_a,
                ["motorcycles"],
            )
    return artifact_id


_COUNT_QUERIES: dict[tuple[str, str], str] = {
    ("sources", "id"): "SELECT count(*) FROM sources WHERE id = $1",
    ("book_categories", "source_id"): "SELECT count(*) FROM book_categories WHERE source_id = $1",
}


async def _count(pool: asyncpg.Pool, table: str, column: str, value: str) -> int:
    async with pool.acquire() as conn:
        result = await conn.fetchval(_COUNT_QUERIES[(table, column)], value)
        return int(result)


def _base_postgres_report(**overrides: int | str) -> dict[str, int | str]:
    """Defaults match `purge_source(..., keep_categories=True)` (the
    default) against a fresh, empty seed -- including `book_categories`,
    which is always present and reports "kept" whenever `keep_categories`
    is True, with no existence check needed."""
    base: dict[str, int | str] = {
        "ideas_unlinked": 0,
        "compliance_log_unlinked": 0,
        "source_ingest_generations": 0,
        "knowledge_artifacts": 0,
        "sources": 0,
        "book_categories": "kept",
    }
    base.update(overrides)
    return base


async def test_dry_run_reports_without_changing_anything(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-a-{marker}"
    source_b = f"itest-purge-b-{marker}"

    await _upsert_by(store, "vo_rag_books_raw_v1", "source_id", source_a, 3)
    await _upsert_by(store, "vo_rag_books_summary_tree_v1", "book_id", source_a, 1)
    node_a = await _seed_node(store, f"n1-{marker}", [source_a])
    node_shared = await _seed_node(store, f"n2-{marker}", [source_a, source_b])
    node_c = await _seed_node(store, f"n3-{marker}", [source_b])
    await _seed_edge(store, node_a, node_shared)
    await _seed_edge(store, node_shared, node_c)
    await _upsert_by(store, "vo_rag_books_raw_v1", "source_id", source_b, 2)

    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b)

    report = await purge_source(store, pg_pool, source_a, apply=False)

    assert report.qdrant["vo_rag_books_raw_v1"] == 3
    assert report.qdrant["vo_rag_books_summary_tree_v1"] == 1
    assert report.qdrant["vo_rag_books_graph_nodes_v1"] == 1
    assert report.graph_nodes_detached == 1
    assert report.qdrant["vo_rag_books_graph_edges_v1"] == 1
    assert report.postgres == _base_postgres_report(
        sources=1, knowledge_artifacts=1, source_ingest_generations=1
    )

    # Nothing actually changed.
    assert await _count_qdrant(store, "vo_rag_books_raw_v1", "source_id", source_a) == 3
    assert await _count_qdrant(store, "vo_rag_books_raw_v1", "source_id", source_b) == 2
    assert await _count_qdrant(store, _GRAPH_NODES, "book_ids", source_a) == 2
    assert await _count_qdrant(store, _GRAPH_EDGES, "source_node_id", node_a) == 1
    assert await _count(pg_pool, "sources", "id", source_a) == 1
    assert await _count(pg_pool, "sources", "id", source_b) == 1


async def test_apply_deletes_and_detaches_leaving_shared_state_intact(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-a-{marker}"
    source_b = f"itest-purge-b-{marker}"

    await _upsert_by(store, "vo_rag_books_raw_v1", "source_id", source_a, 3)
    await _upsert_by(store, "vo_rag_books_summary_tree_v1", "book_id", source_a, 1)
    node_a = await _seed_node(store, f"n1-{marker}", [source_a])
    node_shared = await _seed_node(store, f"n2-{marker}", [source_a, source_b])
    node_c = await _seed_node(store, f"n3-{marker}", [source_b])
    # e1 touches a node that WILL be deleted (n1) -- must be deleted.
    edge_1 = await _seed_edge(store, node_a, node_shared)
    # e2 is only between surviving nodes (n2 detached-not-deleted, n3
    # untouched) -- must be kept.
    edge_2 = await _seed_edge(store, node_shared, node_c)
    await _upsert_by(store, "vo_rag_books_raw_v1", "source_id", source_b, 2)

    await _create_book_categories_table(pg_pool)
    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b, drop_categories_row=True)

    report = await purge_source(store, pg_pool, source_a, apply=True)

    assert report.qdrant["vo_rag_books_raw_v1"] == 3
    assert report.qdrant["vo_rag_books_summary_tree_v1"] == 1
    assert report.qdrant["vo_rag_books_graph_nodes_v1"] == 1
    assert report.graph_nodes_detached == 1
    assert report.qdrant["vo_rag_books_graph_edges_v1"] == 1
    # book_categories is always present -- "kept" here since keep_categories
    # defaults to True (no existence check is needed).
    assert report.postgres["book_categories"] == "kept"

    # A's raw/summary points are gone.
    assert await _count_qdrant(store, "vo_rag_books_raw_v1", "source_id", source_a) == 0
    assert await _count_qdrant(store, "vo_rag_books_summary_tree_v1", "book_id", source_a) == 0
    # n1 deleted.
    assert await _get_node(store, node_a) is None
    # n2 detached: book_ids == [source_b].
    remaining = await _get_node(store, node_shared)
    assert remaining is not None
    assert remaining.payload is not None
    assert remaining.payload["book_ids"] == [source_b]
    # n3 (source_b only) is completely untouched.
    survivor = await _get_node(store, node_c)
    assert survivor is not None
    assert survivor.payload is not None
    assert survivor.payload["book_ids"] == [source_b]
    # e1 (touches deleted n1) is gone; e2 (between two surviving nodes) is kept.
    assert await _get_edge(store, edge_1) is None
    assert await _get_edge(store, edge_2) is not None
    # B's 2 raw points remain untouched.
    assert await _count_qdrant(store, "vo_rag_books_raw_v1", "source_id", source_b) == 2

    # Postgres rows for A are gone except book_categories (kept by default); B's rows remain.
    assert await _count(pg_pool, "sources", "id", source_a) == 0
    assert await _count(pg_pool, "sources", "id", source_b) == 1
    assert await _count(pg_pool, "book_categories", "source_id", source_a) == 1


async def test_apply_with_drop_categories_deletes_book_categories_row(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-cat-{marker}"
    source_b = f"itest-purge-cat-b-{marker}"
    await _create_book_categories_table(pg_pool)
    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b, drop_categories_row=True)

    report = await purge_source(store, pg_pool, source_a, apply=True, keep_categories=False)

    assert report.postgres["book_categories"] == 1
    assert await _count(pg_pool, "book_categories", "source_id", source_a) == 0


async def test_book_categories_table_absent_reports_absent_in_both_modes(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    """No `_create_book_categories_table` call -- the table genuinely
    doesn't exist on this fresh schema. `--drop-categories` (keep_categories=
    False) must report "absent" and not raise, in dry run AND apply."""
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-nocat-{marker}"
    source_b = f"itest-purge-nocat-b-{marker}"
    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b)

    dry_report = await purge_source(store, pg_pool, source_a, apply=False, keep_categories=False)
    assert dry_report.postgres["book_categories"] == "absent"

    apply_report = await purge_source(store, pg_pool, source_a, apply=True, keep_categories=False)
    assert apply_report.postgres["book_categories"] == "absent"
    # Everything else still went through normally.
    assert apply_report.postgres["sources"] == 1
    assert await _count(pg_pool, "sources", "id", source_a) == 0


async def test_book_categories_reports_kept_without_checking_existence(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    """`keep_categories=True` (the default) always reports
    "kept" -- even when the table doesn't exist at all -- since the caller
    asked to leave it alone and no existence check is needed."""
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-keepcat-{marker}"
    source_b = f"itest-purge-keepcat-b-{marker}"
    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b)

    dry_report = await purge_source(store, pg_pool, source_a, apply=False)
    assert dry_report.postgres["book_categories"] == "kept"

    apply_report = await purge_source(store, pg_pool, source_a, apply=True)
    assert apply_report.postgres["book_categories"] == "kept"


async def test_unknown_id_reports_all_zeros_and_changes_nothing(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-a-{marker}"
    source_b = f"itest-purge-b-{marker}"
    await _upsert_by(store, "vo_rag_books_raw_v1", "source_id", source_a, 3)
    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b)

    report = await purge_source(store, pg_pool, "vo-rag-corpus-sha256:nope", apply=True)

    assert report.qdrant == {
        "vo_rag_books_raw_v1": 0,
        "vo_rag_books_sterile_v1": 0,
        "vo_rag_books_contextualized_v1": 0,
        "vo_rag_books_summary_tree_v1": 0,
        "vo_rag_books_raptor_v1": 0,
        "vo_rag_books_dsi_v1": 0,
        "vo_rag_books_graph_nodes_v1": 0,
        "vo_rag_books_graph_edges_v1": 0,
    }
    assert report.graph_nodes_detached == 0
    assert report.postgres == _base_postgres_report()

    # Nothing about A changed.
    assert await _count_qdrant(store, "vo_rag_books_raw_v1", "source_id", source_a) == 3
    assert await _count(pg_pool, "sources", "id", source_a) == 1


async def test_ideas_and_compliance_log_are_never_deleted_only_unlinked(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    """ideas/compliance_log rows survive an apply purge, with their
    FK nulled by Postgres (ON DELETE SET NULL) once knowledge_artifacts is
    deleted -- purge_source itself never deletes these rows."""
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-audit-{marker}"
    source_b = f"itest-purge-audit-b-{marker}"
    artifact_id = await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b)

    idea_id = f"idea-{uuid.uuid4().hex[:12]}"
    compliance_id = f"clog-{uuid.uuid4().hex[:12]}"
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO ideas (id, artifact_id, core_concept) VALUES ($1, $2, $3)",
            idea_id,
            artifact_id,
            "purge audit trail concept",
        )
        await conn.execute(
            "INSERT INTO compliance_log (id, artifact_id, passed, mode) VALUES ($1, $2, $3, $4)",
            compliance_id,
            artifact_id,
            True,
            "internal",
        )

    report = await purge_source(store, pg_pool, source_a, apply=True)

    assert report.postgres["ideas_unlinked"] == 1
    assert report.postgres["compliance_log_unlinked"] == 1

    async with pg_pool.acquire() as conn:
        idea_row = await conn.fetchrow("SELECT artifact_id FROM ideas WHERE id = $1", idea_id)
        compliance_row = await conn.fetchrow(
            "SELECT artifact_id FROM compliance_log WHERE id = $1", compliance_id
        )

    assert idea_row is not None
    assert idea_row["artifact_id"] is None
    assert compliance_row is not None
    assert compliance_row["artifact_id"] is None


async def test_missing_qdrant_collection_reports_absent_instead_of_raising(
    store: QdrantStore, pg_pool: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """count() raising on a missing collection must not abort the whole
    purge -- report "absent" for that collection and keep going.

    Uses a collection name that genuinely doesn't exist in Qdrant rather
    than deleting (and restoring) a real, shared test collection (deleting a real collection is only
    safe when tests run one at a time -- monkeypatching `purge`'s own collection tuple to include a
    name Qdrant has never heard of exercises the exact same "absent" path
    without mutating anything shared)."""
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-missingcoll-{marker}"
    source_b = f"itest-purge-missingcoll-b-{marker}"
    await _upsert_by(store, "vo_rag_books_raw_v1", "source_id", source_a, 3)
    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b)

    fake_collection = f"vo_rag_books_does_not_exist_{marker}_v1"
    monkeypatch.setattr(purge_module, "_BY_SOURCE", (*purge_module._BY_SOURCE, fake_collection))

    report = await purge_source(store, pg_pool, source_a, apply=False)
    assert report.qdrant[fake_collection] == "absent"
    assert report.qdrant["vo_rag_books_raw_v1"] == 3


async def test_postgres_failure_leaves_qdrant_untouched(
    store: QdrantStore, pg_pool: asyncpg.Pool
) -> None:
    """Postgres runs before any Qdrant write. A real Postgres failure
    (here: the `sources` table no longer exists) must leave every Qdrant
    point exactly as it was."""
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-purge-pgfail-{marker}"
    source_b = f"itest-purge-pgfail-b-{marker}"
    await _upsert_by(store, "vo_rag_books_raw_v1", "source_id", source_a, 3)
    await _seed_postgres(pg_pool, source_a=source_a, source_b=source_b)

    async with pg_pool.acquire() as conn:
        await conn.execute("DROP TABLE sources")

    with pytest.raises(Exception):  # noqa: B017 -- real asyncpg UndefinedTableError
        await purge_source(store, pg_pool, source_a, apply=True)

    assert await _count_qdrant(store, "vo_rag_books_raw_v1", "source_id", source_a) == 3


async def _count_qdrant(store: QdrantStore, collection: str, key: str, value: str) -> int:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = await store._client.count(  # noqa: SLF001
        collection_name=collection,
        count_filter=Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))]),
        exact=True,
    )
    return int(result.count)


async def _get_node(store: QdrantStore, node_id: str) -> Any:
    results = await store._client.retrieve(  # noqa: SLF001
        collection_name=_GRAPH_NODES, ids=[node_id], with_payload=True
    )
    return results[0] if results else None


async def _get_edge(store: QdrantStore, edge_id: str) -> Any:
    results = await store._client.retrieve(  # noqa: SLF001
        collection_name=_GRAPH_EDGES, ids=[edge_id], with_payload=True
    )
    return results[0] if results else None
