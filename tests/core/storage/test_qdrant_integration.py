"""Integration tests for `vo_rag.core.storage.qdrant` — gated on a live Qdrant.

Skipped when `VORAG_TEST_QDRANT_URL` is unset (CI default). Exercises the
end-to-end contract on a real Qdrant: `setup_collections` is idempotent, an
upsert + hybrid search round-trip survives the four-vector schema, and the
deterministic point ID survives a re-upsert (no duplicates).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qdrant_url() -> str:
    url = os.environ.get("VORAG_TEST_QDRANT_URL")
    if not url:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_QDRANT_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_QDRANT_URL unset — integration test disabled.")
    return url


@pytest.fixture(scope="module")
def qdrant_api_key() -> str | None:
    return os.environ.get("VORAG_TEST_QDRANT_API_KEY") or None


@pytest.fixture(scope="module", autouse=True)
async def clean_declared_collections(
    qdrant_url: str, qdrant_api_key: str | None
) -> AsyncGenerator[None, None]:
    """Clean only vo-rag's declared test collections before and after this module."""
    from qdrant_client import AsyncQdrantClient

    from vo_rag.core.storage import COLLECTION_NAMES

    client = AsyncQdrantClient(
        url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False, timeout=120
    )

    async def _clean() -> None:
        existing = {item.name for item in (await client.get_collections()).collections}
        for name in COLLECTION_NAMES:
            if name in existing:
                await client.delete_collection(collection_name=name)

    try:
        await _clean()
        yield
    finally:
        await _clean()
        await client.close()


async def test_setup_collections_is_idempotent_against_live_qdrant(
    qdrant_url: str, qdrant_api_key: str | None
) -> None:
    from qdrant_client import AsyncQdrantClient

    from vo_rag.core.storage import COLLECTION_NAMES, setup_collections

    client = AsyncQdrantClient(
        url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False, timeout=120
    )
    try:
        await setup_collections(client)
        await setup_collections(client)  # second call must be a no-op, not raise
        existing = {c.name for c in (await client.get_collections()).collections}
        for name in COLLECTION_NAMES:
            assert name in existing
    finally:
        await client.close()


async def test_upsert_then_hybrid_search_round_trip(
    qdrant_url: str, qdrant_api_key: str | None
) -> None:
    from vo_rag.core.storage import QdrantStore, VoPoint, setup_collections

    store = QdrantStore(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    try:
        await setup_collections(store._client)

        collection = "vo_rag_repos_v1"
        marker = uuid.uuid4().hex
        dense = [0.01] * 1024
        colbert = [[0.01] * 1024]
        point = VoPoint(
            source_id=f"itest-{marker}",
            chunk_index=0,
            stage="raw",
            version_hash="itest_v1",
            dense=dense,
            sparse={1: 0.5, 2: 0.25},
            bm25={3: 1.0},
            colbert=colbert,
            payload={"project_id": "itest", "source_id": f"itest-{marker}"},
        )
        await store.upsert_vectors(collection, [point])

        results = await store.search_hybrid(
            collection=collection,
            query_dense=dense,
            query_sparse={1: 0.5, 2: 0.25},
            query_bm25={3: 1.0},
            query_colbert=colbert,
            top_k=5,
            filters={"source_id": f"itest-{marker}"},
        )
        assert any((r.payload or {}).get("source_id") == f"itest-{marker}" for r in results)

        await store.delete_source(collection, f"itest-{marker}")
    finally:
        await store._client.close()


async def test_warning_write_is_generation_scoped_and_preserves_vectors(
    qdrant_url: str, qdrant_api_key: str | None
) -> None:
    from vo_rag.core.storage import QdrantStore, VoPoint, setup_collections

    store = QdrantStore(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    collection = "vo_rag_repos_v1"
    source_id = "warning-itest-" + uuid.uuid4().hex
    try:
        await setup_collections(store._client)
        points = [
            VoPoint(
                source_id=source_id,
                chunk_index=i,
                stage="raw",
                version_hash="itest",
                dense=[0.01] * 1024,
                sparse={1: 0.5},
                bm25={3: 1.0},
                colbert=[[0.01] * 1024],
                payload={"source_id": source_id, "generation_id": gen, "content": "Usable passage"},
            )
            for i, gen in enumerate(["selected", "other"])
        ]
        await store.upsert_vectors(collection, points)
        before, _ = await store._client.scroll(collection, limit=100, with_vectors=True)
        before = [point for point in before if (point.payload or {}).get("source_id") == source_id]
        warnings = [{"stage": "parse", "code": "low_page_coverage"}]
        await store.set_source_quality_warnings(
            collection, source_id, warnings, generation_id="selected"
        )
        after = await store._client.retrieve(
            collection, [point.id for point in before], with_vectors=True
        )
        old = {point.id: point for point in before}
        for point in after:
            payload = point.payload or {}
            assert point.vector == old[point.id].vector
            assert payload["content"] == "Usable passage"
            if payload["generation_id"] == "selected":
                assert payload["quality_warnings"] == warnings
                assert payload["quality_disposition"] == "warning"
            else:
                assert "quality_warnings" not in payload
    finally:
        await store.delete_source(collection, source_id)
        await store._client.close()


async def test_empty_sparse_vectors_survive_normal_upsert_and_readback(
    qdrant_url: str, qdrant_api_key: str | None
) -> None:
    """The server retains both named empty maps from a symbol-only passage."""
    from qdrant_client.models import SparseVector

    from vo_rag.core.storage import QdrantStore, VoPoint, setup_collections
    from vo_rag.core.storage.qdrant import deterministic_point_id

    store = QdrantStore(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    source_id = "empty-sparse-" + uuid.uuid4().hex
    collection = "vo_rag_repos_v1"
    point_id = deterministic_point_id(source_id, 0, "raw")
    try:
        await setup_collections(store._client)
        point = VoPoint(
            source_id=source_id,
            chunk_index=0,
            stage="raw",
            dense=[0.01] * 1024,
            sparse={},
            bm25={},
            colbert=[[0.01] * 1024],
            payload={"content": "="},
        )
        await store.upsert_vectors(collection, [point])
        rows = await store._client.retrieve(
            collection_name=collection, ids=[point_id], with_payload=True, with_vectors=True
        )
        assert len(rows) == 1
        assert isinstance(rows[0].vector, dict)
        assert set(rows[0].vector) == {"dense", "sparse", "bm25", "colbert"}
        assert rows[0].vector["sparse"] == SparseVector(indices=[], values=[])
        assert rows[0].vector["bm25"] == SparseVector(indices=[], values=[])
    finally:
        await store._client.delete(collection_name=collection, points_selector=[point_id])
        await store._client.close()


async def test_visual_coordinate_payload_survives_exact_normal_rest_readback(
    qdrant_url: str, qdrant_api_key: str | None
) -> None:
    """Actual Qdrant payload storage preserves lossless geometry and all modalities."""
    from vo_rag.core.ingest.representation_receipt import verify_representation
    from vo_rag.core.storage import QdrantStore, VoPoint, setup_collections
    from vo_rag.core.storage.qdrant import build_point_structs

    store = QdrantStore(qdrant_url, qdrant_api_key, prefer_grpc=False)
    source_id = "visual-precision-" + uuid.uuid4().hex
    collection = "vo_rag_repos_v1"
    point = VoPoint(
        source_id=source_id,
        chunk_index=0,
        stage="raw",
        dense=[0.01] * 1024,
        sparse={},
        bm25={},
        colbert=[[0.01] * 1024],
        payload={
            "content": "Source text remains exact",
            "metadata": {
                "equation_visual": {
                    "status": "source_bound",
                    "crop_box": [0.0, 0.0, 458.63995361328125, 249.51023864746094],
                    "original_bbox": {"l": 90.56681823730469, "r": 204.59938400000166},
                }
            },
        },
    )
    expected = build_point_structs([point])[0]
    try:
        await setup_collections(store._client)
        await store.upsert_vectors(collection, [point])
        actual = (
            await store._client.retrieve(
                collection, [expected.id], with_payload=True, with_vectors=True
            )
        )[0]
        receipt = verify_representation(expected, actual)
        assert receipt["writer"]["payload_sha256"] == receipt["stored"]["payload_sha256"]
        assert (
            float.fromhex(actual.payload["metadata"]["equation_visual"]["crop_box"][2])
            == 458.63995361328125
        )
    finally:
        await store._client.delete(collection_name=collection, points_selector=[expected.id])
        await store._client.close()
