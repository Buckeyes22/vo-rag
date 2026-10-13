"""Unit tests for the pure helpers in `vo_rag.core.storage.qdrant`
(VOR-2).

Targets:
- `deterministic_point_id` — F-024 key contract: the ID is built from the
  source ID so retries overwrite; removal filters on payload `source_id`:
    - Stable across calls.
    - `version_hash=None` and `version_hash=""` collapse to the unversioned
      scheme (book_rag raw/sterile).
    - Distinct `version_hash` values produce distinct IDs.
- `bge_sparse_to_qdrant` — sorted-indices invariant; floats coerced.
- `COLLECTION_NAMES` — 18 collections, all `vo_rag_*`-prefixed.
- `_DENSE_ONLY_COLLECTIONS` — exactly the graph-edges collection.
- Module-level `instrument_qdrant_store` wiring exists (storage import
  performed the hook).
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector

from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.storage.qdrant import (
    _DENSE_ONLY_COLLECTIONS,
    COLLECTION_NAMES,
    QdrantStore,
    _build_filter,
    bge_sparse_to_qdrant,
    build_point_structs,
    deterministic_point_id,
)

# ----------------------------------------------------------------------
# deterministic_point_id
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qdrant_store_aclose_closes_its_owned_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.core.storage.qdrant as qdrant_module

    closed: list[bool] = []

    class _Client:
        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_ENABLED", "false")
    monkeypatch.setattr(qdrant_module, "AsyncQdrantClient", lambda **_kwargs: _Client())
    store = QdrantStore("http://qdrant.invalid", prefer_grpc=False)

    await store.aclose()

    assert closed == [True]


def test_point_id_is_stable() -> None:
    a = deterministic_point_id("src-1", 0, "raw")
    b = deterministic_point_id("src-1", 0, "raw")
    assert a == b


def test_point_id_unversioned_path_matches_three_part_uuid() -> None:
    """version_hash=None must produce the exact book_rag raw/sterile scheme."""
    pid = deterministic_point_id("src-1", 7, "raw")
    expected = deterministic_uuid("src-1", "7", "raw")
    assert pid == expected


def test_point_id_versioned_path_matches_four_part_uuid() -> None:
    pid = deterministic_point_id("src-1", 7, "raw", "research_v1")
    expected = deterministic_uuid("src-1", "7", "raw", "research_v1")
    assert pid == expected


def test_point_id_none_and_empty_version_hash_collapse_to_unversioned() -> None:
    """Contract: empty/None version_hash is the unversioned scheme."""
    none_id = deterministic_point_id("src-1", 0, "raw", None)
    empty_id = deterministic_point_id("src-1", 0, "raw", "")
    unversioned_id = deterministic_point_id("src-1", 0, "raw")
    assert none_id == unversioned_id == empty_id


def test_point_id_version_hash_distinguishes() -> None:
    base = deterministic_point_id("src-1", 0, "raw")
    versioned = deterministic_point_id("src-1", 0, "raw", "research_v1")
    other_version = deterministic_point_id("src-1", 0, "raw", "research_v2")
    assert base != versioned
    assert versioned != other_version
    assert base != other_version


def test_point_id_chunk_index_distinguishes() -> None:
    a = deterministic_point_id("src-1", 0, "raw")
    b = deterministic_point_id("src-1", 1, "raw")
    assert a != b


def test_point_id_source_id_and_stage_distinguish() -> None:
    base = deterministic_point_id("src-1", 0, "raw")
    assert deterministic_point_id("src-2", 0, "raw") != base
    assert deterministic_point_id("src-1", 0, "sterile") != base


# ----------------------------------------------------------------------
# bge_sparse_to_qdrant
# ----------------------------------------------------------------------


def test_bge_sparse_to_qdrant_sorts_indices_ascending() -> None:
    sv = bge_sparse_to_qdrant({42: 0.5, 7: 0.25, 9001: 1.0, 0: 0.1})
    assert isinstance(sv, SparseVector)
    assert sv.indices == [0, 7, 42, 9001]
    assert sv.values == [
        pytest.approx(0.1),
        pytest.approx(0.25),
        pytest.approx(0.5),
        pytest.approx(1.0),
    ]


def test_bge_sparse_to_qdrant_handles_empty_dict() -> None:
    sv = bge_sparse_to_qdrant({})
    assert sv.indices == []
    assert sv.values == []


def test_bge_sparse_to_qdrant_coerces_ints_and_floats() -> None:
    sv = bge_sparse_to_qdrant({1: 2, 0: 3})
    assert sv.indices == [0, 1]
    assert all(isinstance(v, float) for v in sv.values)
    assert sv.values == [pytest.approx(3.0), pytest.approx(2.0)]


# ----------------------------------------------------------------------
# Schema invariants — names + counts
# ----------------------------------------------------------------------


def test_collection_list_has_exactly_19_entries() -> None:
    # 19th is vo_rag_reddit_v1 (VOR-1404).
    assert len(COLLECTION_NAMES) == 19


def test_reddit_collection_is_declared() -> None:
    """Declared here is what makes setup_collections create it with the
    standard hybrid schema, so the Reddit archive is searchable the same way
    books are."""
    assert "vo_rag_reddit_v1" in COLLECTION_NAMES


def test_collection_names_are_unique() -> None:
    assert len(set(COLLECTION_NAMES)) == len(COLLECTION_NAMES)


def test_all_collection_names_are_vo_rag_prefixed() -> None:
    for name in COLLECTION_NAMES:
        assert name.startswith("vo_rag_"), name


def test_dense_only_collections_exactly_graph_edges() -> None:
    assert frozenset({"vo_rag_books_graph_edges_v1"}) == _DENSE_ONLY_COLLECTIONS


# ----------------------------------------------------------------------
# Instrumentation hook
# ----------------------------------------------------------------------


def test_qdrantstore_upsert_is_instrumented_by_storage_init() -> None:
    """Importing core.storage must wire metrics onto QdrantStore.upsert_vectors."""
    # Importing the package (already done above transitively) is what calls
    # observability.metrics.instrument_qdrant_store. The replacement function
    # is named `instrumented_upsert` in observability.metrics.
    from vo_rag.core.storage import QdrantStore

    assert QdrantStore.upsert_vectors.__name__ == "instrumented_upsert"


# ----------------------------------------------------------------------
# _build_filter — must be idempotent (regression for the HybridRetriever
# double-build crash: hybrid_search merges the license + user filter into a
# Filter, then search_hybrid called _build_filter on it again → AttributeError
# 'Filter' object has no attribute 'items'. No caller could use a server-side
# filter through HybridRetriever until this guard landed.)
# ----------------------------------------------------------------------


def test_build_filter_none_returns_none() -> None:
    assert _build_filter(None) is None
    assert _build_filter({}) is None


def test_build_filter_dict_builds_match_value() -> None:
    flt = _build_filter({"source_id": "abc"})
    assert isinstance(flt, Filter)
    assert isinstance(flt.must, list) and len(flt.must) == 1


def test_build_filter_is_idempotent_on_prebuilt_filter() -> None:
    """A Filter passed in (the HybridRetriever path) must pass through unchanged."""
    prebuilt = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value="abc"))])
    assert _build_filter(prebuilt) is prebuilt


# ----------------------------------------------------------------------
# count_source — the VOR-1403 duplicate-ingest gate. Must count EXACTLY:
# Qdrant's approximate count can report 0 for a populated source, which
# would re-run a full parse+embed over a book already in the corpus.
# ----------------------------------------------------------------------


async def test_count_source_counts_exactly_and_filters_on_source_id() -> None:
    from types import SimpleNamespace

    from vo_rag.core.storage import QdrantStore

    captured: dict[str, Any] = {}

    class _FakeClient:
        async def count(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(count=412)

    store = QdrantStore.__new__(QdrantStore)
    store._client = _FakeClient()  # noqa: SLF001

    result = await store.count_source("vo_rag_books_raw_v1", "vo-rag-corpus-sha256:abc")

    assert result == 412
    assert captured["exact"] is True
    assert captured["collection_name"] == "vo_rag_books_raw_v1"
    condition = captured["count_filter"].must[0]
    assert condition.key == "source_id"
    assert condition.match.value == "vo-rag-corpus-sha256:abc"


async def test_count_source_returns_zero_for_unseen_source() -> None:
    from types import SimpleNamespace

    from vo_rag.core.storage import QdrantStore

    class _EmptyClient:
        async def count(self, **kwargs: Any) -> Any:
            return SimpleNamespace(count=0)

    store = QdrantStore.__new__(QdrantStore)
    store._client = _EmptyClient()  # noqa: SLF001

    assert await store.count_source("vo_rag_books_raw_v1", "vo-rag-corpus-sha256:new") == 0


# ----------------------------------------------------------------------
# Read-timeout retry (VOR-1427): under ingest load Qdrant answered an exact
# count with 500 "read operations failed: Timeout" and failed whole books.
# ----------------------------------------------------------------------


def _read_timeout() -> Exception:
    import httpx
    from qdrant_client.http.exceptions import UnexpectedResponse

    body = (
        b'{"status":{"error":"Service internal error: 1 of 1 read operations failed:\\n '
        b"Timeout error: Operation 'count' timed out after 60s\"}}"
    )
    return UnexpectedResponse(500, "Internal Server Error", body, httpx.Headers())


async def test_count_source_retries_a_qdrant_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vo_rag.core.storage import QdrantStore
    from vo_rag.core.storage import qdrant as qdrant_mod

    monkeypatch.setattr(qdrant_mod, "_READ_RETRY_BACKOFF_S", 0.0)
    calls = 0

    class _SlowClient:
        async def count(self, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _read_timeout()
            return SimpleNamespace(count=627)

    store = QdrantStore.__new__(QdrantStore)
    store._client = _SlowClient()  # noqa: SLF001

    assert await store.count_source("vo_rag_books_raw_v1", "vo-rag-corpus-sha256:d19") == 627
    assert calls == 3


async def test_count_source_does_not_retry_other_errors(monkeypatch: pytest.MonkeyPatch) -> None:

    import httpx
    from qdrant_client.http.exceptions import UnexpectedResponse

    from vo_rag.core.storage import QdrantStore
    from vo_rag.core.storage import qdrant as qdrant_mod

    monkeypatch.setattr(qdrant_mod, "_READ_RETRY_BACKOFF_S", 0.0)
    calls = 0

    class _BadRequestClient:
        async def count(self, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            raise UnexpectedResponse(
                400, "Bad Request", b'{"status":{"error":"bad filter"}}', httpx.Headers()
            )

    store = QdrantStore.__new__(QdrantStore)
    store._client = _BadRequestClient()  # noqa: SLF001

    with pytest.raises(UnexpectedResponse):
        await store.count_source("vo_rag_books_raw_v1", "vo-rag-corpus-sha256:x")
    assert calls == 1


async def test_count_source_read_retry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:

    from qdrant_client.http.exceptions import UnexpectedResponse

    from vo_rag.core.storage import QdrantStore
    from vo_rag.core.storage import qdrant as qdrant_mod

    monkeypatch.setattr(qdrant_mod, "_READ_RETRY_BACKOFF_S", 0.0)
    calls = 0

    class _AlwaysSlowClient:
        async def count(self, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            raise _read_timeout()

    store = QdrantStore.__new__(QdrantStore)
    store._client = _AlwaysSlowClient()  # noqa: SLF001

    with pytest.raises(UnexpectedResponse):
        await store.count_source("vo_rag_books_raw_v1", "vo-rag-corpus-sha256:x")
    assert calls == qdrant_mod.READ_RETRY_ATTEMPTS


# ----------------------------------------------------------------------
# Payload-index reconcile (VOR-1427): production collections predated the
# generation_id/ingest_state indexes the code declares, and nothing applied
# them, so the duplicate-ingest gate scanned ~2.7M points per book.
# ----------------------------------------------------------------------


async def test_reconcile_adds_only_missing_declared_indexes_on_existing_collections() -> None:
    from types import SimpleNamespace

    from vo_rag.core.storage.qdrant import COMMON_PAYLOAD_INDEXES, reconcile_payload_indexes

    raw = "vo_rag_books_raw_v1"
    have = {field: None for field in COMMON_PAYLOAD_INDEXES if field != "ingest_state"}
    created: list[tuple[str, str]] = []

    class _Client:
        async def get_collections(self) -> Any:
            return SimpleNamespace(collections=[SimpleNamespace(name=raw)])

        async def get_collection(self, name: str) -> Any:
            return SimpleNamespace(payload_schema=dict(have))

        async def create_payload_index(
            self, *, collection_name: str, field_name: str, **_: Any
        ) -> None:
            created.append((collection_name, field_name))
            have[field_name] = None

    added = await reconcile_payload_indexes(_Client())  # type: ignore[arg-type]

    assert ("vo_rag_books_raw_v1", "ingest_state") in created
    assert all(name == raw for name, _ in created)
    assert "ingest_state" in added[raw]
    assert "source_id" not in added[raw]


@pytest.mark.parametrize(
    "name,publication_fields",
    [
        ("vo_rag_books_raw_v1", True),
        ("vo_rag_books_graph_edges_v1", True),
        ("vo_rag_reddit_v1", True),
        ("vo_rag_brain_v1", False),
        ("vo_rag_code_chunks_v1", False),
    ],
)
def test_publication_index_declarations_stay_inside_books_and_reddit(
    name: str, publication_fields: bool
) -> None:
    from vo_rag.core.storage.qdrant import COMMON_PAYLOAD_INDEXES, _extra_indexes_for

    declared = {**COMMON_PAYLOAD_INDEXES, **_extra_indexes_for(name)}
    for field in ("source_revision", "publication_base_id"):
        assert (field in declared) is publication_fields
    assert "dependency_revision_keys" not in declared


# ----------------------------------------------------------------------
# Per-envelope Qdrant call latency (VOR-1434)
# ----------------------------------------------------------------------


async def test_qdrant_store_calls_are_timed_onto_the_ingest() -> None:
    from types import SimpleNamespace

    from vo_rag.core.observability.metrics import ingest_recording
    from vo_rag.core.storage.qdrant import QdrantStore

    class _FakeClient:
        async def count(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(count=3)

        async def query_points(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(points=[])

        async def delete(self, **_: object) -> None:
            return None

        async def set_payload(self, **_: object) -> None:
            return None

    store = QdrantStore("http://qdrant.invalid", prefer_grpc=False)
    store._client = _FakeClient()  # noqa: SLF001

    with ingest_recording() as recorder:
        assert await store.count_source("c", "s") == 3
        await store.search_single_collection("c", [0.0])
        await store.delete_source("c", "s")
        await store.set_generation_state("c", "s", "g", "active")

    summary = recorder.qdrant_summary()
    assert summary is not None
    assert summary["calls"] == 4


async def test_a_failed_qdrant_call_is_still_timed() -> None:
    from vo_rag.core.observability.metrics import ingest_recording
    from vo_rag.core.storage.qdrant import QdrantStore

    class _BoomClient:
        async def delete(self, **_: object) -> None:
            raise RuntimeError("qdrant down")

    store = QdrantStore("http://qdrant.invalid", prefer_grpc=False)
    store._client = _BoomClient()  # noqa: SLF001

    with ingest_recording() as recorder, pytest.raises(RuntimeError):
        await store.delete_source("c", "s")
    assert len(recorder.qdrant_latencies) == 1


@pytest.mark.parametrize("sparse,bm25", [({}, {}), ({7: 0.5}, {}), ({}, {9: 1.0})])
def test_normal_writer_preserves_empty_sparse_modalities(
    sparse: dict[int, float], bm25: dict[int, float]
) -> None:
    """Symbol-only passages retain genuine empty vectors without invented tokens."""
    from vo_rag.core.storage.points import VoPoint

    point = VoPoint(
        source_id="symbol-only",
        chunk_index=0,
        stage="raw",
        dense=[0.01] * 1024,
        sparse=sparse,
        bm25=bm25,
        colbert=[[0.01] * 1024],
        payload={"content": "="},
    )
    stored = build_point_structs([point])[0]
    assert isinstance(stored.vector, dict)
    assert set(stored.vector) == {"dense", "sparse", "bm25", "colbert"}
    assert stored.vector["sparse"] == bge_sparse_to_qdrant(sparse)
    assert stored.vector["bm25"] == bge_sparse_to_qdrant(bm25)


@pytest.mark.parametrize(
    "field", ["equation_visual", "table_visual", "source_page_visuals", "source_table_visuals"]
)
def test_visual_geometry_codec_preserves_exact_float_bits_and_source_values(field: str) -> None:
    """Only source visual coordinates change encoding, reversibly and without mutation."""
    from copy import deepcopy

    from vo_rag.core.storage.qdrant import _encode_source_geometry

    visual: dict[str, Any] = {
        "status": "source_bound",
        "crop_box": [0, -0.0, 458.63995361328125, 249.51023864746094],
        "original_bbox": {
            "l": 90.56681823730469,
            "r": 204.59938400000166,
            "coord_origin": "TOPLEFT",
        },
        "image_ref": "source/unchanged.png",
        "sha256": "a" * 64,
        "symbolic_transcription_verified": False,
    }
    value = [visual] if field.endswith("s") else visual
    payload: dict[str, Any] = {
        "content": "458.63995361328125 is source text",
        "metadata": {field: value, "source_scalar": 458.63995361328125},
    }
    before = deepcopy(payload)
    actual = _encode_source_geometry(payload)
    result = actual["metadata"][field]
    if isinstance(result, list):
        result = result[0]
    assert result["coordinate_encoding"] == "float-hex.v1"
    assert result["crop_box"][0] == 0
    for expected, encoded in zip(visual["crop_box"][1:], result["crop_box"][1:], strict=True):
        assert float.fromhex(encoded).hex() == expected.hex()
    assert float.fromhex(result["original_bbox"]["r"]).hex() == visual["original_bbox"]["r"].hex()
    assert result["original_bbox"]["coord_origin"] == "TOPLEFT"
    assert result["image_ref"] == visual["image_ref"]
    assert actual["content"] == payload["content"]
    assert actual["metadata"]["source_scalar"] == payload["metadata"]["source_scalar"]
    assert payload == before
    assert _encode_source_geometry(actual) == actual


def test_visual_geometry_codec_rejects_nonfinite_coordinates() -> None:
    from vo_rag.core.storage.qdrant import _encode_source_geometry

    with pytest.raises(ValueError, match="finite"):
        _encode_source_geometry(
            {"metadata": {"table_visual": {"status": "source_bound", "crop_box": [float("inf")]}}}
        )


@pytest.mark.parametrize(
    "coordinate", [458.63995361328125, "458.63995361328125", "nan", "0x1p+0", True]
)
def test_visual_geometry_codec_rejects_malformed_existing_marker(coordinate: Any) -> None:
    from vo_rag.core.storage.qdrant import _encode_source_geometry

    with pytest.raises(ValueError, match="coordinate"):
        _encode_source_geometry(
            {
                "metadata": {
                    "table_visual": {
                        "status": "source_bound",
                        "coordinate_encoding": "float-hex.v1",
                        "crop_box": [coordinate],
                    }
                }
            }
        )
