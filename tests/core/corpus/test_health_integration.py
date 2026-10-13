"""Integration tests for `vo_rag.core.corpus.health` (VOR-1427).

Gated on live testinfra (``VORAG_TEST_QDRANT_URL`` -- ``make up`` starts
throwaway local containers, see the Makefile ``test-int`` target). Never
points at production. Only Qdrant is needed here (health has no Postgres
dependency).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny

from vo_rag.core.corpus.health import run_health
from vo_rag.core.storage import VoPoint
from vo_rag.core.storage.qdrant import COLLECTION_NAMES, QdrantStore, setup_collections

pytestmark = pytest.mark.integration

_RAW = "vo_rag_books_raw_v1"
_DENSE = [0.01] * 1024
_COLBERT = [[0.01] * 1024]


def _require_qdrant_url() -> str:
    url = os.environ.get("VORAG_TEST_QDRANT_URL")
    if not url:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_QDRANT_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_QDRANT_URL not set -- skipping integration test")
    return url


@pytest.fixture(scope="module")
def qdrant_url() -> str:
    return _require_qdrant_url()


@pytest.fixture(scope="module", autouse=True)
async def _prepare_qdrant(qdrant_url: str) -> AsyncGenerator[None, None]:
    """Wipe and recreate vo-rag's declared collections once for this module.

    Module-scoped (matching `test_purge_integration.py`): recreating every
    collection (and its payload indexes) per test triples Qdrant churn for
    no isolation benefit here. Per-test isolation instead comes from
    `_cleanup_seeded` below, which deletes exactly the points each test
    seeded -- see that fixture's docstring for why a scroll-based defect
    scan (rather than a `source_id` filter) still needs this.
    """
    from qdrant_client import AsyncQdrantClient

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


@pytest.fixture
def seeded_source_ids() -> list[str]:
    """Every test appends every `source_id` it seeds into `_RAW` here.
    `_cleanup_seeded` deletes exactly those points on teardown -- the
    per-test isolation `test_run_health_ok_on_empty_collection` needs,
    without a `setup_collections()` per test."""
    return []


@pytest.fixture(autouse=True)
async def _cleanup_seeded(
    store: QdrantStore, seeded_source_ids: list[str]
) -> AsyncGenerator[None, None]:
    """Deletes every point under `seeded_source_ids` from `_RAW` after each
    test, regardless of pass/fail.

    Collections are shared (module-scoped `_prepare_qdrant`) but
    `run_health`'s test-points scroll is intentionally *unscoped* by
    `source_id` -- it has to see the whole collection to catch leaked
    canary/rr03/test data anywhere -- so without this, a book (or canary
    point) seeded by one test would still be visible, and skew `ok()`, in
    a later test sharing the module. TDD caught exactly this the first
    time this file was written (see git history): a canary point from an
    earlier test made `test_run_health_ok_on_empty_collection` fail.
    """
    yield
    if not seeded_source_ids:
        return
    await store._client.delete(  # noqa: SLF001
        collection_name=_RAW,
        points_selector=Filter(
            must=[FieldCondition(key="source_id", match=MatchAny(any=seeded_source_ids))]
        ),
        wait=True,
    )


def _points(
    source_id: str,
    n: int,
    *,
    source_book: str,
    content: str | list[str] = "a normal chunk of book content, long enough",
    section_title: str | list[str | None] | None = None,
    page_number: int | list[int] | None = None,
    domain_tags: list[str] | None = None,
    parser_used: str | None = None,
    author: str | None = None,
) -> list[VoPoint]:
    def _at(value: Any, i: int) -> Any:
        return value[i] if isinstance(value, list) else value

    points = []
    for i in range(n):
        payload: dict[str, object] = {
            "project_id": "itest",
            "source_id": source_id,
            "source_book": source_book,
            "content": _at(content, i),
        }
        title = _at(section_title, i)
        if title is not None:
            payload["section_title"] = title
        page = _at(page_number, i)
        if page is not None:
            payload["page_number"] = page
        if domain_tags is not None:
            payload["domain_tags"] = domain_tags
        if parser_used is not None:
            payload["parser_used"] = parser_used
        if author is not None:
            payload["author"] = author
        points.append(
            VoPoint(
                source_id=f"{source_id}-{i}",
                chunk_index=i,
                stage="itest",
                dense=_DENSE,
                sparse={1: 0.5},
                bm25={2: 1.0},
                colbert=_COLBERT,
                payload=payload,
            )
        )
    return points


async def test_run_health_flags_degenerate_dominant_and_test_points(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    marker = uuid.uuid4().hex[:8]
    book_d = f"itest-health-d-{marker}"  # degenerate: 25/30 tiny
    book_s = f"itest-health-s-{marker}"  # dominant section: 200/250 "Bibliography"
    book_g = f"itest-health-g-{marker}"  # clean
    canary_source = f"canary-src-{marker}"
    seeded_source_ids.extend([book_d, book_s, book_g, canary_source])

    d_points = _points(
        book_d,
        30,
        source_book="Book D",
        content=["x"] * 25 + ["a normal, long enough chunk of real content"] * 5,
    )
    s_points = _points(
        book_s,
        250,
        source_book="Book S",
        section_title=["Bibliography"] * 200 + [None] * 50,
    )
    g_points = _points(book_g, 50, source_book="Book G")
    canary_point = VoPoint(
        source_id=f"canary-{marker}",
        chunk_index=0,
        stage="itest",
        dense=_DENSE,
        sparse={1: 0.5},
        bm25={2: 1.0},
        colbert=_COLBERT,
        payload={
            "project_id": "itest",
            "source_id": canary_source,
            "source_book": "Canary",
            "content": "a canary chunk, long enough to not count as tiny",
            "domain_tags": ["rr03", "canary"],
        },
    )

    await store.upsert_vectors(_RAW, d_points)
    await store.upsert_vectors(_RAW, s_points)
    await store.upsert_vectors(_RAW, g_points)
    await store.upsert_vectors(_RAW, [canary_point])

    report = await run_health(
        store,
        source_ids=[book_d, book_s, book_g, canary_source],
    )

    assert report.books == 4
    assert report.degenerate == [book_d]
    assert len(report.dominant_section) == 1
    assert report.dominant_section[0][:2] == (book_s, "Bibliography")
    assert len(report.test_points) == 1
    assert report.ok() is False


async def test_run_health_ok_on_empty_collection(store: QdrantStore) -> None:
    report = await run_health(store, source_ids=[])
    assert report.books == 0
    assert report.ok() is True


async def test_run_health_flags_duplicate_titles(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    marker = uuid.uuid4().hex[:8]
    twin_a = f"itest-health-dup-a-{marker}"
    twin_b = f"itest-health-dup-b-{marker}"
    seeded_source_ids.extend([twin_a, twin_b])

    title = f"Same Title {marker}"
    a_points = _points(twin_a, 10, source_book=title)
    b_points = _points(twin_b, 10, source_book=title)

    await store.upsert_vectors(_RAW, a_points)
    await store.upsert_vectors(_RAW, b_points)

    report = await run_health(store, source_ids=[twin_a, twin_b])

    assert report.duplicate_groups == 1
    assert report.ok() is False


async def test_run_health_flags_unpaginated_page_bucket_titles(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    marker = uuid.uuid4().hex[:8]
    stuck = f"itest-health-stuck-{marker}"
    paginated = f"itest-health-paginated-{marker}"
    ebook = f"itest-health-ebook-{marker}"
    legacy = f"itest-health-legacy-{marker}"
    seeded_source_ids.extend([stuck, paginated, ebook, legacy])

    stuck_points = _points(
        stuck,
        40,
        source_book="Stuck",
        section_title=[f"pp. {i}-{i + 1}" for i in range(40)],
        page_number=1,
    )
    paginated_points = _points(
        paginated,
        300,
        source_book="Paginated",
        section_title=[f"pp. {i}-{i + 1}" for i in range(40)] + [None] * 260,
        page_number=list(range(1, 301)),
        parser_used="docling",
    )
    # A chapter-paginated EPUB: "pages" 1..40 are chapters, so its
    # page-bucket titles are fabricated too.
    ebook_points = _points(
        ebook,
        40,
        source_book="Ebook",
        section_title=[f"pp. {i}-{i + 1}" for i in range(40)],
        page_number=list(range(1, 41)),
        parser_used="epub",
    )
    # A legacy GLM-era PDF: no parser_used, page buckets are its real titles.
    legacy_points = _points(
        legacy,
        40,
        source_book="Legacy",
        section_title=[f"pp. {i}-{i + 1}" for i in range(40)],
        page_number=list(range(1, 41)),
    )

    await store.upsert_vectors(_RAW, stuck_points)
    await store.upsert_vectors(_RAW, paginated_points)
    await store.upsert_vectors(_RAW, ebook_points)
    await store.upsert_vectors(_RAW, legacy_points)

    report = await run_health(store, source_ids=[stuck, paginated, ebook, legacy])

    assert report.unpaginated_page_titles == [stuck, ebook]
    assert report.ok() is False


async def test_run_health_groups_duplicates_by_payload_author(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    marker = uuid.uuid4().hex[:8]
    smith_a, smith_b, jones = (f"itest-health-author-{k}-{marker}" for k in ("a", "b", "c"))
    seeded_source_ids.extend([smith_a, smith_b, jones])
    title = f"Handbook {marker}"
    await store.upsert_vectors(_RAW, _points(smith_a, 5, source_book=title, author="Smith"))
    await store.upsert_vectors(_RAW, _points(jones, 5, source_book=title, author="Jones"))

    alone = await run_health(store, source_ids=[smith_a, jones])
    assert alone.duplicate_groups == 0
    assert alone.ok() is True

    await store.upsert_vectors(_RAW, _points(smith_b, 5, source_book=title, author="Smith"))
    twins = await run_health(store, source_ids=[smith_a, smith_b, jones])
    assert twins.duplicate_groups == 1
    assert twins.ok() is False
