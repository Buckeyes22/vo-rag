from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from vo_rag.pipelines.book_rag.validator import (
    BUCKET_ARTIFACTS,
    DEGENERATE_MIN_CHUNKS_DEFAULT,
    MIN_BOOK_PAGES_DEFAULT,
    RAW_COLLECTION,
    STERILE_COLLECTION,
    IngestionValidator,
    _get_artifact_by_id,
    _get_artifact_by_source,
    non_book_skip_by_pages,
)

SRC = "ab" * 32


class _FakeFigures:
    def __init__(self, present: set[str]) -> None:
        self.present = present

    def exists(self, ref: str) -> bool:
        return ref in self.present


class TestNonBookSkipByPages:
    def test_below_threshold_skips(self) -> None:
        skip, reason = non_book_skip_by_pages(4)
        assert skip is True
        assert reason == f"non_book_too_few_pages:4<{MIN_BOOK_PAGES_DEFAULT}"

    def test_at_or_above_threshold_keeps(self) -> None:
        assert non_book_skip_by_pages(5) == (False, "")
        assert non_book_skip_by_pages(200) == (False, "")

    def test_unknown_page_count_fails_open(self) -> None:
        # None (unreadable count) must never skip a real book.
        assert non_book_skip_by_pages(None) == (False, "")

    def test_threshold_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VO_RAG_BOOK_RAG_MIN_BOOK_PAGES", "20")
        skip, _ = non_book_skip_by_pages(12)
        assert skip is True
        assert non_book_skip_by_pages(20) == (False, "")


class TestConstants:
    def test_degenerate_min_chunks_default_is_11(self) -> None:
        # The degeneracy floor must default to 11 (above the 1-10 bad band
        # that produced false-PASS failures).
        assert DEGENERATE_MIN_CHUNKS_DEFAULT == 11

    def test_min_book_pages_default_is_5(self) -> None:
        assert MIN_BOOK_PAGES_DEFAULT == 5


class TestIsDegenerateChunkOutput:
    def test_one_chunk_100_pages_is_degenerate(self) -> None:
        from vo_rag.pipelines.book_rag.validator import (
            ChunkOutputStats,
            is_degenerate_chunk_output,
        )

        stats = ChunkOutputStats(
            chunk_count=1,
            total_tokens=900,
            distinct_section_paths=1,
            total_pages=100,
            page_span=100,
        )
        is_degenerate, detail = is_degenerate_chunk_output(stats)
        assert is_degenerate is True
        assert "chunks=1" in detail

    def test_two_hundred_chunks_100_pages_is_not_degenerate(self) -> None:
        from vo_rag.pipelines.book_rag.validator import (
            ChunkOutputStats,
            is_degenerate_chunk_output,
        )

        stats = ChunkOutputStats(
            chunk_count=200,
            total_tokens=90_000,
            distinct_section_paths=10,
            total_pages=100,
            page_span=100,
        )
        is_degenerate, detail = is_degenerate_chunk_output(stats)
        assert is_degenerate is False
        assert "chunks=200" in detail


class TestIsRunawayChunkOutput:
    def test_dense_epub_normal_chunk_size_is_not_runaway(self) -> None:
        # VOR-1392: a reflowable EPUB has synthetic-low page numbers, so a valid
        # full book (731 chunks over 6 "pages" = 122/page) trips the raw
        # chunks/page cap — but its chunks are NORMAL size (~290 tok), so it is
        # NOT an explosion and must pass.
        from vo_rag.pipelines.book_rag.validator import (
            ChunkOutputStats,
            is_runaway_chunk_output,
        )

        stats = ChunkOutputStats(
            chunk_count=731,
            total_tokens=731 * 290,
            distinct_section_paths=40,
            total_pages=6,
            page_span=6,
        )
        runaway, _ = is_runaway_chunk_output(stats)
        assert runaway is False

    def test_real_explosion_tiny_chunks_is_runaway(self) -> None:
        # A genuine parse explosion: 194k chunks over 611 real pages, fragmented
        # into ~2-token chunks. Must still be flagged.
        from vo_rag.pipelines.book_rag.validator import (
            ChunkOutputStats,
            is_runaway_chunk_output,
        )

        stats = ChunkOutputStats(
            chunk_count=194_248,
            total_tokens=194_248 * 2,
            distinct_section_paths=50,
            total_pages=611,
            page_span=611,
        )
        runaway, detail = is_runaway_chunk_output(stats)
        assert runaway is True
        assert "194248" in detail

    def test_below_page_floor_never_flagged(self) -> None:
        # Below the 5-page floor the per-page rate is not judged (matches the
        # prior inline guard's ``eff_pages >= 5``).
        from vo_rag.pipelines.book_rag.validator import (
            ChunkOutputStats,
            is_runaway_chunk_output,
        )

        stats = ChunkOutputStats(
            chunk_count=500,
            total_tokens=500,
            distinct_section_paths=3,
            total_pages=4,
            page_span=4,
        )
        runaway, _ = is_runaway_chunk_output(stats)
        assert runaway is False

    def test_fragmentation_floor_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from vo_rag.pipelines.book_rag.validator import (
            ChunkOutputStats,
            is_runaway_chunk_output,
        )

        # 100 chunks/page (over cap) but normal 250-token chunks -> not runaway
        # by default; raising the fragmentation floor above 250 force-flags it.
        stats = ChunkOutputStats(
            chunk_count=1000,
            total_tokens=1000 * 250,
            distinct_section_paths=20,
            total_pages=10,
            page_span=10,
        )
        assert is_runaway_chunk_output(stats)[0] is False
        monkeypatch.setenv("VORAG_RUNAWAY_MIN_MEAN_TOKENS", "300")
        assert is_runaway_chunk_output(stats)[0] is True


class TestMetadataFieldCompleteness:
    def test_license_class_unknown_is_treated_as_missing(self) -> None:
        from vo_rag.pipelines.book_rag.validator import (
            MetadataFieldCompleteness,
            metadata_field_completeness,
        )

        payloads = [
            {"metadata": {"license_class": "unknown"}},
            {"metadata": {"license_class": "unknown"}},
            {"metadata": {"license_class": "licensed_internal"}},
        ]
        status: MetadataFieldCompleteness = metadata_field_completeness(payloads, "license_class")
        assert status.dominant_missing is True
        assert status.missing_count == 2


def _check(report, name: str):
    return next(check for check in report.checks if check.name == name)


def _missing_collection_error(collection: str) -> UnexpectedResponse:
    return UnexpectedResponse(
        404,
        "Not Found",
        f"Collection `{collection}` doesn't exist!".encode(),
        httpx.Headers(),
    )


class _CollectionAwareQdrant:
    def __init__(
        self,
        payloads: list[dict[str, object]],
        *,
        missing_collections: set[str] | None = None,
    ) -> None:
        self.payloads = payloads
        self.missing_collections = missing_collections or set()
        self.count_calls: list[str] = []
        self.scroll_calls: list[str] = []

    async def count(self, *, collection_name: str, **_: object) -> SimpleNamespace:
        self.count_calls.append(collection_name)
        if collection_name in self.missing_collections:
            raise _missing_collection_error(collection_name)
        return SimpleNamespace(count=len(self.payloads))

    async def scroll(
        self,
        *,
        collection_name: str,
        **_: object,
    ) -> tuple[list[SimpleNamespace], None]:
        self.scroll_calls.append(collection_name)
        if collection_name in self.missing_collections:
            raise _missing_collection_error(collection_name)
        return ([SimpleNamespace(payload=payload) for payload in self.payloads], None)


class _RecordingRetriever:
    def __init__(self, hits: list[dict[str, object]]) -> None:
        self.hits = hits
        self.calls: list[dict[str, object]] = []

    async def hybrid_search(
        self,
        query_text: str,
        collection: str,
        *,
        top_k: int,
        filters: dict[str, object],
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "query_text": query_text,
                "collection": collection,
                "top_k": top_k,
                "filters": filters,
            }
        )
        return self.hits


def _raw_payload(
    *,
    source_id: str = "s1",
    page_number: int = 1,
    page_range: str | None = None,
    token_count: int = 500,
    total_pages: int = 100,
    section_path: str = "[Book > Chapter 1]",
    source_book: str | None = "Reference Manual",
    author: str | None = None,
    license_class: str | None = "licensed_internal",
    canonical_source_id: str | None = "vo-rag-corpus-sha256:abc123",
    domain_tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "page_number": page_number,
        "page_range": page_range,
        "section_path": section_path,
        "section_title": "Chapter 1",
        "token_count": token_count,
        "metadata": {
            "source_book": source_book,
            "author": author,
            "license_class": license_class,
            "canonical_source_id": canonical_source_id,
            "domain_tags": [] if domain_tags is None else domain_tags,
            "book_rag_forward_path": {
                "total_pages": total_pages,
                "chunk_count": 1,
            },
        },
    }


def _validator_for_raw_payloads(
    payloads: list[dict[str, object]],
    *,
    raw_count: int | None = None,
    sterile_count: int | None = None,
    sub_count: int | None = None,
) -> Any:
    count = raw_count if raw_count is not None else len(payloads)
    qdrant = SimpleNamespace(
        count=AsyncMock(
            side_effect=[
                SimpleNamespace(count=count),
                SimpleNamespace(count=sterile_count if sterile_count is not None else count),
            ]
        ),
        scroll=AsyncMock(
            return_value=([SimpleNamespace(payload=payload) for payload in payloads], None)
        ),
    )
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "h1"}]))
    validator: Any = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )
    validator._test_sub_count = sub_count if sub_count is not None else count
    return validator


def _patch_artifact_checks(
    monkeypatch: pytest.MonkeyPatch,
    validator: Any,
) -> None:
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=validator._test_sub_count),
    )


@pytest.mark.asyncio
async def test_get_artifact_by_source_casts_jsonb_parameter_to_text() -> None:
    class _Pool:
        query = ""
        args: tuple[object, ...] = ()

        async def fetchrow(self, query: str, *args: object) -> dict[str, str]:
            self.query = query
            self.args = args
            return {"id": "artifact-1", "title": "Reference Manual"}

    pool = _Pool()

    artifact = await _get_artifact_by_source(pool, "source-1")

    assert artifact is not None
    assert artifact.id == "artifact-1"
    assert "jsonb_build_object('source_id', $1::text)" in pool.query
    assert pool.args == ("source-1",)


@pytest.mark.asyncio
async def test_get_artifact_by_id_queries_specific_artifact() -> None:
    class _Pool:
        query = ""
        args: tuple[object, ...] = ()

        async def fetchrow(self, query: str, *args: object) -> dict[str, str]:
            self.query = query
            self.args = args
            return {"id": "artifact-1", "title": "Reference Manual"}

    pool = _Pool()

    artifact = await _get_artifact_by_id(pool, "artifact-1")

    assert artifact is not None
    assert artifact.id == "artifact-1"
    assert "WHERE id = $1" in pool.query
    assert pool.args == ("artifact-1",)


@pytest.mark.asyncio
async def test_raw_sterile_1to1_fails_when_counts_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    qdrant = SimpleNamespace(
        count=AsyncMock(side_effect=[SimpleNamespace(count=4), SimpleNamespace(count=3)]),
        scroll=AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        payload={"source_id": "s1", "page_number": 1, "section_title": "Intro"}
                    )
                ],
                None,
            )
        ),
    )
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "h1"}]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=3),
    )

    report = await validator.validate(document_id="doc-1", source_id="s1")

    assert _check(report, "raw_sterile_1to1").passed is False
    assert report.passed is False


@pytest.mark.asyncio
async def test_count_returns_zero_when_collection_missing() -> None:
    qdrant = _CollectionAwareQdrant([], missing_collections={STERILE_COLLECTION})
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )

    count = await validator._count(STERILE_COLLECTION, "s1")

    assert count == 0


@pytest.mark.asyncio
async def test_validate_skips_sterile_checks_when_sterilize_disabled_and_sterile_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
        for page in range(1, 81)
    ]
    qdrant = _CollectionAwareQdrant(payloads, missing_collections={STERILE_COLLECTION})
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = _RecordingRetriever([{"id": "hit-1"}])
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
        sterilize_enabled=False,
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=len(payloads)),
    )

    report = await validator.validate(document_id="doc-sterile-off", source_id="s1")

    assert _check(report, "raw_sterile_1to1").passed is True
    assert _check(report, "raw_sterile_1to1").detail == "skipped: sterilize disabled"
    assert _check(report, "sample_query_returns").passed is True
    assert _check(report, "sample_query_returns").detail == f"1 hits @ {RAW_COLLECTION}"
    assert STERILE_COLLECTION not in qdrant.count_calls
    assert STERILE_COLLECTION not in qdrant.scroll_calls
    assert retriever.calls == [
        {
            "query_text": "Intro",
            "collection": RAW_COLLECTION,
            "top_k": 5,
            "filters": {"source_id": "s1"},
        }
    ]
    assert report.passed is True


@pytest.mark.asyncio
async def test_validate_fails_sample_query_when_sterilize_disabled_and_raw_returns_no_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
        for page in range(1, 81)
    ]
    qdrant = _CollectionAwareQdrant(payloads, missing_collections={STERILE_COLLECTION})
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = _RecordingRetriever([])
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
        sterilize_enabled=False,
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=len(payloads)),
    )

    report = await validator.validate(document_id="doc-sterile-off-no-hits", source_id="s1")

    assert _check(report, "raw_sterile_1to1").detail == "skipped: sterilize disabled"
    assert _check(report, "sample_query_returns").passed is False
    assert _check(report, "sample_query_returns").detail == f"0 hits @ {RAW_COLLECTION}"
    assert retriever.calls == [
        {
            "query_text": "Intro",
            "collection": RAW_COLLECTION,
            "top_k": 5,
            "filters": {"source_id": "s1"},
        }
    ]
    assert report.passed is False


@pytest.mark.asyncio
async def test_validate_queries_sterile_collection_when_sterilize_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
        for page in range(1, 81)
    ]
    qdrant = _CollectionAwareQdrant(payloads)
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = _RecordingRetriever([{"id": "hit-1"}])
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
        sterilize_enabled=True,
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=len(payloads)),
    )

    report = await validator.validate(document_id="doc-sterile-on", source_id="s1")

    assert _check(report, "sample_query_returns").passed is True
    assert _check(report, "sample_query_returns").detail == f"1 hits @ {STERILE_COLLECTION}"
    assert retriever.calls == [
        {
            "query_text": "Intro",
            "collection": STERILE_COLLECTION,
            "top_k": 5,
            "filters": {"source_id": "s1"},
        }
    ]
    assert report.passed is True


@pytest.mark.asyncio
async def test_validate_skips_pg_checks_when_pool_is_none() -> None:
    payloads = [
        _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
        for page in range(1, 81)
    ]
    qdrant = _CollectionAwareQdrant(payloads)
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "hit-1"}]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=None,
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )

    report = await validator.validate(document_id="doc-no-pg", source_id="s1")

    assert _check(report, "artifact_exists").passed is True
    assert _check(report, "artifact_exists").detail == "skipped: no pg pool"
    assert _check(report, "sub_chunks_present").passed is True
    assert _check(report, "sub_chunks_present").detail == "skipped: no pg pool"
    assert report.passed is True


@pytest.mark.asyncio
async def test_validate_skips_book_rag_compliance_sub_chunk_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
        for page in range(1, 81)
    ]
    qdrant = _CollectionAwareQdrant(payloads)
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "hit-1"}]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )
    count_sub_chunks = AsyncMock(return_value=0)

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        count_sub_chunks,
    )

    report = await validator.validate(document_id="doc-book-rag", source_id="s1")

    assert _check(report, "sub_chunks_present").passed is True
    assert _check(report, "sub_chunks_present").detail == "skipped: compliance sub-chunking not run"
    count_sub_chunks.assert_not_called()
    assert report.passed is True


@pytest.mark.asyncio
async def test_payload_fields_complete_requires_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    qdrant = SimpleNamespace(
        count=AsyncMock(side_effect=[SimpleNamespace(count=2), SimpleNamespace(count=2)]),
        scroll=AsyncMock(
            return_value=(
                [
                    SimpleNamespace(payload={"source_id": "s1", "page_number": 1}),
                ],
                None,
            )
        ),
    )
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "h1"}]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=2),
    )

    report = await validator.validate(document_id="doc-2", source_id="s1")

    assert _check(report, "payload_fields_complete").passed is False
    assert _check(report, "payload_fields_complete").detail == "1 missing"


@pytest.mark.asyncio
async def test_images_resolve_checks_figure_store(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = f"{SRC}/p007_img001.png"
    qdrant = SimpleNamespace(
        count=AsyncMock(side_effect=[SimpleNamespace(count=1), SimpleNamespace(count=1)]),
        scroll=AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        payload={
                            "source_id": "s1",
                            "page_number": 7,
                            "section_title": "Img",
                            "image_ref": ref,
                        }
                    )
                ],
                None,
            )
        ),
    )
    garage = SimpleNamespace(head_object=AsyncMock(return_value=None))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "h1"}]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present={ref}),
        artifacts_bucket="configured-artifacts",
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=1),
    )

    report = await validator.validate(document_id="doc-3", source_id="s1")

    assert _check(report, "images_resolve").passed is True
    assert _check(report, "images_resolve").detail == "0 missing"


@pytest.mark.asyncio
async def test_images_resolve_fails_when_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = f"{SRC}/p007_img001.png"
    qdrant = SimpleNamespace(
        count=AsyncMock(side_effect=[SimpleNamespace(count=1), SimpleNamespace(count=1)]),
        scroll=AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        payload={
                            "source_id": "s1",
                            "page_number": 7,
                            "section_title": "Img",
                            "image_ref": ref,
                        }
                    )
                ],
                None,
            )
        ),
    )
    garage = SimpleNamespace(head_object=AsyncMock(return_value=None))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "h1"}]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
        artifacts_bucket="configured-artifacts",
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=1),
    )

    report = await validator.validate(document_id="doc-3-missing", source_id="s1")

    assert _check(report, "images_resolve").passed is False
    assert _check(report, "images_resolve").detail == "1 missing"


def test_default_artifacts_bucket_matches_canonical_default() -> None:
    # The default must be the canonical vo-rag artifacts bucket name.
    assert BUCKET_ARTIFACTS == "vorag-artifacts"


@pytest.mark.asyncio
async def test_overall_passed_true_when_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    qdrant = SimpleNamespace(
        count=AsyncMock(side_effect=[SimpleNamespace(count=2), SimpleNamespace(count=2)]),
        scroll=AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        payload=_raw_payload(
                            source_id="s1",
                            page_number=1,
                            section_path="[Book > Intro]",
                            total_pages=1,
                        )
                    )
                ],
                None,
            )
        ),
    )
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[{"id": "hit-1"}]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=4),
    )

    report = await validator.validate(document_id="doc-4", source_id="s1")

    assert all(check.passed for check in report.checks)
    assert report.passed is True


@pytest.mark.asyncio
async def test_one_failing_check_sets_report_passed_false(monkeypatch: pytest.MonkeyPatch) -> None:
    qdrant = SimpleNamespace(
        count=AsyncMock(side_effect=[SimpleNamespace(count=2), SimpleNamespace(count=2)]),
        scroll=AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        payload={
                            "source_id": "s1",
                            "page_number": 1,
                            "section_title": "Intro",
                        }
                    )
                ],
                None,
            )
        ),
    )
    garage = SimpleNamespace(head_object=AsyncMock(return_value={"size": 10, "etag": "x"}))
    retriever = SimpleNamespace(hybrid_search=AsyncMock(return_value=[]))
    validator = IngestionValidator(
        qdrant=qdrant,
        pool=SimpleNamespace(),
        garage=garage,
        retriever=retriever,
        figures=_FakeFigures(present=set()),
    )

    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._get_artifact_by_source",
        AsyncMock(return_value=SimpleNamespace(id="a1", title="Intro")),
    )
    monkeypatch.setattr(
        "vo_rag.pipelines.book_rag.validator._count_sub_chunks_for_source",
        AsyncMock(return_value=2),
    )

    report = await validator.validate(document_id="doc-5", source_id="s1")

    assert _check(report, "sample_query_returns").passed is False
    assert report.passed is False


@pytest.mark.asyncio
async def test_degenerate_guard_passes_normal_book(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [
        _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
        for page in range(1, 81)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-normal", source_id="s1")

    check = _check(report, "chunk_output_not_degenerate")
    assert check.passed is True
    assert "chunks=80" in check.detail
    assert report.passed is True


@pytest.mark.asyncio
async def test_degenerate_guard_fails_one_chunk_book(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [_raw_payload(page_number=1, page_range="1-160", total_pages=160, token_count=900)]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-one", source_id="s1")

    check = _check(report, "chunk_output_not_degenerate")
    assert check.passed is False
    assert "chunks=1" in check.detail
    assert report.passed is False


@pytest.mark.asyncio
async def test_degenerate_guard_fails_few_chunks_for_many_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(
            page_number=index * 30 + 1,
            page_range=f"{index * 30 + 1}-{index * 30 + 30}",
            total_pages=240,
            token_count=250,
        )
        for index in range(8)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-few", source_id="s1")

    check = _check(report, "chunk_output_not_degenerate")
    assert check.passed is False
    assert "chunks=8" in check.detail
    assert "pages=240" in check.detail
    assert report.passed is False


@pytest.mark.asyncio
async def test_degenerate_guard_env_override_changes_chunk_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VO_RAG_BOOK_RAG_DEGENERATE_MIN_CHUNKS", "8")
    monkeypatch.setenv("VO_RAG_BOOK_RAG_DEGENERATE_MIN_CHUNKS_PER_PAGE", "0.02")
    payloads = [
        _raw_payload(
            page_number=index * 20 + 1,
            page_range=f"{index * 20 + 1}-{index * 20 + 20}",
            total_pages=160,
            token_count=600,
        )
        for index in range(8)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-override", source_id="s1")

    check = _check(report, "chunk_output_not_degenerate")
    assert check.passed is True
    assert "min_chunks=8" in check.detail
    assert report.passed is True


@pytest.mark.asyncio
async def test_metadata_completeness_passes_when_required_book_identity_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
        for page in range(1, 81)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-metadata-ok", source_id="s1")

    check = _check(report, "metadata_completeness")
    assert check.passed is True
    assert "missing_required=none" in check.detail
    assert report.passed is True


@pytest.mark.asyncio
async def test_metadata_completeness_fails_empty_source_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(
            page_number=page,
            page_range=str(page),
            total_pages=120,
            token_count=450,
            source_book="",
        )
        for page in range(1, 81)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-empty-book", source_id="s1")

    check = _check(report, "metadata_completeness")
    assert check.passed is False
    assert "source_book" in check.detail
    assert report.passed is False


@pytest.mark.asyncio
async def test_metadata_completeness_fails_unknown_license_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        _raw_payload(
            page_number=page,
            page_range=str(page),
            total_pages=120,
            token_count=450,
            license_class="unknown",
        )
        for page in range(1, 81)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-unknown-license", source_id="s1")

    check = _check(report, "metadata_completeness")
    assert check.passed is False
    assert "license_class" in check.detail
    assert report.passed is False


@pytest.mark.asyncio
async def test_metadata_completeness_warn_mode_downgrades_missing_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VO_RAG_BOOK_RAG_METADATA_GUARD", "warn")
    payloads = [
        _raw_payload(
            page_number=page,
            page_range=str(page),
            total_pages=120,
            token_count=450,
            source_book="",
        )
        for page in range(1, 81)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-warn-metadata", source_id="s1")

    check = _check(report, "metadata_completeness")
    assert check.passed is True
    assert "guard=warn" in check.detail
    assert "source_book" in check.detail
    assert report.passed is True


@pytest.mark.asyncio
async def test_metadata_completeness_required_env_override_changes_required_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VO_RAG_BOOK_RAG_METADATA_REQUIRED",
        "license_class,canonical_source_id",
    )
    payloads = [
        _raw_payload(
            page_number=page,
            page_range=str(page),
            total_pages=120,
            token_count=450,
            source_book="",
        )
        for page in range(1, 81)
    ]
    validator = _validator_for_raw_payloads(payloads)
    _patch_artifact_checks(monkeypatch, validator)

    report = await validator.validate(document_id="doc-required-override", source_id="s1")

    check = _check(report, "metadata_completeness")
    assert check.passed is True
    assert "required=license_class,canonical_source_id" in check.detail
    assert report.passed is True


def test_validation_report_has_eight_checks_on_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the IngestionValidator contract: exactly 8 named checks per
    validation report (the 8-check audit). Names are stable because they're
    exported observability surface.
    """
    import asyncio

    async def _run() -> set[str]:
        payloads = [
            _raw_payload(page_number=page, page_range=str(page), total_pages=120, token_count=450)
            for page in range(1, 81)
        ]
        validator = _validator_for_raw_payloads(payloads)
        _patch_artifact_checks(monkeypatch, validator)
        report = await validator.validate(document_id="doc-count", source_id="s1")
        return {check.name for check in report.checks}

    names = asyncio.run(_run())
    expected = {
        "raw_sterile_1to1",
        "chunk_output_not_degenerate",
        "metadata_completeness",
        "payload_fields_complete",
        "images_resolve",
        "artifact_exists",
        "sub_chunks_present",
        "sample_query_returns",
    }
    assert names == expected
