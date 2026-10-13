"""Tests for book_rag chunk + embed + upsert step module (VOR-7).

All external dependencies are faked:
- ``FakeEmbeddingClient`` — returns scripted dense / sparse / ColBERT responses.
- ``FakeQdrantStore`` — records upsert calls for assertions.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

import pytest

from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.points import VoPoint
from vo_rag.pipelines.book_rag.domain_vocab import DOMAIN_VOCABULARY
from vo_rag.pipelines.book_rag.steps.chunk_embed import (
    build_raw_points,
    do_build_chunks,
    do_embed_raw,
    do_upsert_raw,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    *,
    elements: list[ParsedElement] | None = None,
    title: str = "Test Book",
    total_pages: int = 1,
) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id="src-1",
        title=title,
        total_pages=total_pages,
        elements=elements or [],
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="test",
    )


def _el(text: str, etype: ElementType = ElementType.NARRATIVE_TEXT, page: int = 1) -> ParsedElement:
    return ParsedElement(
        element_id=f"el-{text[:20]}",
        element_type=etype,
        text=text,
        page_number=page,
    )


def _quality_warnings(chunks: list[RetrievalChunk]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        value = chunk.metadata.get("quality_warnings", [])
        if isinstance(value, list):
            result.extend(item for item in value if isinstance(item, dict))
    return result


class FakeEmbeddingClient:
    """Records calls and returns scripted responses."""

    def __init__(
        self,
        *,
        dense: list[list[float]] | None = None,
        sparse: list[dict[str, Any]] | None = None,
        colbert: list[list[list[float]]] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.return_colbert_flags: list[bool] = []
        self._dense = dense
        self._sparse = sparse
        self._colbert = colbert

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(texts)
        self.return_colbert_flags.append(return_colbert)
        return {
            "dense": self._dense or [[0.1] * 1024 for _ in texts],
            "sparse": self._sparse or [{"indices": [1, 2], "values": [0.5, 0.6]} for _ in texts],
            "colbert": self._colbert or [[[0.1] * 1024] * 5 for _ in texts],
        }

    async def aclose(self) -> None:
        pass


class FakeQdrantStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[VoPoint]]] = []

    async def upsert_vectors(self, collection: str, points: list[VoPoint]) -> None:
        self.upserts.append((collection, points))


# ---------------------------------------------------------------------------
# do_build_chunks
# ---------------------------------------------------------------------------


def test_build_chunks_empty_doc_raises() -> None:
    """A completely empty document must raise, not return zero chunks silently."""
    doc = _doc(elements=[])
    with pytest.raises(RuntimeError, match="zero chunks"):
        do_build_chunks(doc)


def test_build_chunks_produces_retrieval_chunks() -> None:
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el(
                "This is narrative text that should form a chunk.",
                ElementType.NARRATIVE_TEXT,
                page=1,
            ),
        ]
    )
    chunks = do_build_chunks(doc)
    assert chunks
    assert all(isinstance(c, RetrievalChunk) for c in chunks)
    assert all(c.source_id == "src-1" for c in chunks)


def test_build_chunks_stamps_book_metadata() -> None:
    """do_build_chunks stamps book identity onto every chunk's metadata, alongside
    the forward-path metadata (the fix for empty DSI/graph title/author)."""
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("Narrative text forming a chunk for the metadata test.", page=1),
        ]
    )
    book_meta = {
        "source_book": "Vehicle Dynamics (Casey Marlow)",
        "title": "Vehicle Dynamics",
        "author": "Casey Marlow",
        "domain_tags": ["vehicle-dynamics"],
        "license_class": "unlicensed",
        "canonical_source_id": "vo-rag-corpus-sha256:abc",
    }
    chunks = do_build_chunks(doc, book_metadata=book_meta)
    assert chunks
    for c in chunks:
        assert c.metadata["source_book"] == "Vehicle Dynamics (Casey Marlow)"
        assert c.metadata["title"] == "Vehicle Dynamics"
        assert c.metadata["author"] == "Casey Marlow"
        assert c.metadata["domain_tags"] == ["vehicle-dynamics"]
        assert c.metadata["license_class"] == "unlicensed"
        assert c.metadata["canonical_source_id"] == "vo-rag-corpus-sha256:abc"
        assert "book_rag_forward_path" in c.metadata  # forward-path preserved


def test_build_chunks_fallback_when_zero_chunks() -> None:
    """When HierarchicalChunker returns nothing, fallback text recovery kicks in."""
    # Very short text is dropped by chunker (below_min_words) but accepted by fallback.
    doc = _doc(
        elements=[
            _el("a", ElementType.NARRATIVE_TEXT, page=1),
        ]
    )
    chunks = do_build_chunks(doc)
    assert chunks
    assert all(c.source_id == "src-1" for c in chunks)


def test_build_chunks_publishes_book_scale_degenerate_output_with_warning() -> None:
    doc = _doc(
        total_pages=25,
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("A short paragraph that collapses into too few chunks for a full book.", page=1),
        ],
    )

    chunks = do_build_chunks(doc)
    assert chunks
    warnings = _quality_warnings(chunks)
    assert any(w.get("code") == "chunk_output_degenerate" for w in warnings)


def test_build_chunks_runaway_guard_judges_content_not_section_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1416: 297k one-char chunks slipped past the runaway guard because
    token_count covers embedding_text, whose bloated section path made every
    chunk read ~782 tokens. The guard must size chunks by their content."""
    from vo_rag.core.chunking.hierarchical import HierarchicalChunker

    giant_path = "[Book > INDEX > " + " > ".join(f"Name{n}, Arne {n}" for n in range(200)) + "]"

    def exploded(
        self: HierarchicalChunker, doc: ParsedDocument, composites: Any
    ) -> list[RetrievalChunk]:
        return [
            RetrievalChunk(
                chunk_id=f"c{i}",
                chunk_type="text",
                document_id=doc.document_id,
                source_id=doc.source_id,
                page_number=None,
                section_path=giant_path,
                section_title="INDEX",
                content="abcdefghijklmnopqrstuvwxyz"[i % 26],
                embedding_text=f"{giant_path}\n\n{'abcdefghijklmnopqrstuvwxyz'[i % 26]}",
                token_count=self.count(f"{giant_path}\n\nx"),
            )
            for i in range(216 * 100)
        ]

    monkeypatch.setattr(HierarchicalChunker, "chunk", exploded)
    doc = _doc(total_pages=216, elements=[_el("Some prose on the first page.", page=1)])

    chunks = do_build_chunks(doc)
    assert chunks
    assert len(chunks) < 216 * 100
    assert all(len(c.content) > 1 for c in chunks)
    forward_path = chunks[0].metadata.get("book_rag_forward_path")
    assert isinstance(forward_path, dict)
    assert forward_path.get("chunking_fallback") == "bounded_text_repack_after_runaway"
    warnings = _quality_warnings(chunks)
    assert any(w.get("code") == "runaway_chunk_output_repacked" for w in warnings)


def test_build_chunks_runaway_without_usable_fallback_stays_per_document_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.core.chunking.hierarchical import HierarchicalChunker
    from vo_rag.pipelines.book_rag.steps import chunk_embed

    def exploded(
        self: HierarchicalChunker, doc: ParsedDocument, composites: Any
    ) -> list[RetrievalChunk]:
        return [
            RetrievalChunk(
                chunk_id=f"x{i}",
                chunk_type="text",
                document_id=doc.document_id,
                source_id=doc.source_id,
                page_number=None,
                section_path="x",
                section_title="x",
                content="a",
                embedding_text="x\na",
                token_count=2,
            )
            for i in range(216 * 100)
        ]

    monkeypatch.setattr(HierarchicalChunker, "chunk", exploded)
    monkeypatch.setattr(chunk_embed, "_fallback_text_chunks", lambda *args, **kwargs: [])
    doc = _doc(total_pages=216, elements=[_el("Some prose on the first page.", page=1)])
    with pytest.raises(RuntimeError, match="bounded text repack produced no usable chunks"):
        do_build_chunks(doc)


def test_build_chunks_publishes_bounded_repack_even_if_heuristic_still_flags_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.core.chunking.hierarchical import HierarchicalChunker
    from vo_rag.pipelines.book_rag.steps import chunk_embed

    def exploded(
        self: HierarchicalChunker, doc: ParsedDocument, composites: Any
    ) -> list[RetrievalChunk]:
        return [
            RetrievalChunk(
                chunk_id=f"x{i}",
                chunk_type="text",
                document_id=doc.document_id,
                source_id=doc.source_id,
                page_number=None,
                section_path="x",
                section_title="x",
                content="a",
                embedding_text="x\na",
                token_count=2,
            )
            for i in range(216 * 100)
        ]

    monkeypatch.setattr(HierarchicalChunker, "chunk", exploded)
    monkeypatch.setattr(
        chunk_embed,
        "is_runaway_chunk_output",
        lambda stats: (True, "test heuristic remains flagged"),
    )
    doc = _doc(total_pages=216, elements=[_el("Some usable prose on the first page.", page=1)])

    chunks = do_build_chunks(doc)

    assert chunks
    assert len(chunks) < chunk_embed._RUNAWAY_REPACK_MAX_CHUNKS
    assert any(
        warning.get("code") == "runaway_chunk_output_repacked_with_residual_warning"
        for warning in _quality_warnings(chunks)
    )


def test_build_chunks_allows_small_doc_below_degenerate_page_floor() -> None:
    doc = _doc(
        total_pages=24,
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el(
                "A short paragraph is acceptable for a document below the book-scale floor.",
                page=1,
            ),
        ],
    )

    chunks = do_build_chunks(doc)

    assert chunks


# ---------------------------------------------------------------------------
# do_build_chunks — heuristic domain tagger wiring (VOR-296)
# ---------------------------------------------------------------------------


# Body text that lights up the ``setup_tuning`` vocabulary bucket hard
# (camber / toe / alignment / tuning / setup / caster — six matches).  Repeated
# to force HierarchicalChunker to emit more than one chunk so the
# "identical across every chunk" assertion is meaningful.
_SETUP_TUNING_BODY = (
    "Setup notes cover camber, toe, and alignment adjustments for tuning "
    "the suspension geometry. Caster changes affect straight-line stability. "
) * 8


def test_build_chunks_stamps_heuristic_domain_tags_when_caller_omits_them() -> None:
    """VOR-296: fresh ingests carry no caller-supplied domain_tags.  The
    deterministic heuristic must stamp identical book-level tags onto every
    chunk (domain_tags is a book-level retrieval facet, not per-chunk)."""
    doc = _doc(
        title="Suspension Setup Handbook",
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el(_SETUP_TUNING_BODY, ElementType.NARRATIVE_TEXT, page=1),
            _el("Chapter 2", ElementType.TITLE, page=2),
            _el(_SETUP_TUNING_BODY, ElementType.NARRATIVE_TEXT, page=2),
        ],
    )

    chunks = do_build_chunks(doc)

    assert chunks, "fixture must produce chunks"
    # Every chunk carries domain_tags (per-chunk setdefault stamps it).
    assert all("domain_tags" in c.metadata for c in chunks)
    # Every chunk carries the SAME set of book-level tags.
    tag_sets: set[tuple[str, ...]] = set()
    for c in chunks:
        dt = c.metadata["domain_tags"]
        assert isinstance(dt, list)
        tag_sets.add(tuple(dt))
    assert len(tag_sets) == 1, f"chunks disagree on domain_tags: {tag_sets}"
    tags = chunks[0].metadata["domain_tags"]
    assert isinstance(tags, list)
    assert tags, "heuristic must return a non-empty list for tuning-heavy body"
    assert "setup_tuning" in tags
    # All tags must come from the controlled vocabulary.
    assert set(tags) <= set(DOMAIN_VOCABULARY)


def test_build_chunks_preserves_caller_supplied_domain_tags() -> None:
    """VOR-296: caller-supplied domain_tags (migrated/golden corpus, operator
    backfill) must NEVER be overwritten by the heuristic."""
    doc = _doc(
        title="Suspension Setup Handbook",
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el(_SETUP_TUNING_BODY, ElementType.NARRATIVE_TEXT, page=1),
        ],
    )

    chunks = do_build_chunks(doc, book_metadata={"domain_tags": ["already-set"]})

    assert chunks
    for c in chunks:
        assert c.metadata["domain_tags"] == ["already-set"]


def test_build_chunks_omits_domain_tags_when_heuristic_finds_nothing() -> None:
    """VOR-296: when the body text has no vocabulary keywords, the heuristic
    returns [] and we must NOT stamp an empty list — the key stays absent
    (no crash, no misleading empty-list payload downstream)."""
    # Lorem ipsum has no overlap with the motorsport keyword pack.
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna "
                "aliqua. Ut enim ad minim veniam, quis nostrud exercitation "
                "ullamco laboris nisi ut aliquip ex ea commodo consequat. " * 3,
                ElementType.NARRATIVE_TEXT,
                page=1,
            ),
        ],
    )

    chunks = do_build_chunks(doc)

    assert chunks
    for c in chunks:
        assert not c.metadata.get("domain_tags"), (
            f"expected domain_tags to be absent/empty, got {c.metadata.get('domain_tags')!r}"
        )


# ---------------------------------------------------------------------------
# do_embed_raw
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_raw_calls_client_with_embedding_texts() -> None:
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="hello",
            embedding_text="hello",
            token_count=1,
        ),
        RetrievalChunk(
            chunk_id="c-2",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=2,
            page_range="2",
            section_path="Book",
            section_title="Book",
            content="world",
            embedding_text="world",
            token_count=1,
        ),
    ]
    fake = FakeEmbeddingClient()
    result = await do_embed_raw(chunks, fake)

    assert fake.calls == [["hello", "world"]]
    assert "dense" in result
    assert "sparse" in result
    assert "colbert" in result
    assert len(result["dense"]) == 2
    assert len(result["sparse"]) == 2
    assert len(result["colbert"]) == 2


@pytest.mark.asyncio
async def test_embed_raw_downsamples_colbert_to_192() -> None:
    """ColBERT vectors longer than 192 must be downsampled (Qdrant 1 MiB cap)."""
    long_colbert = [[float(i)] * 1024 for i in range(250)]  # 250 vectors per text
    fake = FakeEmbeddingClient(colbert=[long_colbert, long_colbert])
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="a",
            embedding_text="a",
            token_count=1,
        ),
        RetrievalChunk(
            chunk_id="c-2",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=2,
            page_range="2",
            section_path="Book",
            section_title="Book",
            content="b",
            embedding_text="b",
            token_count=1,
        ),
    ]
    result = await do_embed_raw(chunks, fake)
    assert len(result["colbert"][0]) == 192
    assert len(result["colbert"][1]) == 192


@pytest.mark.asyncio
async def test_embed_raw_empty_colbert_fallback_to_dense() -> None:
    """When BGE-M3 returns empty ColBERT, fallback to dense vector (never silent-drop)."""
    fake = FakeEmbeddingClient(
        dense=[[0.1] * 1024, [0.2] * 1024],
        colbert=[[], []],  # empty for both texts
    )
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="a",
            embedding_text="a",
            token_count=1,
        ),
        RetrievalChunk(
            chunk_id="c-2",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=2,
            page_range="2",
            section_path="Book",
            section_title="Book",
            content="b",
            embedding_text="b",
            token_count=1,
        ),
    ]
    result = await do_embed_raw(chunks, fake)
    # Fallback: each empty colbert becomes [dense_vector]
    assert result["colbert"][0] == [[0.1] * 1024]
    assert result["colbert"][1] == [[0.2] * 1024]


@pytest.mark.asyncio
async def test_embed_raw_default_colbert_opt_out_still_writes_fallback_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VORAG_BOOK_RAG_INCLUDE_COLBERT", raising=False)
    dense = [0.25] * 1024
    fake = FakeEmbeddingClient(dense=[dense], colbert=[[]])
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="a",
            embedding_text="a",
            token_count=1,
        )
    ]

    embeddings = await do_embed_raw(chunks, fake)
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1")

    point = fake_store.upserts[0][1][0]
    assert fake.return_colbert_flags == [False]
    assert point.colbert == [dense]
    assert point.payload["colbert_fallback"] is True


# ---------------------------------------------------------------------------
# do_upsert_raw
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_raw_uses_correct_collection() -> None:
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="hello world",
            embedding_text="hello world",
            token_count=2,
        ),
    ]
    embeddings = {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5}],
        "colbert": [[[0.1] * 1024]],
    }
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1")

    assert len(fake_store.upserts) == 1
    collection, points = fake_store.upserts[0]
    assert collection == "vo_rag_books_raw_v1"
    assert len(points) == 1


@pytest.mark.asyncio
async def test_upsert_raw_returns_writer_point_ids_and_forwards_publication_job() -> None:
    class ReturningStore:
        def __init__(self) -> None:
            self.job: object | None = None

        async def upsert_vectors(
            self,
            _collection: str,
            points: list[VoPoint],
            *,
            publication_job: object | None = None,
        ) -> list[str]:
            self.job = publication_job
            return [f"writer-id-{point.chunk_index}" for point in points]

    chunk = RetrievalChunk(
        chunk_id="c-writer-id",
        chunk_type="text",
        document_id="doc-1",
        source_id="src-1",
        page_number=1,
        page_range="1",
        section_path="Book",
        section_title="Book",
        content="body",
        embedding_text="body",
        token_count=1,
    )
    store = ReturningStore()
    job = object()

    result = await do_upsert_raw(
        [chunk],
        {"dense": [[0.1] * 1024], "sparse": [{}], "colbert": [[]]},
        store,
        "src-1",
        publication_job=job,  # type: ignore[arg-type]
    )

    assert result == {"point_count": 1, "point_ids": ["writer-id-0"]}
    assert store.job is job


@pytest.mark.asyncio
async def test_raw_writer_uses_embedding_text_and_builds_bm25_from_content() -> None:
    embedding_text = "Section heading\nneighbor context used for embedding"
    content = "stored lexical body has tokens"
    chunk = RetrievalChunk(
        chunk_id="c-embed-input",
        chunk_type="text",
        document_id="doc-1",
        source_id="src-1",
        page_number=1,
        page_range="1",
        section_path="Book > Chapter",
        section_title="Chapter",
        content=content,
        embedding_text=embedding_text,
        token_count=8,
    )
    fake = FakeEmbeddingClient(dense=[[0.25] * 1024], sparse=[{}], colbert=[[]])

    embeddings = await do_embed_raw([chunk], fake)
    point = build_raw_points([chunk], embeddings, "src-1")[0]

    assert fake.calls == [[embedding_text]]
    provenance = cast(dict[str, Any], point.payload["embedding_provenance"])
    assert provenance["input_sha256"] == hashlib.sha256(embedding_text.encode()).hexdigest()
    assert point.payload["content"] == content
    assert point.payload["embedding_text"] == embedding_text
    assert point.bm25 == tokens_to_bm25_sparse(tokenize(content))
    assert point.sparse == {}


def test_build_raw_points_accepts_explicit_empty_sparse_map_for_each_chunk() -> None:
    chunks = [
        RetrievalChunk(
            chunk_id=f"c-{index}",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=index + 1,
            page_range=str(index + 1),
            section_path="Book",
            section_title="Book",
            content=f"chunk {index}",
            embedding_text=f"chunk {index}",
            token_count=2,
        )
        for index in range(2)
    ]
    points = build_raw_points(
        chunks,
        {"dense": [[0.1] * 1024, [0.2] * 1024], "sparse": [{}, {}], "colbert": [[], []]},
        "src-1",
    )
    assert [point.sparse for point in points] == [{}, {}]


@pytest.mark.parametrize(
    ("dense", "sparse", "message"),
    [
        ([[0.1] * 1024], [], "sparse response"),
        ([], [{}], "dense response"),
    ],
)
@pytest.mark.asyncio
async def test_embed_raw_rejects_malformed_provider_cardinality(
    dense: list[list[float]], sparse: list[dict[str, Any]], message: str
) -> None:
    class MalformedEmbeddingClient:
        async def embed(self, texts: list[str], **_kwargs: Any) -> dict[str, Any]:
            return {"dense": dense, "sparse": sparse, "colbert": []}

    chunk = RetrievalChunk(
        chunk_id="c-bad-response",
        chunk_type="text",
        document_id="doc-1",
        source_id="src-1",
        page_number=1,
        page_range="1",
        section_path="Book",
        section_title="Book",
        content="body",
        embedding_text="body",
        token_count=1,
    )
    with pytest.raises(RuntimeError, match=message):
        await do_embed_raw([chunk], MalformedEmbeddingClient())


def test_build_raw_points_rejects_missing_sparse_or_dense_rows() -> None:
    chunk = RetrievalChunk(
        chunk_id="c-short-response",
        chunk_type="text",
        document_id="doc-1",
        source_id="src-1",
        page_number=1,
        page_range="1",
        section_path="Book",
        section_title="Book",
        content="body",
        embedding_text="body",
        token_count=1,
    )
    with pytest.raises(ValueError, match="dense embeddings must match"):
        build_raw_points([chunk], {"dense": [], "sparse": [{}]}, "src-1")
    with pytest.raises(ValueError, match="sparse embeddings must match"):
        build_raw_points([chunk], {"dense": [[0.1] * 1024], "sparse": []}, "src-1")


@pytest.mark.asyncio
async def test_upsert_raw_uses_overridden_collection_when_provided() -> None:
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="hello world",
            embedding_text="hello world",
            token_count=2,
        ),
    ]
    embeddings = {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5}],
        "colbert": [[[0.1] * 1024]],
    }
    fake_store = FakeQdrantStore()
    await do_upsert_raw(
        chunks,
        embeddings,
        fake_store,
        "src-1",
        collection="vo_rag_books_raw_v1__ab_semantic",
    )

    assert len(fake_store.upserts) == 1
    collection, points = fake_store.upserts[0]
    assert collection == "vo_rag_books_raw_v1__ab_semantic"
    assert len(points) == 1


@pytest.mark.asyncio
async def test_upsert_raw_deterministic_id_no_version_hash() -> None:
    """Point IDs must be deterministic and unversioned (book_rag invariant)."""
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="a",
            embedding_text="a",
            token_count=1,
        ),
    ]
    embeddings = {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5}],
        "colbert": [[[0.1] * 1024]],
    }
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1")

    point = fake_store.upserts[0][1][0]
    # The point is a VoPoint; the actual Qdrant point ID is computed inside
    # QdrantStore.upsert_vectors. We just need to verify VoPoint fields.
    assert point.source_id == "src-1"
    assert point.chunk_index == 0
    assert point.stage == "raw"
    assert point.version_hash is None


@pytest.mark.asyncio
async def test_upsert_raw_trust_level_defaults_to_quarantined() -> None:
    """F-031: trust_level defaults to 'quarantined' (fail-closed)."""
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="hello",
            embedding_text="hello",
            token_count=1,
        ),
    ]
    embeddings = {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5}],
        "colbert": [[[0.1] * 1024]],
    }
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1")

    point = fake_store.upserts[0][1][0]
    assert point.payload["trust_level"] == "quarantined"


@pytest.mark.asyncio
async def test_upsert_raw_payload_has_expected_fields() -> None:
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1-2",
            section_path="Book > Ch1",
            section_title="Ch1",
            chapter_path="Ch1",
            content="hello",
            embedding_text="hello",
            token_count=5,
            metadata={"foo": "bar"},
        ),
    ]
    embeddings = {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5}],
        "colbert": [[[0.1] * 1024]],
    }
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1")

    point = fake_store.upserts[0][1][0]
    payload = point.payload
    assert payload["chunk_id"] == "c-1"
    assert payload["source_id"] == "src-1"
    assert payload["document_id"] == "doc-1"
    assert payload["section_path"] == "Book > Ch1"
    assert payload["section_title"] == "Ch1"
    assert payload["chapter_path"] == "Ch1"
    assert payload["chunk_type"] == "text"
    assert payload["content"] == "hello"
    assert payload["stage"] == "raw"
    assert payload["page_number"] == 1
    assert payload["page_range"] == "1-2"
    assert payload["token_count"] == 5
    assert payload["metadata"] == {"foo": "bar"}
    assert payload["has_image"] is False
    assert payload["image_id"] is None
    assert payload["image_ref"] is None
    assert payload["vlm_description"] is None
    assert point.bm25  # computed from content, stored on VoPoint not payload


@pytest.mark.asyncio
async def test_upsert_raw_bm25_computed_from_content() -> None:
    """BM25 sparse vector must be derived from chunk content (not embedding text)."""
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="brake bias telemetry",
            embedding_text="Book\n\nbrake bias telemetry",
            token_count=4,
        ),
    ]
    embeddings = {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5}],
        "colbert": [[[0.1] * 1024]],
    }
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1")

    point = fake_store.upserts[0][1][0]
    # BM25 should be non-empty for real words
    assert point.bm25
    assert isinstance(point.bm25, dict)


@pytest.mark.asyncio
async def test_upsert_raw_passes_trust_level_override() -> None:
    """Callers may explicitly pass 'trusted' after compliance gate."""
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content="hello",
            embedding_text="hello",
            token_count=1,
        ),
    ]
    embeddings = {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5}],
        "colbert": [[[0.1] * 1024]],
    }
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1", trust_level="trusted")

    point = fake_store.upserts[0][1][0]
    assert point.payload["trust_level"] == "trusted"


@pytest.mark.asyncio
async def test_embed_raw_sends_full_client_batches() -> None:
    """Each request carries the client's full batch, not 8 chunks: batch 32
    measured 2.5x the throughput of batch 8 on a loaded bge-m3 worker (VOR-1427)."""
    from vo_rag.core.embedding.client import MAX_EMBED_BATCH_TEXTS

    chunks = [
        RetrievalChunk(
            chunk_id=f"c-{i}",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            page_number=1,
            page_range="1",
            section_path="Book",
            section_title="Book",
            content=f"text {i}",
            embedding_text=f"text {i}",
            token_count=2,
        )
        for i in range(40)
    ]
    fake = FakeEmbeddingClient()
    await do_embed_raw(chunks, fake)

    assert [len(call) for call in fake.calls] == [MAX_EMBED_BATCH_TEXTS, 40 - MAX_EMBED_BATCH_TEXTS]


# ---------------------------------------------------------------------------
# Per-chunk OCR quality gate (VOR-1452)
# ---------------------------------------------------------------------------

# Labelled OCR noise sample.
_OCR_NOISE = (
    "Ot Se Bley  OS  oui on: review be désadhed Hi  €  Badiegaty  sBied  qian a  yea am"
    "  Sica  Pang  |  pore,  Me Fle  1A At  te sicry  paws  th ace kyapiet v.  have  5"
    "  he oF.  hg carlin  ah  ¢  a  ti  Ma"
)
_OCR_HALF_READABLE = (
    "T\n\nW\n\n8C\n\nor\n\n8A\n\n3x Easy 10A-Relay-300y or 100y-18\n= Dx 145c 1-4-6B-19-2"
)
_NOTATION = "3p30 x5 (track-spikes)\n120 x3-4 x2 w/ 4' & 8'^s (15.5-16.5)\nskD60 x6 w/ walk back^"


def _chunk(
    content: str, *, chunk_type: str = "text", page: int = 1, idx: int = 0
) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=f"c-{idx}",
        chunk_type=chunk_type,
        document_id="doc-1",
        source_id="src-1",
        page_number=page,
        section_path="Test Book",
        section_title="Test Book",
        content=content,
        embedding_text=content,
        token_count=len(content.split()),
    )


def _patch_chunker_output(
    monkeypatch: pytest.MonkeyPatch, chunks: list[RetrievalChunk]
) -> list[Any]:
    """Make do_build_chunks see *chunks* from the chunker; return a holder
    that receives the HierarchicalChunker instance it built."""
    from vo_rag.pipelines.book_rag.steps import chunk_embed

    built: list[Any] = []
    real_build = chunk_embed._build_chunker

    def build(settings: Any, embedding_client: Any) -> Any:
        chunker = real_build(settings, embedding_client)
        built.append(chunker)
        return chunker

    def fake_chunk_without_boilerplate(
        chunker: Any, parsed_doc: ParsedDocument, composites: Any = None
    ) -> tuple[list[RetrievalChunk], ParsedDocument]:
        return [c.model_copy(deep=True) for c in chunks], parsed_doc

    monkeypatch.setattr(chunk_embed, "_build_chunker", build)
    monkeypatch.setattr(chunk_embed, "chunk_without_boilerplate", fake_chunk_without_boilerplate)
    return built


def test_build_chunks_drops_ocr_noise_keeps_half_readable_and_notation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from structlog.testing import capture_logs

    built = _patch_chunker_output(
        monkeypatch,
        [
            _chunk(_NOTATION, idx=0, page=1),
            _chunk(_OCR_NOISE, idx=1, page=2),
            _chunk(_OCR_HALF_READABLE, idx=2, page=3),
        ],
    )
    with capture_logs() as logs:
        chunks = do_build_chunks(_doc(elements=[_el("placeholder text")], total_pages=3))

    assert [c.chunk_id for c in chunks] == ["c-0", "c-2"]
    assert all("quality_flag" not in c.metadata for c in chunks)  # no flag verdict
    # Counted in the chunker's drop accounting, alongside its own reasons.
    chunker = built[0]
    assert chunker.last_drop_summary.get("ocr_garbage") == 1
    assert any(
        d.page_number == 2 and d.reason == "ocr_garbage" for d in chunker.last_page_drop_ledger
    )
    events = [e for e in logs if e["event"] == "book_rag.chunk.ocr_quality"]
    assert events and events[0]["dropped_by_reason"]["ocr_garbage"] == 1
    assert "flagged" not in events[0]


def test_build_chunks_does_not_assess_figure_composites(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chunker_output(
        monkeypatch,
        [
            _chunk(_NOTATION, idx=0),
            _chunk(_OCR_NOISE, chunk_type="figure_composite", idx=1),
        ],
    )
    chunks = do_build_chunks(_doc(elements=[_el("placeholder text")]))
    assert [c.chunk_id for c in chunks] == ["c-0", "c-1"]
    assert "quality_flag" not in chunks[1].metadata


def test_fallback_chunks_pass_the_ocr_gate() -> None:
    """The zero-chunk fallback used to accept any text with one letter."""
    from vo_rag.pipelines.book_rag.steps.chunk_embed import _fallback_text_chunks

    doc = _doc(
        elements=[
            _el(_OCR_NOISE, ElementType.NARRATIVE_TEXT, page=1),
            _el(_OCR_HALF_READABLE, ElementType.NARRATIVE_TEXT, page=2),
        ]
    )
    drops: dict[str, int] = {}
    chunks = _fallback_text_chunks(
        doc, counter=lambda text: 400, drop_counts=drops
    )  # 400 >= target: one chunk per element
    assert [c.content for c in chunks] == [_OCR_HALF_READABLE]
    assert "quality_flag" not in chunks[0].metadata
    assert drops == {"ocr_garbage": 1}


@pytest.mark.asyncio
async def test_no_quality_flag_reaches_the_qdrant_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate has no flag verdict: kept chunks carry no quality_flag."""
    _patch_chunker_output(monkeypatch, [_chunk(_OCR_HALF_READABLE)])
    chunks = do_build_chunks(_doc(elements=[_el("placeholder text")]))
    embeddings = {"dense": [[0.1] * 1024], "sparse": [{1: 0.5}], "colbert": [[[0.1] * 1024]]}
    fake_store = FakeQdrantStore()
    await do_upsert_raw(chunks, embeddings, fake_store, "src-1")
    payload = fake_store.upserts[0][1][0].payload
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert "quality_flag" not in metadata


def test_zero_chunk_fallback_gates_a_huge_token_in_bounded_time() -> None:
    """A document whose only element is ``"1"*N+"a"``
    gives the chunker 0 chunks, so the fallback builds one N-char chunk and
    gates it: N=32,000 took 41.8 s before the gate's cost was bounded."""
    import contextlib
    import time

    doc = _doc(elements=[_el("1" * 32_000 + "a", ElementType.NARRATIVE_TEXT, page=1)])
    started = time.process_time()
    with contextlib.suppress(RuntimeError):  # zero chunks is fine: only time is tested
        do_build_chunks(doc)
    # 41.8 s before the bound; 5 s absorbs slow CI runners.
    assert time.process_time() - started < 5.0


# A book content hash, not a secret.
_H = "56c9febc95d9356d344a8c4306eadd0d66ce79c902bbde3302a56e2f2b8e7a40"  # pragma: allowlist secret
_BODY = (
    "The javelin is released at an angle of about thirty-five degrees, and the "
    "thrower must keep the tip aligned with the direction of flight throughout."
)


def test_a_hash_title_is_replaced_by_the_book_title_in_section_paths() -> None:
    """VOR-1454: PDFs stored as verified/sha256/<sha>.pdf parse with the sha as their
    title, which became the root of every section path the tagger sees."""
    doc = _doc(title=_H, elements=[_el(_BODY)])
    chunks = do_build_chunks(doc, book_metadata={"title": "biomechanics of the discus spin"})
    assert chunks
    assert all(_H not in c.section_path for c in chunks)
    assert all("biomechanics of the discus spin" in c.section_path for c in chunks)


def test_a_real_parsed_title_is_kept() -> None:
    doc = _doc(title="Applied Strength Methods", elements=[_el(_BODY)])
    chunks = do_build_chunks(doc, book_metadata={"title": "Some Other Title"})
    assert all("Applied Strength Methods" in c.section_path for c in chunks)


@pytest.mark.parametrize(
    "text",
    [
        "a" * 32_000 + "!",  # one huge word: sentence-stop patterns
        "Abc " * 8_000,  # thousands of words, no stop: prose-sentence detection
        "." * 32_000 + " 12",  # a dot run: contents-line detection
        "(1" * 16_000,  # citation volume(issue) shapes
    ],
)
def test_pathological_text_is_chunked_in_bounded_time(text: str) -> None:
    """VOR-1454: heading and back-matter heuristics run on untrusted text, so none
    may be super-linear (a quadratic citation regex took 13 s on CI)."""
    import contextlib
    import time

    doc = _doc(elements=[_el(text, ElementType.NARRATIVE_TEXT, page=1)])
    started = time.process_time()
    with contextlib.suppress(RuntimeError):
        do_build_chunks(doc)
    # Each shape was 3.5-30 s before the fix; ~0.1 s after.
    assert time.process_time() - started < 5.0
