"""End-to-end spine integration test.

Threads the ingest / chunking / retrieval / reranking / answer-context
seams together with in-process fakes:

  1. ``IntakeSubmission(FILE_DROP)`` → ``IntakeEnvelope`` via the
     ``vo_rag.core.ingest.normalize`` pure-function helper.
  2. A tiny ``ParsedDocument`` (from the parsing contract) fed into
     ``HierarchicalChunker(counter=naive_counter)`` to produce
     ``RetrievalChunk`` objects.
  3. BM25 sparse vectors built from ``chunk.embedding_text`` via the
     shared ``tokenize`` + ``tokens_to_bm25_sparse`` (the blake2b-4
     persistence contract — query path and write path must agree).
  4. A fake ``QdrantStore`` that wraps the spine chunks as
     ``ScoredPoint`` rows, plus a fake embedding client; both back
     ``HybridRetriever.cross_collection_search()`` so the canonical
     RRF + visual-mode dispatcher is exercised without a live Qdrant.
  5. A fake reranker that preserves top-k order (the GTE ONNX reranker
     is exercised separately in ``tests/core/reranking/test_gte.py``).
  6. ``format_answer_context()`` over the final results, asserting the
     ``[S1]`` / ``[S2]`` source markers the synthesizer relies on.

Invariants pinned here:

  - ``RetrievalChunk.chunk_id`` is deterministic — re-chunking the same
    document yields the same ids (content-addressed hash key from
    ``vo_rag.core.chunking.ids._chunk_id``).
  - The BM25 sparse vector for a non-trivial chunk is non-empty
    (``tokens_to_bm25_sparse`` returns ``{int: float}`` keyed by the
    blake2b-4 contract).
  - The retriever NEVER reaches past the store into
    ``qdrant._client.query_points`` — the regression
    guard is enforced here at the spine level as well.
  - ``format_answer_context`` emits ``[S1]`` / ``[S2]`` source labels
    matching the chunk index, plus ``source_id`` / ``chunk_id`` /
    ``page`` lines the protected synthesizer prompt depends on.
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client.models import ScoredPoint

from vo_rag.core.answer import format_answer_context
from vo_rag.core.chunking import HierarchicalChunker, RetrievalChunk, naive_counter
from vo_rag.core.ingest import IntakeSubmission, normalize
from vo_rag.core.models.envelope import EnvelopeStatus, IntakeSource
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement
from vo_rag.core.retrieval import (
    HybridRetriever,
    QueryMode,
    SearchResult,
    tokenize,
    tokens_to_bm25_sparse,
)

# ---------------------------------------------------------------------------
# Tiny fixture helpers
# ---------------------------------------------------------------------------


def _tiny_parsed_document() -> ParsedDocument:
    """Three-page tiny ParsedDocument suitable for the spine chunker.

    Two depth-1 TITLE elements force a flush between sections so the
    chunker emits at least one RetrievalChunk per section. The narrative
    text is long enough for the chunker's low-signal gate to keep it.
    """
    elements: list[ParsedElement] = [
        ParsedElement(
            element_id="t1",
            element_type=ElementType.TITLE,
            text="Trail Braking Fundamentals",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Trail braking adjusts brake pressure gradually as the driver "
                "rotates the chassis into the apex, blending corner entry and "
                "rotation in a single motion."
            ),
            page_number=1,
        ),
        ParsedElement(
            element_id="t2",
            element_type=ElementType.TITLE,
            text="Brake Bias Migration",
            page_number=2,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Brake bias migration shifts the balance forward under heavy "
                "deceleration, then rearward as the steering input grows; the "
                "driver feels this through pedal feedback and chassis pitch."
            ),
            page_number=2,
        ),
    ]
    return ParsedDocument(
        document_id="doc-spine",
        source_id="src-spine",
        title="Spine Integration Reference",
        total_pages=2,
        elements=elements,
        images=[],
        page_qa=[PageQA(page_number=p, ocr_confidence=0.95, parser_used="docling") for p in (1, 2)],
        markdown="",
        heading_tree=[],
        parser_used="docling",
    )


def _scored_point_from_chunk(chunk: RetrievalChunk, collection: str) -> ScoredPoint:
    """Wrap a RetrievalChunk as a Qdrant ScoredPoint for the fake store."""
    return ScoredPoint(
        id=chunk.chunk_id,
        score=0.9,
        payload={
            "source_id": chunk.source_id,
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "section_title": chunk.section_title,
            "source_title": "Spine Integration Reference",
            "content": chunk.content,
            "license_class": "cleared",
            "collection": collection,
        },
        vector=None,
        version=0,
        shard_key=None,
    )


# ---------------------------------------------------------------------------
# Fakes — modeled after tests/core/retrieval/test_search.py, scoped to the
# spine test so the integration story is self-contained.
# ---------------------------------------------------------------------------


class _SpineFakeEmbedClient:
    """Returns a fixed dense+sparse+colbert response; records every call.

    The retriever invokes ``embed`` once per ``hybrid_search`` call; the
    sparse vector we return here is independent of the BM25 sparse the
    retriever builds locally from ``tokens_to_bm25_sparse``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        self.calls.append({"texts": list(texts)})
        return {
            "dense": [[0.11, 0.22, 0.33]],
            "sparse": [{"indices": [1, 2], "values": [0.4, 0.5]}],
            "colbert": [[[0.6, 0.7]]],
        }


class _SpineFakeQdrantClient:
    """Bare Qdrant client — HybridRetriever must NEVER call this."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def query_points(self, **kwargs: Any) -> Any:  # pragma: no cover - guarded
        self.calls.append(kwargs)
        raise AssertionError(
            "HybridRetriever must not reach self.qdrant._client.query_points "
            "directly; use QdrantStore.search_hybrid instead."
        )


class _SpineFakeQdrantStore:
    """Records calls to the canonical ``search_hybrid`` seam and replays
    a pre-loaded list of ``ScoredPoint`` rows. ``_client`` exposes the
    bare client so the test can assert the retriever NEVER touches it.
    """

    def __init__(self, points_by_collection: dict[str, list[ScoredPoint]]) -> None:
        self._client = _SpineFakeQdrantClient()
        self._points_by_collection = points_by_collection
        self.search_hybrid_calls: list[dict[str, Any]] = []

    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Any | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        self.search_hybrid_calls.append(
            {
                "collection": collection,
                "query_bm25": dict(query_bm25),
                "top_k": top_k,
                "filters": filters,
            }
        )
        return list(self._points_by_collection.get(collection, []))[:top_k]


def _identity_topk_reranker(top_k: int) -> Any:
    """Return a reranker that preserves input order, truncated to top_k."""

    def _rerank(_query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return list(candidates[:top_k])

    return _rerank


# ---------------------------------------------------------------------------
# Spine: ingest → chunk → BM25 → retrieve → rerank → answer context
# ---------------------------------------------------------------------------


def test_intake_submission_file_drop_normalizes_to_envelope() -> None:
    """Spine step 1: FILE_DROP IntakeSubmission → IntakeEnvelope with
    a fresh ULID, RECEIVED status, and the normalized raw_content."""
    submission = IntakeSubmission(
        source_type=IntakeSource.FILE_DROP,
        raw_content={
            "garage_key": "intake/spine-doc.pdf",
            "filename": "spine-doc.pdf",
            "content_type": "application/pdf",
        },
    )

    envelope = normalize(submission)

    assert envelope.source_type is IntakeSource.FILE_DROP
    assert envelope.status is EnvelopeStatus.RECEIVED
    assert envelope.raw_content["garage_key"] == "intake/spine-doc.pdf"
    assert envelope.raw_content["filename"] == "spine-doc.pdf"
    assert len(envelope.id) == 26  # ULID
    assert envelope.created_at == envelope.updated_at


def test_hierarchical_chunker_emits_deterministic_retrieval_chunks() -> None:
    """Spine step 2: HierarchicalChunker(counter=naive_counter) over the
    tiny ParsedDocument emits at least one RetrievalChunk per section,
    and re-running the chunker on the same input yields identical
    chunk ids (content-addressed via ``_chunk_id``)."""
    doc = _tiny_parsed_document()

    chunker_a = HierarchicalChunker(target=20, max_=200, min_=1, counter=naive_counter)
    chunker_b = HierarchicalChunker(target=20, max_=200, min_=1, counter=naive_counter)

    chunks_a = chunker_a.chunk(doc, composites=[])
    chunks_b = chunker_b.chunk(doc, composites=[])

    assert chunks_a, "chunker must emit at least one RetrievalChunk"
    assert all(isinstance(c, RetrievalChunk) for c in chunks_a)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert all(c.source_id == "src-spine" for c in chunks_a)
    # The two depth-1 titles surface as separate section_paths.
    section_paths = {c.section_path for c in chunks_a}
    assert any("Trail Braking" in path for path in section_paths)
    assert any("Brake Bias" in path for path in section_paths)


def test_bm25_sparse_from_chunk_embedding_text_is_non_empty() -> None:
    """Spine step 3: BM25 sparse payload built from ``chunk.embedding_text``
    via the shared tokenize/tokens_to_bm25_sparse helpers (the blake2b-4
    persistence contract). The retriever's query path uses the same
    helpers — drift breaks every Qdrant point."""
    chunker = HierarchicalChunker(target=20, max_=200, min_=1, counter=naive_counter)
    chunks = chunker.chunk(_tiny_parsed_document(), composites=[])
    assert chunks, "test fixture must produce at least one chunk"

    for chunk in chunks:
        sparse = tokens_to_bm25_sparse(tokenize(chunk.embedding_text))
        assert sparse, f"BM25 sparse must be non-empty for chunk {chunk.chunk_id}"
        assert all(isinstance(k, int) for k in sparse)
        assert all(isinstance(v, float) for v in sparse.values())


@pytest.mark.asyncio
async def test_spine_end_to_end_ingest_through_answer_context() -> None:
    """Full spine traversal: ingest → chunk → fake store/embed →
    HybridRetriever.cross_collection_search → identity reranker →
    format_answer_context.

    Asserts:
      - the chunker produced deterministic ids,
      - the BM25 sparse vector the retriever sent the store is non-empty
        (shared tokenizer with the write path),
      - the bare Qdrant client was NEVER reached,
      - ``format_answer_context`` emits ``[S1]`` / ``[S2]`` source labels.
    """
    # Step 1 — intake → envelope (carries through as test prerequisite).
    submission = IntakeSubmission(
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "intake/spine-doc.pdf", "filename": "spine-doc.pdf"},
    )
    envelope = normalize(submission)
    assert envelope.status is EnvelopeStatus.RECEIVED

    # Step 2 — parsed doc → retrieval chunks via the spine chunker.
    doc = _tiny_parsed_document()
    chunker = HierarchicalChunker(target=20, max_=200, min_=1, counter=naive_counter)
    chunks = chunker.chunk(doc, composites=[])
    assert len(chunks) >= 2, "spine fixture should yield at least two chunks"
    # Deterministic ids — re-run the chunker and confirm identity.
    chunks_again = HierarchicalChunker(target=20, max_=200, min_=1, counter=naive_counter).chunk(
        doc, composites=[]
    )
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunks_again]

    # Step 3 — BM25 sparse from embedding_text is non-empty.
    for chunk in chunks:
        assert tokens_to_bm25_sparse(tokenize(chunk.embedding_text))

    # Step 4 — fake store/embed back the public retrieve path.
    collection = "vo_rag_books_raw_v1"
    points = [_scored_point_from_chunk(chunk, collection) for chunk in chunks]
    store = _SpineFakeQdrantStore({collection: points})
    embed = _SpineFakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)

    query = "trail braking and brake bias migration"
    results = await retriever.cross_collection_search(
        query=query,
        collections=[collection],
        top_k=5,
        mode=QueryMode.INTERNAL,
        visual_mode="baseline",
    )

    assert results, "spine retrieval must return at least one SearchResult"
    assert all(isinstance(r, SearchResult) for r in results)
    # The retriever wired the canonical BM25 tokenizer through to the store.
    expected_query_bm25 = tokens_to_bm25_sparse(tokenize(query))
    assert any(call["query_bm25"] == expected_query_bm25 for call in store.search_hybrid_calls)
    # The bare Qdrant client must NEVER have been called.
    assert store._client.calls == []

    # Step 5 — fake reranker preserving top-k.
    rerank = _identity_topk_reranker(top_k=3)
    reranked = rerank(query, results)
    assert len(reranked) <= 3
    assert all(isinstance(r, SearchResult) for r in reranked)
    # The reranked list is a permutation/prefix of the retrieved list.
    assert {r.id for r in reranked}.issubset({r.id for r in results})

    # Step 6 — format_answer_context emits the source markers the
    # synthesizer prompt depends on.
    context_block = format_answer_context(reranked, max_chunks=5)
    assert "[S1]" in context_block
    if len(reranked) >= 2:
        assert "[S2]" in context_block
    # Source attribution lines are present for each rendered chunk.
    assert "source_id: src-spine" in context_block
    assert "chunk_id:" in context_block
    # Content from the spine fixture made it into the answer context.
    assert "Trail braking" in context_block or "Brake bias" in context_block
