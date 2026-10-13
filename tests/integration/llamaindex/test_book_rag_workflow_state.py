from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.storage.qdrant import deterministic_point_id
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id
from vo_rag.pipelines.book_rag.validator import ValidationReport
from vo_rag.pipelines.book_rag.workflow import PipelineOutput
from vo_rag.pipelines.book_rag.workflow_li import BookRagWorkflow, Context, IngestState

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_BOOK_BYTES = b"plain text fixture"
_HASH = hashlib.sha256(_BOOK_BYTES).hexdigest()
_VERIFICATION = {"status": "verified", "sha256": _HASH, "size_bytes": len(_BOOK_BYTES)}


def _envelope(*, status: EnvelopeStatus = EnvelopeStatus.RECEIVED) -> IntakeEnvelope:
    return IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={
            "garage_key": "drops/book.txt",
            "file_hash": _HASH,
            "object_verification": _VERIFICATION,
        },
        status=status,
        created_at=_TS,
        updated_at=_TS,
    )


def _parsed_doc(*, source_id: str = "src-1") -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id=source_id,
        title="Parsed Book Title",
        author="Parsed Author",
        total_pages=1,
        elements=[
            ParsedElement(
                element_id="e-1",
                element_type=ElementType.TITLE,
                text="Chapter 1",
                page_number=1,
            ),
            ParsedElement(
                element_id="e-2",
                element_type=ElementType.NARRATIVE_TEXT,
                text=("Narrative text forming a chunk for the LlamaIndex workflow state contract."),
                page_number=1,
            ),
        ],
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="test",
    )


class _FakeGarage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        return self.objects[key]

    async def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        self.objects[key] = data


class _FakeParserRouter:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def parse(self, path: str, file_hash: str) -> ParsedDocument:
        self.order.append("parse")
        assert len(file_hash) == 64
        return _parsed_doc()


class _FakeEmbeddingClient:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        self.order.append("chunk_embed")
        dense = [0.1] * 1024
        return {
            "dense": [dense for _ in texts],
            "sparse": [{1: 0.5} for _ in texts],
            "colbert": [[] for _ in texts],
        }


class _FakeQdrantStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[Any]]] = []

    async def upsert_vectors(self, collection: str, points: list[Any]) -> None:
        self.upserts.append((collection, points))

    async def count_source(self, _collection: str, _source_id: str) -> int:
        return 0


class _FakeLLMClient:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any],
    ) -> LLMResponse:
        self.order.append("llm_call")
        assert extra == {"chat_template_kwargs": {"enable_thinking": False}}
        return LLMResponse(text='{"ok": true}', model="fake-llm")


class _FakeRetriever:
    def __init__(self, expected_chunk_id: str = "") -> None:
        self.expected_chunk_id = expected_chunk_id

    async def hybrid_search(
        self,
        query: str,
        *,
        collection: str,
        top_k: int,
        prefetch_depth: int,
    ) -> list[SimpleNamespace]:
        assert collection == "vo_rag_books_raw_v1"
        return [SimpleNamespace(id="point-1", payload={"chunk_id": self.expected_chunk_id})]


def _workflow(*, order: list[str], **overrides: Any) -> BookRagWorkflow:
    base: dict[str, Any] = {
        "garage": _FakeGarage({"drops/book.txt": _BOOK_BYTES}),
        "qdrant_store": _FakeQdrantStore(),
        "embedding_client": _FakeEmbeddingClient(order),
        "llm_client": _FakeLLMClient(order),
        "retriever": _FakeRetriever(),
        "parser_router": _FakeParserRouter(order),
        "config": BookRagConfig(),
        "timeout": 30,
    }
    base.update(overrides)
    return BookRagWorkflow(**base)


@pytest.mark.asyncio
async def test_completed_envelope_short_circuits_before_do_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    def _fail_helper(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("COMPLETED envelope must not invoke do_* helpers")

    async def _fail_async_helper(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("COMPLETED envelope must not invoke do_* helpers")

    monkeypatch.setattr(workflow_mod, "do_parse_document", _fail_async_helper)
    monkeypatch.setattr(workflow_mod, "do_build_chunks", _fail_helper)
    monkeypatch.setattr(workflow_mod, "do_embed_raw", _fail_async_helper)
    monkeypatch.setattr(workflow_mod, "do_upsert_raw", _fail_async_helper)
    monkeypatch.setattr(workflow_mod, "do_sterilize", _fail_async_helper)

    workflow = _workflow(order=[])
    envelope = _envelope(status=EnvelopeStatus.COMPLETED)

    result = await workflow.run(envelope=envelope)

    assert isinstance(result, PipelineOutput)
    assert result.source_id == canonical_source_id(_HASH)
    assert result.status is EnvelopeStatus.COMPLETED
    assert result.qdrant_point_ids == []
    assert result.raw_point_ids == []
    assert result.sterile_point_ids == []
    assert result.smoke_decision is None
    assert result.chunk_count == 0
    assert result.idempotent_noop is True


@pytest.mark.asyncio
async def test_fresh_envelope_populates_ingest_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    order: list[str] = []
    source_id = canonical_source_id(_HASH)
    expected_raw_id = deterministic_point_id(source_id, 0, "raw")
    retriever = _FakeRetriever()

    async def _fake_visual(
        parsed: ParsedDocument,
        *,
        document_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        order.append("visual")
        return []

    class _FakeValidator:
        def __init__(
            self,
            qdrant: Any,
            pool: Any,
            garage: Any,
            retriever: Any,
            *,
            figures: Any = None,
            artifacts_bucket: str,
            sterilize_enabled: bool,
        ) -> None:
            self.sterilize_enabled = sterilize_enabled

        async def validate(self, *, document_id: str, source_id: str) -> Any:
            order.append("validate")
            return ValidationReport(
                document_id=document_id,
                source_id=source_id,
                passed=True,
            )

    async def _fake_insert_artifact(pool: Any, artifact: Any) -> None:
        order.append("register_artifact")
        assert artifact.qdrant_point_ids == [expected_raw_id]

    async def _fake_generate_self_queries(
        *,
        all_chunks: list[Any],
        n: int,
        llm_call: Any,
        seed: int,
    ) -> list[Any]:
        await llm_call("system", "user", 32, 0.0)
        return [
            SimpleNamespace(
                query_id="q-1",
                query_text="chapter query",
                expected_supporting_chunk_ids=[all_chunks[0].chunk_id],
            )
        ]

    async def _fake_score_self_retrieval(
        *,
        queries: list[Any],
        retrieve_fn: Any,
        top_k: int,
        source_id: str,
    ) -> smoke.SmokeReport:
        order.append("retrieval_smoke")
        retriever.expected_chunk_id = queries[0].expected_supporting_chunk_ids[0]
        retrieved = await retrieve_fn(queries[0].query_text, top_k)
        assert retrieved == [queries[0].expected_supporting_chunk_ids[0]]
        return smoke.SmokeReport(
            source_id=source_id,
            n_requested=1,
            n_scored=1,
            top_k=top_k,
            score=1.0,
        )

    monkeypatch.setattr(workflow_mod, "build_extraction_composites", _fake_visual)
    monkeypatch.setattr(workflow_mod, "IngestionValidator", _FakeValidator)
    monkeypatch.setattr(workflow_mod, "insert_artifact", _fake_insert_artifact)
    monkeypatch.setattr(smoke, "generate_self_queries", _fake_generate_self_queries)
    monkeypatch.setattr(smoke, "score_self_retrieval", _fake_score_self_retrieval)

    workflow = _workflow(order=order, retriever=retriever, pg_pool=object())
    ctx: Context[IngestState] = Context(workflow)
    envelope = _envelope()

    result = await workflow.run(ctx=ctx, envelope=envelope)
    state = await ctx.store.get_state()

    assert isinstance(result, PipelineOutput)
    assert isinstance(state, IngestState)
    assert result.status is EnvelopeStatus.COMPLETED
    assert state.source_id == source_id
    assert state.raw_point_ids == [expected_raw_id]
    assert state.sterile_point_ids == []
    assert state.chunk_count > 0
    assert state.smoke_decision is smoke.GateDecision.PASS
    assert order == [
        "parse",
        "visual",
        "chunk_embed",
        "register_artifact",
        "validate",
        "llm_call",
        "retrieval_smoke",
    ]
