from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
)
from vo_rag.core.models.ingest_generation import IngestGenerationState
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.storage.qdrant import deterministic_point_id
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id
from vo_rag.pipelines.book_rag.validator import ValidationReport
from vo_rag.pipelines.book_rag.workflow import BookRagContext, PipelineOutput
from vo_rag.pipelines.book_rag.workflow_li import (
    BookRagWorkflow,
    Context,
    IngestStartEvent,
    IngestState,
    ParsedEvent,
    ParseMeta,
    RegisteredEvent,
    ValidatedEvent,
)

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
                text=(
                    "Narrative text forming a chunk for the LlamaIndex workflow "
                    "orchestration contract."
                ),
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
    def __init__(self, expected_chunk_id: str) -> None:
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


def _workflow(**overrides: Any) -> BookRagWorkflow:
    order: list[str] = overrides.pop("order")
    base: dict[str, Any] = {
        "garage": _FakeGarage({"drops/book.txt": _BOOK_BYTES}),
        "qdrant_store": _FakeQdrantStore(),
        "embedding_client": _FakeEmbeddingClient(order),
        "llm_client": _FakeLLMClient(order),
        "retriever": _FakeRetriever("pending"),
        "parser_router": _FakeParserRouter(order),
        "config": BookRagConfig(),
        "timeout": 30,
    }
    base.update(overrides)
    return BookRagWorkflow(**base)


def test_book_rag_workflow_uses_typed_start_event() -> None:
    workflow = _workflow(order=[])
    register_cfg = getattr(workflow._get_steps()["register_source"], "__step_config", None)

    assert register_cfg is not None
    assert IngestStartEvent in register_cfg.accepted_events


def test_book_rag_workflow_event_payloads_are_typed() -> None:
    assert ParsedEvent.model_fields["parse_meta"].annotation is ParseMeta
    assert ValidatedEvent.model_fields["validation_report"].annotation == ValidationReport | None


def test_book_rag_workflow_advanced_layers_raise_pointer_error() -> None:
    workflow = _workflow(
        order=[],
        config=BookRagConfig(enable_graph=True),
    )

    with pytest.raises(RuntimeError, match="AdvancedBuildWorkflow") as exc_info:
        workflow._raise_if_advanced_layers_enabled()

    assert "NotImplementedError" not in type(exc_info.value).__name__


@pytest.mark.asyncio
async def test_book_rag_workflow_runs_typed_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    order: list[str] = []
    source_id = canonical_source_id(_HASH)
    expected_raw_id = deterministic_point_id(source_id, 0, "raw")

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

    retriever = _FakeRetriever(expected_chunk_id="")
    workflow = _workflow(order=order, retriever=retriever, pg_pool=object())
    envelope = _envelope()

    result = await workflow.run(envelope=envelope)

    assert isinstance(result, PipelineOutput)
    assert result.status is EnvelopeStatus.COMPLETED
    assert envelope.status is EnvelopeStatus.COMPLETED
    assert result.source_id == source_id
    assert result.raw_point_ids == [expected_raw_id]
    assert result.sterile_point_ids == []
    assert result.qdrant_point_ids == [expected_raw_id]
    assert result.chunk_count == 1
    assert result.smoke_decision is smoke.GateDecision.PASS
    assert order == [
        "parse",
        "visual",
        "chunk_embed",
        "register_artifact",
        "validate",
        "llm_call",
        "retrieval_smoke",
    ]


def test_book_rag_workflow_does_not_register_advanced_understanding_step() -> None:
    assert not hasattr(BookRagWorkflow, "advanced_understanding")


@pytest.mark.asyncio
async def test_sse_active_generation_conflict_stops_before_write_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Coordinator:
        async def _admit_source_publication(self, _ctx: BookRagContext, _collection: str) -> None:
            return None

        async def _start_generation(self, ctx: BookRagContext) -> None:
            ctx.generation_id = "gen-active"
            ctx.generation_already_active = True

    qdrant = _FakeQdrantStore()
    workflow = _workflow(order=[], qdrant_store=qdrant, pg_pool=object())
    monkeypatch.setattr(workflow, "_generation_coordinator", lambda: _Coordinator())
    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setenv("VORAG_BOOK_RAG_FORCE_REINGEST", "1")
    ctx: Context[IngestState] = Context(workflow)
    await ctx.store.set("workflow_run_id", "run-overlap")

    stop = await workflow.register_source(ctx, IngestStartEvent(envelope=_envelope()))

    result = stop.result
    assert isinstance(result, PipelineOutput)
    assert result.idempotent_noop is True
    assert result.generation_state == IngestGenerationState.ACTIVE
    assert qdrant.upserts == []


@pytest.mark.asyncio
async def test_li_generation_owner_heartbeat_and_cancel_release_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    released: list[tuple[str | None, str | None]] = []

    class _Coordinator:
        async def _admit_source_publication(self, _ctx: BookRagContext, _collection: str) -> None:
            return None

        async def _start_generation(self, ctx: BookRagContext) -> None:
            ctx.generation_id = "gen-1"
            ctx.generation_owner_token = "owner-1"
            ctx.generation_claim_acquired = True

        async def _maintain_generation_lease(
            self,
            _ctx: BookRagContext,
            **_kwargs: Any,
        ) -> None:
            started.set()
            await asyncio.Event().wait()

        async def _release_generation_lease(self, ctx: BookRagContext) -> None:
            released.append((ctx.generation_id, ctx.generation_owner_token))

    coordinator = _Coordinator()
    workflow = _workflow(order=[], pg_pool=object())
    monkeypatch.setattr(workflow, "_generation_coordinator", lambda: coordinator)
    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setenv("VORAG_BOOK_RAG_FORCE_REINGEST", "1")
    ctx: Context[IngestState] = Context(workflow)
    await ctx.store.set("workflow_run_id", "run-owner")

    event = await workflow.register_source(ctx, IngestStartEvent(envelope=_envelope()))
    await started.wait()

    assert isinstance(event, RegisteredEvent)
    assert await ctx.store.get("generation_owner_token") == "owner-1"
    assert await ctx.store.get("generation_claim_acquired") is True
    assert isinstance(vars(ctx)["generation_lease_task"], asyncio.Task)

    await workflow._cleanup_context_generation_lease(ctx)

    assert released == [("gen-1", "owner-1")]
    assert vars(ctx)["generation_lease_task"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("skip_reason", ["no_client", "disabled"])
async def test_generation_smoke_skip_validates_activates_and_completes(
    monkeypatch: pytest.MonkeyPatch,
    skip_reason: str,
) -> None:
    transitions: list[dict[str, Any]] = []
    activated: list[BookRagContext] = []
    persisted_warnings: list[tuple[str, list[dict[str, Any]]]] = []
    expected_warning = {
        "stage": "chunk",
        "code": "test_advisory",
        "severity": "warning",
        "details": "preserve usable passage",
    }

    class _Coordinator:
        async def _assert_generation_lease(self, _ctx: BookRagContext) -> None:
            return None

        async def _transition_owned_generation(
            self,
            ctx: BookRagContext,
            **kwargs: Any,
        ) -> None:
            transitions.append({"generation_id": ctx.generation_id, **kwargs})

        async def _activate_generation(self, ctx: BookRagContext) -> None:
            activated.append(ctx)

        async def _persist_quality_warnings(self, ctx: BookRagContext) -> None:
            persisted_warnings.append((ctx.source_id, list(ctx.quality_warnings)))

    config = BookRagConfig(retrieval_smoke_enabled=skip_reason != "disabled")
    workflow = _workflow(
        order=[],
        llm_client=None if skip_reason == "no_client" else _FakeLLMClient([]),
        config=config,
        pg_pool=object(),
    )

    async def _record_quality_warning(ctx: BookRagContext) -> None:
        ctx.quality_warnings.append(expected_warning)

    monkeypatch.setattr(workflow, "_record_book_quality", _record_quality_warning)
    monkeypatch.setattr(workflow, "_generation_coordinator", lambda: _Coordinator())
    ctx: Context[IngestState] = Context(workflow)
    envelope = _envelope()
    await ctx.store.set("envelope", envelope)
    await ctx.store.set("generation_id", "gen-1")
    await ctx.store.set("previous_generation_id", "gen-old")
    event = ValidatedEvent(
        source_id="source-1",
        parse_meta={},
        chunks=[],
        raw_point_ids=[],
        sterile_point_ids=[],
        validation_report=None,
    )

    stop = await workflow.retrieval_smoke(ctx, event)

    assert transitions[0]["to_state"] is IngestGenerationState.VALIDATED
    assert transitions[0]["smoke_decision"] == "skipped"
    assert len(activated) == 1
    result = stop.result
    assert isinstance(result, PipelineOutput)
    assert result.status is EnvelopeStatus.COMPLETED
    assert result.generation_state == IngestGenerationState.ACTIVE
    assert persisted_warnings == [("source-1", [expected_warning])]
    assert result.quality_warnings == [expected_warning]
