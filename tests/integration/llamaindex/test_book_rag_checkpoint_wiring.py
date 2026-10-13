from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from workflows.checkpointer import Checkpoint as WorkflowCheckpoint

from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
from vo_rag.core.parsing.contract import ElementType, ExtractedImage, ParsedDocument, ParsedElement
from vo_rag.core.storage.figures import FigureStore
from vo_rag.core.storage.qdrant import deterministic_point_id
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag.checkpoint import CHECKPOINT_PREFIX
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id
from vo_rag.pipelines.book_rag.validator import ValidationReport
from vo_rag.pipelines.book_rag.visual_sidecar import VLMDescription
from vo_rag.pipelines.book_rag.workflow import PipelineOutput
from vo_rag.pipelines.book_rag.workflow_li import (
    BookRagWorkflow,
    BookRagWorkflowCheckpointRunner,
    ImageDescriptionResult,
    ParsedEvent,
    VisualEvent,
    _context_run_id,
    run_book_rag_workflow_checkpointed,
)

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_BOOK_BYTES = b"plain text fixture"
_HASH = hashlib.sha256(_BOOK_BYTES).hexdigest()
_VERIFICATION = {"status": "verified", "sha256": _HASH, "size_bytes": len(_BOOK_BYTES)}
SRC = "ab" * 32
IMAGE_REF = FigureStore.figure_ref(SRC, 7, 0, "png")


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


def _parsed_doc(*, source_id: str = "src-1", with_image: bool = False) -> ParsedDocument:
    images = (
        [
            ExtractedImage(
                image_id="img-1",
                page_number=7,
                image_index=0,
                ref=IMAGE_REF,
                width=64,
                height=64,
                file_size=12,
                element_index=1,
            )
        ]
        if with_image
        else []
    )
    return ParsedDocument(
        document_id="doc-1",
        source_id=source_id,
        title="Parsed Book Title",
        author="Parsed Author",
        total_pages=10,
        elements=[
            ParsedElement(
                element_id="e-1",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Section context about a technical circuit.",
                page_number=7,
                section_title="Circuit Design",
            ),
            ParsedElement(
                element_id="e-2",
                element_type=ElementType.IMAGE,
                text="",
                page_number=7,
                section_title="Circuit Design",
            ),
        ],
        images=images,
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="test",
    )


class _RecordingGarage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.list_calls: list[tuple[str, str]] = []
        self.put_calls: list[tuple[str, str, bytes, str | None]] = []
        self.delete_calls: list[tuple[str, str]] = []

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
        self.put_calls.append((bucket, key, data, content_type))

    async def list_objects(self, bucket: str, prefix: str) -> list[dict[str, str]]:
        self.list_calls.append((bucket, prefix))
        return [{"key": key} for key in sorted(self.objects) if key.startswith(prefix)]

    async def delete_object(self, bucket: str, key: str) -> None:
        self.delete_calls.append((bucket, key))
        self.objects.pop(key, None)


class _FakeContextStore:
    def __init__(self, envelope: IntakeEnvelope) -> None:
        self._envelope = envelope

    async def get(self, key: str, default: Any = ...) -> Any:
        if key == "envelope":
            return self._envelope
        if default is ...:
            raise KeyError(key)
        return default


class _FakeVLMService:
    model = "fake-vlm"

    def __init__(self) -> None:
        self.closed = False

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        image_id: str,
        *,
        max_tokens: int,
    ) -> VLMDescription:
        assert image_bytes == b"fake-image-bytes"
        assert image_id == "img-1"
        assert "Circuit Design" in prompt
        return VLMDescription(
            description="A technical circuit diagram with labeled signal paths.",
            model_used=self.model,
            token_count=12,
            duration_ms=3,
            image_id=image_id,
        )

    async def aclose(self) -> None:
        self.closed = True


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
    async def upsert_vectors(self, collection: str, points: list[Any]) -> None:
        assert collection == "vo_rag_books_raw_v1"
        assert points

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
        return LLMResponse(text='{"ok": true}', model="fake-llm")


class _FakeRetriever:
    def __init__(self) -> None:
        self.expected_chunk_id = ""

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


def _workflow(*, order: list[str], garage: Any | None = None, **overrides: Any) -> BookRagWorkflow:
    base: dict[str, Any] = {
        "garage": garage or _RecordingGarage({"drops/book.txt": _BOOK_BYTES}),
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


def _patch_successful_workflow_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    order: list[str],
    retriever: _FakeRetriever,
) -> None:
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

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
            retriever_arg: Any,
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
    monkeypatch.setattr(smoke, "generate_self_queries", _fake_generate_self_queries)
    monkeypatch.setattr(smoke, "score_self_retrieval", _fake_score_self_retrieval)


@pytest.mark.asyncio
async def test_visual_step_uses_workflow_run_id_for_per_image_checkpoint(tmp_path: Path) -> None:
    workflow_run_id = "workflow-run-202"
    envelope = _envelope()
    figures = FigureStore(tmp_path)
    assert figures.write(SRC, 7, 0, b"fake-image-bytes", ext="png") == IMAGE_REF
    parsed = _parsed_doc(with_image=True)
    parsed_key = "parsed/doc-1/parsed.json"
    garage = _RecordingGarage(
        {
            parsed_key: parsed.model_dump_json().encode("utf-8"),
        }
    )
    vlm_service = _FakeVLMService()
    workflow = _workflow(
        order=[],
        garage=garage,
        config=BookRagConfig(enable_visual_description=True),
        vlm_service_factory=lambda: vlm_service,
        bucket="artifacts",
        figures=figures,
    )
    store_data: dict[str, Any] = {"envelope": envelope}
    sent_events: list[Any] = []

    async def _store_set(key: str, value: Any) -> None:
        store_data[key] = value

    async def _store_get(key: str, default: Any = ...) -> Any:
        if key in store_data:
            return store_data[key]
        if default is ...:
            raise KeyError(key)
        return default

    def _send_event(ev: Any) -> None:
        sent_events.append(ev)

    def _collect_events(
        ev: Any,
        types: list[Any],
        buffer_id: str | None = None,
    ) -> list[Any] | None:
        image_described = [e for e in sent_events if isinstance(e, ImageDescriptionResult)]
        if len(image_described) >= len(types):
            return image_described[: len(types)]
        return None

    ctx = cast(
        Any,
        SimpleNamespace(
            run_id=workflow_run_id,
            store=SimpleNamespace(get=_store_get, set=_store_set),
            # VOR-204 added per-step write_event_to_stream(...) emission; this
            # checkpoint-wiring test only exercises the run_id/store contract,
            # so the event sink is a no-op here.
            write_event_to_stream=lambda *a, **k: None,
            send_event=_send_event,
            collect_events=_collect_events,
        ),
    )

    parsed_ev = ParsedEvent(
        source_id=parsed.source_id,
        parse_meta={"parsed_garage_key": parsed_key},
        parsed=parsed,
    )

    # fan_out_images stores state and sends DescribeImageEvent(s)
    await workflow.fan_out_images(ctx, parsed_ev)
    assert len(sent_events) == 1
    desc_ev = sent_events[0]

    # describe_one processes each image
    img_ev = await workflow.describe_one(ctx, desc_ev)

    # collect_visual joins results and finalizes the checkpoint
    sent_events.append(img_ev)
    result = await workflow.collect_visual(ctx, img_ev)

    checkpoint_prefix = f"{CHECKPOINT_PREFIX}/{workflow_run_id}/images"
    checkpoint_key = f"{checkpoint_prefix}/img-1.json"
    assert isinstance(result, VisualEvent)
    assert result.source_id == parsed.source_id
    # fan_out_images calls checkpoint.load(), collect_visual calls
    # checkpoint.load() and checkpoint.finalize()
    assert garage.list_calls == [
        ("artifacts", checkpoint_prefix),
        ("artifacts", checkpoint_prefix),
        ("artifacts", checkpoint_prefix),
    ]
    assert any(key == checkpoint_key for _, key, _, _ in garage.put_calls)
    assert garage.delete_calls == [("artifacts", checkpoint_key)]
    assert checkpoint_key not in garage.objects
    assert vlm_service.closed is True


@pytest.mark.asyncio
async def test_checkpointed_runner_wraps_workflow_and_keys_checkpoints_by_run_id(
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
            retriever_arg: Any,
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
    monkeypatch.setattr(smoke, "generate_self_queries", _fake_generate_self_queries)
    monkeypatch.setattr(smoke, "score_self_retrieval", _fake_score_self_retrieval)

    workflow_run_id = "workflow-run-smoke"
    envelope = _envelope()
    workflow = _workflow(order=order, retriever=retriever)

    result, runner = await run_book_rag_workflow_checkpointed(
        workflow=workflow,
        envelope=envelope,
        workflow_run_id=workflow_run_id,
    )

    assert isinstance(result, PipelineOutput)
    assert result.status is EnvelopeStatus.COMPLETED
    assert result.raw_point_ids == [expected_raw_id]
    assert workflow_run_id in runner.checkpointer.checkpoints
    assert runner.latest_checkpoint(workflow_run_id) is not None
    assert all(
        checkpoint.last_completed_step is not None
        for checkpoint in runner.checkpointer.checkpoints[workflow_run_id]
    )


@pytest.mark.asyncio
async def test_checkpointed_runner_real_fail_once_resume_keeps_workflow_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod
    from vo_rag.pipelines.book_rag.steps.parse import (
        do_parse_document as original_do_parse_document,
    )

    order: list[str] = []
    workflow_run_id = "workflow-run-fail-once"
    envelope = _envelope()
    retriever = _FakeRetriever()
    workflow = _workflow(order=order, retriever=retriever)
    _patch_successful_workflow_dependencies(monkeypatch, order=order, retriever=retriever)
    parse_attempts = 0

    async def _fail_once_parse_document(**kwargs: Any) -> dict[str, Any]:
        nonlocal parse_attempts
        parse_attempts += 1
        if parse_attempts == 1:
            raise RuntimeError("transient parser outage")
        return await original_do_parse_document(**kwargs)

    monkeypatch.setattr(workflow_mod, "do_parse_document", _fail_once_parse_document)

    result, runner = await run_book_rag_workflow_checkpointed(
        workflow=workflow,
        envelope=envelope,
        workflow_run_id=workflow_run_id,
    )

    assert result.status is EnvelopeStatus.COMPLETED
    assert parse_attempts == 2
    assert set(runner.checkpointer.checkpoints) == {workflow_run_id}
    assert runner.latest_checkpoint(workflow_run_id) is not None


@pytest.mark.asyncio
async def test_context_run_id_uses_restored_state_without_fabricating_id() -> None:
    class _Store:
        async def get(self, key: str, default: Any = ...) -> Any:
            if key == "workflow_run_id":
                return "restored-run-id"
            if default is ...:
                raise KeyError(key)
            return default

    ctx = cast(Any, SimpleNamespace(store=_Store()))

    assert await _context_run_id(ctx) == "restored-run-id"


@pytest.mark.asyncio
async def test_context_run_id_raises_when_run_identity_is_missing() -> None:
    class _Store:
        async def get(self, key: str, default: Any = ...) -> Any:
            if default is ...:
                raise KeyError(key)
            return default

    ctx = cast(Any, SimpleNamespace(store=_Store()))

    with pytest.raises(RuntimeError, match="workflow_run_id"):
        await _context_run_id(ctx)


@pytest.mark.asyncio
async def test_context_run_id_store_failure_does_not_fallback_to_fresh_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    class _Store:
        async def get(self, key: str, default: Any = ...) -> Any:
            raise RuntimeError(f"store read failed for {key}")

    logs: list[tuple[str, dict[str, Any]]] = []

    class _FakeLog:
        def warning(self, event: str, **kwargs: Any) -> None:
            logs.append((event, kwargs))

    ctx = cast(Any, SimpleNamespace(store=_Store(), run_id="fresh-framework-run-id"))
    monkeypatch.setattr(workflow_mod, "log", _FakeLog())

    with pytest.raises(RuntimeError, match="store read failed"):
        await _context_run_id(ctx)

    assert not hasattr(ctx, "vo_rag_workflow_run_id")
    assert logs == [
        (
            "book_rag_workflow_run_id_store_read_failed",
            {"error": "store read failed for workflow_run_id", "exc_info": True},
        )
    ]


@pytest.mark.asyncio
async def test_checkpoint_runner_logs_first_failure_before_auto_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    envelope = _envelope()
    workflow = _workflow(order=[])
    runner = BookRagWorkflowCheckpointRunner.create(workflow)
    expected = PipelineOutput(
        source_id="src-1",
        status=EnvelopeStatus.COMPLETED,
        qdrant_point_ids=[],
        raw_point_ids=[],
        sterile_point_ids=[],
        smoke_decision=None,
        chunk_count=0,
    )
    warnings: list[dict[str, Any]] = []

    async def _failed_handler() -> PipelineOutput:
        raise RuntimeError("transient parse failure")

    def _run_handler(*, envelope: IntakeEnvelope, workflow_run_id: str) -> Any:
        return _failed_handler()

    def _latest_checkpoint(workflow_run_id: str) -> WorkflowCheckpoint:
        return cast(WorkflowCheckpoint, object())

    async def _resume_from_checkpoint(
        *,
        checkpoint: WorkflowCheckpoint,
        envelope: IntakeEnvelope,
        workflow_run_id: str,
    ) -> PipelineOutput:
        return expected

    def _warning(event: str, **kwargs: Any) -> None:
        warnings.append({"event": event, **kwargs})

    monkeypatch.setattr(runner, "run_handler", _run_handler)
    monkeypatch.setattr(runner, "latest_checkpoint", _latest_checkpoint)
    monkeypatch.setattr(runner, "resume_from_checkpoint", _resume_from_checkpoint)
    monkeypatch.setattr(workflow_mod.log, "warning", _warning)

    result = await runner.run(envelope=envelope, workflow_run_id="run-resume")

    assert result is expected
    assert warnings == [
        {
            "event": "book_rag_workflow_resume_after_error",
            "workflow_run_id": "run-resume",
            "error": "transient parse failure",
            "exc_info": True,
        }
    ]


@pytest.mark.asyncio
async def test_checkpoint_runner_reraises_without_resume_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _envelope()
    workflow = _workflow(order=[])
    runner = BookRagWorkflowCheckpointRunner.create(workflow)
    resumed = False

    async def _failed_handler() -> PipelineOutput:
        raise RuntimeError("persistent parse failure")

    def _run_handler(*, envelope: IntakeEnvelope, workflow_run_id: str) -> Any:
        return _failed_handler()

    async def _resume_from_checkpoint(**kwargs: Any) -> PipelineOutput:
        nonlocal resumed
        resumed = True
        raise AssertionError("resume_on_error=False must not resume")

    monkeypatch.setattr(runner, "run_handler", _run_handler)
    monkeypatch.setattr(runner, "resume_from_checkpoint", _resume_from_checkpoint)

    with pytest.raises(RuntimeError, match="persistent parse failure"):
        await runner.run(
            envelope=envelope,
            workflow_run_id="run-no-resume",
            resume_on_error=False,
        )

    assert resumed is False


@pytest.mark.asyncio
async def test_checkpoint_runner_reraises_when_resume_has_no_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _envelope()
    workflow = _workflow(order=[])
    runner = BookRagWorkflowCheckpointRunner.create(workflow)
    resumed = False

    async def _failed_handler() -> PipelineOutput:
        raise RuntimeError("failure before first checkpoint")

    def _run_handler(*, envelope: IntakeEnvelope, workflow_run_id: str) -> Any:
        return _failed_handler()

    def _latest_checkpoint(workflow_run_id: str) -> None:
        return None

    async def _resume_from_checkpoint(**kwargs: Any) -> PipelineOutput:
        nonlocal resumed
        resumed = True
        raise AssertionError("runner must not resume without a checkpoint")

    monkeypatch.setattr(runner, "run_handler", _run_handler)
    monkeypatch.setattr(runner, "latest_checkpoint", _latest_checkpoint)
    monkeypatch.setattr(runner, "resume_from_checkpoint", _resume_from_checkpoint)

    with pytest.raises(RuntimeError, match="failure before first checkpoint"):
        await runner.run(envelope=envelope, workflow_run_id="run-no-checkpoint")

    assert resumed is False
