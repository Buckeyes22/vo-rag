from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("llama_index")

from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.core.storage.figures import FigureStore
from vo_rag.pipelines.book_rag import workflow_li as workflow_mod
from vo_rag.pipelines.book_rag.checkpoint import Checkpoint
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.visual_sidecar import (
    VLMDescription,
    VlmImageJob,
    VLMServiceLike,
    build_vlm_service,
)
from vo_rag.pipelines.book_rag.workflow_li import (
    BookRagWorkflow,
    DescribeImageEvent,
    ImageDescriptionResult,
)

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_HASH = "a" * 64
_VERIFICATION = {"status": "verified", "sha256": _HASH, "size_bytes": 42}
_ENVELOPE_ID = "01KT6XE8M1MW42M0Y0JJKRZSVP"
SRC = "ab" * 32
IMG_1_REF = FigureStore.figure_ref(SRC, 1, 0, "png")
IMG_2_REF = FigureStore.figure_ref(SRC, 1, 1, "png")


def _envelope(*, status: EnvelopeStatus = EnvelopeStatus.PROCESSING) -> IntakeEnvelope:
    return IntakeEnvelope(
        id=_ENVELOPE_ID,
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


def _parsed_doc() -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id="src-1",
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
                element_id="e-img-1",
                element_type=ElementType.IMAGE,
                text="",
                page_number=1,
                metadata={"image_id": "img-1", "image_ref": IMG_1_REF},
            ),
            ParsedElement(
                element_id="e-img-2",
                element_type=ElementType.IMAGE,
                text="",
                page_number=1,
                metadata={"image_id": "img-2", "image_ref": IMG_2_REF},
            ),
        ],
        images=[
            ExtractedImage(
                image_id="img-1",
                page_number=1,
                image_index=0,
                ref=IMG_1_REF,
                element_index=1,
            ),
            ExtractedImage(
                image_id="img-2",
                page_number=1,
                image_index=1,
                ref=IMG_2_REF,
                element_index=2,
            ),
        ],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="test",
    )


class _FakeStore:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    async def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        self._values[key] = value


class _FakeCtx:
    def __init__(
        self,
        *,
        envelope: IntakeEnvelope,
        parsed: ParsedDocument,
        done: list[Any],
    ) -> None:
        self.store = _FakeStore(
            {
                "envelope": envelope,
                "n_pending": len(done),
                "visual_eligible_ids": ["img-1", "img-2"],
                "visual_parsed": parsed,
                "visual_envelope_id": str(envelope.id),
                "visual_source_id": parsed.source_id,
                "visual_parse_meta": {"parsed_garage_key": "parsed/doc-1/parsed.json"},
                "workflow_run_id": "run-visual-test",
            }
        )
        self._done = done
        self.events: list[Any] = []
        self.sent_events: list[Any] = []
        self.collect_buffer_ids: list[str | None] = []

    def collect_events(
        self,
        ev: Any,
        expected: list[type[Any]],
        buffer_id: str | None = None,
    ) -> list[Any] | None:
        assert ev in self._done
        assert len(expected) == len(self._done)
        self.collect_buffer_ids.append(buffer_id)
        return self._done

    def write_event_to_stream(self, ev: Any) -> None:
        self.events.append(ev)

    def send_event(self, ev: Any) -> None:
        self.sent_events.append(ev)


class _FakeGarage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[tuple[str, str, bytes, str | None]] = []
        self.deletes: list[tuple[str, str]] = []

    async def list_objects(self, bucket: str, prefix: str) -> list[dict[str, str]]:
        return [{"key": key} for key in self.objects if key.startswith(prefix)]

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
        self.puts.append((bucket, key, data, content_type))

    async def delete_object(self, bucket: str, key: str) -> None:
        self.objects.pop(key, None)
        self.deletes.append((bucket, key))


class _CorruptingDescriptionGarage(_FakeGarage):
    def __init__(self, payload: bytes | BaseException) -> None:
        super().__init__()
        self._payload = payload

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        if key.endswith("/vlm_descriptions.json"):
            if isinstance(self._payload, BaseException):
                raise self._payload
            return self._payload
        return await super().get_bytes(bucket, key)


class _FakeVLM:
    def __init__(self) -> None:
        self.close_count = 0

    @property
    def model(self) -> str:
        return "fake-vlm"

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        image_id: str,
        *,
        max_tokens: int,
    ) -> VLMDescription:
        return VLMDescription(
            description=f"description of {image_id}",
            model_used=self.model,
            token_count=1,
            duration_ms=1,
            image_id=image_id,
        )

    async def aclose(self) -> None:
        self.close_count += 1


class _FailingCloseVLM(_FakeVLM):
    async def aclose(self) -> None:
        self.close_count += 1
        raise RuntimeError("close failed")


def _figure_store(tmp_path: Path) -> FigureStore:
    store = FigureStore(tmp_path)
    assert store.write(SRC, 1, 0, b"fake-image-1", ext="png") == IMG_1_REF
    assert store.write(SRC, 1, 1, b"fake-image-2", ext="png") == IMG_2_REF
    return store


def _workflow(
    garage: _FakeGarage,
    vlm: _FakeVLM | None,
    *,
    figures: FigureStore | None = None,
) -> BookRagWorkflow:
    vlm_service_factory: Callable[[], VLMServiceLike] = (
        (lambda: vlm) if vlm is not None else build_vlm_service
    )
    kwargs: dict[str, Any] = {}
    if figures is not None:
        kwargs["figures"] = figures
    return BookRagWorkflow(
        garage=garage,
        qdrant_store=SimpleNamespace(),
        embedding_client=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        parser_router=SimpleNamespace(),
        config=BookRagConfig(
            enable_visual_extraction=True,
            enable_visual_description=True,
        ),
        timeout=30,
        vlm_service_factory=vlm_service_factory,
        **kwargs,
    )


def _attach_vlm(ctx: _FakeCtx, vlm: _FakeVLM) -> None:
    setattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR, vlm)


@pytest.mark.asyncio
async def test_collect_visual_low_yield_closes_vlm_and_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    garage = _FakeGarage()
    vlm = _FakeVLM()
    workflow = _workflow(garage, vlm)
    envelope = _envelope()
    done = [ImageDescriptionResult(image_id="img-1", evidence=None)]
    ctx = _FakeCtx(envelope=envelope, parsed=_parsed_doc(), done=done)
    _attach_vlm(ctx, vlm)
    marked: list[tuple[IntakeEnvelope, Exception]] = []

    def _record_mark_failed(failed_envelope: IntakeEnvelope, exc: Exception) -> None:
        marked.append((failed_envelope, exc))

    monkeypatch.setattr(workflow_mod, "_mark_failed", _record_mark_failed)

    with pytest.raises(RuntimeError, match="below threshold"):
        await workflow.collect_visual(ctx, done[0])

    assert vlm.close_count == 1
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is None
    assert len(marked) == 1
    assert marked[0][0] is envelope
    assert isinstance(marked[0][1], RuntimeError)


@pytest.mark.asyncio
async def test_collect_visual_success_closes_vlm_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    garage = _FakeGarage()
    vlm = _FakeVLM()
    workflow = _workflow(garage, vlm)
    envelope = _envelope()
    done = [
        ImageDescriptionResult(image_id="img-1", evidence={"description": "front wing"}),
        ImageDescriptionResult(image_id="img-2", evidence={"description": "rear wing"}),
    ]
    ctx = _FakeCtx(envelope=envelope, parsed=_parsed_doc(), done=done)
    _attach_vlm(ctx, vlm)

    async def _load_visual_composites(
        garage: Any,
        bucket: str,
        parsed: ParsedDocument,
        *,
        descriptions_garage_key: str | None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        return [], {"descriptions_garage_key": descriptions_garage_key}

    monkeypatch.setattr(workflow_mod, "load_visual_composites", _load_visual_composites)

    result = await workflow.collect_visual(ctx, done[0])

    assert result is not None
    assert result.source_id == "src-1"
    assert result.composites == []
    assert vlm.close_count == 1
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is None
    assert ctx.collect_buffer_ids == ["book_rag.visual.collect_visual:run-visual-test"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (RuntimeError("garage read failed"), "garage read failed"),
        (b"{", "vlm_descriptions.json"),
        (b"[]", "vlm_descriptions.json"),
    ],
)
@pytest.mark.asyncio
async def test_collect_visual_marks_failed_when_descriptions_artifact_cannot_load(
    payload: bytes | BaseException,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    garage = _CorruptingDescriptionGarage(payload)
    vlm = _FakeVLM()
    workflow = _workflow(garage, vlm)
    envelope = _envelope()
    done = [
        ImageDescriptionResult(image_id="img-1", evidence={"description": "front wing"}),
        ImageDescriptionResult(image_id="img-2", evidence={"description": "rear wing"}),
    ]
    ctx = _FakeCtx(envelope=envelope, parsed=_parsed_doc(), done=done)
    _attach_vlm(ctx, vlm)
    marked: list[tuple[IntakeEnvelope, Exception]] = []

    def _record_mark_failed(failed_envelope: IntakeEnvelope, exc: Exception) -> None:
        marked.append((failed_envelope, exc))

    monkeypatch.setattr(workflow_mod, "_mark_failed", _record_mark_failed)

    with pytest.raises(RuntimeError, match=match):
        await workflow.collect_visual(ctx, done[0])

    assert len(marked) == 1
    assert marked[0][0] is envelope
    assert "vlm_descriptions.json" in str(marked[0][1])
    assert vlm.close_count == 1
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is None


@pytest.mark.asyncio
async def test_collect_visual_loads_descriptions_artifact_into_composites() -> None:
    garage = _FakeGarage()
    vlm = _FakeVLM()
    workflow = _workflow(garage, vlm)
    done = [
        ImageDescriptionResult(image_id="img-1", evidence={"description": "front wing"}),
        ImageDescriptionResult(image_id="img-2", evidence={"description": "rear wing"}),
    ]
    ctx = _FakeCtx(envelope=_envelope(), parsed=_parsed_doc(), done=done)
    _attach_vlm(ctx, vlm)

    result = await workflow.collect_visual(ctx, done[0])

    assert result is not None
    assert [chunk.chunk_type for chunk in result.composites] == [
        "figure_composite",
        "figure_composite",
    ]
    assert [chunk.vlm_description for chunk in result.composites] == ["front wing", "rear wing"]
    assert vlm.close_count == 1


@pytest.mark.asyncio
async def test_collect_visual_uses_durable_run_id_for_buffer_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    garage = _FakeGarage()
    vlm = _FakeVLM()
    workflow = _workflow(garage, vlm)
    envelope = _envelope()
    done = [
        ImageDescriptionResult(image_id="img-1", evidence={"description": "front wing"}),
        ImageDescriptionResult(image_id="img-2", evidence={"description": "rear wing"}),
    ]
    ctx = _FakeCtx(envelope=envelope, parsed=_parsed_doc(), done=done)
    _attach_vlm(ctx, vlm)
    ctx.store._values["workflow_run_id"] = "durable-run"
    cast(Any, ctx).vo_rag_collect_buffer_scope = "attempt-run-id"

    async def _load_visual_composites(
        garage: Any,
        bucket: str,
        parsed: ParsedDocument,
        *,
        descriptions_garage_key: str | None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        return [], {"descriptions_garage_key": descriptions_garage_key}

    monkeypatch.setattr(workflow_mod, "load_visual_composites", _load_visual_composites)

    await workflow.collect_visual(ctx, done[0])

    assert ctx.collect_buffer_ids == ["book_rag.visual.collect_visual:durable-run"]


@pytest.mark.asyncio
async def test_collect_visual_success_is_not_overridden_by_vlm_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    garage = _FakeGarage()
    vlm = _FailingCloseVLM()
    workflow = _workflow(garage, vlm)
    envelope = _envelope()
    done = [
        ImageDescriptionResult(image_id="img-1", evidence={"description": "front wing"}),
        ImageDescriptionResult(image_id="img-2", evidence={"description": "rear wing"}),
    ]
    ctx = _FakeCtx(envelope=envelope, parsed=_parsed_doc(), done=done)
    _attach_vlm(ctx, vlm)

    async def _load_visual_composites(
        garage: Any,
        bucket: str,
        parsed: ParsedDocument,
        *,
        descriptions_garage_key: str | None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        return [], {"descriptions_garage_key": descriptions_garage_key}

    monkeypatch.setattr(workflow_mod, "load_visual_composites", _load_visual_composites)

    result = await workflow.collect_visual(ctx, done[0])

    assert result is not None
    assert result.source_id == "src-1"
    assert vlm.close_count == 1
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is None


@pytest.mark.asyncio
async def test_describe_one_failure_marks_failed_and_closes_vlm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    garage = _FakeGarage()
    vlm = _FakeVLM()
    workflow = _workflow(garage, vlm, figures=FigureStore(tmp_path))
    envelope = _envelope()
    parsed = _parsed_doc()
    ctx = _FakeCtx(envelope=envelope, parsed=parsed, done=[])
    ctx.store._values.update(
        {
            "visual_envelope_id": str(envelope.id),
            "workflow_run_id": "run-visual-failure",
        }
    )
    _attach_vlm(ctx, vlm)
    marked: list[tuple[IntakeEnvelope, Exception]] = []

    def _record_mark_failed(failed_envelope: IntakeEnvelope, exc: Exception) -> None:
        marked.append((failed_envelope, exc))
        failed_envelope.status = EnvelopeStatus.FAILED
        failed_envelope.error = str(exc)

    monkeypatch.setattr(workflow_mod, "_mark_failed", _record_mark_failed)

    with pytest.raises(FileNotFoundError):
        await workflow.describe_one(
            ctx,
            DescribeImageEvent(
                image_id="img-1",
                image_ref=IMG_1_REF,
                image_role="embedded_image",
                element_index=1,
            ),
        )

    assert vlm.close_count == 1
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is None
    assert len(marked) == 1
    assert marked[0][0] is envelope
    assert envelope.status is EnvelopeStatus.FAILED


@pytest.mark.asyncio
async def test_describe_one_rebuilds_run_scoped_vlm_when_resumed_after_fan_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    garage = _FakeGarage()
    figures = _figure_store(tmp_path)
    vlm = _FakeVLM()
    workflow = _workflow(garage, None, figures=figures)
    workflow.vlm_service_factory = lambda: vlm
    envelope = _envelope()
    parsed = _parsed_doc()
    ctx = _FakeCtx(envelope=envelope, parsed=parsed, done=[])
    ctx.store._values.update(
        {
            "visual_envelope_id": str(envelope.id),
            "workflow_run_id": "run-visual-resume",
        }
    )
    seen_services: list[Any] = []

    async def _describe_one_image(
        **kwargs: Any,
    ) -> dict[str, Any]:
        seen_services.append(kwargs["vlm_service"])
        return {"description": "resumed visual"}

    monkeypatch.setattr(workflow_mod, "_describe_one_image", _describe_one_image)

    result = await workflow.describe_one(
        ctx,
        DescribeImageEvent(
            image_id="img-1",
            image_ref=IMG_1_REF,
            image_role="embedded_image",
            element_index=1,
        ),
    )

    assert result.evidence == {"description": "resumed visual"}
    assert seen_services == [vlm]
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is vlm


@pytest.mark.asyncio
async def test_visual_vlm_services_are_run_scoped_on_shared_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    garage = _FakeGarage()
    figures = _figure_store(tmp_path)
    services = [_FakeVLM(), _FakeVLM()]
    workflow = _workflow(garage, None, figures=figures)
    config = BookRagConfig(
        enable_visual_extraction=True,
        enable_visual_description=True,
    )
    workflow.config = config
    parsed = _parsed_doc()
    envelope_1 = _envelope()
    envelope_2 = _envelope()
    ctx_1 = _FakeCtx(envelope=envelope_1, parsed=parsed, done=[])
    ctx_2 = _FakeCtx(envelope=envelope_2, parsed=parsed, done=[])
    ctx_1.store._values["workflow_run_id"] = "run-1"
    ctx_2.store._values["workflow_run_id"] = "run-2"

    def _build_vlm_service() -> _FakeVLM:
        return services.pop(0)

    workflow.vlm_service_factory = _build_vlm_service

    parsed_event = workflow_mod.ParsedEvent(
        source_id=parsed.source_id,
        parse_meta={"parsed_garage_key": "parsed/doc-1/parsed.json"},
        parsed=parsed,
    )
    assert await workflow.fan_out_images(ctx_1, parsed_event) is None
    assert await workflow.fan_out_images(ctx_2, parsed_event) is None

    service_1 = getattr(ctx_1, workflow_mod._VISUAL_VLM_SERVICE_ATTR)
    service_2 = getattr(ctx_2, workflow_mod._VISUAL_VLM_SERVICE_ATTR)
    assert service_1 is not service_2

    await workflow_mod._close_visual_vlm_service(cast(Any, ctx_1), workflow)

    assert service_1.close_count == 1
    assert service_2.close_count == 0
    assert getattr(ctx_2, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is service_2


@pytest.mark.asyncio
async def test_wrapped_workflow_handler_closes_run_scoped_vlm_before_await_returns() -> None:
    garage = _FakeGarage()
    workflow = _workflow(garage, None)
    vlm = _FakeVLM()
    ctx = _FakeCtx(envelope=_envelope(), parsed=_parsed_doc(), done=[])
    setattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR, vlm)

    class _FakeHandler:
        def __init__(self, ctx: _FakeCtx) -> None:
            self.ctx = ctx

        def __await__(self) -> Any:
            async def _run() -> str:
                return "ok"

            return _run().__await__()

        def add_done_callback(self, callback: Any) -> None:
            self._callback = callback

        def get_loop(self) -> asyncio.AbstractEventLoop:
            return asyncio.get_running_loop()

    handler = _FakeHandler(ctx)

    wrapped = workflow_mod._attach_run_resource_cleanup(handler, workflow)
    result = await wrapped

    assert result == "ok"
    assert vlm.close_count == 1
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is None


@pytest.mark.asyncio
async def test_wrapped_workflow_handler_done_callback_closes_abandoned_run_vlm() -> None:
    garage = _FakeGarage()
    workflow = _workflow(garage, None)
    vlm = _FakeVLM()
    ctx = _FakeCtx(envelope=_envelope(), parsed=_parsed_doc(), done=[])
    setattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR, vlm)

    async def _run() -> str:
        return "ok"

    task = asyncio.create_task(_run())
    cast(Any, task).ctx = ctx

    workflow_mod._attach_run_resource_cleanup(task, workflow)
    await task
    await asyncio.sleep(0)

    assert vlm.close_count == 1
    assert getattr(ctx, workflow_mod._VISUAL_VLM_SERVICE_ATTR) is None


@pytest.mark.asyncio
async def test_describe_one_image_timeout_returns_none() -> None:
    parsed = _parsed_doc()
    image = parsed.images[0]

    class _SlowVLM:
        @property
        def model(self) -> str:
            return "slow-test-vlm"

        async def describe_image(
            self,
            image_bytes: bytes,
            prompt: str,
            image_id: str,
            *,
            max_tokens: int,
        ) -> VLMDescription:
            await asyncio.sleep(0.05)
            return VLMDescription(
                description="late answer",
                model_used="slow-test-vlm",
                token_count=1,
                duration_ms=50,
                image_id=image_id,
            )

        async def aclose(self) -> None:
            return None

    class _Checkpoint:
        async def record_image(self, image_id: str, evidence: dict[str, Any]) -> None:
            raise AssertionError("timed-out image descriptions must not be checkpointed")

    evidence = await workflow_mod._describe_one_image(
        envelope_id=_ENVELOPE_ID,
        job=VlmImageJob(
            image=image,
            image_bytes=b"fake-image",
            image_role="embedded_image",
        ),
        parsed=parsed,
        vlm_service=cast(VLMServiceLike, _SlowVLM()),
        checkpoint=cast(Checkpoint, _Checkpoint()),
        max_tokens=16,
        deadline_s=0.001,
    )

    assert evidence is None


@pytest.mark.asyncio
async def test_describe_one_image_checkpoints_rejected_admission_evidence() -> None:
    """A figure the VLM classifies as non-technical is recorded, not dropped.

    The rejected record carries admitted=False and a normalized reason so the
    yield floor counts it as processed while the composite loader skips it.
    """
    parsed = _parsed_doc()
    image = parsed.images[0]

    class _CoverVLM:
        @property
        def model(self) -> str:
            return "cover-test-vlm"

        async def describe_image(
            self,
            image_bytes: bytes,
            prompt: str,
            image_id: str,
            *,
            max_tokens: int,
        ) -> VLMDescription:
            return VLMDescription(
                description="Book cover with the title in large type.",
                model_used="cover-test-vlm",
                token_count=5,
                duration_ms=10,
                image_id=image_id,
                page_kind="photo",
                confidence=0.9,
                figure_kind="cover",
                is_meaningful_technical_figure=False,
                rejection_reason="Cover art, not a technical figure.",
            )

        async def aclose(self) -> None:
            return None

    recorded: dict[str, dict[str, Any]] = {}

    class _Checkpoint:
        async def record_image(self, image_id: str, evidence: dict[str, Any]) -> None:
            recorded[image_id] = evidence

    evidence = await workflow_mod._describe_one_image(
        envelope_id=_ENVELOPE_ID,
        job=VlmImageJob(
            image=image,
            image_bytes=b"fake-image",
            image_role="embedded_image",
        ),
        parsed=parsed,
        vlm_service=cast(VLMServiceLike, _CoverVLM()),
        checkpoint=cast(Checkpoint, _Checkpoint()),
        max_tokens=16,
        deadline_s=0,
    )

    assert evidence is not None
    assert evidence["admitted"] is False
    assert evidence["figure_kind"] == "cover"
    assert evidence["description"] == ""
    assert evidence["rejection_reason"] == "nontechnical_visual_cover"
    assert recorded[image.image_id] == evidence


@pytest.mark.asyncio
async def test_collect_visual_rejects_spurious_result_before_fan_out() -> None:
    garage = _FakeGarage()
    vlm = _FakeVLM()
    workflow = _workflow(garage, vlm)
    envelope = _envelope()
    done = [ImageDescriptionResult(image_id="img-1", evidence={"description": "front wing"})]
    ctx = _FakeCtx(envelope=envelope, parsed=_parsed_doc(), done=done)
    ctx.store._values["n_pending"] = 0

    with pytest.raises(RuntimeError, match="before fan-out"):
        await workflow.collect_visual(ctx, done[0])
