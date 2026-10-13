from __future__ import annotations

import json
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
)
from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.core.storage.figures import FigureStore
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.visual_sidecar import VLMDescription
from vo_rag.pipelines.book_rag.workflow import PipelineOutput
from vo_rag.pipelines.book_rag.workflow_li import (
    BookRagWorkflow,
    ParsedEvent,
    VisualEvent,
)

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_HASH = "a" * 64
_VERIFICATION = {"status": "verified", "sha256": _HASH, "size_bytes": 42}
SRC = "ab" * 32


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


def _parsed_doc(
    *,
    n_images: int = 3,
    source_id: str = "src-1",
    refs: list[str] | None = None,
) -> ParsedDocument:
    # Each image follows its own paragraph, so every image has distinct text
    # context: images with none are not emitted (VOR-1427), and consecutive
    # images with identical context collapse into one figure (VOR-1441).
    elements = [
        ParsedElement(
            element_id="e-title",
            element_type=ElementType.TITLE,
            text="Chapter 1",
            page_number=1,
        )
    ]
    images = []
    for i in range(n_images):
        elements.append(
            ParsedElement(
                element_id=f"e-text-{i}",
                element_type=ElementType.NARRATIVE_TEXT,
                text=f"Paragraph {i} introduces figure {i}.",
                page_number=1,
            )
        )
        elements.append(
            ParsedElement(
                element_id=f"e-image-{i}",
                element_type=ElementType.IMAGE,
                text=f"![](figure-{i}.png)",
                page_number=1,
            )
        )
        images.append(
            ExtractedImage(
                image_id=f"img-{i}",
                page_number=1,
                image_index=i,
                ref=(refs[i] if refs is not None else FigureStore.figure_ref(SRC, 1, i, "png")),
                element_index=len(elements) - 1,
            )
        )
    return ParsedDocument(
        document_id="doc-1",
        source_id=source_id,
        title="Parsed Book Title",
        author="Parsed Author",
        total_pages=1,
        elements=elements,
        images=images,
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

    async def list_objects(self, bucket: str, prefix: str) -> list[dict[str, Any]]:
        return [{"key": k} for k in self.objects if k.startswith(prefix)]

    async def delete_object(self, bucket: str, key: str) -> None:
        self.objects.pop(key, None)


class _FakeParserRouter:
    def __init__(self, parsed: ParsedDocument) -> None:
        self._parsed = parsed

    async def parse(self, path: str, file_hash: str) -> ParsedDocument:
        return self._parsed


class _FakeEmbeddingClient:
    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
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
    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any],
    ) -> LLMResponse:
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
        return [SimpleNamespace(id="point-1", payload={"chunk_id": self.expected_chunk_id})]


class _FakeVLMService:
    def __init__(self, descriptions: dict[str, str] | None = None) -> None:
        self._descriptions = descriptions or {}
        self.model = "fake-vlm"
        self.closed = False

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        image_id: str,
        *,
        max_tokens: int,
    ) -> VLMDescription:
        desc = self._descriptions.get(image_id, f"description of {image_id}")
        return VLMDescription(
            description=desc,
            model_used=self.model,
            token_count=10,
            duration_ms=100,
            image_id=image_id,
        )

    async def aclose(self) -> None:
        self.closed = True


def _patch_step(
    workflow: BookRagWorkflow,
    name: str,
    replacement: Any,
) -> None:
    """Monkeypatch a workflow step while preserving __step_config."""
    original = getattr(workflow, name)
    replacement.__step_config = original.__step_config
    setattr(workflow, name, types.MethodType(replacement, workflow))


def _build_garage(
    tmp_path: Path,
    n_images: int = 3,
) -> tuple[_FakeGarage, ParsedDocument, FigureStore]:
    figures = FigureStore(tmp_path)
    refs = [
        figures.write(SRC, 1, i, f"fake-image-{i}".encode(), ext="png") for i in range(n_images)
    ]
    parsed = _parsed_doc(n_images=n_images, refs=refs)
    objects: dict[str, bytes] = {
        "drops/book.txt": b"plain text fixture",
        f"parsed/{_envelope().id}/parsed.json": parsed.model_dump_json().encode(),
    }
    return _FakeGarage(objects), parsed, figures


def _workflow(
    *,
    garage: _FakeGarage,
    figures: FigureStore,
    config: BookRagConfig | None = None,
    vlm_service: _FakeVLMService | None = None,
) -> BookRagWorkflow:
    vlm_service_factory = (lambda: vlm_service) if vlm_service is not None else None
    workflow_kwargs: dict[str, Any] = {}
    if vlm_service_factory is not None:
        workflow_kwargs["vlm_service_factory"] = vlm_service_factory
    return BookRagWorkflow(
        garage=garage,
        qdrant_store=_FakeQdrantStore(),
        embedding_client=_FakeEmbeddingClient(),
        llm_client=_FakeLLMClient(),
        retriever=_FakeRetriever("pending"),
        parser_router=_FakeParserRouter(_parsed_doc()),
        config=config or BookRagConfig(),
        figures=figures,
        timeout=30,
        **workflow_kwargs,
    )


@pytest.mark.asyncio
async def test_visual_fan_out_describes_all_images(tmp_path: Path) -> None:
    """M images + fake vlm_service -> all M described, one VisualEvent with M composites."""
    n_images = 3
    garage, parsed, figures = _build_garage(tmp_path, n_images=n_images)
    vlm = _FakeVLMService()
    config = BookRagConfig(
        enable_visual_extraction=True,
        enable_visual_description=True,
    )
    workflow = _workflow(garage=garage, figures=figures, config=config, vlm_service=vlm)
    envelope = _envelope()

    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    async def _fake_parse(
        *,
        envelope_id: str,
        garage_key: str,
        source_id: str,
        garage: Any,
        bucket: str,
        parser_router: Any,
    ) -> dict[str, Any]:
        return {
            "parsed_garage_key": f"parsed/{envelope_id}/parsed.json",
            "document_id": "doc-1",
            "source_id": source_id,
            "title": parsed.title,
            "total_pages": parsed.total_pages,
            "image_count": len(parsed.images),
            "element_count": len(parsed.elements),
            "parser_used": "test",
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_mod, "do_parse_document", _fake_parse)
    garage.objects[f"parsed/{envelope.id}/parsed.json"] = parsed.model_dump_json().encode()

    visual_events: list[VisualEvent] = []

    async def _capturing_chunk_embed(self: Any, ctx: Any, ev: VisualEvent) -> Any:
        visual_events.append(ev)
        from vo_rag.pipelines.book_rag.workflow_li import EmbeddedEvent

        return EmbeddedEvent(
            source_id=ev.source_id,
            parse_meta=ev.parse_meta,
            chunks=[],
            raw_point_ids=[],
            raw_embeddings={},
        )

    _patch_step(workflow, "chunk_embed", _capturing_chunk_embed)

    async def _noop_register_artifact(self: Any, ctx: Any, ev: Any) -> Any:
        from vo_rag.pipelines.book_rag.workflow_li import RegisteredArtifactEvent

        return RegisteredArtifactEvent(**ev.model_dump())

    _patch_step(workflow, "register_artifact", _noop_register_artifact)

    async def _noop_validate(self: Any, ctx: Any, ev: Any) -> Any:
        from vo_rag.pipelines.book_rag.workflow_li import ValidatedEvent

        return ValidatedEvent(**ev.model_dump(), validation_report=None)

    _patch_step(workflow, "validate", _noop_validate)

    async def _noop_smoke(self: Any, ctx: Any, ev: Any) -> Any:
        from llama_index.core.workflow import StopEvent

        return StopEvent(
            result=PipelineOutput(
                source_id=ev.source_id,
                status=EnvelopeStatus.COMPLETED,
                qdrant_point_ids=[],
                raw_point_ids=ev.raw_point_ids,
                sterile_point_ids=ev.sterile_point_ids,
                smoke_decision=None,
                chunk_count=0,
            )
        )

    _patch_step(workflow, "retrieval_smoke", _noop_smoke)

    workflow._disable_validation = True
    result = await workflow.run(envelope=envelope)

    assert isinstance(result, PipelineOutput)
    assert len(visual_events) == 1
    visual_ev = visual_events[0]
    assert len(visual_ev.composites) == n_images
    image_ids = {img.image_id for img in parsed.images}
    composite_image_ids = {c.image_id for c in visual_ev.composites}
    assert composite_image_ids == image_ids
    assert vlm.closed
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_visual_fan_out_resumes_from_checkpoint(tmp_path: Path) -> None:
    """K already-done images -> only M-K DescribeImageEvents sent."""
    n_images = 5
    k_done = 2
    garage, parsed, figures = _build_garage(tmp_path, n_images=n_images)
    run_id = "test-run-resume"

    checkpoint_prefix = f"vo_rag/checkpoints/book_rag/{run_id}/images"
    for i in range(k_done):
        image_id = f"img-{i}"
        payload = json.dumps(
            {
                "image_id": image_id,
                "description": {
                    "description": f"already-processed-{i}",
                    "model_used": "fake",
                    "token_count": 10,
                    "duration_ms": 100,
                    "image_id": image_id,
                    "image_role": "embedded_image",
                    "image_ref": f"images/{image_id}.png",
                    "page_number": 1,
                    "schema_version": "vo_rag_visual_evidence_v0",
                },
            }
        ).encode()
        garage.objects[f"{checkpoint_prefix}/{image_id}.json"] = payload

    vlm = _FakeVLMService()
    config = BookRagConfig(
        enable_visual_extraction=True,
        enable_visual_description=True,
    )
    workflow = _workflow(garage=garage, figures=figures, config=config, vlm_service=vlm)
    envelope = _envelope()

    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    async def _fake_parse(
        *,
        envelope_id: str,
        garage_key: str,
        source_id: str,
        garage: Any,
        bucket: str,
        parser_router: Any,
    ) -> dict[str, Any]:
        return {
            "parsed_garage_key": f"parsed/{envelope_id}/parsed.json",
            "document_id": "doc-1",
            "source_id": source_id,
            "title": parsed.title,
            "total_pages": parsed.total_pages,
            "image_count": len(parsed.images),
            "element_count": len(parsed.elements),
            "parser_used": "test",
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_mod, "do_parse_document", _fake_parse)
    garage.objects[f"parsed/{envelope.id}/parsed.json"] = parsed.model_dump_json().encode()

    visual_events: list[VisualEvent] = []

    async def _capturing_chunk_embed(self: Any, ctx: Any, ev: VisualEvent) -> Any:
        visual_events.append(ev)
        from vo_rag.pipelines.book_rag.workflow_li import EmbeddedEvent

        return EmbeddedEvent(
            source_id=ev.source_id,
            parse_meta=ev.parse_meta,
            chunks=[],
            raw_point_ids=[],
            raw_embeddings={},
        )

    _patch_step(workflow, "chunk_embed", _capturing_chunk_embed)

    async def _noop_register_artifact(self: Any, ctx: Any, ev: Any) -> Any:
        from vo_rag.pipelines.book_rag.workflow_li import RegisteredArtifactEvent

        return RegisteredArtifactEvent(**ev.model_dump())

    _patch_step(workflow, "register_artifact", _noop_register_artifact)

    async def _noop_validate(self: Any, ctx: Any, ev: Any) -> Any:
        from vo_rag.pipelines.book_rag.workflow_li import ValidatedEvent

        return ValidatedEvent(**ev.model_dump(), validation_report=None)

    _patch_step(workflow, "validate", _noop_validate)

    async def _noop_smoke(self: Any, ctx: Any, ev: Any) -> Any:
        from llama_index.core.workflow import StopEvent

        return StopEvent(
            result=PipelineOutput(
                source_id=ev.source_id,
                status=EnvelopeStatus.COMPLETED,
                qdrant_point_ids=[],
                raw_point_ids=ev.raw_point_ids,
                sterile_point_ids=ev.sterile_point_ids,
                smoke_decision=None,
                chunk_count=0,
            )
        )

    _patch_step(workflow, "retrieval_smoke", _noop_smoke)

    from vo_rag.pipelines.book_rag.workflow_li import _workflow_context_for_run

    ctx = _workflow_context_for_run(workflow, run_id)

    workflow._disable_validation = True
    result = await workflow.run(ctx=ctx, envelope=envelope)

    assert isinstance(result, PipelineOutput)
    assert len(visual_events) == 1
    visual_ev = visual_events[0]
    assert len(visual_ev.composites) == n_images
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_visual_fan_out_all_images_checkpointed_does_not_open_vlm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    n_images = 2
    garage, parsed, figures = _build_garage(tmp_path, n_images=n_images)
    run_id = "test-run-all-checkpointed"
    checkpoint_prefix = f"vo_rag/checkpoints/book_rag/{run_id}/images"
    for image in parsed.images:
        garage.objects[f"{checkpoint_prefix}/{image.image_id}.json"] = json.dumps(
            {
                "image_id": image.image_id,
                "description": {
                    "description": f"already-processed-{image.image_id}",
                    "model_used": "fake",
                    "token_count": 10,
                    "duration_ms": 100,
                    "image_id": image.image_id,
                    "image_role": "embedded_image",
                    "image_ref": image.ref,
                    "page_number": 1,
                    "schema_version": "vo_rag_visual_evidence_v0",
                },
            }
        ).encode()

    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    def _build_vlm_service() -> Any:
        raise AssertionError("VLM service should not be built when no images are pending")

    monkeypatch.setattr(workflow_mod, "build_vlm_service", _build_vlm_service)

    config = BookRagConfig(
        enable_visual_extraction=True,
        enable_visual_description=True,
    )
    workflow = _workflow(garage=garage, figures=figures, config=config, vlm_service=None)
    envelope = _envelope()
    values: dict[str, Any] = {"envelope": envelope, "workflow_run_id": run_id}
    sent_events: list[Any] = []

    async def _get(key: str, default: Any = ...) -> Any:
        if key in values:
            return values[key]
        if default is ...:
            raise KeyError(key)
        return default

    async def _set(key: str, value: Any) -> None:
        values[key] = value

    ctx = SimpleNamespace(
        store=SimpleNamespace(get=_get, set=_set),
        write_event_to_stream=lambda ev: None,
        send_event=sent_events.append,
    )

    result = await workflow.fan_out_images(
        ctx,
        ParsedEvent(
            source_id=parsed.source_id,
            parse_meta={"parsed_garage_key": f"parsed/{envelope.id}/parsed.json"},
            parsed=parsed,
        ),
    )

    assert isinstance(result, VisualEvent)
    assert sent_events == []
    assert "visual_vlm_service" not in values


@pytest.mark.asyncio
async def test_visual_no_description_uses_extraction_composites(tmp_path: Path) -> None:
    """enable_visual_description=False -> build_extraction_composites path, no VLM events."""
    n_images = 3
    garage, parsed, figures = _build_garage(tmp_path, n_images=n_images)
    config = BookRagConfig(
        enable_visual_extraction=True,
        enable_visual_description=False,
    )
    workflow = _workflow(garage=garage, figures=figures, config=config, vlm_service=None)
    envelope = _envelope()

    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    async def _fake_parse(
        *,
        envelope_id: str,
        garage_key: str,
        source_id: str,
        garage: Any,
        bucket: str,
        parser_router: Any,
    ) -> dict[str, Any]:
        return {
            "parsed_garage_key": f"parsed/{envelope_id}/parsed.json",
            "document_id": "doc-1",
            "source_id": source_id,
            "title": parsed.title,
            "total_pages": parsed.total_pages,
            "image_count": len(parsed.images),
            "element_count": len(parsed.elements),
            "parser_used": "test",
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_mod, "do_parse_document", _fake_parse)
    garage.objects[f"parsed/{envelope.id}/parsed.json"] = parsed.model_dump_json().encode()

    visual_events: list[VisualEvent] = []

    async def _capturing_chunk_embed(self: Any, ctx: Any, ev: VisualEvent) -> Any:
        visual_events.append(ev)
        from vo_rag.pipelines.book_rag.workflow_li import EmbeddedEvent

        return EmbeddedEvent(
            source_id=ev.source_id,
            parse_meta=ev.parse_meta,
            chunks=[],
            raw_point_ids=[],
            raw_embeddings={},
        )

    _patch_step(workflow, "chunk_embed", _capturing_chunk_embed)

    async def _noop_register_artifact(self: Any, ctx: Any, ev: Any) -> Any:
        from vo_rag.pipelines.book_rag.workflow_li import RegisteredArtifactEvent

        return RegisteredArtifactEvent(**ev.model_dump())

    _patch_step(workflow, "register_artifact", _noop_register_artifact)

    async def _noop_validate(self: Any, ctx: Any, ev: Any) -> Any:
        from vo_rag.pipelines.book_rag.workflow_li import ValidatedEvent

        return ValidatedEvent(**ev.model_dump(), validation_report=None)

    _patch_step(workflow, "validate", _noop_validate)

    async def _noop_smoke(self: Any, ctx: Any, ev: Any) -> Any:
        from llama_index.core.workflow import StopEvent

        return StopEvent(
            result=PipelineOutput(
                source_id=ev.source_id,
                status=EnvelopeStatus.COMPLETED,
                qdrant_point_ids=[],
                raw_point_ids=ev.raw_point_ids,
                sterile_point_ids=ev.sterile_point_ids,
                smoke_decision=None,
                chunk_count=0,
            )
        )

    _patch_step(workflow, "retrieval_smoke", _noop_smoke)

    workflow._disable_validation = True
    result = await workflow.run(envelope=envelope)

    assert isinstance(result, PipelineOutput)
    assert len(visual_events) == 1
    visual_ev = visual_events[0]
    assert len(visual_ev.composites) == n_images
    for comp in visual_ev.composites:
        assert comp.vlm_description == ""
    monkeypatch.undo()
