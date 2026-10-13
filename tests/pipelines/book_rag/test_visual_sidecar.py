"""Contract tests for `vo_rag.pipelines.book_rag.visual_sidecar` (VOR-7).

The sidecar is the operator backfill path: it reads a parsed document from
Garage, runs VLM description on eligible images, persists a
`vlm_descriptions.json` artifact, and re-builds `RetrievalChunk` records
with `chunk_type="figure_composite"` (the seam consumed by the core
`HierarchicalChunker`).

This test module locks the public surface that downstream book_rag steps
will depend on:

- ``describe_visual_images`` returns a `vlm_descriptions.json`-shaped dict
  (with the keys operators index on: ``descriptions_garage_key``,
  ``described_count``, ``eligible_image_count``, ``skipped_count``,
  ``skipped_by_reason``, ``model``, ``vlm_url``).
- ``build_visual_chunks`` produces ``RetrievalChunk`` records with
  ``chunk_type="figure_composite"`` (the seam the chunker emits into Qdrant).
- ``_assert_vlm_healthy`` probes RunPod serverless via
  ``vo_rag.core.cloud.serverless.ServerlessClient`` (NOT raw httpx) and
  raises on a non-serving response.
- The three operator env vars
  ``VO_RAG_VLM_DESCRIBE_CONCURRENCY`` / ``VO_RAG_VLM_MAX_TOKENS`` /
  ``VO_RAG_VLM_PER_IMAGE_DEADLINE_S`` are read with the documented defaults.

Fakes are used for Garage + the VLM service so the tests stay hermetic and
do not touch the network. The ``ServerlessClient`` is exercised via a
duck-typed double that records the probe payload.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import respx

# Pillow is runtime-extra-only (src lazy-imports it); the llamaindex-absent
# light CI lane must skip rather than fail collection.
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from vo_rag.core.chunking.artifact_filters import visual_composite_skip_reason  # noqa: E402
from vo_rag.core.cloud.serverless import ServerlessClient  # noqa: E402
from vo_rag.core.parsing.contract import (  # noqa: E402
    ElementType,
    ExtractedImage,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.core.storage.figures import FigureStore  # noqa: E402
from vo_rag.core.storage.garage import GarageClient  # noqa: E402
from vo_rag.pipelines.book_rag import visual_sidecar  # noqa: E402

SRC = "ab" * 32
IMG_A_REF = FigureStore.figure_ref(SRC, 1, 0, "png")
IMG_B_REF = FigureStore.figure_ref(SRC, 2, 1, "png")


def _png_with_dimensions(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeGarage:
    """In-memory Garage stand-in — bytes go in, bytes come out, all async."""

    def __init__(self, initial: dict[str, bytes] | None = None) -> None:
        self._objects: dict[str, bytes] = dict(initial or {})
        self.puts: list[tuple[str, str, bytes, str | None]] = []
        self.gets: list[tuple[str, str]] = []

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        self.gets.append((bucket, key))
        return self._objects[key]

    async def put_bytes(
        self, bucket: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        self.puts.append((bucket, key, data, content_type))
        self._objects[key] = data

    async def list_objects(self, bucket: str, prefix: str = "") -> list[dict[str, Any]]:
        return [{"key": key} for key in sorted(self._objects) if key.startswith(prefix)]

    async def delete_object(self, bucket: str, key: str) -> None:
        self._objects.pop(key, None)


class _FakeVLMDescription:
    """Stand-in for the VLM return type — has the four fields the sidecar reads."""

    def __init__(
        self,
        *,
        description: str,
        model_used: str = "qwen2-vl-7b",
        token_count: int = 12,
        duration_ms: int = 100,
        image_id: str = "",
    ) -> None:
        self.description = description
        self.model_used = model_used
        self.token_count = token_count
        self.duration_ms = duration_ms
        self.image_id = image_id


class _FakeVLMService:
    """Duck-typed VLM service: records calls, returns canned descriptions."""

    def __init__(self, *, model: str = "qwen2-vl-7b") -> None:
        self.model = model
        self.calls: list[tuple[bytes, str, str, int]] = []
        self.aclose_called = False

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        image_id: str,
        *,
        max_tokens: int = 1024,
    ) -> _FakeVLMDescription:
        self.calls.append((image_bytes, prompt, image_id, max_tokens))
        return _FakeVLMDescription(
            description=f"diagram showing {image_id} — a wing cross-section with pressure vectors",
            image_id=image_id,
        )

    async def aclose(self) -> None:
        self.aclose_called = True


class _RejectingVLMService(_FakeVLMService):
    """VLM fake that returns a structured non-figure admission result."""

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        image_id: str,
        *,
        max_tokens: int = 1024,
    ) -> _FakeVLMDescription:
        self.calls.append((image_bytes, prompt, image_id, max_tokens))
        return _FakeVLMDescription(
            description=json.dumps(
                {
                    "figure_kind": "logo",
                    "is_meaningful_technical_figure": False,
                    "description": "",
                    "rejection_reason": None,
                }
            ),
            image_id=image_id,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _element(
    text: str, page: int, *, element_type: ElementType = ElementType.NARRATIVE_TEXT
) -> ParsedElement:
    return ParsedElement(
        element_id=f"e-{page}-{abs(hash(text)) % 9999}",
        element_type=element_type,
        text=text,
        page_number=page,
    )


def _parsed_doc(*, images: list[ExtractedImage] | None = None) -> ParsedDocument:
    default_images = [
        ExtractedImage(
            image_id="img-aaaa",
            page_number=1,
            image_index=0,
            ref=IMG_A_REF,
            element_index=1,
        ),
        ExtractedImage(
            image_id="img-bbbb",
            page_number=2,
            image_index=1,
            ref=IMG_B_REF,
            element_index=4,
        ),
    ]
    return ParsedDocument(
        document_id="doc-1",
        source_id="src-1",
        title="Wing Aerodynamics",
        author="Author",
        total_pages=2,
        elements=[
            _element("Preceding prose for image one.", 1),
            _element("Image one shows the wing planform.", 1),
            _element("Following text on page one.", 1),
            _element("Preceding prose for image two on page two.", 2),
            _element("Following text after image two.", 2),
        ],
        images=list(images) if images is not None else default_images,
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="pymupdf",
    )


@pytest.fixture
def fake_garage() -> _FakeGarage:
    parsed = _parsed_doc()
    return _FakeGarage(
        {
            "parsed/doc-1/parsed.json": parsed.model_dump_json().encode("utf-8"),
        }
    )


@pytest.fixture
def figure_store(tmp_path: Path) -> FigureStore:
    store = FigureStore(tmp_path)
    assert store.write(SRC, 1, 0, b"PNG-A", ext="png") == IMG_A_REF
    assert store.write(SRC, 2, 1, b"PNG-B", ext="png") == IMG_B_REF
    return store


@pytest.fixture
def fake_vlm() -> _FakeVLMService:
    return _FakeVLMService()


# ---------------------------------------------------------------------------
# Visual evidence admission
# ---------------------------------------------------------------------------


def test_visual_evidence_from_vlm_parses_structured_admission() -> None:
    image = _parsed_doc().images[0]
    response = visual_sidecar.VLMDescription(
        description=json.dumps(
            {
                "figure_kind": "chart",
                "is_meaningful_technical_figure": True,
                "description": "Line chart showing lift coefficient versus angle of attack.",
                "rejection_reason": None,
            }
        ),
        model_used="qwen2-vl-7b",
        token_count=20,
        duration_ms=50,
        image_id=image.image_id,
    )

    evidence = visual_sidecar.visual_evidence_from_vlm(
        response,
        image=image,
        image_role="embedded_image",
    )

    assert evidence.admitted is True
    assert evidence.figure_kind == "chart"
    assert evidence.is_meaningful_technical_figure is True
    assert "lift coefficient" in evidence.description
    assert evidence.rejection_reason is None
    assert evidence.raw_response == response.description


def test_visual_evidence_from_vlm_repairs_jsonish_admission_with_inch_marks() -> None:
    image = _parsed_doc().images[0]
    response = visual_sidecar.VLMDescription(
        description="""{
  "figure_kind": "chart",
  "is_meaningful_technical_figure": true,
  "description": "A line chart titled 'Figure 13: Aerodynamic Performance vs Pitch'. \
The horizontal axis shows pitch conditions: 'nose down 1"', 'Baseline', \
'tail down 1"', and 'tail down 2"'.",
  "rejection_reason": ""
}""",
        model_used="qwen2-vl-7b",
        token_count=20,
        duration_ms=50,
        image_id=image.image_id,
    )

    evidence = visual_sidecar.visual_evidence_from_vlm(
        response,
        image=image,
        image_role="embedded_image",
    )

    assert evidence.admitted is True
    assert evidence.figure_kind == "chart"
    assert evidence.description.startswith("A line chart titled")
    assert "nose down 1" in evidence.description
    assert not evidence.description.startswith("{")
    assert evidence.rejection_reason is None


def test_visual_evidence_from_vlm_normalizes_structured_rejection_reason() -> None:
    image = _parsed_doc().images[0]
    response = visual_sidecar.VLMDescription(
        description=json.dumps(
            {
                "figure_kind": "cover",
                "is_meaningful_technical_figure": False,
                "description": "Book cover with title typography.",
                "rejection_reason": "This is a book cover, not a technical figure.",
            }
        ),
        model_used="qwen2-vl-7b",
        token_count=20,
        duration_ms=50,
        image_id=image.image_id,
    )

    evidence = visual_sidecar.visual_evidence_from_vlm(
        response,
        image=image,
        image_role="embedded_image",
    )

    assert evidence.admitted is False
    assert evidence.figure_kind == "cover"
    assert evidence.description == ""
    assert evidence.rejection_reason == "nontechnical_visual_cover"


def test_build_describe_prompt_admits_unlabeled_motorsport_schematics() -> None:
    parsed = _parsed_doc()
    prompt = visual_sidecar._build_describe_prompt(
        image=parsed.images[0],
        image_role="embedded_image",
        parsed=parsed,
        context=visual_sidecar.ContextWindow(section_title="Vehicle Dynamics"),
    )

    assert "racing lines" in prompt
    assert "concentric-circle or spiral exercises" in prompt
    assert "concentric-arrow patterns" in prompt
    assert "rather than rejected as decorative" in prompt
    assert "numbered or lettered callouts" in prompt
    assert "use the exact mapped terms" in prompt
    assert "Do not transcribe exact formulas" in prompt


# ---------------------------------------------------------------------------
# describe_visual_images — vlm_descriptions.json shape
# ---------------------------------------------------------------------------


async def test_describe_visual_images_returns_vlm_descriptions_shaped_dict(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
    figure_store: FigureStore,
    fake_vlm: _FakeVLMService,
) -> None:
    """The return dict mirrors the on-disk vlm_descriptions.json shape so
    operators can index the same fields in both places (descriptions_garage_key,
    described_count, eligible_image_count, skipped_count, skipped_by_reason,
    model, vlm_url)."""

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)
    monkeypatch.setattr(visual_sidecar, "build_vlm_service", lambda: fake_vlm)
    monkeypatch.setenv("VO_RAG_VLM_DESCRIBE_CONCURRENCY", "1")

    result = await visual_sidecar.describe_visual_images(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        figures=figure_store,
    )

    # The contract: the keys operators reach for
    assert set(result).issuperset(
        {
            "descriptions_garage_key",
            "described_count",
            "vlm_processed_count",
            "admitted_description_count",
            "rejected_visual_evidence_count",
            "eligible_image_count",
            "skipped_count",
            "skipped_by_reason",
            "model",
            "vlm_url",
        }
    )
    # Two images, two VLM evidence records, both admitted as usable descriptions.
    assert result["described_count"] == 2
    assert result["vlm_processed_count"] == 2
    assert result["admitted_description_count"] == 2
    assert result["rejected_visual_evidence_count"] == 0
    assert result["visual_evidence_admitted_count"] == 2
    assert result["visual_evidence_rejected_count"] == 0
    assert result["eligible_image_count"] == 2
    assert result["skipped_count"] == 0
    assert result["model"] == "qwen2-vl-7b"
    # descriptions_garage_key follows the canonical layout
    assert result["descriptions_garage_key"] == "parsed/doc-1/vlm_descriptions.json"

    # The on-disk artifact is a {image_id: evidence} dict
    raw = fake_garage._objects["parsed/doc-1/vlm_descriptions.json"]
    persisted = json.loads(raw)
    assert set(persisted) == {"img-aaaa", "img-bbbb"}
    # Each evidence payload carries the description + model provenance
    assert persisted["img-aaaa"]["description"].startswith("diagram showing")
    assert persisted["img-aaaa"]["model_used"] == "qwen2-vl-7b"
    assert persisted["img-aaaa"]["image_id"] == "img-aaaa"

    # VLM service was closed
    assert fake_vlm.aclose_called is True


async def test_describe_visual_images_persists_rejected_visual_evidence(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
    figure_store: FigureStore,
) -> None:
    rejecting_vlm = _RejectingVLMService()
    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)
    monkeypatch.setattr(visual_sidecar, "build_vlm_service", lambda: rejecting_vlm)
    monkeypatch.setenv("VO_RAG_VLM_DESCRIBE_CONCURRENCY", "1")

    result = await visual_sidecar.describe_visual_images(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        figures=figure_store,
    )

    assert result["described_count"] == 2
    assert result["vlm_processed_count"] == 2
    assert result["admitted_description_count"] == 0
    assert result["rejected_visual_evidence_count"] == 2
    assert result["visual_evidence_admitted_count"] == 0
    assert result["visual_evidence_rejected_count"] == 2
    assert result["skipped_by_reason"] == {"nontechnical_visual_logo": 2}

    raw = fake_garage._objects["parsed/doc-1/vlm_descriptions.json"]
    persisted = json.loads(raw)
    assert persisted["img-aaaa"]["admitted"] is False
    assert persisted["img-aaaa"]["figure_kind"] == "logo"
    assert persisted["img-aaaa"]["description"] == ""
    assert persisted["img-aaaa"]["rejection_reason"] == "nontechnical_visual_logo"

    composites, skipped = await visual_sidecar.load_visual_composites(
        cast(GarageClient, fake_garage),
        "artifacts",
        _parsed_doc(),
        descriptions_garage_key="parsed/doc-1/vlm_descriptions.json",
    )
    assert composites == []
    assert skipped["nontechnical_visual_logo"] == 2


async def test_describe_visual_images_does_not_close_caller_owned_vlm_service(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
    figure_store: FigureStore,
    fake_vlm: _FakeVLMService,
) -> None:
    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)
    monkeypatch.setenv("VO_RAG_VLM_DESCRIBE_CONCURRENCY", "1")

    result = await visual_sidecar.describe_visual_images(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        figures=figure_store,
        vlm_service=cast(visual_sidecar.VLMServiceLike, fake_vlm),
    )

    assert result["described_count"] == 2
    assert result["vlm_processed_count"] == 2
    assert result["admitted_description_count"] == 2
    assert result["rejected_visual_evidence_count"] == 0
    assert fake_vlm.aclose_called is False


async def test_describe_visual_images_with_zero_images_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
    figure_store: FigureStore,
) -> None:
    """A parsed doc with no images must not open a VLM session. Operators
    expect described_count=0 and no descriptions_garage_key in that case."""

    parsed = _parsed_doc(images=[])
    fake_garage._objects["parsed/doc-0/parsed.json"] = parsed.model_dump_json().encode("utf-8")

    build_called = False

    def _build() -> _FakeVLMService:
        nonlocal build_called
        build_called = True
        return _FakeVLMService()

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)
    monkeypatch.setattr(visual_sidecar, "build_vlm_service", _build)

    result = await visual_sidecar.describe_visual_images(
        envelope_id="doc-0",
        parsed_garage_key="parsed/doc-0/parsed.json",
        figures=figure_store,
    )

    assert result["described_count"] == 0
    assert result["vlm_processed_count"] == 0
    assert result["admitted_description_count"] == 0
    assert result["rejected_visual_evidence_count"] == 0
    assert result["eligible_image_count"] == 0
    assert result["skipped_count"] == 0
    assert result["skipped_by_reason"] == {}
    assert result["visual_evidence_count"] == 0
    assert result["visual_evidence_admitted_count"] == 0
    assert result["visual_evidence_rejected_count"] == 0
    assert result["descriptions_garage_key"] is None
    assert build_called is False


async def test_describe_visual_images_resumes_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
    figure_store: FigureStore,
    fake_vlm: _FakeVLMService,
) -> None:
    """A checkpoint prefix holding image_id evidence should cause the
    sidecar to skip those images and only describe the rest."""

    # Pre-seed a checkpoint entry for img-aaaa
    fake_garage._objects["checkpoints/book_rag/visual-sidecar-doc-1/images/img-aaaa.json"] = (
        json.dumps(
            {
                "image_id": "img-aaaa",
                "description": {
                    "description": "already done",
                    "image_id": "img-aaaa",
                    "model_used": "prior-run",
                    "token_count": 0,
                    "duration_ms": 0,
                    "schema_version": "v0",
                },
            }
        ).encode("utf-8")
    )

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)
    monkeypatch.setattr(visual_sidecar, "build_vlm_service", lambda: fake_vlm)
    monkeypatch.setenv("VO_RAG_VLM_DESCRIBE_CONCURRENCY", "1")

    result = await visual_sidecar.describe_visual_images(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        figures=figure_store,
    )

    # Only the non-checkpointed image was described by the VLM
    assert len(fake_vlm.calls) == 1
    assert fake_vlm.calls[0][2] == "img-bbbb"
    # Both still appear in eligible_image_count and in the persisted artifact
    assert result["eligible_image_count"] == 2
    assert result["described_count"] == 2
    assert result["vlm_processed_count"] == 2
    assert result["admitted_description_count"] == 2
    assert result["rejected_visual_evidence_count"] == 0


# ---------------------------------------------------------------------------
# build_visual_chunks — figure_composite chunk_type
# ---------------------------------------------------------------------------


async def test_build_visual_chunks_emits_figure_composite_chunks(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
) -> None:
    """build_visual_chunks must return a list whose every element is a
    RetrievalChunk with chunk_type="figure_composite" when the parsed doc
    has at least one eligible image and a descriptions artifact exists."""

    descriptions = {
        "img-aaaa": {
            "description": "wing cross section showing pressure vectors",
            "image_id": "img-aaaa",
            "image_role": "embedded_image",
            "image_ref": IMG_A_REF,
            "page_kind": "diagram",
            "needs_burst": True,
            "model_used": "qwen2-vl-7b",
            "token_count": 12,
            "duration_ms": 100,
            "schema_version": "v0",
        },
        "img-bbbb": {
            "description": "drag polar chart with annotations",
            "image_id": "img-bbbb",
            "image_role": "embedded_image",
            "image_ref": IMG_B_REF,
            "page_kind": "chart",
            "needs_burst": True,
            "model_used": "qwen2-vl-7b",
            "token_count": 14,
            "duration_ms": 110,
            "schema_version": "v0",
        },
    }
    fake_garage._objects["parsed/doc-1/vlm_descriptions.json"] = json.dumps(descriptions).encode(
        "utf-8"
    )

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)

    chunks = await visual_sidecar.build_visual_chunks(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        descriptions_garage_key="parsed/doc-1/vlm_descriptions.json",
    )

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    # The contract: the chunker emits at least one figure_composite chunk
    # for the eligible image (the chunker also produces text chunks for
    # the surrounding prose — the contract is that figure_composite
    # chunks are present, not that every chunk is a composite).
    figure_chunks = [c for c in chunks if c.chunk_type == "figure_composite"]
    assert figure_chunks, "expected at least one figure_composite chunk"
    # And those composite chunks carry the eligible image's image_id
    assert any(c.image_id in {"img-aaaa", "img-bbbb"} for c in figure_chunks)


async def test_build_visual_chunks_writes_chunks_artifact(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
) -> None:
    """The function must persist a chunks JSON to Garage so downstream
    book_rag steps (embed_and_upsert) can read it."""

    descriptions = {
        "img-aaaa": {
            "description": "wing cross section",
            "image_id": "img-aaaa",
            "image_role": "embedded_image",
            "image_ref": IMG_A_REF,
            "page_kind": "diagram",
            "needs_burst": True,
            "model_used": "qwen2-vl-7b",
            "token_count": 12,
            "duration_ms": 100,
            "schema_version": "v0",
        },
    }
    fake_garage._objects["parsed/doc-1/vlm_descriptions.json"] = json.dumps(descriptions).encode(
        "utf-8"
    )

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)

    chunks = await visual_sidecar.build_visual_chunks(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        descriptions_garage_key="parsed/doc-1/vlm_descriptions.json",
        variant_label="visual_sidecar_test",
    )

    assert chunks, "expected at least one chunk to be written"
    # The output key follows the variant pattern
    assert "parsed/doc-1/chunks_visual_sidecar_test.json" in fake_garage._objects
    # The on-disk artifact is a list of dicts
    raw = fake_garage._objects["parsed/doc-1/chunks_visual_sidecar_test.json"]
    persisted = json.loads(raw)
    assert isinstance(persisted, list)
    # The figure_composite chunk for the eligible image is persisted
    assert any(c["chunk_type"] == "figure_composite" for c in persisted)


async def test_build_visual_chunks_skips_artifact_text_only_images(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
) -> None:
    """Front-matter / distribution-artifact images are skipped via
    core.chunking.artifact_filters.visual_composite_skip_reason. With
    such descriptions only, the chunker still emits a chunks artifact but
    no figure_composite chunks are produced — and the call must not raise
    the 'zero chunks' guard."""

    descriptions = {
        "img-aaaa": {
            "description": "downloaded from sae copyright all rights reserved",
            "image_id": "img-aaaa",
            "image_role": "embedded_image",
            "image_ref": IMG_A_REF,
            "page_kind": "text",
            "needs_burst": False,
            "model_used": "qwen2-vl-7b",
            "token_count": 0,
            "duration_ms": 0,
            "schema_version": "v0",
        },
    }
    fake_garage._objects["parsed/doc-1/vlm_descriptions.json"] = json.dumps(descriptions).encode(
        "utf-8"
    )

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)

    chunks = await visual_sidecar.build_visual_chunks(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        descriptions_garage_key="parsed/doc-1/vlm_descriptions.json",
    )
    # All composites are skipped — figure_composite count is zero,
    # but text chunks still come through
    assert all(c.chunk_type != "figure_composite" for c in chunks)


async def test_build_visual_chunks_skips_logo_descriptions(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
) -> None:
    descriptions = {
        "img-aaaa": {
            "description": "MVP publisher logo with stylized lettering",
            "image_id": "img-aaaa",
            "image_role": "embedded_image",
            "image_ref": IMG_A_REF,
            "page_kind": "photo",
            "model_used": "qwen2-vl-7b",
            "token_count": 12,
            "duration_ms": 100,
            "schema_version": "v0",
        },
    }
    fake_garage._objects["parsed/doc-1/vlm_descriptions.json"] = json.dumps(descriptions).encode(
        "utf-8"
    )

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)

    chunks = await visual_sidecar.build_visual_chunks(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        descriptions_garage_key="parsed/doc-1/vlm_descriptions.json",
    )

    assert all(c.chunk_type != "figure_composite" for c in chunks)


async def test_vlm_image_jobs_prefilters_front_matter_rights_context(
    figure_store: FigureStore,
) -> None:
    parsed = ParsedDocument(
        document_id="doc-rights",
        source_id="src-rights",
        title="Rights Page",
        total_pages=1,
        elements=[
            _element("Copyright 2026. All rights reserved. No part may be reproduced.", 1),
            _element("", 1, element_type=ElementType.IMAGE),
        ],
        images=[
            ExtractedImage(
                image_id="img-rights",
                page_number=1,
                image_index=0,
                ref=IMG_A_REF,
                element_index=1,
            )
        ],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="pymupdf",
    )

    jobs, eligible_ids, skipped = await visual_sidecar._vlm_image_jobs(
        figure_store,
        parsed,
        existing_descriptions={},
    )

    assert jobs == []
    assert eligible_ids == set()
    assert skipped["front_matter_nontechnical_image"] == 1


async def test_vlm_image_jobs_keeps_collapsed_epub_page_one_images(
    figure_store: FigureStore,
) -> None:
    parsed = ParsedDocument(
        document_id="doc-epub-collapsed",
        source_id="src-epub-collapsed",
        title="Car Suspension",
        total_pages=65,
        elements=[
            _element("Jordan Avery has written more than one technical book.", 1),
            _element("", 1, element_type=ElementType.IMAGE),
        ],
        images=[
            ExtractedImage(
                image_id="img-technical-suspension",
                page_number=1,
                image_index=0,
                ref=IMG_A_REF,
                element_index=1,
            )
        ],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="epub",
    )

    jobs, eligible_ids, skipped = await visual_sidecar._vlm_image_jobs(
        figure_store,
        parsed,
        existing_descriptions={},
    )

    assert len(jobs) == 1
    assert jobs[0].image.image_id == "img-technical-suspension"
    assert eligible_ids == {"img-technical-suspension"}
    assert skipped["front_matter_nontechnical_image"] == 0


async def test_vlm_image_jobs_prefilters_likely_front_cover_before_vlm(tmp_path: Path) -> None:
    store = FigureStore(tmp_path)
    cover_ref = store.write(SRC, 1, 0, _png_with_dimensions(579, 751), ext="png")
    parsed = ParsedDocument(
        document_id="doc-cover",
        source_id="src-cover",
        title="Track Day Orientation Beginners Guide",
        total_pages=14,
        elements=[
            _element("", 1, element_type=ElementType.IMAGE),
            _element("Track Day Orientation Beginners Guide - 101", 1),
        ],
        images=[
            ExtractedImage(
                image_id="img-cover",
                page_number=1,
                image_index=0,
                ref=cover_ref,
                element_index=0,
            )
        ],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="pymupdf",
    )

    jobs, eligible_ids, skipped = await visual_sidecar._vlm_image_jobs(
        store,
        parsed,
        existing_descriptions={},
    )

    assert jobs == []
    assert eligible_ids == set()
    assert skipped["nontechnical_visual_cover"] == 1


async def test_vlm_image_jobs_prefilters_duplicate_image_ids(figure_store: FigureStore) -> None:
    parsed = _parsed_doc(
        images=[
            ExtractedImage(
                image_id="img-same",
                page_number=3,
                image_index=0,
                ref=IMG_A_REF,
                element_index=1,
            ),
            ExtractedImage(
                image_id="img-same",
                page_number=4,
                image_index=1,
                ref=IMG_B_REF,
                element_index=4,
            ),
        ]
    )

    jobs, eligible_ids, skipped = await visual_sidecar._vlm_image_jobs(
        figure_store,
        parsed,
        existing_descriptions={},
    )

    assert len(jobs) == 1
    assert jobs[0].image.image_id == "img-same"
    assert eligible_ids == {"img-same"}
    assert skipped["duplicate_image_id"] == 1


def test_payload_spend_filter_rejects_tiny_images() -> None:
    assert (
        visual_sidecar.native_pdf_image_vlm_spend_rejection_reason(
            image_bytes=_png_with_dimensions(16, 16),
            element_metadata={},
        )
        == "image_too_small"
    )


# ---------------------------------------------------------------------------
# _assert_vlm_healthy — uses ServerlessClient, raises on UNHEALTHY
# ---------------------------------------------------------------------------


def _make_serverless_client() -> ServerlessClient:
    """Build a real ServerlessClient for the test — the sidecar takes the
    concrete type, so we exercise the same chat-completion path the
    production code does. The httpx transport is mocked via respx so no
    socket is opened."""

    return ServerlessClient(
        base_url="https://api.runpod.ai/v2/test-endpoint/openai/v1",
        api_key="test-key",  # pragma: allowlist secret
        model="qwen2-vl-7b",
        timeout_s=10,
        retry_delays=(0.0,),
    )


async def test_assert_vlm_healthy_passes_on_serving_response() -> None:
    """A successful chat-completion probe means the endpoint is serving.
    The function returns None on success and never raises."""

    client = _make_serverless_client()
    try:
        with respx.mock(base_url="https://api.runpod.ai/v2/test-endpoint/openai/v1") as mock:
            route = mock.post("/chat/completions").respond(
                200,
                json={
                    "model": "qwen2-vl-7b",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
            await visual_sidecar._assert_vlm_healthy(client)
        assert route.called
    finally:
        await client.aclose()


async def test_assert_vlm_healthy_raises_on_http_error() -> None:
    """A 5xx on the probe must surface as RuntimeError — the operator needs
    to see the failure path loudly, not as a swallowed warning."""

    client = _make_serverless_client()
    try:
        with respx.mock(base_url="https://api.runpod.ai/v2/test-endpoint/openai/v1") as mock:
            mock.post("/chat/completions").respond(503, json={"error": "unhealthy"})
            with pytest.raises(RuntimeError, match="VLM health check failed"):
                await visual_sidecar._assert_vlm_healthy(client)
    finally:
        await client.aclose()


async def test_assert_vlm_healthy_raises_on_empty_response() -> None:
    """An empty content reply means the worker responded with a 200 but
    nothing useful. The sidecar should treat that as UNHEALTHY."""

    client = _make_serverless_client()
    try:
        with respx.mock(base_url="https://api.runpod.ai/v2/test-endpoint/openai/v1") as mock:
            mock.post("/chat/completions").respond(
                200,
                json={
                    "model": "qwen2-vl-7b",
                    "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
            )
            with pytest.raises(RuntimeError, match="empty"):
                await visual_sidecar._assert_vlm_healthy(client)
    finally:
        await client.aclose()


async def test_assert_vlm_healthy_sends_minimal_probe() -> None:
    """The probe is a single 1-token user message — not a real describe
    call. The sidecar must not waste tokens during a health check."""

    client = _make_serverless_client()
    try:
        with respx.mock(base_url="https://api.runpod.ai/v2/test-endpoint/openai/v1") as mock:
            route = mock.post("/chat/completions").respond(
                200,
                json={
                    "model": "qwen2-vl-7b",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
            await visual_sidecar._assert_vlm_healthy(client)

        assert route.called
        request_payload = route.calls[0].request
        import json as _json

        body = _json.loads(request_payload.content)
        assert body["max_tokens"] <= 4
        assert isinstance(body["messages"], list)
        assert body["messages"][0]["role"] == "user"
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Env var wiring
# ---------------------------------------------------------------------------


async def test_describe_visual_images_reads_vo_rag_vlm_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
    figure_store: FigureStore,
    fake_vlm: _FakeVLMService,
) -> None:
    """The three operator env vars are read with documented defaults
    (concurrency=1, max_tokens=1024, deadline_s=300). They must
    be VO_RAG_* namespaced."""

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)
    monkeypatch.setattr(visual_sidecar, "build_vlm_service", lambda: fake_vlm)
    monkeypatch.setenv("VO_RAG_VLM_DESCRIBE_CONCURRENCY", "2")
    monkeypatch.setenv("VO_RAG_VLM_MAX_TOKENS", "256")
    monkeypatch.setenv("VO_RAG_VLM_PER_IMAGE_DEADLINE_S", "5")

    await visual_sidecar.describe_visual_images(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        figures=figure_store,
    )

    # The describe_image call picked up max_tokens=256
    assert all(call[3] == 256 for call in fake_vlm.calls)
    assert len(fake_vlm.calls) == 2


async def test_vo_rag_vlm_env_var_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fake_garage: _FakeGarage,
    figure_store: FigureStore,
    fake_vlm: _FakeVLMService,
) -> None:
    """With the env vars unset, the defaults are:
    concurrency=1, max_tokens=1024, deadline_s=300."""

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: fake_garage)
    monkeypatch.setattr(visual_sidecar, "build_vlm_service", lambda: fake_vlm)
    for var in (
        "VO_RAG_VLM_DESCRIBE_CONCURRENCY",
        "VO_RAG_VLM_MAX_TOKENS",
        "VO_RAG_VLM_PER_IMAGE_DEADLINE_S",
    ):
        monkeypatch.delenv(var, raising=False)

    await visual_sidecar.describe_visual_images(
        envelope_id="doc-1",
        parsed_garage_key="parsed/doc-1/parsed.json",
        figures=figure_store,
    )

    # All calls used max_tokens=1024 (the documented default)
    assert all(call[3] == 1024 for call in fake_vlm.calls)


# ---------------------------------------------------------------------------
# build_extraction_composites — extraction-only path (VOR-53)
# ---------------------------------------------------------------------------


async def test_build_extraction_composites_produces_composites_without_vlm() -> None:
    """build_extraction_composites must return a CompositeChunk for every image
    in parsed.images with image_ref set and vlm_description empty."""

    parsed = _parsed_doc()
    composites = await visual_sidecar.build_extraction_composites(parsed)

    assert len(composites) == 2
    assert composites[0].image_id == "img-aaaa"
    assert composites[0].image_ref == IMG_A_REF
    assert composites[0].vlm_description == ""
    assert composites[0].chunk_type == "figure_composite"
    assert composites[1].image_id == "img-bbbb"
    assert composites[1].image_ref == IMG_B_REF
    assert composites[1].vlm_description == ""


async def test_build_extraction_composites_empty_when_no_images() -> None:
    """A parsed doc with zero images must produce an empty composites list."""

    parsed = _parsed_doc(images=[])
    composites = await visual_sidecar.build_extraction_composites(parsed)

    assert composites == []


class _StructuredClient:
    """Record structured VLM calls and return configured OpenAI responses."""

    model = "Qwen3VL-8B-Instruct-Q8_0.gguf"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        payload = self.payloads[len(self.calls) - 1]
        return SimpleNamespace(
            content=json.dumps(payload),
            model=self.model,
            output_tokens=42,
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None


def _large_test_png() -> bytes:
    image = Image.new("RGB", (2048, 1024), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


async def test_structured_vlm_service_parses_full_visual_evidence() -> None:
    client = _StructuredClient(
        [
            {
                "description": "Plot of lateral acceleration against time.",
                "page_kind": "chart",
                "visible_text": ["Lateral acceleration", "Time (s)"],
                "tables": [],
                "equations": [],
                "diagrams": ["A red trace rises after 2 s"],
                "confidence": 0.92,
                "needs_burst": False,
                "needs_burst_reasons": [],
            }
        ]
    )
    service = visual_sidecar._ServerlessVLMService(client)  # type: ignore[arg-type]

    result = await service.describe_image(
        _large_test_png(),
        "Parser context is untrusted.",
        "img-structured",
        max_tokens=512,
    )

    assert result.page_kind == "chart"
    assert result.visible_text == ("Lateral acceleration", "Time (s)")
    assert result.confidence == 0.92
    assert result.attempt_count == 1
    assert client.calls[0]["extra"]["json_schema"]["additionalProperties"] is False
    assert set(client.calls[0]["extra"]["json_schema"]["required"]) == {
        "description",
        "page_kind",
        "visible_text",
        "tables",
        "equations",
        "diagrams",
        "confidence",
        "needs_burst",
        "needs_burst_reasons",
        # VOR-1413 admission fields extend the VOR-1409 contract.
        "figure_kind",
        "is_meaningful_technical_figure",
        "rejection_reason",
    }
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert "untrusted" in client.calls[0]["messages"][0]["content"]


def test_structured_vlm_schema_avoids_llama_cpp_unsafe_string_repetition() -> None:
    """Generation schema omits the bound that llama.cpp cannot compile to GBNF."""
    schema = visual_sidecar._structured_vlm_schema()

    assert "maxLength" not in schema["properties"]["description"]
    assert schema["properties"]["description"]["minLength"] == 1


def test_structured_vlm_parser_still_enforces_description_runtime_bound() -> None:
    """Removing the generation hint must not weaken response validation."""
    payload = {
        "description": "x" * 6001,
        "page_kind": "chart",
        "visible_text": [],
        "tables": [],
        "equations": [],
        "diagrams": [],
        "confidence": 0.5,
        "needs_burst": False,
        "needs_burst_reasons": [],
    }

    with pytest.raises(ValueError, match="outside the visual-evidence schema"):
        visual_sidecar._parse_structured_vlm_output(json.dumps(payload))


async def test_structured_vlm_service_retries_low_confidence_at_higher_resolution() -> None:
    client = _StructuredClient(
        [
            {
                "description": "The labels are unclear.",
                "page_kind": "unclear",
                "visible_text": [],
                "tables": [],
                "equations": [],
                "diagrams": [],
                "confidence": 0.4,
                "needs_burst": True,
                "needs_burst_reasons": ["small labels"],
            },
            {
                "description": "Suspension geometry diagram with camber labels.",
                "page_kind": "diagram",
                "visible_text": ["camber"],
                "tables": [],
                "equations": [],
                "diagrams": ["Suspension geometry"],
                "confidence": 0.88,
                "needs_burst": False,
                "needs_burst_reasons": [],
            },
        ]
    )
    service = visual_sidecar._ServerlessVLMService(client)  # type: ignore[arg-type]

    result = await service.describe_image(
        _large_test_png(),
        "Figure context.",
        "img-retry",
        max_tokens=512,
    )

    assert result.attempt_count == 2
    assert result.needs_burst is False
    assert len(client.calls) == 2
    image_sizes: list[tuple[int, int]] = []
    for call in client.calls:
        url = call["messages"][1]["content"][1]["image_url"]["url"]
        encoded = url.split(",", maxsplit=1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            image_sizes.append(image.size)
    assert image_sizes[1][0] * image_sizes[1][1] > image_sizes[0][0] * image_sizes[0][1]


# ---------------------------------------------------------------------------
# Figure admission on the structured VOR-1409 contract (VOR-1413)
# ---------------------------------------------------------------------------


def _structured_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "description": "Suspension geometry diagram with camber labels.",
        "page_kind": "diagram",
        "visible_text": ["camber"],
        "tables": [],
        "equations": [],
        "diagrams": ["Suspension geometry"],
        "confidence": 0.9,
        "needs_burst": False,
        "needs_burst_reasons": [],
    }
    payload.update(overrides)
    return payload


def _evidence_for(response: visual_sidecar.VLMDescription) -> visual_sidecar.VisualEvidence:
    return visual_sidecar.visual_evidence_from_vlm(
        response,
        image=_parsed_doc().images[0],
        image_role="embedded_image",
    )


def test_structured_vlm_schema_requires_admission_fields() -> None:
    schema = visual_sidecar._structured_vlm_schema()
    properties = schema["properties"]

    assert {"figure_kind", "is_meaningful_technical_figure", "rejection_reason"} <= set(
        schema["required"]
    )
    assert properties["figure_kind"]["type"] == "string"
    assert set(properties["figure_kind"]["enum"]) == {
        "diagram",
        "chart",
        "table",
        "technical_photo",
        "logo",
        "cover",
        "portrait",
        "decorative",
        "unknown",
    }
    assert properties["is_meaningful_technical_figure"] == {
        "title": "Is Meaningful Technical Figure",
        "type": "boolean",
    }
    # rejection_reason stays nullable (null when admitted) but has no default
    # and no GBNF-unsafe string bound.
    assert "default" not in properties["rejection_reason"]
    assert {"type": "null"} in properties["rejection_reason"]["anyOf"]
    assert all("maxLength" not in branch for branch in properties["rejection_reason"]["anyOf"])


async def test_structured_vlm_service_carries_admission_fields_and_rejects_cover() -> None:
    client = _StructuredClient(
        [
            _structured_payload(
                description="Book cover with title typography.",
                page_kind="photo",
                visible_text=["APPLIED CHASSIS DYNAMICS"],
                diagrams=[],
                figure_kind="cover",
                is_meaningful_technical_figure=False,
                rejection_reason="This is a book cover, not a technical figure.",
            )
        ]
    )
    service = visual_sidecar._ServerlessVLMService(client)  # type: ignore[arg-type]

    result = await service.describe_image(
        _large_test_png(), "Parser context.", "img-cover", max_tokens=512
    )

    assert result.figure_kind == "cover"
    assert result.is_meaningful_technical_figure is False
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert "figure_kind" in client.calls[0]["messages"][0]["content"]

    evidence = _evidence_for(result)
    assert evidence.admitted is False
    assert evidence.figure_kind == "cover"
    assert evidence.is_meaningful_technical_figure is False
    assert evidence.description == ""
    assert evidence.rejection_reason == "nontechnical_visual_cover"
    # VOR-1409 fields survive alongside the admission fields.
    assert evidence.page_kind == "photo"
    assert evidence.visible_text == ("APPLIED CHASSIS DYNAMICS",)
    assert evidence.confidence == 0.9
    assert json.loads(evidence.raw_response)["figure_kind"] == "cover"
    dumped = evidence.model_dump()
    assert dumped["schema_version"] == "vo_rag_visual_evidence_v2"
    assert dumped["admitted"] is False
    assert dumped["needs_burst_reasons"] == []


async def test_structured_vlm_service_admits_meaningful_technical_figure() -> None:
    client = _StructuredClient(
        [
            _structured_payload(
                figure_kind="diagram",
                is_meaningful_technical_figure=True,
                rejection_reason=None,
            )
        ]
    )
    service = visual_sidecar._ServerlessVLMService(client)  # type: ignore[arg-type]

    evidence = _evidence_for(
        await service.describe_image(_large_test_png(), "ctx", "img-diag", max_tokens=512)
    )

    assert evidence.admitted is True
    assert evidence.figure_kind == "diagram"
    assert evidence.rejection_reason is None
    assert evidence.description == "Suspension geometry diagram with camber labels."


@pytest.mark.parametrize(
    ("page_kind", "expected_kind"),
    [
        ("diagram", "diagram"),
        ("chart", "chart"),
        ("table", "table"),
        ("photo", "unknown"),
        ("equation_heavy", "unknown"),
        ("mixed", "unknown"),
        ("text", "unknown"),
        ("unclear", "unknown"),
    ],
)
def test_missing_admission_fields_derive_kind_from_page_kind_and_admit(
    page_kind: str, expected_kind: str
) -> None:
    """Responses without admission fields keep the pre-admission behavior."""
    parsed = visual_sidecar._parse_structured_vlm_output(
        json.dumps(_structured_payload(page_kind=page_kind))
    )
    assert parsed.figure_kind is None

    evidence = _evidence_for(
        visual_sidecar.VLMDescription(
            description=parsed.description,
            model_used="m",
            token_count=1,
            duration_ms=1,
            image_id="img-aaaa",
            page_kind=parsed.page_kind,
        )
    )

    assert evidence.admitted is True
    assert evidence.figure_kind == expected_kind
    assert evidence.rejection_reason is None


def test_structured_parser_rejects_unknown_figure_kind() -> None:
    with pytest.raises(ValueError, match="outside the visual-evidence schema"):
        visual_sidecar._parse_structured_vlm_output(
            json.dumps(_structured_payload(figure_kind="clipart"))
        )


def test_classifier_admitted_description_keeps_label_uncertainty() -> None:
    """The prompt asks the model to say when labels are unreadable; that must
    not trip the whole-image decode-failure filter."""
    evidence = _evidence_for(
        visual_sidecar.VLMDescription(
            description="Brake bias diagram; the small axis labels are unreadable.",
            model_used="m",
            token_count=1,
            duration_ms=1,
            image_id="img-aaaa",
            page_kind="diagram",
            figure_kind="diagram",
            is_meaningful_technical_figure=True,
        )
    )
    assert evidence.admitted is True


def test_classifier_admitted_engine_front_cover_plate_is_not_a_book_cover() -> None:
    evidence = _evidence_for(
        visual_sidecar.VLMDescription(
            description="Exploded diagram of the engine front cover plate and timing chain.",
            model_used="m",
            token_count=1,
            duration_ms=1,
            image_id="img-aaaa",
            page_kind="diagram",
            figure_kind="diagram",
            is_meaningful_technical_figure=True,
        )
    )
    assert evidence.admitted is True
    assert evidence.figure_kind == "diagram"


def test_self_declaration_overrides_classifier_admission() -> None:
    evidence = _evidence_for(
        visual_sidecar.VLMDescription(
            description="This image is a book cover with a racing car photograph.",
            model_used="m",
            token_count=1,
            duration_ms=1,
            image_id="img-aaaa",
            page_kind="photo",
            figure_kind="technical_photo",
            is_meaningful_technical_figure=True,
        )
    )
    assert evidence.admitted is False
    assert evidence.figure_kind == "cover"
    assert evidence.rejection_reason == "vlm_self_declaration_nonfigure"
    assert evidence.description == ""


def test_free_text_self_declaration_rejects_legacy_description() -> None:
    evidence = _evidence_for(
        visual_sidecar.VLMDescription(
            description="The front cover of the book, showing the title and a car.",
            model_used="m",
            token_count=1,
            duration_ms=1,
            image_id="img-aaaa",
        )
    )
    assert evidence.admitted is False
    assert evidence.figure_kind == "cover"
    assert evidence.rejection_reason == "vlm_self_declaration_nonfigure"


def test_decode_failure_declaration_is_rejected() -> None:
    assert (
        visual_sidecar.vlm_self_declaration_reject_reason(
            "I cannot decode this image; it appears corrupted."
        )
        == "vlm_self_declaration_nonfigure"
    )
    assert (
        visual_sidecar.vlm_self_declaration_reject_reason(
            "Wiring diagram; some pin labels are unreadable."
        )
        is None
    )


def test_meaningful_kind_with_empty_description_is_rejected() -> None:
    evidence = _evidence_for(
        visual_sidecar.VLMDescription(
            description=json.dumps(
                {
                    "figure_kind": "chart",
                    "is_meaningful_technical_figure": True,
                    "description": "",
                    "rejection_reason": None,
                }
            ),
            model_used="m",
            token_count=1,
            duration_ms=1,
            image_id="img-aaaa",
        )
    )
    assert evidence.admitted is False
    assert evidence.rejection_reason == "meaningful_visual_description_empty"


def test_v1_artifact_without_admission_fields_hydrates_as_admitted() -> None:
    image = _parsed_doc().images[0]
    evidence = visual_sidecar.visual_evidence_from_artifact(
        image.image_id,
        {
            "description": "Chart of tyre temperature against lap number.",
            "page_kind": "chart",
            "confidence": 0.8,
            "schema_version": "vo_rag_visual_evidence_v1",
        },
        fallback_image=image,
    )
    assert evidence is not None
    assert evidence.admitted is True
    assert evidence.figure_kind == "chart"
    assert evidence.schema_version == "vo_rag_visual_evidence_v1"


def test_rejected_artifact_hydrates_and_is_skipped_at_composite_time() -> None:
    image = _parsed_doc().images[0]
    evidence = visual_sidecar.visual_evidence_from_artifact(
        image.image_id,
        {
            "description": "",
            "page_kind": "photo",
            "figure_kind": "portrait",
            "is_meaningful_technical_figure": False,
            "admitted": False,
            "rejection_reason": "nontechnical_visual_portrait",
            "schema_version": "vo_rag_visual_evidence_v2",
        },
        fallback_image=image,
    )
    assert evidence is not None
    assert evidence.admitted is False
    assert evidence.figure_kind == "portrait"
    assert (
        visual_composite_skip_reason(
            page_number=image.page_number,
            description=evidence.description,
            context_text="",
            total_pages=2,
            figure_kind=evidence.figure_kind,
            admitted=evidence.admitted,
            rejection_reason=evidence.rejection_reason,
        )
        == "nontechnical_visual_portrait"
    )


async def test_vlm_image_jobs_keeps_tall_collapsed_epub_figures_out_of_cover_gate(
    tmp_path: Path,
) -> None:
    """Collapsed EPUBs put every image on page 1; the page-position cover
    heuristic must not reject their tall technical figures before the VLM."""
    store = FigureStore(tmp_path)
    tall_ref = store.write(SRC, 1, 0, _png_with_dimensions(579, 751), ext="png")
    tiny_ref = store.write(SRC, 1, 1, _png_with_dimensions(16, 16), ext="png")
    parsed = ParsedDocument(
        document_id="doc-epub-tall",
        source_id="src-epub-tall",
        title="Car Suspension",
        total_pages=65,
        elements=[
            _element("", 1, element_type=ElementType.IMAGE),
            _element("Double wishbone layout with the upright and spring.", 1),
            _element("", 1, element_type=ElementType.IMAGE),
        ],
        images=[
            ExtractedImage(
                image_id="img-tall-suspension",
                page_number=1,
                image_index=0,
                ref=tall_ref,
                element_index=0,
            ),
            ExtractedImage(
                image_id="img-tiny-icon",
                page_number=1,
                image_index=1,
                ref=tiny_ref,
                element_index=2,
            ),
        ],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="epub",
    )

    jobs, eligible_ids, skipped = await visual_sidecar._vlm_image_jobs(
        store,
        parsed,
        existing_descriptions={},
    )

    assert [job.image.image_id for job in jobs] == ["img-tall-suspension"]
    assert eligible_ids == {"img-tall-suspension"}
    assert skipped["nontechnical_visual_cover"] == 0
    assert skipped["image_too_small"] == 1


# ---------------------------------------------------------------------------
# VOR-1429: per-document running-line scan, not per image
# ---------------------------------------------------------------------------


async def test_extraction_composites_scan_running_lines_once_per_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The running-line scan walks every element; doing it per image made a
    5,001-image book take 3.2 h in this step and starve the dispatcher."""
    from vo_rag.core.chunking.section_packer import SectionPacker

    calls: list[int] = []
    real = SectionPacker.repeated_short_page_text

    def counting(elements: Any, page_count: int) -> set[str]:
        calls.append(len(elements))
        return real(elements, page_count)

    monkeypatch.setattr(SectionPacker, "repeated_short_page_text", staticmethod(counting))
    monkeypatch.setattr(visual_sidecar, "_REPEATED_SHORT_TEXT_CACHE", None)
    images = [
        ExtractedImage(
            image_id=f"img-{n:04d}",
            page_number=1 + n % 2,
            image_index=n,
            ref=IMG_A_REF,
            element_index=n % 5,
        )
        for n in range(40)
    ]
    parsed = _parsed_doc(images=images)

    composites = await visual_sidecar.build_extraction_composites(parsed)

    assert len(composites) == 40
    assert calls == [5]  # one scan for the document, not one per image

    other = _parsed_doc(images=images[:3])  # a different document recomputes
    await visual_sidecar.build_extraction_composites(other)
    assert calls == [5, 5]


def test_running_line_cache_matches_uncached_result() -> None:
    parsed = _parsed_doc()
    visual_sidecar._REPEATED_SHORT_TEXT_CACHE = None
    first = visual_sidecar._repeated_short_page_text(parsed.elements, parsed.total_pages)
    again = visual_sidecar._repeated_short_page_text(parsed.elements, parsed.total_pages)
    from vo_rag.core.chunking.section_packer import SectionPacker

    assert first is again
    assert first == SectionPacker.repeated_short_page_text(parsed.elements, parsed.total_pages)


def test_sidecar_bucket_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.pipelines.book_rag import visual_sidecar

    monkeypatch.setenv("VO_RAG_GARAGE_BUCKET", "custom-bucket")
    assert visual_sidecar._bucket() == "custom-bucket"


def test_sidecar_bucket_defaults_to_the_settings_artifacts_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.core import config as config_module
    from vo_rag.pipelines.book_rag import visual_sidecar

    monkeypatch.delenv("VO_RAG_GARAGE_BUCKET", raising=False)
    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: SimpleNamespace(
            garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="from-settings"))
        ),
    )
    assert visual_sidecar._bucket() == "from-settings"


def test_sidecar_bucket_falls_back_to_vorag_artifacts_without_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.core import config as config_module
    from vo_rag.pipelines.book_rag import visual_sidecar

    def boom() -> Any:
        raise RuntimeError("no settings")

    monkeypatch.delenv("VO_RAG_GARAGE_BUCKET", raising=False)
    monkeypatch.setattr(config_module, "get_settings", boom)
    assert visual_sidecar._bucket() == "vorag-artifacts"
