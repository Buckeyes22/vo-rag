"""A figure chunk's neighbour window survives storage (VOR-1452).

A figure chunk's ``content`` is its own text only; its ``embedding_text`` is the
wider window (neighbour prose included). The raw payload stores that window
for figures and the exact input for every other raw point. API hit payloads
strip the raw embedding-only field; rebuilds and sterile embeddings reuse it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from vo_rag.core.chunking.models import RetrievalChunk, hit_payload
from vo_rag.core.compliance.sterilizer import sterilize_chunks
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.storage.points import VoPoint
from vo_rag.pipelines.book_rag.steps.chunk_embed import do_upsert_raw

WINDOW = (
    "[Book > Throws]\n\nCaption: Figure 4.3 The glide.\n\n"
    "Preceding: The athlete starts low at the back of the circle.\n\n"
    "Following: The hips drive the chest round before the arm strikes."
)
FIGURE_CONTENT = "Caption: Figure 4.3 The glide."


def _figure() -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="f-1",
        chunk_type="figure_composite",
        document_id="d",
        source_id="s",
        page_number=4,
        section_path="[Book > Throws]",
        section_title="Throws",
        content=FIGURE_CONTENT,
        embedding_text=WINDOW,
        token_count=40,
        has_image=True,
        image_id="img-1",
        image_ref="s/img-1.png",
    )


def _text() -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="t-1",
        chunk_type="text",
        document_id="d",
        source_id="s",
        page_number=4,
        section_path="[Book > Throws]",
        section_title="Throws",
        content="The glide is taught in three stages.",
        embedding_text="[Book > Throws]\n\nThe glide is taught in three stages.",
        token_count=10,
    )


class _Store:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[VoPoint]]] = []

    async def upsert_vectors(self, collection: str, points: list[VoPoint]) -> None:
        self.upserts.append((collection, points))


async def _payloads(chunks: list[RetrievalChunk]) -> list[dict[str, Any]]:
    store = _Store()
    embeddings = {
        "dense": [[0.1] * 1024 for _ in chunks],
        "sparse": [{1: 0.5} for _ in chunks],
        "colbert": [[[0.1] * 1024] for _ in chunks],
    }
    await do_upsert_raw(chunks, embeddings, store, "s")
    return [point.payload for point in store.upserts[0][1]]


# --- raw payload ---------------------------------------------------------------


async def test_raw_payload_stores_exact_embedding_input_for_every_chunk() -> None:
    figure_payload, text_payload = await _payloads([_figure(), _text()])

    assert figure_payload["embedding_text"] == WINDOW
    assert figure_payload["content"] == FIGURE_CONTENT
    assert text_payload["embedding_text"] == (
        "[Book > Throws]\n\nThe glide is taught in three stages."
    )


async def test_raw_hit_payload_hides_persisted_embedding_input() -> None:
    figure_payload, text_payload = await _payloads([_figure(), _text()])

    assert "embedding_text" not in hit_payload(figure_payload)
    assert "embedding_text" not in hit_payload(text_payload)
    assert hit_payload(text_payload)["content"] == text_payload["content"]


async def test_raw_payload_skips_a_figure_window_equal_to_its_content() -> None:
    figure = _figure().model_copy(update={"embedding_text": FIGURE_CONTENT})
    [payload] = await _payloads([figure])
    assert "embedding_text" not in payload


# --- rebuild from Qdrant (RAPTOR leaf embeddings) --------------------------------


class _ScrollClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads

    async def scroll(self, **_: Any) -> tuple[list[SimpleNamespace], None]:
        points = [SimpleNamespace(id=f"p{i}", payload=p) for i, p in enumerate(self.payloads)]
        return points, None


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str], **_: Any) -> dict[str, Any]:
        self.texts.extend(texts)
        return {"dense": [[0.0, 1.0] for _ in texts]}


async def test_cli_rebuild_embeds_the_figure_window_and_text_as_before() -> None:
    from vo_rag.cli.build import _embed_leaf_embeddings, _load_chunks_for_source

    payloads = await _payloads([_figure(), _text()])
    store: Any = SimpleNamespace(_client=_ScrollClient(payloads))
    chunks = await _load_chunks_for_source(store, "s", "raw")
    embedder = _RecordingEmbedder()
    await _embed_leaf_embeddings(chunks, embedder)

    assert chunks[0].content == FIGURE_CONTENT
    assert embedder.texts == [
        WINDOW,
        "[Book > Throws]\n\nThe glide is taught in three stages.",
    ]


async def test_build_workflow_rebuild_embeds_the_figure_window_and_text_as_before() -> None:
    pytest.importorskip("llama_index.core")  # the adapter needs the llamaindex extra
    from vo_rag.core.storage.qdrant import QdrantStore
    from vo_rag.integrations.llamaindex.build_workflow import AdvancedBuildWorkflow

    payloads = await _payloads([_figure(), _text()])
    embedder = _RecordingEmbedder()
    llm = AsyncMock(spec=LLMClient)
    workflow = AdvancedBuildWorkflow(
        store=cast(QdrantStore, SimpleNamespace(_client=_ScrollClient(payloads))),
        embedding_client=embedder,
        llm_clients=cast(dict[str, LLMClient], {"text": llm, "dsi": llm}),
        settings=object(),
    )
    chunks = await workflow._load_chunks("s")
    await workflow._embed_leaf_embeddings(chunks)

    assert chunks[0].content == FIGURE_CONTENT
    assert embedder.texts == [
        WINDOW,
        "[Book > Throws]\n\nThe glide is taught in three stages.",
    ]


# --- sterile path --------------------------------------------------------------


class _Sterile(BaseModel):
    sterile_content: str
    facts: list[str] = []
    equations: list[str] = []
    concepts: list[str] = []
    injection_suspected: bool = False
    injection_reason: str | None = None


def _fake_complete(calls: list[str]) -> Any:
    async def _fake(
        client: LLMClient,
        *,
        schema: type[Any],
        untrusted_content: str,
        **_: Any,
    ) -> tuple[Any, LLMResponse, ValidationResult, bool]:
        calls.append(untrusted_content)
        if "Preceding:" in untrusted_content:
            text = "Sterile window: athlete low in circle; hips rotate before release."
        else:
            text = f"Sterile summary {len(calls)}."
        payload = _Sterile(sterile_content=text)
        resp = LLMResponse(
            text=json.dumps(payload.model_dump()),
            model="fake-model",
            input_tokens=1,
            output_tokens=1,
            provider="fake",
        )
        return (
            schema.model_validate(payload.model_dump()),
            resp,
            ValidationResult(ok=True, signals=[]),
            False,
        )

    return _fake


class _Provider:
    name = "fake"
    model = "fake-model"

    async def complete_raw(self, **_: Any) -> LLMResponse:  # pragma: no cover
        raise AssertionError("patched")

    async def aclose(self) -> None:
        return None


async def test_sterile_figure_embeds_a_sterilized_neighbour_window() -> None:
    calls: list[str] = []
    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _fake_complete(calls)):
        figure, text = await sterilize_chunks([_figure(), _text()], LLMClient(provider=_Provider()))

    # Figure: content sterilized from its own text, embedding from the window.
    assert figure.content == "Sterile summary 1."
    assert figure.embedding_text == (
        "[Book > Throws]\n\nSterile window: athlete low in circle; hips rotate before release."
    )
    # The raw window never reaches the sterile chunk (payload = model_dump).
    assert "back of the circle" not in json.dumps(figure.model_dump())
    # Text: one call, embedding as before.
    assert text.embedding_text == f"[Book > Throws]\n\n{text.content}"
    assert len(calls) == 3
    assert "Preceding:" in calls[1] and "[Book > Throws]\n\nCaption" not in calls[1]


async def test_sterile_figure_falls_back_to_its_content_when_the_window_fails() -> None:
    calls: list[str] = []
    inner = _fake_complete(calls)

    async def _failing_window(client: LLMClient, **kwargs: Any) -> Any:
        if "Preceding:" in kwargs["untrusted_content"]:
            calls.append(kwargs["untrusted_content"])
            payload = _Sterile(sterile_content="")
            resp = LLMResponse(
                text="{}", model="m", input_tokens=1, output_tokens=1, provider="fake"
            )
            return (
                _Sterile.model_validate(payload.model_dump()),
                resp,
                ValidationResult(ok=True, signals=[]),
                False,
            )
        return await inner(client, **kwargs)

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _failing_window):
        [figure] = await sterilize_chunks([_figure()], LLMClient(provider=_Provider()))

    assert figure.embedding_text == f"[Book > Throws]\n\n{figure.content}"


@pytest.mark.parametrize("chunk_type", ["text", "table"])
async def test_sterile_non_figure_makes_one_call(chunk_type: str) -> None:
    calls: list[str] = []
    chunk = _text().model_copy(update={"chunk_type": chunk_type})
    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _fake_complete(calls)):
        await sterilize_chunks([chunk], LLMClient(provider=_Provider()))
    assert len(calls) == 1


async def test_sterile_figure_carries_the_window_calls_quarantine_verdict() -> None:
    """The window is untrusted neighbour prose embedded as the
    sterile ``embedding_text``, but its protected call's quarantine verdict was
    discarded. Like the content call's (the text-chunk precedent), it lands in
    ``metadata["_sterilization_quarantine"]``: either call wanting quarantine
    quarantines the chunk."""
    calls: list[str] = []
    inner = _fake_complete(calls)

    async def _quarantine_window(client: LLMClient, **kwargs: Any) -> Any:
        parsed, resp, validation, quarantine = await inner(client, **kwargs)
        return parsed, resp, validation, quarantine or "Preceding:" in kwargs["untrusted_content"]

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _quarantine_window):
        figure, text = await sterilize_chunks([_figure(), _text()], LLMClient(provider=_Provider()))

    assert figure.metadata["_sterilization_quarantine"] is True
    assert text.metadata["_sterilization_quarantine"] is False

    calls.clear()
    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _fake_complete(calls)):
        [clean] = await sterilize_chunks([_figure()], LLMClient(provider=_Provider()))
    assert clean.metadata["_sterilization_quarantine"] is False
