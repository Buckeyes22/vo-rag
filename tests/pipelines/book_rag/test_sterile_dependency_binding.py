"""Regression checks for sterile representations bound to raw-book pointers."""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.ingest.source_publication import PublicationState, SourcePublicationJob
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.response import LLMResponse
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.steps.sterilize import do_sterilize


class _Provider:
    name = "fake"
    model = "fake-model"

    async def complete_raw(self, **_: Any) -> LLMResponse:
        return LLMResponse(
            text='{"sterile_content":"plain text"}',
            model=self.model,
            input_tokens=1,
            output_tokens=1,
            provider=self.name,
        )

    async def aclose(self) -> None:
        return None


class _Store:
    source_publication_enabled = True

    def __init__(self) -> None:
        self.write: tuple[str, list[Any], dict[str, Any]] | None = None

    async def upsert_vectors(self, collection: str, points: list[Any], **kwargs: Any) -> list[str]:
        self.write = (collection, points, kwargs)
        return ["published-point"]


def _chunk() -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="chunk-1",
        chunk_type="text",
        document_id="doc-1",
        source_id="src-1",
        section_path="1",
        section_title="Section",
        chapter_path="Chapter",
        content="source text",
        embedding_text="source text",
        token_count=2,
    )


@pytest.mark.asyncio
async def test_sterile_writer_stamps_and_forwards_captured_raw_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag.steps import sterilize as module

    async def keep_chunks(chunks: list[RetrievalChunk], _: LLMClient) -> list[RetrievalChunk]:
        return chunks

    monkeypatch.setattr(module, "sterilize_chunks", keep_chunks)
    source_id = "vo-rag-corpus-sha256:" + "a" * 64
    chunk = _chunk().model_copy(update={"source_id": source_id})
    raw_revision = 12
    dependencies = {source_id: raw_revision}
    job = SourcePublicationJob(
        job_id="sterile-job",
        corpus="books",
        collection=module.STERILE_COLLECTION,
        source_id=source_id,
        revision=1,
        expected_active_revision=None,
        source_sha256="a" * 64,
        state=PublicationState.PREPARING,
        checkpoint={},
        retention_disposition=None,
        recipe={"expected_dependencies": dependencies},
    )
    store = _Store()

    async def embed(texts: list[str]) -> dict[str, Any]:
        return {
            "dense": [[0.25] * 1024 for _ in texts],
            "sparse": [{"indices": [1], "values": [1.0]} for _ in texts],
            "colbert": [[[0.25] * 1024] for _ in texts],
        }

    await do_sterilize(
        raw_chunks=[chunk],
        llm_client=LLMClient(provider=_Provider()),
        config=BookRagConfig(sterilize_enabled=True),
        store=store,  # type: ignore[arg-type]
        embed=embed,
        publication_job=job,
        expected_dependencies=dependencies,
    )

    assert store.write is not None
    collection, points, kwargs = store.write
    assert collection == module.STERILE_COLLECTION
    assert points[0].payload["dependency_revisions"] == dependencies
    assert kwargs == {"publication_job": job, "expected_dependencies": dependencies}
