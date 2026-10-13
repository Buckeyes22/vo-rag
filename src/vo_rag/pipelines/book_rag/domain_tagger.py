"""Domain tagging for book_rag retrieval chunks.

The LLM client is an injected `vo_rag.core.llm.client.LLMClient`. The
vocabulary / keyword pack is overridable at construction so operators can
plug a non-motorsport pack without forking the leaf.

The LLM call goes through `complete_json_protected` (F-012 injection
trio) so the chunk text — which is attacker-controllable in the
operator-supplied backfill path — never lands in the system prompt and
a canary leak is conclusive evidence the model was hijacked.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.errors import StructuredOutputError
from vo_rag.core.llm.protected import complete_json_protected
from vo_rag.pipelines.book_rag.domain_vocab import DOMAIN_KEYWORDS, DOMAIN_VOCABULARY


class _DomainTagsResult(BaseModel):
    """Pydantic schema for the LLM's tag response. Empty list is a valid
    response — the LLM may legitimately have no confidence in any tag."""

    tags: list[str] = Field(default_factory=list)


def heuristic_domain_tags(text: str, *, max_tags: int = 6) -> list[str]:
    """Fast controlled-vocabulary tags for large ingestion batches.

    Counts substring matches in lowercased text and returns the top
    `max_tags` by (descending score, ascending vocabulary order). Ties
    are broken by the vocabulary's declaration order so the same input
    always produces the same output (idempotent across runs).
    """
    normalized = text.lower()
    scored: list[tuple[int, int, str]] = []
    for order, tag in enumerate(DOMAIN_VOCABULARY):
        keywords = DOMAIN_KEYWORDS.get(tag, ())
        score = sum(normalized.count(keyword) for keyword in keywords)
        if score:
            scored.append((-score, order, tag))
    scored.sort()
    return [tag for _, _, tag in scored[:max_tags]]


class DomainTagger:
    """Classify chunk text into the (overridable) controlled domain vocabulary.

    The LLM path is best-effort: a parse failure, structured-output
    retry exhaustion, or a validation quarantine surfaces as `[]`
    rather than raising. The heuristic path is independent — it is the
    fallback the orchestrator uses on LLM failure.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        vocabulary: list[str] | None = None,
        keywords: dict[str, tuple[str, ...]] | None = None,
        max_tags_per_chunk: int = 6,
        concurrency: int = 8,
    ) -> None:
        self._llm_client = llm_client
        self._vocabulary: list[str] = (
            list(vocabulary) if vocabulary is not None else list(DOMAIN_VOCABULARY)
        )
        self._keywords: dict[str, tuple[str, ...]] = (
            dict(keywords) if keywords is not None else dict(DOMAIN_KEYWORDS)
        )
        self.max_tags = max_tags_per_chunk
        self._sem = asyncio.Semaphore(concurrency)

    @property
    def vocabulary(self) -> list[str]:
        return list(self._vocabulary)

    def heuristic_tags(self, text: str, *, max_tags: int = 6) -> list[str]:
        """Heuristic path with the operator-overridden pack (if any)."""
        normalized = text.lower()
        scored: list[tuple[int, int, str]] = []
        for order, tag in enumerate(self._vocabulary):
            keywords = self._keywords.get(tag, ())
            score = sum(normalized.count(keyword) for keyword in keywords)
            if score:
                scored.append((-score, order, tag))
        scored.sort()
        return [tag for _, _, tag in scored[:max_tags]]

    async def tag(self, chunk: RetrievalChunk) -> list[str]:
        prompt = (
            f"Tag the following chunk with up to {self.max_tags} domain tags.\n"
            f"You MUST pick only from this vocabulary: {self._vocabulary}\n"
            'Return JSON: {"tags": ["tag1", "tag2"]}\n\n'
            f"Chunk:\n{chunk.embedding_text[:2000]}\n"
        )
        try:
            parsed, _resp, _validation, _quarantine = await complete_json_protected(
                self._llm_client,
                schema=_DomainTagsResult,
                untrusted_content=chunk.embedding_text[:2000],
                pipeline="book_rag",
                allowed_domains=[],
                system=(
                    "You classify technical text into a controlled vocabulary. "
                    "You will see a chunk wrapped in <document> tags. Return "
                    "ONLY JSON matching the schema."
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
        except (ValueError, TypeError, KeyError, AttributeError, StructuredOutputError):
            return []

        raw_tags: list[Any] = parsed.tags
        return [tag for tag in raw_tags[: self.max_tags] if tag in self._vocabulary]

    async def tag_batch(self, chunks: list[RetrievalChunk]) -> dict[str, list[str]]:
        async def one(chunk: RetrievalChunk) -> tuple[str, list[str]]:
            async with self._sem:
                return chunk.chunk_id, await self.tag(chunk)

        pairs = await asyncio.gather(*(one(chunk) for chunk in chunks), return_exceptions=False)
        return dict(pairs)


__all__ = [
    "DOMAIN_KEYWORDS",
    "DOMAIN_VOCABULARY",
    "DomainTagger",
    "heuristic_domain_tags",
]
