"""Summary tree builder for Layer 1A — per-book, per-section -> per-chapter -> per-book.

Implements the Ou & Lapata Extract-Support pattern at each recursive level:

  - Section: raw chunks -> section summary (primary) + extractive snippets (support)
  - Chapter: section summaries -> chapter summary (primary) + RAW chunk snippets (support)
  - Book: chapter summaries -> book summary (primary) + RAW chunk snippets (support)

Invariant: supporting evidence always comes from RAW chunks, never from
already-computed summary text.

Notes:
- ``BuildConfig.embedding_version`` and ``source_collection_version`` default
  to explicit values (``bge-m3@unversioned`` / ``vo_rag_books_raw_v1``) that
  match the canonical constants, never a placeholder.
- LLM calls retry on ``RETRY_DELAYS = (10.0, 30.0, 90.0)`` with
  ``enable_thinking=False``; ``_strip_reasoning`` and JSON truncation salvage
  clean up the response.
- All LLM calls go through the injected ``llm_client.complete`` callable which
  returns ``LLMResponse``; no raw provider construction inside summary.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.chunking.chapter_path import derive_chapter_path
from vo_rag.core.embedding.point_vectors import (
    fallback_colbert_vectors,
    include_colbert_at_ingest,
)
from vo_rag.core.ingest.representation_receipt import canonical_sha256, representation_receipt
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.storage.derived_publication import (
    DerivedSourceBinding,
    abort_derived_source,
    admit_derived_source,
    publish_derived_source,
)
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter
from vo_rag.core.summary.extract_support import (
    build_extract_support_prompt,
    select_evidence_snippets,
)
from vo_rag.core.summary.persistence import (
    SUMMARY_TREE_COLLECTION,
    ensure_summary_tree_collection,
    get_summary_node,
    summary_node_point,
    upsert_summary_node,
)
from vo_rag.core.summary.prompts import PROMPT_HASHES
from vo_rag.core.summary.schemas import NodeLevel, SummaryNode, node_id_for

log = structlog.get_logger(__name__)


@dataclass
class _SummaryPublicationContext:
    binding: DerivedSourceBinding
    collecting: bool = True
    points: list[Any] | None = None


_SUMMARY_PUBLICATION_CONTEXT: contextvars.ContextVar[_SummaryPublicationContext | None] = (
    contextvars.ContextVar("summary_publication_context", default=None)
)

SUMMARY_PROMPT_VERSIONS = {
    NodeLevel.SECTION: "l1a_section_extract_support_v1",
    NodeLevel.CHAPTER: "l1a_chapter_extract_support_v1",
    NodeLevel.BOOK: "l1a_book_extract_support_v1",
}

# Output caps must fit the full JSON schema (summary_text + key_points +
# figure_dependent_points + technical_terms + caveats + used_evidence_span_ids +
# unsupported_or_uncertain), not just the prose. At 350 the trailing arrays were
# truncated -> unparseable JSON (the _parse_summary_response salvage still recovers
# clean summary_text, but the structured extract is lost). These caps fit a
# typical section's full object; salvage covers the rare overflow.
MAX_TOKENS_BY_LEVEL = {
    NodeLevel.SECTION: 1024,
    NodeLevel.CHAPTER: 1536,
    NodeLevel.BOOK: 2560,
}

TIMEOUT_SECONDS_BY_LEVEL = {
    NodeLevel.SECTION: 120,
    NodeLevel.CHAPTER: 180,
    NodeLevel.BOOK: 240,
}

RETRY_DELAYS = (10.0, 30.0, 90.0)

# Qwen3 hybrid-reasoning models leak chain-of-thought into message.content unless
# thinking is disabled. Layer 1A summaries are extractive and never need reasoning,
# so we disable it per-request — this also yields ~10x fewer output tokens for the
# same answer. The RunPod serverless provider forwards this into the OpenAI request
# body; llama.cpp/Anthropic/Kimi providers ignore it (see _strip_reasoning backstop).
SUMMARY_LLM_EXTRA: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

_THINK_CLOSE = "</think>"


# Matches the (complete) "summary_text": "..." field even inside truncated JSON.
_SUMMARY_TEXT_RE = re.compile(r'"summary_text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _parse_summary_response(raw: str) -> tuple[str, dict[str, Any]]:
    """Extract clean summary text from the LLM's JSON response.

    The Layer-1A prompts ask for a JSON object
    ``{summary_text, key_points, figure_dependent_points, technical_terms,
    caveats, used_evidence_span_ids, unsupported_or_uncertain}``. We store
    ``summary_text`` as the node's text (so embeddings see clean prose, not JSON)
    and stash the remaining structured fields under ``metadata.summary_extract``.

    Falls back to the raw (stripped) response with a ``summary_parse_fallback``
    marker if the response is not valid JSON, isn't an object, lacks
    ``summary_text``, or yields an empty summary — so an occasional malformed
    response degrades gracefully instead of crashing the build.
    """
    text = raw.strip()
    if text.startswith("```"):  # tolerate ```json ... ``` fences
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # 1) Full parse — keeps the structured extract (key_points, technical_terms, …).
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            summary = str(parsed.get("summary_text") or "").strip()
            if summary:
                extract = {k: v for k, v in parsed.items() if k != "summary_text"}
                return summary, ({"summary_extract": extract} if extract else {})
    except (json.JSONDecodeError, ValueError):
        pass
    # 2) Salvage summary_text from truncated/malformed JSON. It is the first field
    #    in the schema, so it survives even when trailing arrays are cut at
    #    max_tokens (the common case). We lose the structured extract but keep
    #    clean prose — never the raw JSON blob.
    m = _SUMMARY_TEXT_RE.search(text)
    if m:
        try:
            summary = json.loads('"' + m.group(1) + '"').strip()
        except (json.JSONDecodeError, ValueError):
            summary = m.group(1).strip()
        if summary:
            log.warning("summary_response_salvaged", reason="truncated_or_malformed_json")
            return summary, {"summary_parse_fallback": "salvaged_summary_text"}
    # 3) Unrecoverable — store raw, clearly flagged.
    log.warning("summary_response_unparseable", head=text[:80])
    return raw.strip(), {"summary_parse_fallback": "unparseable"}


def _strip_reasoning(text: str) -> str:
    """Defensively drop any leaked reasoning preamble before the real answer.

    The Qwen3 chat template injects the opening ``<think>`` tag into the prompt, so
    leaked content carries the trailing ``</think>`` followed by the answer. The
    ``enable_thinking=False`` flag prevents leakage at the source; this is a backstop
    for providers/models that ignore the flag. No-op when no closing tag is present.
    """
    if _THINK_CLOSE in text:
        return text.rsplit(_THINK_CLOSE, 1)[-1].strip()
    return text


@dataclass
class BuildConfig:
    summary_model: str = "qwen3.6-27b-fp8"
    embedding_model: str = "BAAI/bge-m3"
    # Defaults point at the canonical vo-rag names (bge-m3@unversioned
    # matches vo_rag.core.embedding.cache; vo_rag_books_raw_v1 matches
    # vo_rag.core.storage.qdrant.COLLECTION_NAMES) so persisted points survive
    # a fresh-cluster migration; placeholder versions are never used.
    embedding_version: str = "bge-m3@unversioned"
    source_collection_version: str = "vo_rag_books_raw_v1"
    raw_collection: str = "vo_rag_books_raw_v1"
    summary_collection: str = SUMMARY_TREE_COLLECTION
    section_max_tokens: int = 800
    chapter_max_tokens: int = 1200
    book_max_tokens: int = 2000
    section_evidence_k: int = 6
    chapter_evidence_k: int = 10
    book_evidence_k: int = 15
    summary_temperature: float = 0.2
    max_concurrent_llm_calls: int = 8
    # Balanced-tree fan-out controls (VOR — summary leaf-grouping fix). Measured
    # on the migrated corpus, grouping leaves by the full ``section_path`` gives
    # 1.52 chunks/section (a near-1:1 explosion → ~22k useless single-chunk
    # summaries), while grouping by ``chapter_path`` overflows on outliers (one
    # book has a 281-chunk chapter). So leaves are packed to a bounded size and
    # every upper tier to a bounded fan-out instead:
    #   - leaf_min_chunks: a run of tiny chapters coalesces UP to at least this.
    #   - leaf_max_chunks: an oversized chapter splits DOWN to at most this
    #     (caps the raw text handed to one section-summary LLM call).
    #   - max_fanout: an internal/book node aggregates at most this many child
    #     summaries; tiers are inserted until the top fits under it.
    leaf_min_chunks: int = 3
    leaf_max_chunks: int = 12
    max_fanout: int = 10


class SummaryTreeBuilder:
    def __init__(
        self,
        *,
        llm_client: Any,
        embed_client: Any,
        store: QdrantStore,
        config: BuildConfig,
    ) -> None:
        self.llm = llm_client
        self.embed = embed_client
        self.store = store
        self.config = config

    async def build_book_tree(
        self,
        *,
        book_id: str,
        chunks: list[RetrievalChunk],
        expected_dependencies: dict[str, int | None] | None = None,
    ) -> list[SummaryNode]:
        """Build and publish one complete, raw-revision-bound book tree."""
        if not getattr(
            self.store,
            "source_visibility_enabled",
            getattr(self.store, "_publication_enabled", False),
        ):
            return await self._build_book_tree_unpublished(book_id=book_id, chunks=chunks)

        body = [c for c in chunks if not (c.metadata or {}).get("front_matter_kind")]
        if not body:
            return []
        if expected_dependencies is None:
            raise ValueError(
                "publication-enabled summary builds require raw dependency revisions captured "
                "before loading chunks"
            )
        chunk_manifest = canonical_sha256(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "section_path": chunk.section_path,
                    "chapter_path": getattr(chunk, "chapter_path", ""),
                    "metadata": chunk.metadata or {},
                }
                for chunk in sorted(body, key=lambda item: (item.page_number or 0, item.chunk_id))
            ]
        )
        binding = await admit_derived_source(
            self.store,
            collection=self.config.summary_collection,
            source_id=book_id,
            dependency_sources=[book_id],
            expected_dependencies=expected_dependencies,
            producer="summary-tree-v1",
            input_manifest={"book_id": book_id, "chunk_manifest_sha256": chunk_manifest},
            producer_recipe={
                "summary_model": self.config.summary_model,
                "embedding_model": self.config.embedding_model,
                "embedding_version": self.config.embedding_version,
                "prompt_versions": {
                    level.value: version for level, version in SUMMARY_PROMPT_VERSIONS.items()
                },
                "source_collection_version": self.config.source_collection_version,
                "leaf_min_chunks": self.config.leaf_min_chunks,
                "leaf_max_chunks": self.config.leaf_max_chunks,
                "max_fanout": self.config.max_fanout,
            },
        )
        if binding is None:
            if not getattr(self.store, "source_publication_enabled", False):
                return await self._build_book_tree_unpublished(book_id=book_id, chunks=chunks)
            raise RuntimeError("summary publication admission unexpectedly disabled")
        current = await binding.coordinator.read_active(binding.collection, book_id)
        durable_job = await binding.coordinator.read_job(binding.job.job_id)
        if durable_job is not None and durable_job.state.value == "published":
            if current is None or current.revision != durable_job.revision:
                raise RuntimeError("published summary build was superseded by a newer revision")
            from vo_rag.core.summary.persistence import get_book_tree

            return await get_book_tree(book_id, store=self.store, collection=binding.collection)

        try:
            before = await SourceRevisionWriter(self.store._raw_client, binding.coordinator).census(
                binding.collection, book_id, current.revision if current is not None else None
            )
        except BaseException:
            await abort_derived_source(self.store, binding, "summary before-census failed")
            raise
        before_receipts = {str(row.id): representation_receipt(row) for row in before}
        context = _SummaryPublicationContext(binding=binding, points=[])
        token = _SUMMARY_PUBLICATION_CONTEXT.set(context)
        try:
            nodes = await self._build_book_tree_unpublished(book_id=book_id, chunks=chunks)
            context.collecting = False
            for node in nodes:
                await self._embed_and_persist(node)
            await publish_derived_source(
                self.store,
                binding,
                context.points or [],
                expected_before_receipts=before_receipts,
            )
            return nodes
        except BaseException:
            await abort_derived_source(
                self.store, binding, "summary build failed before publication"
            )
            raise
        finally:
            _SUMMARY_PUBLICATION_CONTEXT.reset(token)

    async def _build_book_tree_unpublished(
        self, *, book_id: str, chunks: list[RetrievalChunk]
    ) -> list[SummaryNode]:
        """Build a balanced summary tree for one book and persist to Qdrant.

        The tree is built bottom-up with a **bounded fan-out at every tier**, so
        it stays well-formed across the whole range of real books (a 1-chunk
        pamphlet to a 3,374-chunk / 437-chapter manual):

          1. LEAVES — pack chunks (reading order) into SECTION nodes of
             ``leaf_min..leaf_max`` chunks, soft-breaking at ``chapter_path``
             boundaries. This fixes BOTH chunk-path degeneracies measured on the
             corpus: fine ``section_path``s / tiny chapters merge UP to a useful
             size (no more 1.52-chunks-per-section 1:1 explosion), and oversized
             chapters split DOWN to ``leaf_max`` (no 281-chunk leaf that would
             blow the summariser's context window).
          2. INTERNAL TIERS — while a tier has more than ``max_fanout`` nodes,
             pack it into parent CHAPTER nodes of ``<= max_fanout`` children and
             recurse. Trailing singletons are merged up, so no parent ever
             re-summarises exactly one child.
          3. BOOK — a single root aggregates the (``<= max_fanout``) top tier.

        Every per-node LLM call therefore sees a bounded input: a leaf sees
        ``<= leaf_max`` raw chunks; an internal/book node sees ``<= max_fanout``
        child summaries.

        Returns all SummaryNode objects (sections + any internal tiers + book).

        Idempotency: per-node deterministic node_id over the SORTED source chunk
        ids. Packing is deterministic (reading order), so re-runs reproduce the
        same node ids and skip already-built nodes.
        """
        # Front/back-matter never becomes its own summary node.
        body = [c for c in chunks if not (c.metadata or {}).get("front_matter_kind")]
        if not body:
            return []

        reading_order = sorted(body, key=lambda c: (c.page_number or 0, c.chunk_id))
        chapter_of = {
            c.chunk_id: (getattr(c, "chapter_path", "") or _extract_chapter_path(c.section_path))
            for c in reading_order
        }
        by_id = {c.chunk_id: c for c in reading_order}
        doc_title = _doc_title(chapter_of, book_id)

        sem = asyncio.Semaphore(self.config.max_concurrent_llm_calls)

        async def _leaf(group: list[RetrievalChunk]) -> SummaryNode:
            async with sem:
                return await self._build_section_node(
                    book_id=book_id,
                    section_path=_leaf_label(group, chapter_of, doc_title),
                    section_chunks=group,
                )

        async def _internal(group: list[SummaryNode]) -> SummaryNode:
            underlying = [
                by_id[cid] for node in group for cid in node.source_chunk_ids if cid in by_id
            ]
            async with sem:
                return await self._build_chapter_node(
                    book_id=book_id,
                    chapter_path=_leaf_label(underlying, chapter_of, doc_title),
                    section_nodes=group,
                    underlying_chunks=underlying,
                )

        # 1. Leaves.
        leaf_groups = _pack_leaves(
            reading_order,
            chapter_of,
            leaf_min=self.config.leaf_min_chunks,
            leaf_max=self.config.leaf_max_chunks,
        )
        tier: list[SummaryNode] = list(await asyncio.gather(*(_leaf(g) for g in leaf_groups)))
        all_nodes: list[SummaryNode] = list(tier)

        # 2. Internal tiers, built tier by tier (each tier depends on the prior).
        while len(tier) > self.config.max_fanout:
            parent_groups = _pack_nodes(tier, self.config.max_fanout)
            tier = list(await asyncio.gather(*(_internal(g) for g in parent_groups)))
            all_nodes.extend(tier)

        # 3. Book root aggregates the (<= max_fanout) top tier.
        book_node = await self._build_book_node(
            book_id=book_id,
            chapter_nodes=tier,
            all_underlying_chunks=reading_order,
        )
        all_nodes.append(book_node)

        return all_nodes

    async def _build_section_node(
        self,
        *,
        book_id: str,
        section_path: str,
        section_chunks: list[RetrievalChunk],
    ) -> SummaryNode:
        sorted_chunks = sorted(section_chunks, key=lambda c: (c.page_number or 0, c.chunk_id))
        source_chunk_ids = {c.chunk_id for c in sorted_chunks}

        prompt_hash = _prompt_hash_for(NodeLevel.SECTION)
        node_id = node_id_for(
            book_id=book_id,
            level=NodeLevel.SECTION,
            source_chunk_ids=sorted(source_chunk_ids),
            prompt_hash=prompt_hash,
        )

        existing = await self._maybe_load_existing(node_id)
        if existing is not None:
            log.info("summary_node_skipped_existing", node_id=node_id, level="section")
            return existing

        evidence_spans = select_evidence_snippets(
            sorted_chunks,
            topic_or_query=section_path,
            k=self.config.section_evidence_k,
        )

        # Section-level Extract-Support: primary_input is the raw chunk content
        # itself; supporting_evidence is a top-k snippet selection over the same
        # chunks. Both halves come from raw chunks (Ou & Lapata invariant).
        system, user = build_extract_support_prompt(
            level=NodeLevel.SECTION,
            primary_input="\n\n".join(c.content for c in sorted_chunks),
            supporting_evidence=evidence_spans,
            display_path=section_path,
        )

        t0 = time.perf_counter()
        resp = await self._call_llm_with_retry(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS_BY_LEVEL[NodeLevel.SECTION],
            temperature=self.config.summary_temperature,
            node_id=node_id,
            level=NodeLevel.SECTION,
            book_id=book_id,
        )
        wall_time_ms = int((time.perf_counter() - t0) * 1000)

        log.info(
            "summary_build_token_account",
            node_id=node_id,
            level="section",
            book_id=book_id,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            wall_time_ms=wall_time_ms,
        )

        summary_text, summary_meta = _parse_summary_response(resp.text)

        summary_node = SummaryNode(
            node_id=node_id,
            book_id=book_id,
            level=NodeLevel.SECTION,
            node_type="section_summary",
            parent_node_id=None,
            child_node_ids=[],
            canonical_path=section_path,
            display_path=section_path,
            source_chunk_ids=sorted(source_chunk_ids),
            source_section_paths=[section_path],
            evidence_spans=evidence_spans,
            summary_text=summary_text,
            summary_model=self.config.summary_model,
            prompt_hash=prompt_hash,
            prompt_version=SUMMARY_PROMPT_VERSIONS[NodeLevel.SECTION],
            embedding_model=self.config.embedding_model,
            embedding_version=self.config.embedding_version,
            created_at=datetime.now(UTC),
            source_collection_version=self.config.source_collection_version,
            metadata={
                "input_token_count": resp.input_tokens,
                "output_token_count": resp.output_tokens,
                "wall_time_ms": wall_time_ms,
                "retry_count": 0,
                **summary_meta,
            },
        )

        await self._embed_and_persist(summary_node)

        return summary_node

    async def _build_chapter_node(
        self,
        *,
        book_id: str,
        chapter_path: str,
        section_nodes: list[SummaryNode],
        underlying_chunks: list[RetrievalChunk],
    ) -> SummaryNode:
        source_chunk_ids: set[str] = set()
        for sn in section_nodes:
            source_chunk_ids.update(sn.source_chunk_ids)

        prompt_hash = _prompt_hash_for(NodeLevel.CHAPTER)
        node_id = node_id_for(
            book_id=book_id,
            level=NodeLevel.CHAPTER,
            source_chunk_ids=sorted(source_chunk_ids),
            prompt_hash=prompt_hash,
        )

        existing = await self._maybe_load_existing(node_id)
        if existing is not None:
            log.info("summary_node_skipped_existing", node_id=node_id, level="chapter")
            return existing

        sorted_chunks = sorted(underlying_chunks, key=lambda c: (c.page_number or 0, c.chunk_id))
        evidence_spans = select_evidence_snippets(
            sorted_chunks,
            topic_or_query=chapter_path,
            k=self.config.chapter_evidence_k,
        )

        ordered_summaries = [
            sn.summary_text for sn in sorted(section_nodes, key=lambda n: n.canonical_path)
        ]

        # Ou & Lapata recursion: primary_input is the child section summaries;
        # supporting_evidence comes from underlying RAW chunks (already selected
        # above via select_evidence_snippets), never from section summaries.
        system, user = build_extract_support_prompt(
            level=NodeLevel.CHAPTER,
            primary_input="\n\n".join(ordered_summaries),
            supporting_evidence=evidence_spans,
            display_path=chapter_path,
        )

        t0 = time.perf_counter()
        resp = await self._call_llm_with_retry(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS_BY_LEVEL[NodeLevel.CHAPTER],
            temperature=self.config.summary_temperature,
            node_id=node_id,
            level=NodeLevel.CHAPTER,
            book_id=book_id,
        )
        wall_time_ms = int((time.perf_counter() - t0) * 1000)

        log.info(
            "summary_build_token_account",
            node_id=node_id,
            level="chapter",
            book_id=book_id,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            wall_time_ms=wall_time_ms,
        )

        summary_text, summary_meta = _parse_summary_response(resp.text)

        summary_node = SummaryNode(
            node_id=node_id,
            book_id=book_id,
            level=NodeLevel.CHAPTER,
            node_type="chapter_summary",
            parent_node_id=None,
            child_node_ids=[n.node_id for n in section_nodes],
            canonical_path=chapter_path,
            display_path=chapter_path,
            source_chunk_ids=sorted(source_chunk_ids),
            source_section_paths=list({sn.display_path for sn in section_nodes}),
            evidence_spans=evidence_spans,
            summary_text=summary_text,
            summary_model=self.config.summary_model,
            prompt_hash=prompt_hash,
            prompt_version=SUMMARY_PROMPT_VERSIONS[NodeLevel.CHAPTER],
            embedding_model=self.config.embedding_model,
            embedding_version=self.config.embedding_version,
            created_at=datetime.now(UTC),
            source_collection_version=self.config.source_collection_version,
            metadata={
                "input_token_count": resp.input_tokens,
                "output_token_count": resp.output_tokens,
                "wall_time_ms": wall_time_ms,
                "retry_count": 0,
                "child_section_node_ids": [n.node_id for n in section_nodes],
                **summary_meta,
            },
        )

        await self._embed_and_persist(summary_node)

        return summary_node

    async def _build_book_node(
        self,
        *,
        book_id: str,
        chapter_nodes: list[SummaryNode],
        all_underlying_chunks: list[RetrievalChunk],
    ) -> SummaryNode:
        source_chunk_ids: set[str] = set()
        for cn in chapter_nodes:
            source_chunk_ids.update(cn.source_chunk_ids)

        prompt_hash = _prompt_hash_for(NodeLevel.BOOK)
        node_id = node_id_for(
            book_id=book_id,
            level=NodeLevel.BOOK,
            source_chunk_ids=sorted(source_chunk_ids),
            prompt_hash=prompt_hash,
        )

        existing = await self._maybe_load_existing(node_id)
        if existing is not None:
            log.info("summary_node_skipped_existing", node_id=node_id, level="book")
            return existing

        sorted_chunks = sorted(
            all_underlying_chunks, key=lambda c: (c.page_number or 0, c.chunk_id)
        )
        evidence_spans = select_evidence_snippets(
            sorted_chunks,
            topic_or_query=f"book {book_id}",
            k=self.config.book_evidence_k,
        )

        ordered_summaries = [
            cn.summary_text for cn in sorted(chapter_nodes, key=lambda n: n.canonical_path)
        ]

        # Ou & Lapata recursion: primary_input is the child chapter summaries;
        # supporting_evidence (passed below) comes from RAW chunks via
        # select_evidence_snippets, never from already-summarised text.
        system, user = build_extract_support_prompt(
            level=NodeLevel.BOOK,
            primary_input="\n\n".join(ordered_summaries),
            supporting_evidence=evidence_spans,
            display_path=f"Book {book_id}",
        )

        t0 = time.perf_counter()
        resp = await self._call_llm_with_retry(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS_BY_LEVEL[NodeLevel.BOOK],
            temperature=self.config.summary_temperature,
            node_id=node_id,
            level=NodeLevel.BOOK,
            book_id=book_id,
        )
        wall_time_ms = int((time.perf_counter() - t0) * 1000)

        log.info(
            "summary_build_token_account",
            node_id=node_id,
            level="book",
            book_id=book_id,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            wall_time_ms=wall_time_ms,
        )

        summary_text, summary_meta = _parse_summary_response(resp.text)

        summary_node = SummaryNode(
            node_id=node_id,
            book_id=book_id,
            level=NodeLevel.BOOK,
            node_type="book_summary",
            parent_node_id=None,
            child_node_ids=[n.node_id for n in chapter_nodes],
            canonical_path=f"book/{book_id}",
            display_path=f"Book {book_id}",
            source_chunk_ids=sorted(source_chunk_ids),
            source_section_paths=list({cn.display_path for cn in chapter_nodes}),
            evidence_spans=evidence_spans,
            summary_text=summary_text,
            summary_model=self.config.summary_model,
            prompt_hash=prompt_hash,
            prompt_version=SUMMARY_PROMPT_VERSIONS[NodeLevel.BOOK],
            embedding_model=self.config.embedding_model,
            embedding_version=self.config.embedding_version,
            created_at=datetime.now(UTC),
            source_collection_version=self.config.source_collection_version,
            metadata={
                "input_token_count": resp.input_tokens,
                "output_token_count": resp.output_tokens,
                "wall_time_ms": wall_time_ms,
                "retry_count": 0,
                "child_chapter_node_ids": [n.node_id for n in chapter_nodes],
                **summary_meta,
            },
        )

        await self._embed_and_persist(summary_node)

        return summary_node

    async def _call_llm_with_retry(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        node_id: str,
        level: NodeLevel,
        book_id: str,
    ) -> LLMResponse:
        last_error: Exception | None = None
        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                resp = await self.llm.complete(
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra=SUMMARY_LLM_EXTRA,
                )
                return cast(
                    LLMResponse,
                    resp.model_copy(update={"text": _strip_reasoning(resp.text)}),
                )
            except Exception as exc:
                last_error = exc
                log.warning(
                    "summary_llm_retry",
                    node_id=node_id,
                    level=level.value if level else level,
                    book_id=book_id,
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(exc),
                )
                if attempt < len(RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        log.error(
            "summary_llm_failed",
            node_id=node_id,
            level=level.value if level else level,
            book_id=book_id,
            error=str(last_error),
        )
        raise RuntimeError(
            f"LLM call failed for node {node_id} after {len(RETRY_DELAYS)} attempts: {last_error}"
        ) from last_error

    async def _embed_and_persist(self, node: SummaryNode) -> None:
        publication_context = _SUMMARY_PUBLICATION_CONTEXT.get()
        if publication_context is not None and publication_context.collecting:
            return
        embedding_text = f"{node.display_path}\n\n{node.summary_text}"

        emb = await self.embed.embed(
            texts=[embedding_text],
            return_dense=True,
            return_sparse=True,
            return_colbert=include_colbert_at_ingest(),  # opt-in; fallback below
        )

        dense = emb["dense"][0]
        sparse_raw = emb["sparse"][0]
        if isinstance(sparse_raw, dict):
            sparse = {
                int(i): v for i, v in zip(sparse_raw["indices"], sparse_raw["values"], strict=True)
            }
        else:
            sparse = {int(i): v for i, v in zip(sparse_raw.indices, sparse_raw.values, strict=True)}
        colbert = emb["colbert"][0] if emb.get("colbert") else []
        if not colbert:
            colbert = fallback_colbert_vectors(dense)  # schema needs ≥1 colbert vector

        if publication_context is not None:
            if publication_context.points is None:
                raise RuntimeError("derived source output buffer is unavailable")
            publication_context.points.append(
                summary_node_point(
                    node,
                    dense_embedding=dense,
                    sparse_embedding=sparse,
                    colbert_embedding=colbert,
                )
            )
            return
        await upsert_summary_node(
            node,
            dense_embedding=dense,
            sparse_embedding=sparse,
            colbert_embedding=colbert,
            store=self.store,
            collection=self.config.summary_collection,
        )

    async def _maybe_load_existing(self, node_id: str) -> SummaryNode | None:
        try:
            return await get_summary_node(
                node_id,
                store=self.store,
                collection=self.config.summary_collection,
            )
        except Exception:
            return None


def _extract_chapter_path(canonical_path: str) -> str:
    """Coarse chapter from a ``[Title > H1 > ...]`` section path.

    Legacy fallback for chunks persisted before the ingestion-owned
    ``chapter_path`` field existed. Delegates to ``derive_chapter_path`` so
    legacy and new chunks group into chapters identically. (The previous
    ``split("/")`` never matched the ``>``-delimited path, so every section
    became its own chapter — the 1:1 flat-tree defect.)
    """
    s = canonical_path.strip()
    inner = s[1:-1] if s.startswith("[") and s.endswith("]") else s
    title = inner.split(" > ", 1)[0].strip()
    return derive_chapter_path(canonical_path, title)


def _pack_leaves(
    chunks: list[RetrievalChunk],
    chapter_of: dict[str, str],
    *,
    leaf_min: int,
    leaf_max: int,
) -> list[list[RetrievalChunk]]:
    """Pack chunks (already in reading order) into leaf groups of bounded size.

    Per arriving chunk, flush the current group when it has reached ``leaf_max``
    (hard split of an oversized chapter) OR it has reached ``leaf_min`` AND the
    arriving chunk starts a new chapter. The two thresholds are the whole point:
    a normally-sized chapter (``>= leaf_min``) becomes its own leaf at its
    boundary, while a run of tiny chapters coalesces up to ``leaf_min`` before
    breaking — so neither the 1:1 explosion nor the giant-chapter overflow can
    happen.
    """
    groups: list[list[RetrievalChunk]] = []
    cur: list[RetrievalChunk] = []
    for c in chunks:
        crosses_chapter = bool(cur) and chapter_of[c.chunk_id] != chapter_of[cur[-1].chunk_id]
        if cur and (len(cur) >= leaf_max or (crosses_chapter and len(cur) >= leaf_min)):
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    return groups


def _pack_nodes(nodes: list[SummaryNode], fanout: int) -> list[list[SummaryNode]]:
    """Split nodes into consecutive groups, each of size in ``[2, fanout]``.

    Groups are ``fanout``-sized slices in reading order, except a trailing
    singleton is merged into the previous group so that no parent node is ever
    built over exactly one child (a wasted re-summary, and a node_id that would
    collide with that child's chunk-set at a different level). That merge is the
    one case a group can reach ``fanout + 1``.
    """
    groups = [nodes[i : i + fanout] for i in range(0, len(nodes), fanout)]
    if len(groups) >= 2 and len(groups[-1]) == 1:
        groups[-2].extend(groups.pop())
    return groups


def _path_title(path: str) -> str:
    """First segment ("doc title") of a ``[Title > H1 > ...]`` bracketed path."""
    s = path.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return s.split(" > ", 1)[0].strip()


def _path_inner(path: str, doc_title: str) -> str:
    """The heading portion of a bracketed path, minus the leading doc title."""
    s = path.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    parts = [p.strip() for p in s.split(" > ")]
    if parts and parts[0] == doc_title:
        parts = parts[1:]
    return " > ".join(p for p in parts if p)


def _doc_title(chapter_of: dict[str, str], book_id: str) -> str:
    for path in chapter_of.values():
        title = _path_title(path)
        if title:
            return title
    return f"Book {book_id}"


def _page_span(chunks: list[RetrievalChunk]) -> str:
    pages = [c.page_number for c in chunks if c.page_number]
    if not pages:
        return ""
    lo, hi = min(pages), max(pages)
    return f"pp. {lo}" if lo == hi else f"pp. {lo}-{hi}"


def _leaf_label(
    chunks: list[RetrievalChunk],
    chapter_of: dict[str, str],
    doc_title: str,
) -> str:
    """A human + locator label for a packed group: chapter span + page range.

    e.g. ``[SCCA Manual > Section 2: Preparation (pp. 12-16)]`` for one chapter,
    or ``[SCCA Manual > Section 2 … Section 4 (pp. 12-28)]`` when a run of small
    chapters coalesced. The label feeds the node's ``display_path`` / embedding
    text and the evidence-selection topic, so it carries real semantics, not just
    a node index.
    """
    chapters: list[str] = []
    for c in chunks:
        cp = chapter_of[c.chunk_id]
        if cp not in chapters:
            chapters.append(cp)
    if len(chapters) == 1:
        semantic = _path_inner(chapters[0], doc_title)
    else:
        semantic = f"{_path_inner(chapters[0], doc_title)} … {_path_inner(chapters[-1], doc_title)}"
    span = _page_span(chunks)
    pieces = [p for p in (semantic, f"({span})" if span else "") if p]
    inner = " ".join(pieces)
    return f"[{doc_title} > {inner}]" if inner else f"[{doc_title}]"


def _prompt_hash_for(level: NodeLevel) -> str:
    return PROMPT_HASHES.get(level, f"unknown_{level.value}")


# ---------------------------------------------------------------------------
# Corpus-level builder
# ---------------------------------------------------------------------------


class SummaryCorpusStats(VoBaseModel):
    """Per-book outcome record emitted by ``build_summary_for_corpus``."""

    book_id: str
    node_count: int
    wall_time_ms: int


async def build_summary_for_corpus(
    *,
    books: list[tuple[str, list[RetrievalChunk]]],
    builder: SummaryTreeBuilder,
    max_concurrent_books: int,
    expected_dependencies: Mapping[str, int | None] | None = None,
) -> tuple[list[SummaryNode], list[SummaryCorpusStats]]:
    """Build summary trees for a corpus of books with bounded concurrency.

    Each element of ``books`` is ``(book_id, chunks)``.

    Processes up to ``max_concurrent_books`` books in parallel via an
    ``asyncio.Semaphore``. Calls ``ensure_summary_tree_collection`` once up
    front.

    Returns ``(list_of_all_nodes, list_of_per_book_stats)``.
    """
    await ensure_summary_tree_collection(
        builder.store._client,
        collection=builder.config.summary_collection,
    )

    semaphore = asyncio.Semaphore(max_concurrent_books)
    all_nodes: list[SummaryNode] = []
    all_stats: list[SummaryCorpusStats] = []

    async def process_book(book_id: str, chunks: list[RetrievalChunk]) -> None:
        async with semaphore:
            t0 = time.monotonic()
            dependency_token = (
                {book_id: expected_dependencies[book_id]}
                if expected_dependencies is not None and book_id in expected_dependencies
                else None
            )
            nodes = await builder.build_book_tree(
                book_id=book_id,
                chunks=chunks,
                expected_dependencies=dependency_token,
            )
            wall_time_ms = int((time.monotonic() - t0) * 1000)
            all_nodes.extend(nodes)
            all_stats.append(
                SummaryCorpusStats(
                    book_id=book_id,
                    node_count=len(nodes),
                    wall_time_ms=wall_time_ms,
                )
            )

    await asyncio.gather(*(process_book(book_id, chunks) for book_id, chunks in books))

    log.info(
        "summary.build_corpus.complete",
        book_count=len(books),
        total_nodes=len(all_nodes),
        stage="build_summary_for_corpus",
    )
    return all_nodes, all_stats


__all__ = [
    "BuildConfig",
    "MAX_TOKENS_BY_LEVEL",
    "RETRY_DELAYS",
    "SUMMARY_LLM_EXTRA",
    "SUMMARY_PROMPT_VERSIONS",
    "SummaryCorpusStats",
    "SummaryTreeBuilder",
    "TIMEOUT_SECONDS_BY_LEVEL",
    "_extract_chapter_path",
    "_leaf_label",
    "_pack_leaves",
    "_pack_nodes",
    "_parse_summary_response",
    "_prompt_hash_for",
    "_strip_reasoning",
    "build_summary_for_corpus",
]
