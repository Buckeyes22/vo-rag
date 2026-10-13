"""Advanced build workflow for DSI, graph, summary, and RAPTOR layers.

This workflow is the ADAPTER-TIER (post-ingest) orchestration for agent/adapter
surfaces that depend on the optional ``llamaindex`` extra.  It is NOT the
operator CLI's build engine — the CLI in ``src/vo_rag/cli/build.py`` runs the
core builders directly to remain import-safe when the extra is absent.

The ``_build_dsi``, ``_build_summary``, ``_build_raptor``, and ``_extract_graph``
bodies below mirror the CLI implementations.  The shared core builders
(``vo_rag.core.dsi/graph/summary/raptor``) remain the single source of truth.

The graph branch ships the STOP/GO direction gate, and ``load_corpus`` recovers
book metadata from persisted chunks before fanning out the requested build
layers.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, Literal, cast, get_args

from pydantic import BaseModel, Field, field_validator

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.chunking.models import payload_embedding_text
from vo_rag.core.dsi.builder import (
    _DSI_COLLECTION_DEFAULT,
    DsiBuildConfig,
    admit_dsi_source_revision,
    build_dsi_for_book,
    ensure_dsi_collection,
    persist_dsi_records,
)
from vo_rag.core.dsi.extraction import ExtractionConfig
from vo_rag.core.dsi.prompts import DSI_EXTRACTION_PROMPT_HASH
from vo_rag.core.embedding.point_vectors import include_colbert_at_ingest
from vo_rag.core.graph.builder import (
    GraphBuildConfig,
    GraphBuildStats,
    build_graph_for_book,
    persist_graph,
)
from vo_rag.core.graph.direction import orient_low_confidence_via_llm
from vo_rag.core.graph.extraction import GraphExtractionConfig, graph_response_json_schema
from vo_rag.core.graph.persistence import (
    GRAPH_EDGES_COLLECTION,
    GRAPH_NODES_COLLECTION,
    GraphQdrantClient,
    ensure_graph_collections,
)
from vo_rag.core.graph.prompts import GRAPH_EXTRACTION_PROMPT_HASH
from vo_rag.core.graph.schemas import GraphEdge, GraphNode
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.guided import structured_output_extra
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.raptor.builder import RaptorBuildConfig, build_raptor_for_corpus
from vo_rag.core.raptor.persistence import RAPTOR_COLLECTION
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.summary.builder import BuildConfig, SummaryTreeBuilder, build_summary_for_corpus
from vo_rag.core.summary.persistence import SUMMARY_TREE_COLLECTION
from vo_rag.integrations.llamaindex import require_llamaindex

try:
    require_llamaindex()
    from llama_index.core.workflow import (
        Context as LlamaIndexContext,
    )
    from llama_index.core.workflow import (
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )
except ImportError:
    raise


type BuildLayer = Literal["dsi", "summary", "raptor", "graph"]
SUPPORTED_BUILD_LAYERS: Final[frozenset[str]] = frozenset(
    cast(tuple[str, ...], get_args(BuildLayer.__value__))
)


def _validate_build_layers(raw_layers: Any) -> list[BuildLayer]:
    if isinstance(raw_layers, (str, bytes)) or not isinstance(raw_layers, Sequence):
        raise ValueError("layers must be a sequence of layer names")

    layers = list(raw_layers)
    if not layers:
        raise ValueError("at least one build layer is required")
    if any(not isinstance(layer, str) for layer in layers):
        raise ValueError("layers must be a sequence of layer names")
    layer_names = cast(list[str], layers)

    seen: set[str] = set()
    duplicates: set[str] = set()
    for layer in layer_names:
        if layer in seen:
            duplicates.add(layer)
        seen.add(layer)
    if duplicates:
        raise ValueError(f"duplicate build layer(s): {', '.join(sorted(duplicates))}")

    unknown = sorted(set(layer_names) - SUPPORTED_BUILD_LAYERS)
    if unknown:
        raise ValueError(f"unknown build layer(s): {', '.join(unknown)}")
    return cast(list[BuildLayer], layer_names)


class BuildLayerEvent(Event):
    layer: BuildLayer


class BuildStartEvent(StartEvent):
    source_id: str
    version_hash: str | None = None
    layers: list[BuildLayer]

    @field_validator("layers")
    @classmethod
    def _layers_supported(cls, value: Any) -> list[BuildLayer]:
        return _validate_build_layers(value)


class LayerBuiltEvent(Event):
    layer: BuildLayer
    summary: dict[str, Any]


class GraphBuiltEvent(Event):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    coverage: float
    low_confidence_edge_ids: list[str]
    summary: dict[str, Any]


class GraphOrientedEvent(LayerBuiltEvent):
    pass


class RaptorPreflightEvent(Event):
    source_id: str
    chunks: list[RetrievalChunk]


class RaptorBuildEvent(Event):
    source_id: str
    chunks: list[RetrievalChunk]


class BuildState(BaseModel):
    source_id: str = ""
    version_hash: str | None = None
    dsi_dependency_revision: int | None = None
    chunks: list[RetrievalChunk] = Field(default_factory=list)
    title: str = ""
    author: str = ""
    category_ids: list[str] = Field(default_factory=list)
    layers: list[BuildLayer] = Field(default_factory=list)
    results: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Context[StateT: BaseModel](LlamaIndexContext[StateT]):
    """Typed Context annotation for native LlamaIndex state-store inference."""


def _collect_buffer_id(ctx: LlamaIndexContext[Any], state: BuildState) -> str:
    version_hash = state.version_hash or ""
    layers = ",".join(state.layers)
    return f"advanced_build.collect:{state.source_id}:{version_hash}:{layers}"


def _collection(name: str) -> str:
    return name + os.environ.get("VORAG_BUILD_COLLECTION_SUFFIX", "")


def _structured_output_extra(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Local alias for :func:`vo_rag.core.llm.guided.structured_output_extra`."""
    return structured_output_extra(schema)


async def _llm_call_from_client(
    client: LLMClient,
    *,
    guided_json: dict[str, Any] | None = None,
) -> Callable[[str, str, int, float], Awaitable[LLMResponse]]:
    extra = _structured_output_extra(guided_json)

    async def _llm_call(system: str, user: str, max_tokens: int, temperature: float) -> LLMResponse:
        resp: LLMResponse = await client.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra=extra,
        )
        return resp

    return _llm_call


_RAW_COLLECTION = "vo_rag_books_raw_v1"
_LEAF_EMBED_BATCH_SIZE = 8


class AdvancedBuildWorkflow(Workflow):
    """Corpus advanced-understanding builds as one workflow.

    Inline book_rag ingestion still refuses advanced-understanding layers; this
    workflow is the corpus-level harness that supplies the build configs and
    book metadata those layers require.
    """

    def __init__(
        self,
        *,
        store: QdrantStore,
        embedding_client: Any,
        llm_clients: dict[str, LLMClient],
        settings: Any,
        direction_threshold: float = 0.85,
        graph_27b_fallback: bool = False,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.store = store
        self.embed = embedding_client
        self.llm = llm_clients
        self.settings = settings
        self.direction_threshold = direction_threshold
        self.graph_27b_fallback = graph_27b_fallback

    def run(
        self,
        ctx: LlamaIndexContext[Any] | None = None,
        stepwise: bool = False,
        checkpoint_callback: Any | None = None,
        start_event: StartEvent | None = None,
        **kwargs: Any,
    ) -> Any:
        if start_event is None:
            source_id = str(kwargs.pop("source_id", ""))
            version_hash = kwargs.pop("version_hash", None)
            layers = _validate_build_layers(kwargs.pop("layers", []))
            start_event = BuildStartEvent(
                source_id=source_id,
                version_hash=version_hash,
                layers=layers,
            )
        handler = super().run(
            ctx=ctx,
            stepwise=stepwise,
            checkpoint_callback=checkpoint_callback,
            start_event=start_event,
            **kwargs,
        )
        return handler

    # ------------------------------------------------------------------
    # Fallback / gate helpers (VOR-207)
    # ------------------------------------------------------------------

    def _runpod_text_available(self) -> bool:
        serverless = getattr(self.settings, "serverless", None)
        if serverless is None:
            return False
        if not getattr(serverless, "api_key", ""):
            return False
        try:
            serverless.base_url("text", ignore_override=True)
        except (KeyError, ValueError):
            return False
        return True

    def _build_runpod_fallback_client(self) -> LLMClient:
        from vo_rag.core.cloud.serverless import ServerlessClient
        from vo_rag.core.llm.providers.runpod_serverless_provider import (
            RunpodServerlessProvider,
        )

        serverless = self.settings.serverless
        client = ServerlessClient(
            base_url=serverless.base_url("text", ignore_override=True),
            api_key=serverless.api_key,
            model=serverless.model_name("text"),
            timeout_s=serverless.request_timeout_s,
        )
        return LLMClient(provider=RunpodServerlessProvider(client=client))

    # ------------------------------------------------------------------
    # VOR-208 helper seams
    # ------------------------------------------------------------------

    def _summary_model(self, role: str) -> str:
        """Resolve the LLM model name for *role* from settings, with a safe fallback.

        The CLI build commands historically hard-code ``qwen3.6-27b-fp8``; the
        Workflow prefers ``settings.serverless.{role}_model_name`` when present
        so a cluster cutover propagates without a code change.
        """
        serverless = getattr(self.settings, "serverless", None)
        if serverless is not None:
            value = getattr(serverless, f"{role}_model_name", None)
            if value:
                return str(value)
        return "qwen3.6-27b-fp8"

    async def _load_chunks(
        self, source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        """Scroll the raw collection for *source_id*, optionally scoped to *version_hash*.

        The ``version_hash`` filter is load-bearing for deterministic idempotency:
        a re-ingested book must not mix old- and current-version chunks into one
        build.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks: list[RetrievalChunk] = []
        offset: Any = None
        must: list[Any] = [FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        if version_hash is not None:
            must.append(FieldCondition(key="version_hash", match=MatchValue(value=version_hash)))
        scroll_filter = Filter(must=must)
        while True:
            results, next_offset = await self.store._client.scroll(
                collection_name=_RAW_COLLECTION,
                scroll_filter=scroll_filter,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in results:
                payload = point.payload or {}
                meta = payload.get("metadata") or {}
                if not isinstance(meta, dict):
                    meta = {}
                chunks.append(
                    RetrievalChunk(
                        chunk_id=str(payload.get("chunk_id", point.id)),
                        chunk_type=str(payload.get("chunk_type", "text")),
                        document_id=str(payload.get("document_id", source_id)),
                        source_id=str(payload.get("source_id", source_id)),
                        page_number=payload.get("page_number"),
                        page_range=payload.get("page_range"),
                        section_path=str(payload.get("section_path", "")),
                        section_title=str(payload.get("section_title", "")),
                        chapter_path=str(payload.get("chapter_path", "")),
                        content=str(payload.get("content", "")),
                        embedding_text=payload_embedding_text(payload),
                        token_count=int(payload.get("token_count", 0) or 0),
                        has_image=bool(payload.get("has_image", False)),
                        image_id=payload.get("image_id"),
                        image_ref=payload.get("image_ref"),
                        vlm_description=payload.get("vlm_description"),
                        metadata=dict(meta),
                    )
                )
            if next_offset is None:
                break
            offset = next_offset
        if not chunks:
            raise ValueError(f"no chunks for source_id={source_id}")
        return chunks

    def _book_meta(self, chunks: list[RetrievalChunk]) -> tuple[str, str, list[str]]:
        """Extract (title, author, category_ids) from the first chunk's stamped metadata."""
        if not chunks:
            return "", "", []
        meta = chunks[0].metadata
        if not isinstance(meta, dict):
            return "", "", []
        title = str(meta.get("title") or meta.get("source_book") or "")
        author = str(meta.get("author") or "")
        domain_tags = meta.get("domain_tags") or []
        if not isinstance(domain_tags, list):
            domain_tags = []
        return title, author, [str(t) for t in domain_tags]

    # ------------------------------------------------------------------
    # Build helpers — mirror the cli/build.py implementations
    # ------------------------------------------------------------------

    async def _build_dsi(self, st: BuildState) -> dict[str, Any]:
        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(
                summary_model=self._summary_model("dsi"),
                temperature=0.2,
                max_tokens=3000,
                n_representative_chunks=16,
                prompt_hash=DSI_EXTRACTION_PROMPT_HASH,
            ),
        )
        dsi_collection = _collection(_DSI_COLLECTION_DEFAULT)
        expected_dependencies = (
            {st.source_id: st.dsi_dependency_revision}
            if getattr(
                self.store,
                "source_visibility_enabled",
                getattr(self.store, "source_publication_enabled", False),
            )
            else {}
        )
        publication_job = await admit_dsi_source_revision(
            store=self.store,
            source_id=st.source_id,
            collection=dsi_collection,
            config=config,
            source_collection_version=_RAW_COLLECTION,
            expected_dependencies=expected_dependencies,
        )
        try:
            llm_call = await _llm_call_from_client(self.llm["dsi"])
            record, stats = await build_dsi_for_book(
                book_id=st.source_id,
                title=st.title,
                author=st.author,
                category_ids=st.category_ids,
                chunks=st.chunks,
                config=config,
                llm_call=llm_call,
                source_collection_version=_RAW_COLLECTION,
            )
            n_records = 0
            if record is not None:
                await ensure_dsi_collection(self.store._client, collection=dsi_collection)
                n_records = await persist_dsi_records(
                    records=[record],
                    embed=lambda texts: self.embed.embed(
                        texts,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert=include_colbert_at_ingest(),
                    ),
                    store=self.store,
                    collection=dsi_collection,
                    publication_job=publication_job,
                    expected_dependencies=expected_dependencies,
                )
            elif publication_job is not None:
                await self.store.abort_source_revision(
                    publication_job, "DSI extraction produced no record"
                )
        except BaseException:
            if publication_job is not None:
                await self.store.abort_source_revision(publication_job, "DSI producer failed")
            raise
        stats_payload: Any
        if isinstance(stats, dict):
            stats_payload = stats
        elif hasattr(stats, "model_dump"):
            stats_payload = stats.model_dump(mode="json")
        elif hasattr(stats, "dict"):
            stats_payload = stats.dict()
        else:
            stats_payload = stats
        return {
            "book_id": st.source_id,
            "n_records": n_records,
            "n_chunks": len(st.chunks),
            "stats": stats_payload,
        }

    async def _build_summary(self, st: BuildState) -> dict[str, Any]:
        config = BuildConfig(
            summary_model=self._summary_model("text"),
            max_concurrent_llm_calls=8,
            summary_collection=_collection(SUMMARY_TREE_COLLECTION),
        )
        builder = SummaryTreeBuilder(
            llm_client=self.llm["text"],
            embed_client=self.embed,
            store=self.store,
            config=config,
        )
        nodes, _stats = await build_summary_for_corpus(
            books=[(st.source_id, st.chunks)],
            builder=builder,
            max_concurrent_books=1,
            expected_dependencies={st.source_id: st.dsi_dependency_revision},
        )
        return {
            "book_id": st.source_id,
            "n_nodes": len(nodes),
            "n_chunks": len(st.chunks),
        }

    async def _embed_leaf_embeddings(
        self, chunks: list[RetrievalChunk], *, embed_batch: int = _LEAF_EMBED_BATCH_SIZE
    ) -> list[list[float]]:
        """Return dense leaf vectors aligned 1:1 with *chunks* for RAPTOR clustering."""
        texts = [chunk.embedding_text or chunk.content for chunk in chunks]
        embed_batch = max(1, embed_batch)
        leaf_embeddings: list[list[float]] = []

        for start in range(0, len(texts), embed_batch):
            batch_texts = texts[start : start + embed_batch]
            resp: dict[str, Any] = await self.embed.embed(
                batch_texts,
                return_dense=True,
                return_sparse=False,
                return_colbert=False,
            )
            dense_batch = resp.get("dense") or []
            for vector in dense_batch:
                leaf_embeddings.append([float(value) for value in vector])

        if len(leaf_embeddings) != len(chunks):
            raise RuntimeError(
                f"leaf embedding count ({len(leaf_embeddings)}) does not match "
                f"chunk count ({len(chunks)})"
            )
        return leaf_embeddings

    async def _build_raptor(
        self,
        *,
        source_id: str,
        chunks: list[RetrievalChunk],
        expected_dependency_revision: int | None = None,
    ) -> dict[str, Any]:
        leaf_embeddings = await self._embed_leaf_embeddings(chunks)
        config = RaptorBuildConfig(
            summary_model=self._summary_model("text"),
            summary_max_tokens=2500,
            raptor_collection=_collection(RAPTOR_COLLECTION),
        )
        llm_call = await _llm_call_from_client(self.llm["text"])
        results, _stats = await build_raptor_for_corpus(
            books=[(source_id, chunks, leaf_embeddings)],
            config=config,
            llm_call=llm_call,
            embed_client=self.embed,
            store=self.store,
            max_concurrent_books=1,
            expected_dependencies={source_id: expected_dependency_revision},
        )
        n_nodes = sum(len(result.nodes) for result in results)
        return {
            "book_id": source_id,
            "n_nodes": n_nodes,
            "n_chunks": len(chunks),
        }

    async def _extract_graph(
        self, st: BuildState
    ) -> tuple[list[GraphNode], list[GraphEdge], GraphBuildStats]:
        nodes_collection = _collection(GRAPH_NODES_COLLECTION)
        edges_collection = _collection(GRAPH_EDGES_COLLECTION)
        await ensure_graph_collections(
            cast(GraphQdrantClient, self.store._client),
            nodes_collection=nodes_collection,
            edges_collection=edges_collection,
        )
        config = GraphBuildConfig(
            extraction_config=GraphExtractionConfig(
                summary_model=self._summary_model("text"),
                temperature=0.2,
                max_tokens=4000,
                prompt_hash=GRAPH_EXTRACTION_PROMPT_HASH,
            ),
            chunks_per_extraction_call=6,
            max_concurrent_extraction_calls=24,
        )
        llm_call = await _llm_call_from_client(
            self.llm["text"], guided_json=graph_response_json_schema()
        )
        result, stats = await build_graph_for_book(
            book_id=st.source_id,
            title=st.title,
            author=st.author,
            chunks=st.chunks,
            config=config,
            llm_call=llm_call,
        )
        return result.nodes, result.edges, stats

    async def _persist_graph(
        self, nodes: list[GraphNode], edges: list[GraphEdge]
    ) -> tuple[int, int]:
        nodes_collection = _collection(GRAPH_NODES_COLLECTION)
        edges_collection = _collection(GRAPH_EDGES_COLLECTION)
        n_nodes, n_edges = await persist_graph(
            nodes=nodes,
            edges=edges,
            embedding_client=self.embed,
            store=self.store,
            nodes_collection=nodes_collection,
            edges_collection=edges_collection,
        )
        return n_nodes, n_edges

    # ------------------------------------------------------------------
    # Workflow steps
    # ------------------------------------------------------------------

    @step
    async def load_corpus(
        self, ctx: Context[BuildState], ev: BuildStartEvent
    ) -> BuildLayerEvent | None:
        source_id = ev.source_id
        version_hash = ev.version_hash
        layers = ev.layers
        dsi_dependency_revision = (
            await self.store.active_source_revision(_RAW_COLLECTION, source_id)
            if getattr(
                self.store,
                "source_visibility_enabled",
                getattr(self.store, "source_publication_enabled", False),
            )
            else None
        )
        chunks = await self._load_chunks(source_id, version_hash)
        if not chunks:
            raise ValueError(f"no chunks for source_id={source_id}")
        title, author, cats = self._book_meta(chunks)
        async with ctx.store.edit_state() as st:
            st.source_id = source_id
            st.version_hash = version_hash
            st.dsi_dependency_revision = dsi_dependency_revision
            st.chunks = chunks
            st.title = title
            st.author = author
            st.category_ids = cats
            st.layers = list(layers)
        for layer in layers:
            ctx.send_event(BuildLayerEvent(layer=layer))
        return None

    @step(num_workers=4)
    async def build_layer(
        self, ctx: Context[BuildState], ev: BuildLayerEvent
    ) -> LayerBuiltEvent | GraphBuiltEvent | RaptorPreflightEvent:
        st = await ctx.store.get_state()
        if ev.layer == "dsi":
            summary = await self._build_dsi(st)
            return LayerBuiltEvent(layer="dsi", summary=summary)
        if ev.layer == "summary":
            summary = await self._build_summary(st)
            return LayerBuiltEvent(layer="summary", summary=summary)
        if ev.layer == "raptor":
            return RaptorPreflightEvent(
                source_id=st.source_id,
                chunks=st.chunks,
            )
        if ev.layer == "graph" and getattr(
            self.store,
            "source_visibility_enabled",
            getattr(self.store, "source_publication_enabled", False),
        ):
            return GraphOrientedEvent(
                layer="graph",
                summary={
                    "book_id": st.source_id,
                    "skipped": True,
                    "reason": (
                        "graph publication requires one complete corpus dependency snapshot; "
                        "the per-book workflow cannot prove the shared graph roster"
                    ),
                },
            )
        if ev.layer == "graph":
            nodes, edges, stats = await self._extract_graph(st)
            return GraphBuiltEvent(
                nodes=nodes,
                edges=edges,
                coverage=stats.direction_coverage,
                low_confidence_edge_ids=stats.low_confidence_edge_ids,
                summary={
                    "book_id": st.source_id,
                    "n_low_confidence_edges": stats.n_low_confidence_edges,
                },
            )
        raise ValueError(f"unknown build layer: {ev.layer}")

    @step
    async def graph_direction_gate(
        self, ctx: Context[BuildState], ev: GraphBuiltEvent
    ) -> GraphOrientedEvent:
        """STOP/GO gate — build.py:402-479 branch ladder."""
        edges = ev.edges
        n_reoriented = 0
        reorientation_status = "not_needed"

        if (
            self.graph_27b_fallback
            and ev.coverage < self.direction_threshold
            and ev.low_confidence_edge_ids
            and self._runpod_text_available()
        ):
            runpod_client = self._build_runpod_fallback_client()
            try:
                fallback_call = await _llm_call_from_client(runpod_client, guided_json=None)
                corrected_edges = await orient_low_confidence_via_llm(
                    ev.nodes,
                    ev.edges,
                    ev.low_confidence_edge_ids,
                    fallback_call,
                )
                n_reoriented = sum(
                    1
                    for before, after in zip(ev.edges, corrected_edges, strict=True)
                    if before.source_node_id != after.source_node_id
                    or before.target_node_id != after.target_node_id
                )
                edges = corrected_edges
                reorientation_status = "applied"
            finally:
                await runpod_client.aclose()
        elif ev.coverage < self.direction_threshold:
            if not ev.low_confidence_edge_ids:
                reorientation_status = "no_candidates"
            elif self.graph_27b_fallback:
                reorientation_status = "unavailable"
            else:
                reorientation_status = "disabled"

        n_nodes, n_edges = await self._persist_graph(ev.nodes, edges)
        return GraphOrientedEvent(
            layer="graph",
            summary={
                **ev.summary,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "direction_coverage": ev.coverage,
                "n_reoriented": n_reoriented,
                "reorientation_status": reorientation_status,
            },
        )

    @step
    async def raptor_preflight_gate(
        self, ctx: Context[BuildState], ev: RaptorPreflightEvent
    ) -> LayerBuiltEvent | RaptorBuildEvent:
        """STOP/GO gate — abort RAPTOR when corpus is too small to build a tree."""
        if len(ev.chunks) < 3:
            return LayerBuiltEvent(
                layer="raptor",
                summary={
                    "skipped": True,
                    "reason": "corpus too small to build a tree",
                    "n_chunks": len(ev.chunks),
                },
            )
        return RaptorBuildEvent(
            source_id=ev.source_id,
            chunks=ev.chunks,
        )

    @step
    async def raptor_build(self, ctx: Context[BuildState], ev: RaptorBuildEvent) -> LayerBuiltEvent:
        """Build RAPTOR tree for a corpus that passed preflight."""
        if getattr(
            self.store,
            "source_visibility_enabled",
            getattr(self.store, "source_publication_enabled", False),
        ):
            st = await ctx.store.get_state()
            summary = await self._build_raptor(
                source_id=ev.source_id,
                chunks=ev.chunks,
                expected_dependency_revision=st.dsi_dependency_revision,
            )
        else:
            summary = await self._build_raptor(source_id=ev.source_id, chunks=ev.chunks)
        return LayerBuiltEvent(layer="raptor", summary=summary)

    @step
    async def collect(
        self, ctx: Context[BuildState], ev: LayerBuiltEvent | GraphOrientedEvent
    ) -> StopEvent | None:
        st = await ctx.store.get_state()
        expected: list[type[Event]] = []
        for layer in st.layers:
            if layer == "graph":
                expected.append(GraphOrientedEvent)
            else:
                expected.append(LayerBuiltEvent)
        done = ctx.collect_events(ev, expected, buffer_id=_collect_buffer_id(ctx, st))
        if done is None:
            return None
        results = {d.layer: d.summary for d in done}
        async with ctx.store.edit_state() as st_edit:
            st_edit.results = results
        return StopEvent(result=results)
