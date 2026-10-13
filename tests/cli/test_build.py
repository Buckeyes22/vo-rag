from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import click
import pytest
from click.testing import CliRunner

from vo_rag.cli.build import register_build_commands
from vo_rag.cli.main import _LazyBuildGroup, cli
from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.dsi.builder import DsiBuildStats
from vo_rag.core.dsi.schemas import FacetedBookRecord
from vo_rag.core.graph.builder import GraphBuildStats
from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode, GraphNodeType


def test_lazy_build_group_command_listing_matches_registered_build_commands() -> None:
    root = click.Group()
    register_build_commands(root)
    actual_build = root.commands["build"]
    assert isinstance(actual_build, click.Group)

    lazy_build = _LazyBuildGroup(name="build")
    ctx = click.Context(lazy_build)

    assert lazy_build.list_commands(ctx) == actual_build.list_commands(click.Context(actual_build))


def test_collection_returns_base_name_when_suffix_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.cli.build import _collection

    monkeypatch.delenv("VORAG_BUILD_COLLECTION_SUFFIX", raising=False)

    assert _collection("x") == "x"


def test_collection_appends_env_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.cli.build import _collection

    monkeypatch.setenv("VORAG_BUILD_COLLECTION_SUFFIX", "__ab14b")

    assert _collection("x") == "x__ab14b"


def _fake_chunk(source_id: str = "book-1") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="c1",
        chunk_type="text",
        document_id=source_id,
        source_id=source_id,
        page_number=1,
        page_range="1",
        section_path="[Book > Ch1]",
        section_title="Ch1",
        chapter_path="[Book > Ch1]",
        content="hello world",
        embedding_text="hello world",
        token_count=2,
        metadata={"title": "Test Book", "author": "Test Author", "domain_tags": ["tag-a"]},
    )


def _fake_settings() -> Any:
    return SimpleNamespace(
        qdrant=SimpleNamespace(
            url="http://localhost:6333",
            api_key="",
            prefer_grpc=True,
        ),
        embedding=SimpleNamespace(
            batch_url="http://localhost:8001",
            timeout_s=30,
        ),
    )


class _FakeEmbedder:
    async def embed(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        return {"dense": [], "sparse": [], "colbert": []}

    async def aclose(self) -> None:
        pass


class _FakeLLM:
    async def complete(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            text="fake",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        pass


class _FakeStore:
    _client: Any = None


class _ClosingFakeEmbedder:
    def __init__(self) -> None:
        self.closed = False

    async def embed(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        return {"dense": [], "sparse": [], "colbert": []}

    async def aclose(self) -> None:
        self.closed = True


class _ClosingFakeLLM:
    def __init__(self) -> None:
        self.closed = False

    async def complete(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            text="fake",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.closed = True


class _CapturingScrollClient:
    def __init__(self) -> None:
        self.scroll_filters: list[Any] = []

    async def scroll(self, **kwargs: Any) -> tuple[list[SimpleNamespace], None]:
        self.scroll_filters.append(kwargs["scroll_filter"])
        return (
            [
                SimpleNamespace(
                    id="point-1",
                    payload={
                        "chunk_id": "c1",
                        "chunk_type": "text",
                        "document_id": "doc-1",
                        "source_id": "book-1",
                        "page_number": 1,
                        "page_range": "1",
                        "section_path": "Book > Ch1",
                        "section_title": "Ch1",
                        "chapter_path": "Book > Ch1",
                        "content": "hello world",
                        "token_count": 2,
                        "metadata": {"title": "Test Book"},
                    },
                )
            ],
            None,
        )


def _filter_pairs(scroll_filter: Any) -> list[tuple[str, Any]]:
    return [(condition.key, condition.match.value) for condition in scroll_filter.must]


@pytest.fixture(autouse=True)
def _patch_build_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch heavy constructors so every test runs offline."""
    monkeypatch.setattr("vo_rag.cli.build.get_settings", _fake_settings)
    monkeypatch.setattr("vo_rag.cli.build.QdrantStore", lambda *args, **kwargs: _FakeStore())
    monkeypatch.setattr(
        "vo_rag.cli.build.build_embedding_client", lambda *args, **kwargs: _FakeEmbedder()
    )
    monkeypatch.setattr("vo_rag.cli.build.LLMClient", lambda *args, **kwargs: _FakeLLM())


def test_build_dsi_parses_args_and_prints_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunk = _fake_chunk("book-1")
    captured_load_args: dict[str, Any] = {}

    async def fake_load_chunks(
        store: Any,
        source_id: str,
        collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        captured_load_args.update(
            {
                "store": store,
                "source_id": source_id,
                "collection": collection,
                "version_hash": version_hash,
            }
        )
        return [fake_chunk]

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    captured_dsi_args: dict[str, Any] = {}

    async def fake_build_dsi_for_book(
        **kwargs: Any,
    ) -> tuple[FacetedBookRecord | None, DsiBuildStats]:
        captured_dsi_args.update(kwargs)
        record = FacetedBookRecord(
            book_id="book-1",
            title="Test Book",
            author="Test Author",
            category_ids=["tag-a"],
            short_abstract="abs",
            technical_domains=["domain"],
            key_concepts=[],
            key_components=[],
            key_methods=[],
            key_equations_or_models=[],
            figure_heavy_topics=[],
            synonyms_and_aliases=[],
            prerequisite_topics=[],
            chapter_outline="",
            representative_chunk_ids=["c1"],
            source_collection_version="vo_rag_books_raw_v1",
            summary_model="m",
            prompt_hash="h",
            created_at=datetime.now(UTC),
        )
        stats = DsiBuildStats(
            book_id="book-1",
            extraction_attempts=1,
            extraction_succeeded=True,
            validation_passed=True,
            validation_violations=0,
            final_action="accept",
            input_tokens=10,
            output_tokens=20,
            wall_time_ms=100,
        )
        return record, stats

    monkeypatch.setattr("vo_rag.cli.build.build_dsi_for_book", fake_build_dsi_for_book)

    captured_persist_args: dict[str, Any] = {}

    async def fake_persist_dsi_records(**kwargs: Any) -> int:
        captured_persist_args.update(kwargs)
        return 1

    captured_ensure_args: dict[str, Any] = {}

    async def fake_ensure_dsi(_client: Any, **kwargs: Any) -> bool:
        captured_ensure_args.update(kwargs)
        return True

    monkeypatch.setattr("vo_rag.cli.build.ensure_dsi_collection", fake_ensure_dsi)
    monkeypatch.setattr("vo_rag.cli.build.persist_dsi_records", fake_persist_dsi_records)

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "dsi", "--source-id", "book-1"])
    assert result.exit_code == 0, result.output
    assert "book_id: book-1" in result.output
    assert "n_records: 1" in result.output
    assert "n_chunks: 1" in result.output

    assert captured_dsi_args["book_id"] == "book-1"
    assert captured_dsi_args["title"] == "Test Book"
    assert captured_dsi_args["author"] == "Test Author"
    assert captured_dsi_args["category_ids"] == ["tag-a"]
    assert captured_dsi_args["chunks"] == [fake_chunk]
    assert captured_dsi_args["source_collection_version"] == "vo_rag_books_raw_v1"

    assert captured_load_args["collection"] == "vo_rag_books_raw_v1"
    assert captured_ensure_args["collection"] == "vo_rag_books_dsi_v1"
    assert captured_persist_args["collection"] == "vo_rag_books_dsi_v1"
    assert captured_persist_args["records"][0].book_id == "book-1"


def test_build_dsi_uses_suffixed_output_collection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_BUILD_COLLECTION_SUFFIX", "__ab14b")
    fake_chunk = _fake_chunk("book-dsi-suffix")
    captured_load_args: dict[str, Any] = {}

    async def fake_load_chunks(
        store: Any,
        source_id: str,
        collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        captured_load_args.update(
            {
                "store": store,
                "source_id": source_id,
                "collection": collection,
                "version_hash": version_hash,
            }
        )
        return [fake_chunk]

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    async def fake_build_dsi_for_book(
        **_kwargs: Any,
    ) -> tuple[FacetedBookRecord | None, DsiBuildStats]:
        record = FacetedBookRecord(
            book_id="book-dsi-suffix",
            title="Test Book",
            author="Test Author",
            category_ids=["tag-a"],
            short_abstract="abs",
            technical_domains=["domain"],
            key_concepts=[],
            key_components=[],
            key_methods=[],
            key_equations_or_models=[],
            figure_heavy_topics=[],
            synonyms_and_aliases=[],
            prerequisite_topics=[],
            chapter_outline="",
            representative_chunk_ids=["c1"],
            source_collection_version="vo_rag_books_raw_v1",
            summary_model="m",
            prompt_hash="h",
            created_at=datetime.now(UTC),
        )
        stats = DsiBuildStats(
            book_id="book-dsi-suffix",
            extraction_attempts=1,
            extraction_succeeded=True,
            validation_passed=True,
            validation_violations=0,
            final_action="accept",
            input_tokens=10,
            output_tokens=20,
            wall_time_ms=100,
        )
        return record, stats

    monkeypatch.setattr("vo_rag.cli.build.build_dsi_for_book", fake_build_dsi_for_book)

    captured_ensure_args: dict[str, Any] = {}
    captured_persist_args: dict[str, Any] = {}

    async def fake_ensure_dsi(_client: Any, **kwargs: Any) -> bool:
        captured_ensure_args.update(kwargs)
        return True

    async def fake_persist_dsi_records(**kwargs: Any) -> int:
        captured_persist_args.update(kwargs)
        return 1

    monkeypatch.setattr("vo_rag.cli.build.ensure_dsi_collection", fake_ensure_dsi)
    monkeypatch.setattr("vo_rag.cli.build.persist_dsi_records", fake_persist_dsi_records)

    result = CliRunner().invoke(cli, ["build", "dsi", "--source-id", "book-dsi-suffix"])

    assert result.exit_code == 0, result.output
    assert captured_load_args["collection"] == "vo_rag_books_raw_v1"
    assert captured_ensure_args["collection"] == "vo_rag_books_dsi_v1__ab14b"
    assert captured_persist_args["collection"] == "vo_rag_books_dsi_v1__ab14b"


def test_build_dsi_custom_collection_version(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunk = _fake_chunk("book-2")

    async def fake_load_chunks(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        return [fake_chunk]

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    captured: dict[str, Any] = {}

    async def fake_build_dsi_for_book(
        **kwargs: Any,
    ) -> tuple[FacetedBookRecord | None, DsiBuildStats]:
        captured.update(kwargs)
        return None, DsiBuildStats(
            book_id="book-2",
            extraction_attempts=1,
            extraction_succeeded=False,
            validation_passed=False,
            validation_violations=0,
            final_action="extraction_failed",
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    monkeypatch.setattr("vo_rag.cli.build.build_dsi_for_book", fake_build_dsi_for_book)

    async def fake_persist_dsi_records_zero(**kwargs: Any) -> int:
        return 0

    async def fake_ensure_dsi_z(_client: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr("vo_rag.cli.build.ensure_dsi_collection", fake_ensure_dsi_z)
    monkeypatch.setattr("vo_rag.cli.build.persist_dsi_records", fake_persist_dsi_records_zero)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "dsi",
            "--source-id",
            "book-2",
            "--collection-version",
            "vo_rag_books_sterile_v1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "n_records: 0" in result.output
    assert captured["source_collection_version"] == "vo_rag_books_sterile_v1"


def test_build_dsi_custom_extraction_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    """--n-representative-chunks / --max-tokens flow into the ExtractionConfig (VOR-98)."""
    fake_chunk = _fake_chunk("book-sz")

    async def fake_load_chunks(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        return [fake_chunk]

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    captured: dict[str, Any] = {}

    async def fake_build_dsi_for_book(
        **kwargs: Any,
    ) -> tuple[FacetedBookRecord | None, DsiBuildStats]:
        captured.update(kwargs)
        return None, DsiBuildStats(
            book_id="book-sz",
            extraction_attempts=1,
            extraction_succeeded=False,
            validation_passed=False,
            validation_violations=0,
            final_action="extraction_failed",
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    async def fake_persist_zero(**kwargs: Any) -> int:
        return 0

    async def fake_ensure(_client: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr("vo_rag.cli.build.build_dsi_for_book", fake_build_dsi_for_book)
    monkeypatch.setattr("vo_rag.cli.build.ensure_dsi_collection", fake_ensure)
    monkeypatch.setattr("vo_rag.cli.build.persist_dsi_records", fake_persist_zero)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "dsi",
            "--source-id",
            "book-sz",
            "--n-representative-chunks",
            "8",
            "--max-tokens",
            "1500",
        ],
    )
    assert result.exit_code == 0, result.output
    extraction = captured["config"].extraction_config
    assert extraction.n_representative_chunks == 8
    assert extraction.max_tokens == 1500


def test_build_dsi_default_sizing_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults are right-sized to 16 chunks / 3000 tokens (was 20 / 4000)."""
    fake_chunk = _fake_chunk("book-def")

    async def fake_load_chunks(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        return [fake_chunk]

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)
    captured: dict[str, Any] = {}

    async def fake_build_dsi_for_book(
        **kwargs: Any,
    ) -> tuple[FacetedBookRecord | None, DsiBuildStats]:
        captured.update(kwargs)
        return None, DsiBuildStats(
            book_id="book-def",
            extraction_attempts=1,
            extraction_succeeded=False,
            validation_passed=False,
            validation_violations=0,
            final_action="extraction_failed",
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    async def fake_persist_zero(**kwargs: Any) -> int:
        return 0

    async def fake_ensure(_client: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr("vo_rag.cli.build.build_dsi_for_book", fake_build_dsi_for_book)
    monkeypatch.setattr("vo_rag.cli.build.ensure_dsi_collection", fake_ensure)
    monkeypatch.setattr("vo_rag.cli.build.persist_dsi_records", fake_persist_zero)

    result = CliRunner().invoke(cli, ["build", "dsi", "--source-id", "book-def"])
    assert result.exit_code == 0, result.output
    extraction = captured["config"].extraction_config
    assert extraction.n_representative_chunks == 16
    assert extraction.max_tokens == 3000


def test_book_meta_from_chunks_reads_source_book_fallback() -> None:
    """_book_meta_from_chunks accepts the `source_book` key (golden corpus) and
    prefers a clean `title` when present (the metadata-gap fix)."""
    from vo_rag.cli.build import _book_meta_from_chunks

    base = _fake_chunk("b")
    c1 = base.model_copy(
        update={
            "metadata": {
                "source_book": "Vehicle Dynamics (Casey Marlow)",
                "author": "Casey Marlow",
                "domain_tags": ["vd"],
            }
        }
    )
    title, author, tags = _book_meta_from_chunks([c1])
    assert title == "Vehicle Dynamics (Casey Marlow)"
    assert author == "Casey Marlow"
    assert tags == ["vd"]

    c2 = base.model_copy(
        update={"metadata": {"title": "Vehicle Dynamics", "source_book": "VD (X)"}}
    )
    assert _book_meta_from_chunks([c2])[0] == "Vehicle Dynamics"


@pytest.mark.asyncio
async def test_load_chunks_for_source_defaults_to_source_id_only_filter() -> None:
    from vo_rag.cli.build import _load_chunks_for_source

    client = _CapturingScrollClient()
    store: Any = SimpleNamespace(_client=client)

    chunks = await _load_chunks_for_source(store, "book-1", "raw")

    assert chunks[0].source_id == "book-1"
    assert _filter_pairs(client.scroll_filters[0]) == [("source_id", "book-1")]


@pytest.mark.asyncio
async def test_load_chunks_for_source_adds_version_hash_filter() -> None:
    from vo_rag.cli.build import _load_chunks_for_source

    client = _CapturingScrollClient()
    store: Any = SimpleNamespace(_client=client)

    chunks = await _load_chunks_for_source(
        store,
        "book-1",
        "raw",
        version_hash="vh-1",
    )

    assert chunks[0].source_id == "book-1"
    assert _filter_pairs(client.scroll_filters[0]) == [
        ("source_id", "book-1"),
        ("version_hash", "vh-1"),
    ]


def test_build_graph_parses_args_and_prints_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunk = _fake_chunk("book-3")
    captured_load_args: dict[str, Any] = {}

    async def fake_load_chunks(
        store: Any,
        source_id: str,
        collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        captured_load_args.update(
            {
                "store": store,
                "source_id": source_id,
                "collection": collection,
                "version_hash": version_hash,
            }
        )
        return [fake_chunk]

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    captured_graph_args: dict[str, Any] = {}

    from vo_rag.core.graph.extraction import GraphExtractionResult

    async def fake_build_graph_for_book(
        **kwargs: Any,
    ) -> tuple[GraphExtractionResult, GraphBuildStats]:
        captured_graph_args.update(kwargs)
        result = GraphExtractionResult(
            book_id="book-3",
            nodes=[
                GraphNode(
                    node_id="n1",
                    node_type=GraphNodeType.CONCEPT,
                    label="Node1",
                    book_ids=["book-3"],
                    chunk_ids=["c1"],
                    aliases=[],
                    created_at=datetime.now(UTC),
                )
            ],
            edges=[
                GraphEdge(
                    edge_id="e1",
                    edge_type=GraphEdgeType.RELATES_TO,
                    source_node_id="n1",
                    target_node_id="n1",
                    book_ids=["book-3"],
                    evidence_chunk_ids=["c1"],
                    weight=1.0,
                    created_at=datetime.now(UTC),
                )
            ],
            validation_errors=[],
            prompt_hash="h",
            input_tokens=5,
            output_tokens=5,
            wall_time_ms=50,
        )
        stats = GraphBuildStats(
            book_id="book-3",
            n_nodes_extracted=1,
            n_edges_extracted=1,
            n_nodes_after_dedup=1,
            n_edges_after_dedup=1,
            validation_error_count=0,
            wall_time_ms=50,
        )
        return result, stats

    monkeypatch.setattr("vo_rag.cli.build.build_graph_for_book", fake_build_graph_for_book)

    captured_persist_args: dict[str, Any] = {}

    async def fake_persist_graph(**kwargs: Any) -> tuple[int, int]:
        captured_persist_args.update(kwargs)
        return 1, 1

    monkeypatch.setattr("vo_rag.cli.build.persist_graph", fake_persist_graph)

    captured_ensure: list[Any] = []
    captured_ensure_args: dict[str, Any] = {}

    async def fake_ensure_graph_collections(_client: Any, **kwargs: Any) -> tuple[bool, bool]:
        captured_ensure.append(_client)
        captured_ensure_args.update(kwargs)
        return False, False

    monkeypatch.setattr("vo_rag.cli.build.ensure_graph_collections", fake_ensure_graph_collections)

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "graph", "--source-id", "book-3"])
    assert result.exit_code == 0, result.output
    assert "book_id: book-3" in result.output
    assert "n_nodes: 1" in result.output
    assert "n_edges: 1" in result.output

    assert captured_graph_args["book_id"] == "book-3"
    assert captured_graph_args["title"] == "Test Book"
    assert captured_graph_args["author"] == "Test Author"
    assert captured_graph_args["chunks"] == [fake_chunk]

    assert captured_load_args["collection"] == "vo_rag_books_raw_v1"
    assert captured_ensure_args == {
        "nodes_collection": "vo_rag_books_graph_nodes_v1",
        "edges_collection": "vo_rag_books_graph_edges_v1",
    }
    assert captured_persist_args["nodes"][0].node_id == "n1"
    assert captured_persist_args["edges"][0].edge_id == "e1"
    assert captured_persist_args["nodes_collection"] == "vo_rag_books_graph_nodes_v1"
    assert captured_persist_args["edges_collection"] == "vo_rag_books_graph_edges_v1"
    assert len(captured_ensure) == 1


@pytest.mark.asyncio
async def test_build_graph_impl_persists_oriented_edges_and_reports_directional_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_graph_impl

    fake_chunk = _fake_chunk("book-directed")
    payload = json.dumps(
        {
            "nodes": [
                {
                    "node_type": "component",
                    "label": "Generator",
                    "aliases": [],
                    "chunk_ids": ["c0"],
                },
                {
                    "node_type": "component",
                    "label": "Honda EU2000i Generator",
                    "aliases": [],
                    "chunk_ids": ["c0"],
                },
            ],
            "edges": [
                {
                    "source_label": "Generator",
                    "source_type": "component",
                    "target_label": "Honda EU2000i Generator",
                    "target_type": "component",
                    "edge_type": "is_a",
                    "evidence_chunk_ids": ["c0"],
                }
            ],
        }
    )

    class JsonGraphLLM:
        async def complete(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                text=payload,
                input_tokens=10,
                output_tokens=20,
                latency_ms=1,
                finish_reason="stop",
            )

        async def aclose(self) -> None:
            pass

    async def fake_load_chunks(
        store: Any,
        source_id: str,
        collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[RetrievalChunk]:
        return [fake_chunk]

    async def fake_ensure_graph_collections(_client: Any, **kwargs: Any) -> tuple[bool, bool]:
        return False, False

    captured_persist_args: dict[str, Any] = {}

    async def fake_persist_graph(**kwargs: Any) -> tuple[int, int]:
        captured_persist_args.update(kwargs)
        return len(kwargs["nodes"]), len(kwargs["edges"])

    monkeypatch.setattr("vo_rag.cli.build.LLMClient", lambda *args, **kwargs: JsonGraphLLM())
    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)
    monkeypatch.setattr("vo_rag.cli.build.ensure_graph_collections", fake_ensure_graph_collections)
    monkeypatch.setattr("vo_rag.cli.build.persist_graph", fake_persist_graph)

    summary = await _build_graph_impl("book-directed")

    node_labels = {node.node_id: node.label for node in captured_persist_args["nodes"]}
    persisted_edge = captured_persist_args["edges"][0]
    assert node_labels[persisted_edge.source_node_id] == "Honda EU2000i Generator"
    assert node_labels[persisted_edge.target_node_id] == "Generator"
    assert summary["direction_coverage"] == 1.0
    assert summary["n_low_confidence_edges"] == 0


@pytest.mark.asyncio
async def test_build_graph_impl_skips_fallback_when_both_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_graph_impl
    from vo_rag.core.graph.extraction import GraphExtractionResult

    # Both the RunPod-27B fallback and the local-14B reorientation are off, so no
    # reorientation should run (the "27B off but local on -> applied_local" case is
    # covered by tests/core/graph/test_cli_build_direction_fallback.py).
    monkeypatch.delenv("VORAG_GRAPH_27B_FALLBACK", raising=False)
    monkeypatch.setenv("VORAG_GRAPH_LOCAL_REORIENT", "0")
    fake_chunk = _fake_chunk("book-fallback-off")
    parent = GraphNode(
        node_id="n-parent",
        node_type=GraphNodeType.COMPONENT,
        label="Suspension System",
        book_ids=["book-fallback-off"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    child = GraphNode(
        node_id="n-child",
        node_type=GraphNodeType.COMPONENT,
        label="Spring",
        book_ids=["book-fallback-off"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    edge = GraphEdge(
        edge_id="edge-low",
        edge_type=GraphEdgeType.PART_OF,
        source_node_id=parent.node_id,
        target_node_id=child.node_id,
        book_ids=["book-fallback-off"],
        evidence_chunk_ids=["c1"],
        weight=1.0,
        created_at=datetime.now(UTC),
    )

    async def fake_build_graph_for_book(
        **_kwargs: Any,
    ) -> tuple[GraphExtractionResult, GraphBuildStats]:
        return (
            GraphExtractionResult(
                book_id="book-fallback-off",
                nodes=[parent, child],
                edges=[edge],
                validation_errors=[],
                prompt_hash="h",
                input_tokens=1,
                output_tokens=1,
                wall_time_ms=1,
            ),
            GraphBuildStats(
                book_id="book-fallback-off",
                n_nodes_extracted=2,
                n_edges_extracted=1,
                n_nodes_after_dedup=2,
                n_edges_after_dedup=1,
                validation_error_count=0,
                wall_time_ms=1,
                direction_coverage=0.5,
                n_low_confidence_edges=1,
                low_confidence_edge_ids=[edge.edge_id],
            ),
        )

    async def fail_fallback(*_args: Any, **_kwargs: Any) -> list[GraphEdge]:
        raise AssertionError("no reorientation should run when both fallbacks are disabled")

    captured_persist_args: dict[str, Any] = {}

    async def fake_load_chunks(*_args: Any, **_kwargs: Any) -> list[RetrievalChunk]:
        return [fake_chunk]

    async def fake_ensure_graph_collections(*_args: Any, **_kwargs: Any) -> tuple[bool, bool]:
        return False, False

    async def fake_persist_graph(**kwargs: Any) -> tuple[int, int]:
        captured_persist_args.update(kwargs)
        return len(kwargs["nodes"]), len(kwargs["edges"])

    monkeypatch.setattr("vo_rag.cli.build.build_graph_for_book", fake_build_graph_for_book)
    monkeypatch.setattr("vo_rag.cli.build.orient_low_confidence_via_llm", fail_fallback)
    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)
    monkeypatch.setattr("vo_rag.cli.build.ensure_graph_collections", fake_ensure_graph_collections)
    monkeypatch.setattr("vo_rag.cli.build.persist_graph", fake_persist_graph)

    summary = await _build_graph_impl("book-fallback-off")

    assert captured_persist_args["edges"] == [edge]
    assert summary["direction_coverage"] == 0.5
    assert summary["low_confidence_edge_ids"] == [edge.edge_id]


@pytest.mark.asyncio
async def test_build_graph_impl_runs_27b_fallback_and_persists_corrected_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_graph_impl
    from vo_rag.core.graph.extraction import GraphExtractionResult

    monkeypatch.setenv("VORAG_GRAPH_27B_FALLBACK", "1")
    monkeypatch.setenv("VORAG_GRAPH_DIRECTION_MIN_COVERAGE", "0.85")
    fake_chunk = _fake_chunk("book-fallback-on")
    parent = GraphNode(
        node_id="n-parent",
        node_type=GraphNodeType.COMPONENT,
        label="Suspension System",
        book_ids=["book-fallback-on"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    child = GraphNode(
        node_id="n-child",
        node_type=GraphNodeType.COMPONENT,
        label="Spring",
        book_ids=["book-fallback-on"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    edge = GraphEdge(
        edge_id="edge-low",
        edge_type=GraphEdgeType.PART_OF,
        source_node_id=parent.node_id,
        target_node_id=child.node_id,
        book_ids=["book-fallback-on"],
        evidence_chunk_ids=["c1"],
        weight=1.0,
        created_at=datetime.now(UTC),
    )
    corrected_edge = edge.model_copy(
        update={"source_node_id": child.node_id, "target_node_id": parent.node_id}
    )

    class FakeRunpodClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    runpod_client = FakeRunpodClient()

    async def fake_build_graph_for_book(
        **_kwargs: Any,
    ) -> tuple[GraphExtractionResult, GraphBuildStats]:
        return (
            GraphExtractionResult(
                book_id="book-fallback-on",
                nodes=[parent, child],
                edges=[edge],
                validation_errors=[],
                prompt_hash="h",
                input_tokens=1,
                output_tokens=1,
                wall_time_ms=1,
            ),
            GraphBuildStats(
                book_id="book-fallback-on",
                n_nodes_extracted=2,
                n_edges_extracted=1,
                n_nodes_after_dedup=2,
                n_edges_after_dedup=1,
                validation_error_count=0,
                wall_time_ms=1,
                direction_coverage=0.5,
                n_low_confidence_edges=1,
                low_confidence_edge_ids=[edge.edge_id],
            ),
        )

    async def fake_orient_low_confidence_via_llm(
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        low_confidence_edge_ids: list[str],
        llm_call: Any,
    ) -> list[GraphEdge]:
        assert nodes == [parent, child]
        assert edges == [edge]
        assert low_confidence_edge_ids == [edge.edge_id]
        assert llm_call == "fallback-call"
        return [corrected_edge]

    async def fake_load_chunks(*_args: Any, **_kwargs: Any) -> list[RetrievalChunk]:
        return [fake_chunk]

    async def fake_ensure_graph_collections(*_args: Any, **_kwargs: Any) -> tuple[bool, bool]:
        return False, False

    captured_persist_args: dict[str, Any] = {}

    async def fake_persist_graph(**kwargs: Any) -> tuple[int, int]:
        captured_persist_args.update(kwargs)
        return len(kwargs["nodes"]), len(kwargs["edges"])

    async def fake_llm_call_from_client(client: Any, *, guided_json: Any = None) -> str:
        if client is runpod_client:
            assert guided_json is None
            return "fallback-call"
        assert guided_json is not None
        return "primary-call"

    class FakeServerlessSettings:
        api_key = "runpod-key"  # pragma: allowlist secret

        def base_url(self, role: str, *, ignore_override: bool = False) -> str:
            assert role == "text"
            assert ignore_override is True
            return "https://api.runpod.ai/v2/text-endpoint/openai/v1"

    def fake_settings_with_serverless() -> Any:
        settings = _fake_settings()
        settings.serverless = FakeServerlessSettings()
        return settings

    monkeypatch.setattr("vo_rag.cli.build.get_settings", fake_settings_with_serverless)
    monkeypatch.setattr("vo_rag.cli.build.build_graph_for_book", fake_build_graph_for_book)
    monkeypatch.setattr(
        "vo_rag.cli.build.orient_low_confidence_via_llm",
        fake_orient_low_confidence_via_llm,
    )
    monkeypatch.setattr("vo_rag.cli.build._build_runpod_text_client", lambda: runpod_client)
    monkeypatch.setattr("vo_rag.cli.build._llm_call_from_client", fake_llm_call_from_client)
    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)
    monkeypatch.setattr("vo_rag.cli.build.ensure_graph_collections", fake_ensure_graph_collections)
    monkeypatch.setattr("vo_rag.cli.build.persist_graph", fake_persist_graph)

    summary = await _build_graph_impl("book-fallback-on")

    assert captured_persist_args["edges"] == [corrected_edge]
    assert runpod_client.closed is True
    assert summary["direction_coverage"] == 0.5
    assert summary["low_confidence_edge_ids"] == [edge.edge_id]


def test_build_graph_uses_suffixed_output_collections_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_BUILD_COLLECTION_SUFFIX", "__ab14b")
    fake_chunk = _fake_chunk("book-graph-suffix")
    captured_load_args: dict[str, Any] = {}

    async def fake_load_chunks(
        store: Any,
        source_id: str,
        collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        captured_load_args.update(
            {
                "store": store,
                "source_id": source_id,
                "collection": collection,
                "version_hash": version_hash,
            }
        )
        return [fake_chunk]

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    from vo_rag.core.graph.extraction import GraphExtractionResult

    async def fake_build_graph_for_book(
        **_kwargs: Any,
    ) -> tuple[GraphExtractionResult, GraphBuildStats]:
        result = GraphExtractionResult(
            book_id="book-graph-suffix",
            nodes=[],
            edges=[],
            validation_errors=[],
            prompt_hash="h",
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )
        stats = GraphBuildStats(
            book_id="book-graph-suffix",
            n_nodes_extracted=0,
            n_edges_extracted=0,
            n_nodes_after_dedup=0,
            n_edges_after_dedup=0,
            validation_error_count=0,
            wall_time_ms=0,
        )
        return result, stats

    monkeypatch.setattr("vo_rag.cli.build.build_graph_for_book", fake_build_graph_for_book)

    captured_ensure_args: dict[str, Any] = {}
    captured_persist_args: dict[str, Any] = {}

    async def fake_ensure_graph_collections(_client: Any, **kwargs: Any) -> tuple[bool, bool]:
        captured_ensure_args.update(kwargs)
        return False, False

    async def fake_persist_graph(**kwargs: Any) -> tuple[int, int]:
        captured_persist_args.update(kwargs)
        return 0, 0

    monkeypatch.setattr("vo_rag.cli.build.ensure_graph_collections", fake_ensure_graph_collections)
    monkeypatch.setattr("vo_rag.cli.build.persist_graph", fake_persist_graph)

    result = CliRunner().invoke(cli, ["build", "graph", "--source-id", "book-graph-suffix"])

    assert result.exit_code == 0, result.output
    assert captured_load_args["collection"] == "vo_rag_books_raw_v1"
    assert captured_ensure_args == {
        "nodes_collection": "vo_rag_books_graph_nodes_v1__ab14b",
        "edges_collection": "vo_rag_books_graph_edges_v1__ab14b",
    }
    assert captured_persist_args["nodes_collection"] == "vo_rag_books_graph_nodes_v1__ab14b"
    assert captured_persist_args["edges_collection"] == "vo_rag_books_graph_edges_v1__ab14b"


def test_build_summary_is_registered_and_exposes_help() -> None:
    runner = CliRunner()

    build_help = runner.invoke(cli, ["build", "--help"])
    assert build_help.exit_code == 0, build_help.output
    assert "summary" in build_help.output

    summary_help = runner.invoke(cli, ["build", "summary", "--help"])
    assert summary_help.exit_code == 0, summary_help.output
    assert "--source-id" in summary_help.output
    assert "--max-concurrent" in summary_help.output
    assert "--version-hash" in summary_help.output


@pytest.mark.asyncio
async def test_build_summary_impl_wires_builder_and_closes_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_summary_impl

    source_id = "book-summary"
    chunks = [_fake_chunk(source_id), _fake_chunk(source_id)]
    embedder = _ClosingFakeEmbedder()
    llm = _ClosingFakeLLM()
    store = _FakeStore()
    constructor_args: dict[str, Any] = {}

    monkeypatch.setattr("vo_rag.cli.build.QdrantStore", lambda *args, **kwargs: store)
    monkeypatch.setattr("vo_rag.cli.build.build_embedding_client", lambda *args, **kwargs: embedder)
    monkeypatch.setattr("vo_rag.cli.build.LLMClient", lambda *args, **kwargs: llm)

    async def fake_load_chunks(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        assert _store is store
        assert _source_id == source_id
        assert _collection == "vo_rag_books_raw_v1"
        assert version_hash == "vh-summary"
        return chunks

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    class FakeSummaryTreeBuilder:
        def __init__(self, **kwargs: Any) -> None:
            constructor_args.update(kwargs)

    monkeypatch.setattr("vo_rag.cli.build.SummaryTreeBuilder", FakeSummaryTreeBuilder)

    captured_summary_args: dict[str, Any] = {}
    fake_node = SimpleNamespace(node_id="summary-node")

    async def fake_build_summary_for_corpus(**kwargs: Any) -> tuple[list[Any], list[Any]]:
        captured_summary_args.update(kwargs)
        return [fake_node, fake_node], []

    monkeypatch.setattr("vo_rag.cli.build.build_summary_for_corpus", fake_build_summary_for_corpus)

    result = await _build_summary_impl(source_id, max_concurrent=3, version_hash="vh-summary")

    assert result == {"book_id": source_id, "n_nodes": 2, "n_chunks": len(chunks)}
    assert constructor_args["llm_client"] is llm
    assert constructor_args["embed_client"] is embedder
    assert constructor_args["store"] is store
    assert constructor_args["config"].max_concurrent_llm_calls == 3
    assert constructor_args["config"].summary_collection == "vo_rag_books_summary_tree_v1"
    assert captured_summary_args["books"] == [(source_id, chunks)]
    assert captured_summary_args["builder"].__class__ is FakeSummaryTreeBuilder
    assert captured_summary_args["max_concurrent_books"] == 1
    assert embedder.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_build_summary_uses_suffixed_output_collection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_summary_impl

    monkeypatch.setenv("VORAG_BUILD_COLLECTION_SUFFIX", "__ab14b")
    source_id = "book-summary-suffix"
    chunks = [_fake_chunk(source_id)]
    store = _FakeStore()
    constructor_args: dict[str, Any] = {}
    captured_load_args: dict[str, Any] = {}

    monkeypatch.setattr("vo_rag.cli.build.QdrantStore", lambda *args, **kwargs: store)

    async def fake_load_chunks(
        store_arg: Any,
        loaded_source_id: str,
        collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        captured_load_args.update(
            {
                "store": store_arg,
                "source_id": loaded_source_id,
                "collection": collection,
                "version_hash": version_hash,
            }
        )
        return chunks

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    class FakeSummaryTreeBuilder:
        def __init__(self, **kwargs: Any) -> None:
            constructor_args.update(kwargs)

    monkeypatch.setattr("vo_rag.cli.build.SummaryTreeBuilder", FakeSummaryTreeBuilder)

    async def fake_build_summary_for_corpus(**_kwargs: Any) -> tuple[list[Any], list[Any]]:
        return [SimpleNamespace(node_id="summary-node")], []

    monkeypatch.setattr("vo_rag.cli.build.build_summary_for_corpus", fake_build_summary_for_corpus)

    result = await _build_summary_impl(source_id, version_hash="vh-summary")

    assert result == {"book_id": source_id, "n_nodes": 1, "n_chunks": len(chunks)}
    assert captured_load_args["collection"] == "vo_rag_books_raw_v1"
    assert captured_load_args["version_hash"] == "vh-summary"
    assert constructor_args["config"].summary_collection == "vo_rag_books_summary_tree_v1__ab14b"


@pytest.mark.asyncio
async def test_build_summary_impl_no_chunks_raises_click_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_summary_impl

    embedder = _ClosingFakeEmbedder()
    llm = _ClosingFakeLLM()

    monkeypatch.setattr("vo_rag.cli.build.build_embedding_client", lambda *args, **kwargs: embedder)
    monkeypatch.setattr("vo_rag.cli.build.LLMClient", lambda *args, **kwargs: llm)

    async def fake_load_chunks_empty(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        return []

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks_empty)

    with pytest.raises(click.ClickException, match="No chunks found"):
        await _build_summary_impl("missing")

    assert embedder.closed is True
    assert llm.closed is True


def test_build_raptor_is_registered_and_exposes_help() -> None:
    runner = CliRunner()

    build_help = runner.invoke(cli, ["build", "--help"])
    assert build_help.exit_code == 0, build_help.output
    assert "raptor" in build_help.output

    raptor_help = runner.invoke(cli, ["build", "raptor", "--help"])
    assert raptor_help.exit_code == 0, raptor_help.output
    assert "--source-id" in raptor_help.output
    assert "--max-tokens" in raptor_help.output
    assert "--version-hash" in raptor_help.output


@pytest.mark.parametrize(
    ("args", "expected_max_tokens"),
    [
        (["build", "raptor", "--source-id", "book-raptor"], 2500),
        (
            ["build", "raptor", "--source-id", "book-raptor", "--max-tokens", "3200"],
            3200,
        ),
    ],
)
def test_build_raptor_cli_forwards_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expected_max_tokens: int,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_build_raptor_impl(
        source_id: str,
        *,
        max_tokens: int,
        version_hash: str | None = None,
    ) -> dict[str, Any]:
        captured.update(
            {
                "source_id": source_id,
                "max_tokens": max_tokens,
                "version_hash": version_hash,
            }
        )
        return {"book_id": source_id, "n_nodes": 0, "n_chunks": 0}

    monkeypatch.setattr("vo_rag.cli.build._build_raptor_impl", fake_build_raptor_impl)

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert captured == {
        "source_id": "book-raptor",
        "max_tokens": expected_max_tokens,
        "version_hash": None,
    }


@pytest.mark.parametrize(
    ("impl_kwargs", "expected_summary_max_tokens"),
    [
        ({"version_hash": "vh-raptor"}, 2500),
        ({"version_hash": "vh-raptor", "max_tokens": 3200}, 3200),
    ],
)
@pytest.mark.asyncio
async def test_build_raptor_impl_wires_builder_and_closes_clients(
    monkeypatch: pytest.MonkeyPatch,
    impl_kwargs: dict[str, Any],
    expected_summary_max_tokens: int,
) -> None:
    from vo_rag.cli.build import _build_raptor_impl

    source_id = "book-raptor"
    chunks = [
        _fake_chunk(source_id).model_copy(
            update={
                "chunk_id": f"c{i}",
                "content": f"content-{i}",
                "embedding_text": f"embed-{i}" if i != 3 else "",
            }
        )
        for i in range(10)
    ]
    embedder = _ClosingFakeEmbedder()
    llm = _ClosingFakeLLM()
    store = _FakeStore()

    class RaptorEmbedder(_ClosingFakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[list[str], dict[str, Any]]] = []

        async def embed(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            self.calls.append((list(texts), dict(kwargs)))
            return {"dense": [[float(len(self.calls)), float(i)] for i in range(len(texts))]}

    embedder = RaptorEmbedder()

    monkeypatch.setattr("vo_rag.cli.build.QdrantStore", lambda *args, **kwargs: store)
    monkeypatch.setattr("vo_rag.cli.build.build_embedding_client", lambda *args, **kwargs: embedder)
    monkeypatch.setattr("vo_rag.cli.build.LLMClient", lambda *args, **kwargs: llm)

    async def fake_load_chunks(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        assert _store is store
        assert _source_id == source_id
        assert _collection == "vo_rag_books_raw_v1"
        assert version_hash == "vh-raptor"
        return chunks

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    captured_raptor_args: dict[str, Any] = {}
    fake_nodes = [SimpleNamespace(node_id=f"raptor-node-{i}") for i in range(4)]

    async def fake_build_raptor_for_corpus(**kwargs: Any) -> tuple[list[Any], list[Any]]:
        captured_raptor_args.update(kwargs)
        return [SimpleNamespace(nodes=fake_nodes, root=fake_nodes[-1])], []

    monkeypatch.setattr("vo_rag.cli.build.build_raptor_for_corpus", fake_build_raptor_for_corpus)

    result = await _build_raptor_impl(source_id, **impl_kwargs)

    leaf_embeddings = captured_raptor_args["books"][0][2]
    assert result == {"book_id": source_id, "n_nodes": len(fake_nodes), "n_chunks": len(chunks)}
    assert captured_raptor_args["books"] == [(source_id, chunks, leaf_embeddings)]
    assert len(leaf_embeddings) == len(chunks)
    assert captured_raptor_args["embed_client"] is embedder
    assert captured_raptor_args["store"] is store
    assert captured_raptor_args["max_concurrent_books"] == 1
    assert captured_raptor_args["config"].summary_model == "qwen3.6-27b-fp8"
    assert captured_raptor_args["config"].summary_max_tokens == expected_summary_max_tokens
    assert captured_raptor_args["config"].raptor_collection == "vo_rag_books_raptor_v1"
    assert embedder.calls == [
        (
            [
                "embed-0",
                "embed-1",
                "embed-2",
                "content-3",
                "embed-4",
                "embed-5",
                "embed-6",
                "embed-7",
            ],
            {"return_dense": True, "return_sparse": False, "return_colbert": False},
        ),
        (
            ["embed-8", "embed-9"],
            {"return_dense": True, "return_sparse": False, "return_colbert": False},
        ),
    ]
    assert embedder.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_build_raptor_uses_suffixed_output_collection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_raptor_impl

    monkeypatch.setenv("VORAG_BUILD_COLLECTION_SUFFIX", "__ab14b")
    source_id = "book-raptor-suffix"
    chunks = [_fake_chunk(source_id)]
    store = _FakeStore()
    captured_load_args: dict[str, Any] = {}
    captured_raptor_args: dict[str, Any] = {}

    class RaptorEmbedder(_ClosingFakeEmbedder):
        async def embed(self, texts: list[str], **_kwargs: Any) -> dict[str, Any]:
            return {"dense": [[1.0, float(i)] for i in range(len(texts))]}

    embedder = RaptorEmbedder()

    monkeypatch.setattr("vo_rag.cli.build.QdrantStore", lambda *args, **kwargs: store)
    monkeypatch.setattr("vo_rag.cli.build.build_embedding_client", lambda *args, **kwargs: embedder)

    async def fake_load_chunks(
        store_arg: Any,
        loaded_source_id: str,
        collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        captured_load_args.update(
            {
                "store": store_arg,
                "source_id": loaded_source_id,
                "collection": collection,
                "version_hash": version_hash,
            }
        )
        return chunks

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)

    async def fake_build_raptor_for_corpus(**kwargs: Any) -> tuple[list[Any], list[Any]]:
        captured_raptor_args.update(kwargs)
        node = SimpleNamespace(node_id="raptor-node")
        return [SimpleNamespace(nodes=[node], root=node)], []

    monkeypatch.setattr("vo_rag.cli.build.build_raptor_for_corpus", fake_build_raptor_for_corpus)

    result = await _build_raptor_impl(source_id, version_hash="vh-raptor")

    assert result == {"book_id": source_id, "n_nodes": 1, "n_chunks": len(chunks)}
    assert captured_load_args["collection"] == "vo_rag_books_raw_v1"
    assert captured_load_args["version_hash"] == "vh-raptor"
    assert captured_raptor_args["config"].raptor_collection == "vo_rag_books_raptor_v1__ab14b"


@pytest.mark.asyncio
async def test_build_raptor_impl_no_chunks_raises_click_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_raptor_impl

    embedder = _ClosingFakeEmbedder()
    llm = _ClosingFakeLLM()

    monkeypatch.setattr("vo_rag.cli.build.build_embedding_client", lambda *args, **kwargs: embedder)
    monkeypatch.setattr("vo_rag.cli.build.LLMClient", lambda *args, **kwargs: llm)

    async def fake_load_chunks_empty(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        return []

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks_empty)

    with pytest.raises(click.ClickException, match="No chunks found"):
        await _build_raptor_impl("missing")

    assert embedder.closed is True
    assert llm.closed is True


def test_build_graph_cli_uses_direct_core_path_not_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI ``build graph`` must invoke ``_build_graph_impl`` directly."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_build_graph_impl(
        source_id: str,
        *,
        max_tokens: int,
        chunks_per_call: int,
        max_concurrent: int,
        version_hash: str | None = None,
    ) -> dict[str, Any]:
        calls.append(
            (
                "direct",
                {
                    "source_id": source_id,
                    "max_tokens": max_tokens,
                    "chunks_per_call": chunks_per_call,
                    "max_concurrent": max_concurrent,
                    "version_hash": version_hash,
                },
            )
        )
        return {
            "book_id": source_id,
            "n_nodes": 1,
            "n_edges": 0,
            "direction_coverage": 1.0,
            "n_low_confidence_edges": 0,
        }

    monkeypatch.setattr("vo_rag.cli.build._build_graph_impl", fake_build_graph_impl)

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "graph", "--source-id", "book-direct"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][0] == "direct"
    assert calls[0][1]["source_id"] == "book-direct"
    assert "book_id: book-direct" in result.output


def test_build_summary_cli_uses_direct_core_path_not_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI ``build summary`` must invoke ``_build_summary_impl`` directly."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_build_summary_impl(
        source_id: str,
        *,
        max_concurrent: int,
        version_hash: str | None = None,
    ) -> dict[str, Any]:
        calls.append(
            (
                "direct",
                {
                    "source_id": source_id,
                    "max_concurrent": max_concurrent,
                    "version_hash": version_hash,
                },
            )
        )
        return {"book_id": source_id, "n_nodes": 2, "n_chunks": 3}

    monkeypatch.setattr("vo_rag.cli.build._build_summary_impl", fake_build_summary_impl)

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "summary", "--source-id", "book-direct"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][0] == "direct"
    assert calls[0][1]["source_id"] == "book-direct"
    assert "book_id: book-direct" in result.output


def test_build_dsi_no_chunks_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_load_chunks_empty(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        return []

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks_empty)

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "dsi", "--source-id", "missing"])
    assert result.exit_code != 0
    assert "No chunks found" in result.output


def test_build_graph_no_chunks_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_load_chunks_empty(
        _store: Any,
        _source_id: str,
        _collection: str,
        *,
        version_hash: str | None = None,
    ) -> list[Any]:
        return []

    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks_empty)

    async def fake_ensure_graph_collections(_client: Any, **_kwargs: Any) -> tuple[bool, bool]:
        return False, False

    monkeypatch.setattr("vo_rag.cli.build.ensure_graph_collections", fake_ensure_graph_collections)

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "graph", "--source-id", "missing"])
    assert result.exit_code != 0
    assert "No chunks found" in result.output


class _CapturingLLM:
    """Records the kwargs of the last complete() call for adapter assertions."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    async def complete(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return SimpleNamespace(text="{}", input_tokens=1, output_tokens=1, latency_ms=1)

    async def aclose(self) -> None:
        pass


def test_structured_output_extra_defaults_schema_to_response_format_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _structured_output_extra

    monkeypatch.delenv("VORAG_GUIDED_DECODING_MODE", raising=False)
    schema = {"type": "object", "required": ["kind"]}

    extra = _structured_output_extra(schema)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": schema},
    }
    assert "guided_json" not in extra


def test_structured_output_extra_guided_json_mode_uses_legacy_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _structured_output_extra

    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "guided_json")
    schema = {"type": "object", "required": ["kind"]}

    extra = _structured_output_extra(schema)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["guided_json"] == schema
    assert "response_format" not in extra


def test_structured_output_extra_json_object_mode_drops_schema_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _structured_output_extra

    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "json_object")
    schema = {"type": "object", "required": ["kind"]}

    extra = _structured_output_extra(schema)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["response_format"] == {"type": "json_object"}
    assert "guided_json" not in extra


def test_structured_output_extra_native_json_schema_mode_uses_top_level_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp native mode sends the schema as a top-level ``json_schema`` field.

    The local daily-driver (llama.cpp build b8234) ignores OpenAI-compatible
    ``response_format: json_schema`` for grammar enforcement (it decodes free text
    and 500s on the strict parse), but honors its native top-level ``json_schema``
    request field. This mode emits that field so advanced-layer extraction stays
    grammar-constrained against the local endpoint (VOR-302).
    """
    from vo_rag.cli.build import _structured_output_extra

    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "native_json_schema")
    schema = {"type": "object", "required": ["kind"]}

    extra = _structured_output_extra(schema)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["json_schema"] == schema
    assert "response_format" not in extra
    assert "guided_json" not in extra


def test_structured_output_extra_without_schema_uses_json_object_in_any_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _structured_output_extra

    monkeypatch.setenv("VORAG_GUIDED_DECODING_MODE", "guided_json")

    extra = _structured_output_extra(None)

    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["response_format"] == {"type": "json_object"}
    assert "guided_json" not in extra


@pytest.mark.asyncio
async def test_llm_adapter_passes_schema_response_format_and_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a schema is supplied, the adapter defaults to response_format json_schema."""
    from vo_rag.cli.build import _llm_call_from_client

    monkeypatch.delenv("VORAG_GUIDED_DECODING_MODE", raising=False)
    client = _CapturingLLM()
    schema = {"type": "object", "required": ["nodes", "edges"]}
    llm_call = await _llm_call_from_client(client, guided_json=schema)  # type: ignore[arg-type]
    await llm_call("sys", "usr", 100, 0.2)

    extra = client.last_kwargs["extra"]
    assert extra["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": schema},
    }
    assert "guided_json" not in extra
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_llm_adapter_falls_back_to_json_object_without_schema() -> None:
    """Without a schema, the adapter still enforces valid JSON via response_format
    json_object (the minimum structured-output discipline DSI/summary inherit)."""
    from vo_rag.cli.build import _llm_call_from_client

    client = _CapturingLLM()
    llm_call = await _llm_call_from_client(client)  # type: ignore[arg-type]
    await llm_call("sys", "usr", 100, 0.2)

    extra = client.last_kwargs["extra"]
    assert extra["response_format"] == {"type": "json_object"}
    assert "guided_json" not in extra
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
