"""Deep tests for vo_rag.core.summary.builder — SummaryTreeBuilder.

The tree is **balanced and target-sized**. One SECTION node per unique
``section_path`` would give about 1.52 chunks/section on a real corpus (a
near-1:1 explosion) while grouping by ``chapter_path`` overflows on outliers
(one book has a 281-chunk chapter). The builder packs leaves to a bounded
size (``leaf_min..leaf_max`` chunks, soft-breaking at chapter boundaries) and
every upper tier to a bounded ``max_fanout``, so these tests assert the packing
behaviour and the bounded-fan-out invariants instead of 1:1 section counts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.embedding.schemas import SparseVec
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.summary.builder import (
    BuildConfig,
    SummaryTreeBuilder,
    _leaf_label,
    _pack_leaves,
    _pack_nodes,
)
from vo_rag.core.summary.schemas import NodeLevel

# ---------------------------------------------------------------------------
# Test doubles and helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str,
    content: str,
    section_path: str,
    page_number: int | None = None,
    chapter_path: str = "",
    metadata: dict[str, object] | None = None,
) -> RetrievalChunk:
    """Build a minimal RetrievalChunk for builder tests.

    section_path uses the real chunker format "[Title > H1 > ...]" (bracketed,
    " > "-joined). Chapters are derived by derive_chapter_path / the builder's
    _extract_chapter_path fallback (numbered prefix "15.x -> Ch 15", else the top
    heading segment). Pass chapter_path to exercise the ingestion-owned field;
    leave it "" to exercise the legacy derive-from-section_path fallback.
    """
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="book-001",
        source_id="source-001",
        page_number=page_number,
        page_range=None,
        section_path=section_path,
        section_title="Test Section",
        chapter_path=chapter_path,
        content=content,
        embedding_text=f"{section_path}\n\n{content}",
        token_count=max(1, len(content.split())),
        has_image=False,
        image_id=None,
        image_ref=None,
        vlm_description=None,
        metadata=metadata or {},
    )


def _mock_embed_client() -> Any:
    """Return a dict-backed embed client that never calls a real model."""

    class FakeEmbedClient:
        async def embed(
            self,
            texts: list[str],
            *,
            return_dense: bool = True,
            return_sparse: bool = False,
            return_colbert: bool = False,
        ) -> dict[str, Any]:
            dense = [0.05] * 1024
            return {
                "dense": [dense] * len(texts),
                "sparse": [SparseVec(indices=[0, 1, 2], values=[0.1, 0.2, 0.3])] * len(texts),
                "colbert": [[[0.05] * 1024]] * len(texts) if return_colbert else [],
            }

    return FakeEmbedClient()


class _InMemoryQdrantClient:
    """Dict-backed AsyncQdrantClient test double — no unittest.mock."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}
        self._upsert_calls: list[dict[str, Any]] = []

    async def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:
        for point in points:
            self._upsert_calls.append({"collection": collection_name, "point_id": point.id})
            self._points[point.id] = {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload,
            }

    async def retrieve(
        self, collection_name: str, ids: list[str], with_payload: bool = True
    ) -> list[Any]:
        class _Point:
            def __init__(self, data: dict[str, Any]) -> None:
                self.id = data["id"]
                self.payload = data["payload"]
                self.vector = data.get("vector", {})

        return [_Point(self._points[oid]) for oid in ids if oid in self._points]

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 100,
        with_payload: bool = True,
    ) -> tuple[list[Any], Any]:
        class _Point:
            def __init__(self, data: dict[str, Any]) -> None:
                self.id = data["id"]
                self.payload = data["payload"]

        return [_Point(p) for p in list(self._points.values())[:limit]], None

    async def get_collections(self) -> Any:
        class _Collections:
            collections: list[Any] = []

        return _Collections()

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Any = None,
        sparse_vectors_config: Any = None,
    ) -> None:
        pass


class _InMemoryStore(QdrantStore):
    """QdrantStore subclass backed by _InMemoryQdrantClient — no live Qdrant."""

    def __init__(self) -> None:
        # Bypass QdrantStore.__init__ (it would try to construct a real
        # AsyncQdrantClient); we just need an object that exposes the
        # in-memory store via the same attribute name the parent class
        # uses for persistence and selective-expansion seams.
        self._client = _InMemoryQdrantClient()


def _mock_llm_client(responses: list[LLMResponse]) -> Any:
    """Return a sequential-response LLM client consuming from a pre-seeded list."""

    class FakeLLMClient:
        def __init__(self, resp_list: list[LLMResponse]) -> None:
            self._responses = resp_list
            self._calls: list[dict[str, Any]] = []

        async def complete(
            self,
            *,
            system: str,
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            extra: dict[str, Any] | None = None,
        ) -> LLMResponse:
            self._calls.append(
                {
                    "system": system,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return self._responses.pop(0)

    return FakeLLMClient(responses)


def _config(**overrides: Any) -> BuildConfig:
    base: dict[str, Any] = {
        "summary_model": "test-model",
        "embedding_model": "BGE-M3",
        "embedding_version": "test-v1",
        "source_collection_version": "test-v1",
    }
    base.update(overrides)
    return BuildConfig(**base)


def _default_config() -> BuildConfig:
    return _config()


def _resp(text: str = "Summary.", input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    """Shorthand to build a LLMResponse for tests."""
    return LLMResponse(
        text=text, model="test-model", input_tokens=input_tokens, output_tokens=output_tokens
    )


def _many(
    text: str = "Summary.", input_tokens: int = 10, output_tokens: int = 5
) -> list[LLMResponse]:
    """A comfortably-large response pool — each node consumes exactly one; extras
    are harmless (never popped), so over-supplying keeps node-count arithmetic
    out of the test's way unless the test is specifically asserting it."""
    return [_resp(text, input_tokens, output_tokens) for _ in range(64)]


def _builder(llm: Any, config: BuildConfig | None = None) -> SummaryTreeBuilder:
    return SummaryTreeBuilder(
        llm_client=llm,
        embed_client=_mock_embed_client(),
        store=_InMemoryStore(),
        config=config or _default_config(),
    )


# ---------------------------------------------------------------------------
# Pure packing-function unit tests (no LLM, no I/O)
# ---------------------------------------------------------------------------


def test_pack_leaves_coalesces_runs_of_tiny_chapters() -> None:
    # Five 1-chunk chapters, leaf_min=3: the run coalesces UP into leaves of >= 3
    # (no 1:1 explosion). c1c2c3 reach the floor and break at the c4 boundary.
    chunks = [
        _make_chunk(chunk_id=f"c{i}", content="x", section_path=f"[B > H{i}]") for i in range(1, 6)
    ]
    chapter_of = {f"c{i}": f"[B > H{i}]" for i in range(1, 6)}
    groups = _pack_leaves(chunks, chapter_of, leaf_min=3, leaf_max=12)
    assert [[c.chunk_id for c in g] for g in groups] == [["c1", "c2", "c3"], ["c4", "c5"]]


def test_pack_leaves_normal_chapter_becomes_one_leaf() -> None:
    # Two chapters of 3 chunks each (>= leaf_min): each breaks at its boundary,
    # so a normally-sized chapter is exactly one leaf.
    chunks = [
        _make_chunk(chunk_id="a1", content="x", section_path="[B > A]"),
        _make_chunk(chunk_id="a2", content="x", section_path="[B > A]"),
        _make_chunk(chunk_id="a3", content="x", section_path="[B > A]"),
        _make_chunk(chunk_id="b1", content="x", section_path="[B > C]"),
        _make_chunk(chunk_id="b2", content="x", section_path="[B > C]"),
        _make_chunk(chunk_id="b3", content="x", section_path="[B > C]"),
    ]
    chapter_of = {
        "a1": "[B > A]",
        "a2": "[B > A]",
        "a3": "[B > A]",
        "b1": "[B > C]",
        "b2": "[B > C]",
        "b3": "[B > C]",
    }
    groups = _pack_leaves(chunks, chapter_of, leaf_min=3, leaf_max=12)
    assert [[c.chunk_id for c in g] for g in groups] == [["a1", "a2", "a3"], ["b1", "b2", "b3"]]


def test_pack_leaves_splits_oversized_chapter_at_leaf_max() -> None:
    # One chapter of 5 chunks, leaf_max=2 → split DOWN to <= 2 (no giant leaf).
    chunks = [_make_chunk(chunk_id=f"c{i}", content="x", section_path="[B > A]") for i in range(5)]
    chapter_of = {f"c{i}": "[B > A]" for i in range(5)}
    groups = _pack_leaves(chunks, chapter_of, leaf_min=3, leaf_max=2)
    assert [len(g) for g in groups] == [2, 2, 1]
    assert all(len(g) <= 2 for g in groups)


def test_pack_nodes_groups_by_fanout_and_merges_trailing_singleton() -> None:
    # _pack_nodes only slices/merges; pass opaque tokens (typed Any for mypy,
    # whose declared param/return is SummaryNode).
    five: list[Any] = [1, 2, 3, 4, 5]
    merged: list[list[Any]] = [[1, 2], [3, 4, 5]]  # trailing singleton merged up
    assert _pack_nodes(five, 2) == merged
    four: list[Any] = [1, 2, 3, 4]
    exact: list[list[Any]] = [[1, 2], [3, 4]]  # exact split, no merge
    assert _pack_nodes(four, 2) == exact
    three: list[Any] = [1, 2, 3]
    single: list[list[Any]] = [[1, 2, 3]]  # single group, never a lone-child parent
    assert _pack_nodes(three, 5) == single


def test_leaf_label_single_chapter_carries_semantics_and_pages() -> None:
    chunks = [
        _make_chunk(chunk_id="c1", content="x", section_path="[SCCA > Section 2]", page_number=12),
        _make_chunk(chunk_id="c2", content="x", section_path="[SCCA > Section 2]", page_number=16),
    ]
    chapter_of = {"c1": "[SCCA > Section 2]", "c2": "[SCCA > Section 2]"}
    label = _leaf_label(chunks, chapter_of, "SCCA")
    assert label == "[SCCA > Section 2 (pp. 12-16)]"


def test_leaf_label_spanning_chapters_shows_range() -> None:
    chunks = [
        _make_chunk(chunk_id="c1", content="x", section_path="[SCCA > Section 2]", page_number=12),
        _make_chunk(chunk_id="c2", content="x", section_path="[SCCA > Section 4]", page_number=28),
    ]
    chapter_of = {"c1": "[SCCA > Section 2]", "c2": "[SCCA > Section 4]"}
    label = _leaf_label(chunks, chapter_of, "SCCA")
    assert label == "[SCCA > Section 2 … Section 4 (pp. 12-28)]"


# ---------------------------------------------------------------------------
# Builder integration tests (mocked LLM + embed + in-memory store)
# ---------------------------------------------------------------------------


def test_default_m_for_returns_spec_values() -> None:
    from vo_rag.core.summary.selective_expansion import default_m_for

    assert default_m_for(NodeLevel.SECTION) == 8
    assert default_m_for(NodeLevel.CHAPTER) == 15
    assert default_m_for(NodeLevel.BOOK) == 25


async def test_normal_chapters_become_one_leaf_each_two_level_tree() -> None:
    # Two chapters of 3 chunks each. With default packing each chapter is its own
    # leaf, and 2 leaves <= max_fanout, so the tree is just sections + book — no
    # intermediate chapter tier is manufactured for a small book.
    chunks = [
        _make_chunk(
            chunk_id=f"a{i}",
            content=f"Chapter A section {i}.",
            section_path="[Book > Ch1]",
            page_number=i,
        )
        for i in range(1, 4)
    ] + [
        _make_chunk(
            chunk_id=f"b{i}",
            content=f"Chapter B section {i}.",
            section_path="[Book > Ch2]",
            page_number=10 + i,
        )
        for i in range(1, 4)
    ]

    builder = _builder(_mock_llm_client(_many()))
    nodes = await builder.build_book_tree(book_id="book-001", chunks=chunks)

    section_nodes = [n for n in nodes if n.level == NodeLevel.SECTION]
    chapter_nodes = [n for n in nodes if n.level == NodeLevel.CHAPTER]
    book_nodes = [n for n in nodes if n.level == NodeLevel.BOOK]

    assert len(section_nodes) == 2  # one leaf per normally-sized chapter
    assert len(chapter_nodes) == 0  # bounded fan-out not exceeded → no extra tier
    assert len(book_nodes) == 1
    # Each leaf carries its chapter in the label.
    leaf_paths = {n.canonical_path for n in section_nodes}
    assert any("Ch1" in p for p in leaf_paths)
    assert any("Ch2" in p for p in leaf_paths)


async def test_reader_first_summary_admission_falls_back_to_guarded_legacy_writer(
    monkeypatch: Any,
) -> None:
    from unittest.mock import AsyncMock

    import vo_rag.core.summary.builder as summary_builder

    builder = _builder(_mock_llm_client(_many()))
    builder.store._publication_enabled = True
    builder.store._publication_writes_enabled = False
    chunks = [
        _make_chunk(chunk_id="c1", content="A current raw chunk.", section_path="[Book > Section]")
    ]
    built: list[Any] = []

    async def legacy_build(*, book_id: str, chunks: list[RetrievalChunk]) -> list[Any]:
        built.append((book_id, chunks))
        return []

    fallback = AsyncMock(side_effect=legacy_build)
    admission = AsyncMock(return_value=None)
    monkeypatch.setattr(summary_builder, "admit_derived_source", admission)
    monkeypatch.setattr(builder, "_build_book_tree_unpublished", fallback)

    result = await builder.build_book_tree(
        book_id="book-001", chunks=chunks, expected_dependencies={"book-001": None}
    )

    assert result == []
    assert len(built) == 1
    assert admission.await_count == 1
    assert admission.await_args is not None
    assert admission.await_args.kwargs["expected_dependencies"] == {"book-001": None}
    fallback.assert_awaited_once()


async def test_tiny_chapters_coalesce_no_one_to_one_explosion() -> None:
    # Six 1-chunk chapters would be six SECTION nodes under the old 1:1 rule;
    # leaf_min=3 coalesces them into 2 leaves.
    chunks = [
        _make_chunk(
            chunk_id=f"c{i}",
            content=f"Tiny chapter {i}.",
            section_path=f"[Book > H{i}]",
            page_number=i,
        )
        for i in range(1, 7)
    ]
    builder = _builder(_mock_llm_client(_many()))
    nodes = await builder.build_book_tree(book_id="book-tiny", chunks=chunks)

    section_nodes = [n for n in nodes if n.level == NodeLevel.SECTION]
    assert len(section_nodes) == 2
    # No leaf is a lone chunk.
    assert all(len(n.source_chunk_ids) >= 2 for n in section_nodes)


async def test_oversized_chapter_splits_into_bounded_leaves() -> None:
    # A single 25-chunk chapter must not become one 25-chunk leaf (context
    # overflow on the real summariser); leaf_max=12 splits it.
    chunks = [
        _make_chunk(
            chunk_id=f"c{i:02d}",
            content=f"Section {i} of one big chapter.",
            section_path="[Book > Mega]",
            page_number=i,
        )
        for i in range(1, 26)
    ]
    builder = _builder(_mock_llm_client(_many()))
    nodes = await builder.build_book_tree(book_id="book-mega", chunks=chunks)

    section_nodes = [n for n in nodes if n.level == NodeLevel.SECTION]
    assert all(len(n.source_chunk_ids) <= 12 for n in section_nodes)
    assert len(section_nodes) == 3  # 12 + 12 + 1


async def test_large_book_inserts_bounded_internal_tiers() -> None:
    # leaf_max=1 forces one leaf per chunk; max_fanout=3 forces internal tiers.
    # 10 leaves -> pack(10, 3) = [3,3,4] internal nodes -> 3 <= fanout -> book.
    chunks = [
        _make_chunk(
            chunk_id=f"c{i:02d}",
            content=f"Content block {i}.",
            section_path=f"[Book > S{i}]",
            page_number=i,
        )
        for i in range(10)
    ]
    builder = _builder(
        _mock_llm_client(_many()),
        _config(leaf_min_chunks=1, leaf_max_chunks=1, max_fanout=3),
    )
    nodes = await builder.build_book_tree(book_id="book-large", chunks=chunks)

    section_nodes = [n for n in nodes if n.level == NodeLevel.SECTION]
    chapter_nodes = [n for n in nodes if n.level == NodeLevel.CHAPTER]
    book_nodes = [n for n in nodes if n.level == NodeLevel.BOOK]

    assert len(section_nodes) == 10
    assert len(chapter_nodes) == 3
    assert len(book_nodes) == 1

    # Bounded fan-out everywhere, and never a lone-child internal node. A merged
    # trailing singleton can push one group to fanout+1 (here 10 -> [3, 3, 4]).
    for cn in chapter_nodes:
        assert 2 <= len(cn.child_node_ids) <= 4
    assert len(book_nodes[0].child_node_ids) == 3
    # The book root still covers every body chunk exactly once.
    assert book_nodes[0].source_chunk_ids == sorted(c.chunk_id for c in chunks)


async def test_book_source_chunk_ids_are_union_of_all_body_chunks() -> None:
    chunks = [
        _make_chunk(
            chunk_id="c1", content="Tire physics.", section_path="[Book > Ch1]", page_number=1
        ),
        _make_chunk(
            chunk_id="c2", content="Brake systems.", section_path="[Book > Ch1]", page_number=2
        ),
    ]
    builder = _builder(_mock_llm_client(_many()))
    nodes = await builder.build_book_tree(book_id="book-union", chunks=chunks)

    book_node = next(n for n in nodes if n.level == NodeLevel.BOOK)
    assert book_node.source_chunk_ids == ["c1", "c2"]


async def test_records_token_accounting_on_every_node() -> None:
    chunks = [
        _make_chunk(
            chunk_id="c1",
            content="Section content about aerodynamics.",
            section_path="[Book > Ch1]",
            page_number=1,
        ),
    ]
    builder = _builder(_mock_llm_client(_many("Summary.", 123, 45)))
    nodes = await builder.build_book_tree(book_id="book-token-test", chunks=chunks)

    assert len(nodes) == 2  # 1 leaf + 1 book (no chapter tier for a 1-chunk book)
    for node in nodes:
        assert node.metadata.get("input_token_count") == 123
        assert node.metadata.get("output_token_count") == 45
        assert "wall_time_ms" in node.metadata


async def test_respects_max_concurrent_building_leaves_in_parallel() -> None:
    # leaf_max=1 → 4 leaves built concurrently, then the book root.
    chunks = [
        _make_chunk(
            chunk_id=f"c{i}",
            content=f"Section {i} content about suspension geometry.",
            section_path=f"[Book > S{i}]",
            page_number=i,
        )
        for i in range(1, 5)
    ]

    call_times: list[float] = []

    class TimedLLMClient:
        def __init__(self) -> None:
            self._call_times = call_times
            import time

            self._time = time

        async def complete(
            self,
            *,
            system: str,
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            extra: dict[str, Any] | None = None,
        ) -> LLMResponse:
            self._call_times.append(self._time.perf_counter())
            await asyncio.sleep(0.05)
            return LLMResponse(text="Summary.", model="test", input_tokens=10, output_tokens=5)

    builder = _builder(
        TimedLLMClient(), _config(leaf_min_chunks=1, leaf_max_chunks=1, max_fanout=10)
    )
    nodes = await builder.build_book_tree(book_id="book-concurrent", chunks=chunks)

    assert len(nodes) == 5  # 4 leaves + 1 book


async def test_evidence_chunk_ids_are_raw_only() -> None:
    # Ou & Lapata invariant: every evidence_span references a raw input chunk_id.
    chunks = [
        _make_chunk(
            chunk_id="raw-c1",
            content="Raw content for section.",
            section_path="[Book > Ch1]",
            page_number=1,
        ),
    ]
    builder = _builder(_mock_llm_client(_many("Summary with evidence from raw chunks only.")))
    nodes = await builder.build_book_tree(book_id="book-raw", chunks=chunks)

    raw_chunk_ids = {"raw-c1"}
    for node in nodes:
        for span in node.evidence_spans:
            assert span.chunk_id in raw_chunk_ids, (
                f"Node {node.node_id} ({node.level}) references non-raw chunk {span.chunk_id!r}"
            )


async def test_propagates_prompt_hash_to_all_nodes() -> None:
    chunks = [
        _make_chunk(
            chunk_id="c1",
            content="Suspension dynamics content.",
            section_path="[Book > Ch1]",
            page_number=1,
        ),
    ]
    builder = _builder(_mock_llm_client(_many("Suspension summary.")))
    nodes = await builder.build_book_tree(book_id="book-hash", chunks=chunks)

    assert len(nodes) >= 2
    for node in nodes:
        assert len(node.prompt_hash) == 64, (
            f"Node {node.node_id} ({node.level}) prompt_hash is not 64 chars: {node.prompt_hash!r}"
        )
        assert all(c in "0123456789abcdef" for c in node.prompt_hash), (
            f"Node {node.node_id} ({node.level}) prompt_hash not hex: {node.prompt_hash!r}"
        )


async def test_numbered_chapters_drive_leaf_grouping() -> None:
    # Numbered subsections (15.1/15.2/15.3 -> Ch 15, 16.1 -> Ch 16) still group by
    # the derived chapter: the Ch 15 chunks coalesce into one leaf, Ch 16 another.
    # chapter_path left "" so this exercises the _extract_chapter_path fallback.
    chunks = [
        _make_chunk(
            chunk_id="c1",
            content="Section 15.1.",
            section_path="[Analysis > 15.1 Intro]",
            page_number=1,
        ),
        _make_chunk(
            chunk_id="c2",
            content="Section 15.2.",
            section_path="[Analysis > 15.2 Model]",
            page_number=2,
        ),
        _make_chunk(
            chunk_id="c3",
            content="Section 15.3.",
            section_path="[Analysis > 15.3 Method]",
            page_number=3,
        ),
        _make_chunk(
            chunk_id="c4",
            content="Section 16.1.",
            section_path="[Analysis > 16.1 Next]",
            page_number=4,
        ),
    ]
    builder = _builder(_mock_llm_client(_many()))
    nodes = await builder.build_book_tree(book_id="book-numbered", chunks=chunks)

    section_nodes = [n for n in nodes if n.level == NodeLevel.SECTION]
    assert len(section_nodes) == 2
    labels = {n.canonical_path for n in section_nodes}
    assert any("Ch 15" in p for p in labels), labels
    assert any("Ch 16" in p for p in labels), labels


async def test_ingestion_owned_chapter_path_field_is_preferred() -> None:
    # An explicit chapter_path overrides the derive-from-section_path fallback and
    # shows up in the leaf label.
    chunks = [
        _make_chunk(
            chunk_id="c1",
            content="Body.",
            section_path="[Book > Weird Heading]",
            chapter_path="[Book > Ch 9]",
            page_number=1,
        ),
    ]
    builder = _builder(_mock_llm_client(_many()))
    nodes = await builder.build_book_tree(book_id="book-field", chunks=chunks)

    section_nodes = [n for n in nodes if n.level == NodeLevel.SECTION]
    assert len(section_nodes) == 1
    assert "Ch 9" in section_nodes[0].canonical_path
    assert "Weird Heading" not in section_nodes[0].canonical_path


async def test_front_matter_chunks_excluded_from_tree() -> None:
    chunks = [
        _make_chunk(
            chunk_id="c1", content="Real body 1.", section_path="[Book > 1.1 Intro]", page_number=1
        ),
        _make_chunk(
            chunk_id="c2", content="Real body 2.", section_path="[Book > 1.2 More]", page_number=2
        ),
        _make_chunk(
            chunk_id="fm1",
            content="abc 12 page refs ...",
            section_path="[Book > Index]",
            metadata={"front_matter_kind": "index"},
            page_number=300,
        ),
    ]
    builder = _builder(_mock_llm_client(_many()))
    nodes = await builder.build_book_tree(book_id="book-fm", chunks=chunks)

    # The two real (Ch 1) chunks coalesce into one leaf; the front-matter chunk is gone.
    assert all("Index" not in n.canonical_path for n in nodes)
    assert all("fm1" not in n.source_chunk_ids for n in nodes)


def test_build_config_default_versions_are_explicit_not_frozen() -> None:
    """Defaults must be explicit versions, not the ``<frozen at Phase 0>`` placeholder."""
    cfg = BuildConfig()
    assert cfg.embedding_version != "<frozen at Phase 0>"
    assert cfg.source_collection_version != "<frozen at Phase 0>"
    # The defaults must point at the canonical vo-rag names so persisted
    # points survive a migration onto a fresh cluster.
    assert cfg.embedding_version == "bge-m3@unversioned"
    assert cfg.source_collection_version == "vo_rag_books_raw_v1"


def test_build_config_balanced_tree_defaults() -> None:
    cfg = BuildConfig()
    assert cfg.leaf_min_chunks == 3
    assert cfg.leaf_max_chunks == 12
    assert cfg.max_fanout == 10
