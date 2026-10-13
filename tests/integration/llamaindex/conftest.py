"""Shared fixtures + skip-guard for the LlamaIndex contract-test harness.

This subtree mirrors the adapter package under
``src/vo_rag/integrations/llamaindex/``; the tests are contract tests
(round-trip identity + parity). It carries the SHARED skip-guard
that lets the light unit CI collect the whole subtree as a no-op when
the ``llamaindex`` extra is absent:

    Light job (``uv sync --extra dev``): runs the existing ~1,618 tests;
    the LlamaIndex contract tests self-skip when the extra is absent.
    This job is the tripwire that the tier stays optional.

    Adapter job (``uv sync --extra dev --extra llamaindex``): runs
    Layers 1–2 and the wrapped eval-gate run (Layer 3).

The contract tests for the actual converters (VOR-176/177/178/179/180)
plug into the factories below; this conftest is the substrate they
share. The CI job wiring itself is VOR-175's unit — not this one.

Skip-guard semantics
--------------------
When the optional ``llamaindex`` extra is ABSENT (the light unit job),
every test module here that imports ``llama_index`` / the convert layer
at module top would raise ``ModuleNotFoundError`` at collection. The
``collect_ignore`` computed at the bottom of this module drops those
modules — but deliberately KEEPS ``test_import_safety.py`` (and anything
else in ``_LIGHT_JOB_SAFE``), the tripwire that must RUN in the light
job to prove ``import vo_rag.integrations.llamaindex`` stays
side-effect-free with the extra absent. The adapter job
(``uv sync --extra dev --extra llamaindex``) has the extra, so
``collect_ignore`` is empty and the full contract suite runs.

This is a curated ``collect_ignore`` (NOT a blanket
``collect_ignore_glob = ["*.py"]``): a blanket glob would also drop the
import-safety tripwire, leaving import safety unverified in the only job where the
extra is actually absent. New adapter tests need no per-file guard —
they auto-drop in the light job and run in the adapter job.

House style: fakes at seams, no GPU / no live services
-----------------------------------------------------
The harness is the substrate for Layer-1 round-trip ID-fidelity tests,
so the ``make_chunk`` factory MUST produce a real content-addressed
``chunk_id`` (via ``core/chunking/ids.py:_chunk_id``) — the
deterministic content-addressed handle that
``chunk_to_text_node`` is supposed to round-trip onto
``TextNode.id_``. Likewise the ``make_search_result`` factory
populates ``payload["chunk_id"]`` so the converter tests can assert
``node.id_ == payload["chunk_id"]`` — the load-bearing invariant
carried by ``search_result_to_node_with_score`` (``SearchResult.id`` is
the Qdrant point id, NOT the logical chunk id).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.retrieval.search import SearchResult

# Curated subtree skip-guard for the light unit job (optional `llamaindex`
# extra ABSENT). Every test module here that imports llama_index / the convert
# layer at module top would raise ModuleNotFoundError at collection, so we drop
# them — but deliberately KEEP the import-safety tripwire(s) listed in
# ``_LIGHT_JOB_SAFE``, which MUST run in the light job to prove
# ``import vo_rag.integrations.llamaindex`` stays side-effect-free with the
# extra absent (non-negotiable). A blanket ``collect_ignore_glob = ["*.py"]``
# would wrongly drop that tripwire too. The adapter job
# (``uv sync --extra dev --extra llamaindex``) has the extra, so
# ``collect_ignore`` is empty and the full contract suite runs. New adapter
# tests need no per-file guard: they auto-drop in the light job, run in the
# adapter job.
_LIGHT_JOB_SAFE = frozenset({"test_import_safety.py"})


def _llamaindex_extra_present() -> bool:
    try:
        return importlib.util.find_spec("llama_index.core") is not None
    except ModuleNotFoundError:
        return False


if not _llamaindex_extra_present():
    collect_ignore = [
        path.name
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name not in _LIGHT_JOB_SAFE
    ]

_DEFAULT_STAGE = "text"
_DEFAULT_SECTION_PATH = "Chapter 1 > Test"
_DEFAULT_CONTENT = "harness fixture body"
_DEFAULT_SOURCE_ID = "harness-source"
_DEFAULT_DOCUMENT_ID = "harness-doc"
_DEFAULT_COLLECTION = "harness_collection"
_DEFAULT_QDRANT_POINT_ID = "harness-point-1"


def _build_chunk(**overrides: Any) -> RetrievalChunk:
    """Build a fully-populated :class:`RetrievalChunk` with a real
    content-addressed ``chunk_id`` (deterministic, content-addressed via
    ``_chunk_id``; the same triple ``(source_id, content, stage)`` always
    yields the same ``chunk_id`` — the invariant the converter contract
    tests assert against).
    """
    source_id: str = overrides.pop("source_id", _DEFAULT_SOURCE_ID)
    document_id: str = overrides.pop("document_id", _DEFAULT_DOCUMENT_ID)
    section_path: str = overrides.pop("section_path", _DEFAULT_SECTION_PATH)
    section_title: str = overrides.pop("section_title", "Harness Section")
    chapter_path: str = overrides.pop("chapter_path", "Chapter 1")
    content: str = overrides.pop("content", _DEFAULT_CONTENT)
    stage: str = overrides.pop("stage", _DEFAULT_STAGE)
    chunk_type: str = overrides.pop("chunk_type", stage)
    embedding_text: str = overrides.pop("embedding_text", f"{section_path}\n\n{content}")
    token_count: int = overrides.pop("token_count", len(content.split()))

    chunk_id = _chunk_id(source_id, content, stage)
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        document_id=document_id,
        source_id=source_id,
        section_path=section_path,
        section_title=section_title,
        chapter_path=chapter_path,
        content=content,
        embedding_text=embedding_text,
        token_count=token_count,
        **overrides,
    )


def _build_search_result(**overrides: Any) -> SearchResult:
    """Build a :class:`SearchResult` whose ``payload`` carries the
    canonical fields the converter contract tests assert against:
    ``chunk_id``, ``content``, ``section_path``.

    If ``payload_chunk_id`` / ``payload_content`` / ``payload_section_path``
    are not provided, the factory synthesises a chunk-shaped triple and
    derives a matching content-addressed ``chunk_id`` so the resulting
    payload is byte-compatible with the contract tests.
    """
    qdrant_point_id: str = overrides.pop("id", _DEFAULT_QDRANT_POINT_ID)
    score: float = overrides.pop("score", 0.91)
    collection: str = overrides.pop("collection", _DEFAULT_COLLECTION)
    payload: dict[str, Any] | None = overrides.pop("payload", None)

    if payload is None:
        payload_chunk_id: str = overrides.pop(
            "payload_chunk_id", _chunk_id(_DEFAULT_SOURCE_ID, _DEFAULT_CONTENT, _DEFAULT_STAGE)
        )
        payload_content: str = overrides.pop("payload_content", _DEFAULT_CONTENT)
        payload_section_path: str = overrides.pop("payload_section_path", _DEFAULT_SECTION_PATH)
        extra_payload: dict[str, Any] = overrides.pop("extra_payload", {})
        payload = {
            "chunk_id": payload_chunk_id,
            "content": payload_content,
            "section_path": payload_section_path,
            **extra_payload,
        }
    else:
        for k in (
            "payload_chunk_id",
            "payload_content",
            "payload_section_path",
            "extra_payload",
        ):
            overrides.pop(k, None)

    return SearchResult(
        id=qdrant_point_id,
        score=score,
        collection=collection,
        payload=payload,
        **overrides,
    )


@pytest.fixture
def make_chunk() -> Callable[..., RetrievalChunk]:
    """Factory fixture: ``make_chunk(**overrides) -> RetrievalChunk``.

    The returned chunk carries a REAL content-addressed ``chunk_id``
    (deterministic over ``source_id | sha256(content) | stage`` via
    ``_chunk_id``) — so the converter contract tests can assert
    ``node.id_ == chunk.chunk_id`` (Layer-1 ID-fidelity invariant).

    Keyword overrides land directly on the ``RetrievalChunk`` model
    (e.g. ``section_path="A > B"``, ``content="body text"``) or in
    ``payload``-shaped factory helpers (``payload_*`` /
    ``extra_payload`` for ``make_search_result``).
    """
    return _build_chunk


@pytest.fixture
def make_search_result() -> Callable[..., SearchResult]:
    """Factory fixture: ``make_search_result(**overrides) -> SearchResult``.

    The returned result carries ``payload["chunk_id"]`` populated with a
    real content-addressed chunk id — so the converter contract tests
    can assert ``node.id_ == payload["chunk_id"]`` (``SearchResult.id``
    is the Qdrant point id and must NOT be used as ``node.id_``).
    """
    return _build_search_result


@pytest.fixture
def harness_factory() -> Iterator[Callable[..., RetrievalChunk | SearchResult]]:
    """Convenience: a single factory handle that dispatches on
    ``type=`` to either the chunk or the search-result builder.

    Mostly sugar — the converter tests typically want one or the other
    directly via ``make_chunk`` / ``make_search_result``. Kept here so
    the smoke test and any future shared "make a pair" helpers can
    call a single name.
    """

    def _build(kind: str = "chunk", **overrides: Any) -> RetrievalChunk | SearchResult:
        if kind == "chunk":
            return _build_chunk(**overrides)
        if kind == "search_result":
            return _build_search_result(**overrides)
        raise ValueError(f"harness_factory: unknown kind {kind!r}")

    yield _build


__all__ = [
    "harness_factory",
    "make_chunk",
    "make_search_result",
]

# NOTE: the research-agent fixtures plugin is registered via ``pytest_plugins``
# in the TOP-LEVEL ``tests/conftest.py`` (gated on the llamaindex extra). pytest
# no longer honours ``pytest_plugins`` in a non-top-level conftest — declaring it
# here raised a collection error on a full ``pytest tests/`` run. The top-level
# conftest is loaded as an ancestor even for ``pytest tests/integration/llamaindex/``,
# so the fixtures remain available to this subtree in the adapter job.
