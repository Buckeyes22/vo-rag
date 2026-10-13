"""Tests for synthetic_query_prompts, synthetic_queries, and fixtures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.llm.response import LLMResponse

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "eval" / "book_rag"


# ── Fixture existence ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename",
    [
        "queries.jsonl",
        "qrels.tsv",
        "baseline.json",
        "retrieval_smoke.json",
    ],
)
def test_fixture_exists(filename: str) -> None:
    path = FIXTURES / filename
    assert path.exists(), f"Fixture {filename} not found at {path}"


# ── Prompt hash ────────────────────────────────────────────────────────────


def test_prompt_hash_is_64_char_hex() -> None:
    from vo_rag.pipelines.book_rag.synthetic_query_prompts import (
        SYNTHETIC_QUERY_PROMPT_HASH,
    )

    assert len(SYNTHETIC_QUERY_PROMPT_HASH) == 64
    assert int(SYNTHETIC_QUERY_PROMPT_HASH, 16) >= 0


def test_changing_prompt_changes_hash() -> None:
    from hashlib import sha256

    from vo_rag.pipelines.book_rag.synthetic_query_prompts import (
        SYNTHETIC_QUERY_PROMPT_HASH,
        SYNTHETIC_QUERY_SYSTEM_PROMPT,
        SYNTHETIC_QUERY_USER_TEMPLATE,
    )

    original = sha256(
        (SYNTHETIC_QUERY_SYSTEM_PROMPT + SYNTHETIC_QUERY_USER_TEMPLATE).encode()
    ).hexdigest()
    assert original == SYNTHETIC_QUERY_PROMPT_HASH

    modified = sha256(
        (SYNTHETIC_QUERY_SYSTEM_PROMPT + SYNTHETIC_QUERY_USER_TEMPLATE + "\nextra").encode()
    ).hexdigest()
    assert modified != original


# ── SyntheticQueryConfig prompt_hash validation ────────────────────────────


def test_config_raises_on_hash_mismatch() -> None:
    from vo_rag.pipelines.book_rag.synthetic_queries import SyntheticQueryConfig

    with pytest.raises(ValueError, match="prompt_hash"):
        SyntheticQueryConfig(prompt_hash="badhash")


# ── generate_query_batch shape + evidence subset ───────────────────────────


@pytest.mark.asyncio
async def test_generate_query_batch_returns_n_queries() -> None:
    from vo_rag.pipelines.book_rag.synthetic_queries import (
        SyntheticQueryConfig,
        generate_query_batch,
    )

    chunks = [
        _make_chunk("c1", "sec1", "content one"),
        _make_chunk("c2", "sec2", "content two"),
        _make_chunk("c3", "sec1", "content three"),
    ]

    async def fake_llm_call(
        system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        # Return a minimal valid JSON response
        return LLMResponse(
            text=json.dumps(
                {
                    "query_text": "What is content?",
                    "query_type": "factual",
                    "expected_answer_type": "short_fact",
                    "ambiguity_level": "low",
                    "requires_multi_chunk": False,
                    "requires_multi_book": False,
                    "requires_graph": False,
                    "requires_figure": False,
                    "requires_equation": False,
                    "requires_definition": False,
                    "domain": "general",
                    "difficulty": 2,
                    "evidence_chunk_ids": ["c1"],
                }
            ),
            model="fake",
        )

    records = await generate_query_batch(
        all_chunks=chunks,
        n_queries=3,
        config=SyntheticQueryConfig(),
        seed=42,
        llm_call=fake_llm_call,
        max_concurrent=4,
    )
    assert len(records) == 3
    for record in records:
        assert record.query is not None
        assert set(record.query.expected_supporting_chunk_ids) <= {"c1", "c2", "c3"}


# ── Bounded concurrency ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_concurrent_bounded() -> None:
    from vo_rag.pipelines.book_rag.synthetic_queries import (
        SyntheticQueryConfig,
        generate_query_batch,
    )

    chunks = [
        _make_chunk("c1", "sec1", "content one"),
        _make_chunk("c2", "sec2", "content two"),
        _make_chunk("c3", "sec1", "content three"),
    ]

    concurrency_log: list[int] = []
    max_seen = 0

    async def fake_llm_call(
        system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        concurrency_log.append(1)
        nonlocal max_seen
        max_seen = max(max_seen, len(concurrency_log))
        await asyncio.sleep(0.01)
        concurrency_log.pop()
        return _valid_llm_response()

    await generate_query_batch(
        all_chunks=chunks,
        n_queries=8,
        config=SyntheticQueryConfig(),
        seed=42,
        llm_call=fake_llm_call,
        max_concurrent=2,
    )
    assert max_seen <= 2


@pytest.mark.asyncio
async def test_generate_query_batch_cancels_sibling_tasks_on_first_failure() -> None:
    from vo_rag.pipelines.book_rag.synthetic_queries import (
        SyntheticQueryConfig,
        generate_query_batch,
    )

    chunks = [
        _make_chunk("c1", "sec1", "content one"),
        _make_chunk("c2", "sec2", "content two"),
        _make_chunk("c3", "sec1", "content three"),
    ]
    all_started = asyncio.Event()
    release = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    calls = 0

    async def fake_llm_call(
        system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        nonlocal calls
        calls += 1
        call_number = calls
        if calls == 2:
            all_started.set()
        if call_number == 1:
            await all_started.wait()
            raise RuntimeError("synthetic query failure")
        try:
            await release.wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        return _valid_llm_response()

    try:
        with pytest.raises(RuntimeError, match="synthetic query failure"):
            await generate_query_batch(
                all_chunks=chunks,
                n_queries=2,
                config=SyntheticQueryConfig(),
                seed=42,
                llm_call=fake_llm_call,
                max_concurrent=2,
            )
        assert sibling_cancelled.is_set()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_generate_query_batch_logs_secondary_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import synthetic_queries

    chunks = [
        _make_chunk("c1", "sec1", "content one"),
        _make_chunk("c2", "sec2", "content two"),
        _make_chunk("c3", "sec3", "content three"),
    ]
    all_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    warnings: list[tuple[str, dict[str, str]]] = []

    class _FakeLog:
        def warning(self, event: str, **kwargs: str) -> None:
            warnings.append((event, kwargs))

    async def fake_llm_call(
        system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        nonlocal calls
        calls += 1
        call_number = calls
        if calls == 3:
            all_started.set()
        await all_started.wait()
        if call_number in (1, 2):
            raise RuntimeError(f"synthetic query failure {call_number}")
        await release.wait()
        return _valid_llm_response()

    monkeypatch.setattr(synthetic_queries, "log", _FakeLog())

    try:
        with pytest.raises(RuntimeError) as exc_info:
            await synthetic_queries.generate_query_batch(
                all_chunks=chunks,
                n_queries=3,
                config=synthetic_queries.SyntheticQueryConfig(),
                seed=42,
                llm_call=fake_llm_call,
                max_concurrent=3,
            )
        logged_errors = {kwargs["error"] for _, kwargs in warnings}
        assert str(exc_info.value) not in logged_errors
        assert logged_errors <= {"synthetic query failure 1", "synthetic query failure 2"}
        assert logged_errors
        assert {event for event, _ in warnings} == {"book_rag.synthetic_query_straggler_failed"}
    finally:
        release.set()


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_chunk(chunk_id: str, section_path: str, content: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="d1",
        source_id="s1",
        section_path=section_path,
        section_title="Title",
        content=content,
        embedding_text=content,
        token_count=10,
    )


def _valid_llm_response() -> LLMResponse:
    import json

    return LLMResponse(
        text=json.dumps(
            {
                "query_text": "What is content?",
                "query_type": "factual",
                "expected_answer_type": "short_fact",
                "ambiguity_level": "low",
                "requires_multi_chunk": False,
                "requires_multi_book": False,
                "requires_graph": False,
                "requires_figure": False,
                "requires_equation": False,
                "requires_definition": False,
                "domain": "general",
                "difficulty": 2,
                "evidence_chunk_ids": ["c1"],
            }
        ),
        model="fake",
    )
