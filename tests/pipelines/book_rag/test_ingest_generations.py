"""Cross-store ingest-generation activation and replacement invariants."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.idempotency import deterministic_hex
from vo_rag.core.models.envelope import IntakeEnvelope, IntakeSource
from vo_rag.core.models.ingest_generation import (
    IngestGenerationState,
    SourceIngestGeneration,
)
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag import workflow as workflow_module
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.validator import ValidationReport
from vo_rag.pipelines.book_rag.workflow import (
    BookRagContext,
    BookRagPipeline,
    IngestGenerationClaimed,
)

_UNSET = object()


async def _async_value(value: Any) -> Any:
    return value


class _Transaction:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    async def __aenter__(self) -> _Transaction:
        return self

    async def __aexit__(self, *args: object) -> None:
        exc_type = args[0]
        if exc_type is None:
            if self.conn.durable_after_exit is not _UNSET:
                self.conn.durable_active = self.conn.durable_after_exit
            elif self.conn.exit_error is None:
                self.conn.durable_active = "gen-new"
        if self.conn.exit_error is not None and exc_type is None:
            error = self.conn.exit_error
            self.conn.exit_error = None
            raise error
        return None


@dataclass
class _Conn:
    old_generation: str | None = "gen-old"
    fail_activation: bool = False
    exit_error: BaseException | None = None
    reconciliation_exit_error: BaseException | None = None
    durable_after_exit: object = _UNSET
    durable_active: object = _UNSET
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.durable_active is _UNSET:
            self.durable_active = self.old_generation

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        if "SELECT generation_id" in query and "LIMIT 1" in query:
            return (
                {"generation_id": self.durable_active}
                if isinstance(self.durable_active, str)
                else None
            )
        if "SELECT generation_id" in query:
            return (
                {"generation_id": self.old_generation} if self.old_generation is not None else None
            )
        if "SELECT state" in query:
            return {"state": "validated"}
        raise AssertionError(query)

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        if self.fail_activation and "SET state = 'active'" in query:
            return "UPDATE 0"
        return "UPDATE 1"


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    async def __aenter__(self) -> _Conn:
        return self.conn

    async def __aexit__(self, *args: object) -> None:
        return None


class _Pool:
    def __init__(self, conn: _Conn, *, fail_cleanup_record: bool = False) -> None:
        self.conn = conn
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.acquire_count = 0
        self.fail_cleanup_record = fail_cleanup_record

    def acquire(self) -> _Acquire:
        self.acquire_count += 1
        if self.acquire_count == 2 and self.conn.reconciliation_exit_error is not None:
            self.conn.exit_error = self.conn.reconciliation_exit_error
            self.conn.reconciliation_exit_error = None
        return _Acquire(self.conn)

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        if self.fail_cleanup_record and "cleanup_required = $2" in query:
            raise OSError("cleanup marker unavailable")
        return "UPDATE 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if "SELECT EXISTS" in query:
            return {"current": True}
        if "COUNT(*)" in query:
            return {"count": 1}
        assert "state = 'active'" in query
        assert args == ("source-1", "vo_rag_books_raw_v1")
        durable = self.conn.durable_active
        return {"generation_id": durable} if isinstance(durable, str) else None


class _Store:
    def __init__(
        self,
        *,
        cleanup_fails: bool = False,
        cancel_on_state_call: int | None = None,
        cancel_on_delete_call: int | None = None,
    ) -> None:
        self.state_calls: list[tuple[str, str, str, str]] = []
        self.count_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.legacy_state_calls: list[tuple[str, str, str]] = []
        self.delete_calls: list[tuple[str, str, str]] = []
        self.delete_legacy_calls: list[tuple[str, str]] = []
        self.cleanup_fails = cleanup_fails
        self.cancel_on_state_call = cancel_on_state_call
        self.cancel_on_delete_call = cancel_on_delete_call
        self.delete_call_count = 0
        self.deleted_generations: set[tuple[str, str]] = set()
        self.actual_states: dict[tuple[str, str], str] = {
            ("vo_rag_books_raw_v1", "gen-old"): "active",
            ("vo_rag_books_sterile_v1", "gen-old"): "active",
            ("vo_rag_books_raw_v1", "gen-new"): "staging",
            ("vo_rag_books_sterile_v1", "gen-new"): "staging",
        }

    async def set_generation_state(
        self,
        collection: str,
        source_id: str,
        generation_id: str,
        state: str,
    ) -> None:
        self.state_calls.append((collection, source_id, generation_id, str(state)))
        self.actual_states[(collection, generation_id)] = str(state)
        if self.cancel_on_state_call == len(self.state_calls):
            raise asyncio.CancelledError

    async def count_source(self, collection: str, source_id: str, **kwargs: Any) -> int:
        self.count_calls.append((collection, source_id, dict(kwargs)))
        del source_id
        generation_id = str(kwargs.get("generation_id", ""))
        if (collection, generation_id) in self.deleted_generations:
            return 0
        if (collection, generation_id) not in self.actual_states:
            return 0
        return 2

    async def set_legacy_source_state(
        self,
        collection: str,
        source_id: str,
        state: str,
    ) -> None:
        self.legacy_state_calls.append((collection, source_id, str(state)))

    async def delete_generation(
        self,
        collection: str,
        source_id: str,
        generation_id: str,
    ) -> None:
        self.delete_calls.append((collection, source_id, generation_id))
        self.deleted_generations.add((collection, generation_id))
        self.delete_call_count += 1
        if self.cancel_on_delete_call == self.delete_call_count:
            raise asyncio.CancelledError
        if self.cleanup_fails:
            raise OSError("cleanup unavailable")

    async def delete_legacy_source(self, collection: str, source_id: str) -> None:
        self.delete_legacy_calls.append((collection, source_id))
        self.delete_call_count += 1
        if self.cancel_on_delete_call == self.delete_call_count:
            raise asyncio.CancelledError
        if self.cleanup_fails:
            raise OSError("cleanup unavailable")


def _context() -> BookRagContext:
    envelope = IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "intake/book.pdf"},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return BookRagContext(
        envelope=envelope,
        config=BookRagConfig(sterilize_enabled=True),
        source_id="source-1",
        generation_id="gen-new",
        generation_owner_token="owner-1",
        generation_claim_acquired=True,
        raw_point_ids=["r1", "r2"],
        sterile_point_ids=["s1", "s2"],
    )


def _pipeline(conn: _Conn, store: _Store) -> BookRagPipeline:
    return BookRagPipeline(
        garage=SimpleNamespace(),
        qdrant_store=store,
        embedding_client=SimpleNamespace(),
        llm_client=None,
        retriever=SimpleNamespace(),
        parser_router=SimpleNamespace(),
        pg_pool=_Pool(conn),
        config=BookRagConfig(sterilize_enabled=True),
    )


def _generation_pool(pipeline: BookRagPipeline) -> _Pool:
    assert isinstance(pipeline.pg_pool, _Pool)
    return pipeline.pg_pool


@pytest.mark.asyncio
async def test_n_to_m_replacement_switches_then_deletes_only_old_generation() -> None:
    conn = _Conn()
    store = _Store()
    ctx = _context()
    pipeline = _pipeline(conn, store)

    await pipeline._activate_generation(ctx)

    assert store.state_calls[:2] == [
        ("vo_rag_books_raw_v1", "source-1", "gen-old", "superseded"),
        ("vo_rag_books_sterile_v1", "source-1", "gen-old", "superseded"),
    ]
    assert store.state_calls[2:4] == [
        ("vo_rag_books_raw_v1", "source-1", "gen-new", "active"),
        ("vo_rag_books_sterile_v1", "source-1", "gen-new", "active"),
    ]
    assert store.delete_calls == [
        ("vo_rag_books_raw_v1", "source-1", "gen-old"),
        ("vo_rag_books_sterile_v1", "source-1", "gen-old"),
    ]


@pytest.mark.asyncio
async def test_native_ingest_handler_keeps_ambiguously_committed_generation_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        exit_error=OSError("commit acknowledgement lost"),
        durable_after_exit="gen-new",
    )
    store = _Store()
    pipeline = _pipeline(conn, store)
    envelope = _context().envelope

    async def run_step(
        ctx: BookRagContext,
        step_name: str,
        _step: object,
    ) -> None:
        if step_name == "register_source":
            ctx.source_id = "source-1"
        elif step_name == "chunk_embed":
            ctx.raw_point_ids = ["r1", "r2"]
            ctx.sterile_point_ids = ["s1", "s2"]

    async def not_duplicate(_source_id: str) -> bool:
        return False

    async def start(ctx: BookRagContext) -> None:
        ctx.generation_id = "gen-new"
        ctx.generation_owner_token = "owner-1"
        ctx.generation_claim_acquired = True

    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setattr(pipeline, "_run_ingest_step", run_step)
    monkeypatch.setattr(pipeline, "_source_already_ingested", not_duplicate)
    monkeypatch.setattr(pipeline, "_start_generation", start)

    output = await pipeline.ingest(envelope)

    assert output.generation_state == IngestGenerationState.ACTIVE
    assert envelope.status.value == "completed"
    assert not any(
        generation_id == "gen-new" and state == "failed"
        for _collection, _source_id, generation_id, state in store.state_calls
    )
    assert store.legacy_state_calls == []
    assert store.delete_legacy_calls == [
        ("vo_rag_books_raw_v1", "source-1"),
        ("vo_rag_books_sterile_v1", "source-1"),
    ]
    sql = "\n".join(query for query, _args in conn.calls)
    assert "pg_advisory_xact_lock" in sql
    assert "SET state = 'active'" in sql
    assert "cleanup_required = TRUE" in sql
    assert any(
        args == ("gen-new", False, None) for _query, args in _generation_pool(pipeline).calls
    )


@pytest.mark.asyncio
async def test_double_activation_never_deletes_active_target_generation() -> None:
    conn = _Conn()
    store = _Store()
    pipeline = _pipeline(conn, store)

    await pipeline._activate_generation(_context())
    conn.old_generation = "gen-new"
    conn.durable_active = "gen-new"
    store.state_calls.clear()
    store.count_calls.clear()
    store.delete_calls.clear()
    store.delete_legacy_calls.clear()

    await pipeline._activate_generation(_context())

    assert store.delete_calls == []
    assert store.delete_legacy_calls == []
    assert store.state_calls == []
    assert [call[0] for call in store.count_calls] == [
        "vo_rag_books_raw_v1",
        "vo_rag_books_sterile_v1",
    ]
    assert all(call[2]["generation_id"] == "gen-new" for call in store.count_calls)
    assert store.actual_states[("vo_rag_books_raw_v1", "gen-new")] == "active"


@pytest.mark.asyncio
async def test_cleanup_failure_cannot_deactivate_new_generation() -> None:
    store = _Store(cleanup_fails=True)
    pipeline = _pipeline(_Conn(), store)

    await pipeline._activate_generation(_context())

    assert ("vo_rag_books_raw_v1", "source-1", "gen-new", "active") in store.state_calls
    assert all(call[2] != "gen-new" for call in store.delete_calls)
    generation_id, cleanup_required, cleanup_error = _generation_pool(pipeline).calls[-1][1]
    assert generation_id == "gen-new"
    assert cleanup_required is True
    assert cleanup_error == (
        "vo_rag_books_raw_v1: OSError; vo_rag_books_sterile_v1: OSError; "
        "legacy/vo_rag_books_raw_v1: OSError; legacy/vo_rag_books_sterile_v1: OSError"
    )


@pytest.mark.asyncio
async def test_cleanup_and_marker_failure_leave_transactional_retry_marker() -> None:
    conn = _Conn()
    store = _Store(cleanup_fails=True)
    pipeline = _pipeline(conn, store)
    _generation_pool(pipeline).fail_cleanup_record = True

    await pipeline._activate_generation(_context())

    activation_sql = "\n".join(query for query, _args in conn.calls)
    assert "cleanup_required = TRUE" in activation_sql
    assert "post-activation cleanup pending" in activation_sql
    assert any(
        "cleanup_required = $2" in query for query, _args in _generation_pool(pipeline).calls
    )


@pytest.mark.asyncio
async def test_pending_cleanup_is_retried_without_touching_active_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def superseded(
        _pool: object,
        _source_id: str,
        _raw_collection: str,
    ) -> list[str]:
        return ["gen-old"]

    monkeypatch.setattr(
        workflow_module,
        "get_superseded_generation_ids",
        superseded,
    )
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    pipeline.config.sterilize_enabled = False
    active = SourceIngestGeneration(
        generation_id="gen-new",
        source_id="source-1",
        raw_collection="vo_rag_books_raw_v1",
        envelope_id="env-new",
        schema_version="book-rag-points-v1",
        chunking_version="default",
        state=IngestGenerationState.ACTIVE,
        cleanup_required=True,
    )

    await pipeline._retry_pending_generation_cleanup(active)

    assert store.delete_calls == [
        ("vo_rag_books_raw_v1", "source-1", "gen-old"),
        ("vo_rag_books_sterile_v1", "source-1", "gen-old"),
    ]
    assert all(call[2] != "gen-new" for call in store.state_calls)
    assert any(
        args == ("gen-new", False, None) for _query, args in _generation_pool(pipeline).calls
    )


@pytest.mark.asyncio
async def test_enabled_to_disabled_replacement_retires_old_sterile_generation() -> None:
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    pipeline.config.sterilize_enabled = False
    ctx = _context()
    ctx.config.sterilize_enabled = False
    ctx.sterile_point_ids = []

    await pipeline._sterilize(ctx)
    await pipeline._activate_generation(ctx)

    assert (
        "vo_rag_books_sterile_v1",
        "source-1",
        "gen-old",
        "superseded",
    ) in store.state_calls
    assert (
        "vo_rag_books_sterile_v1",
        "source-1",
        "gen-old",
    ) in store.delete_calls


@pytest.mark.asyncio
async def test_disabled_retry_deletes_same_generation_sterile_residue_and_counts_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    pipeline.config.sterilize_enabled = False
    ctx = _context()
    ctx.config.sterilize_enabled = False
    active = SourceIngestGeneration(
        generation_id="gen-new",
        source_id="source-1",
        raw_collection="vo_rag_books_raw_v1",
        envelope_id="env-new",
        schema_version="book-rag-points-v1",
        chunking_version="default",
        sterilize_enabled=False,
        state=IngestGenerationState.ACTIVE,
        expected_raw_points=2,
        actual_raw_points=2,
        expected_sterile_points=0,
        actual_sterile_points=0,
        expected_artifacts=1,
        actual_artifacts=1,
    )

    async def current(
        _pool: object,
        _source_id: str,
        _raw_collection: str,
    ) -> SourceIngestGeneration:
        return active

    monkeypatch.setattr(workflow_module, "get_active_generation", current)
    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")

    await pipeline._sterilize(ctx)

    assert (
        "vo_rag_books_sterile_v1",
        "source-1",
        "gen-new",
    ) in store.delete_calls
    assert await pipeline._source_already_ingested("source-1") is True


@pytest.mark.asyncio
async def test_duplicate_requires_matching_sterilize_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(_Conn(), _Store())
    pipeline.config.sterilize_enabled = False
    active = SourceIngestGeneration(
        generation_id="gen-active",
        source_id="source-1",
        raw_collection="vo_rag_books_raw_v1",
        envelope_id="env-active",
        schema_version="book-rag-points-v1",
        chunking_version="default",
        sterilize_enabled=True,
        state=IngestGenerationState.ACTIVE,
    )

    async def current(
        _pool: object,
        _source_id: str,
        _raw_collection: str,
    ) -> SourceIngestGeneration:
        return active

    monkeypatch.setattr(workflow_module, "get_active_generation", current)
    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")

    assert await pipeline._source_already_ingested("source-1") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("visible_count", "expected_duplicate"),
    [(0, False), (1, True)],
)
async def test_flag_off_duplicate_gate_uses_only_active_or_legacy_points(
    monkeypatch: pytest.MonkeyPatch,
    visible_count: int,
    expected_duplicate: bool,
) -> None:
    class _VisibilityStore(_Store):
        async def count_active_or_legacy_source(
            self,
            collection: str,
            source_id: str,
        ) -> int:
            del collection, source_id
            return visible_count

        async def count_source(self, *args: object, **kwargs: object) -> int:
            raise AssertionError("mixed rollout must not count all source states")

    monkeypatch.delenv("VORAG_INGEST_GENERATIONS", raising=False)
    pipeline = _pipeline(_Conn(), _VisibilityStore())

    assert await pipeline._source_already_ingested("source-1") is expected_duplicate


@pytest.mark.asyncio
async def test_custom_collection_rejects_shared_sterile_generation() -> None:
    pipeline = _pipeline(_Conn(), _Store())
    pipeline.raw_collection = "vo_rag_reddit_v1"

    with pytest.raises(RuntimeError, match="only supported for the canonical"):
        await pipeline._start_generation(_context())


@pytest.mark.asyncio
async def test_same_source_generation_identity_is_collection_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[SourceIngestGeneration] = []

    async def no_active(
        _pool: object,
        _source_id: str,
        _raw_collection: str,
    ) -> None:
        return None

    async def capture(
        _pool: object,
        generation: SourceIngestGeneration,
    ) -> bool:
        created.append(generation)
        return True

    monkeypatch.setattr(workflow_module, "get_active_generation", no_active)
    monkeypatch.setattr(workflow_module, "create_generation", capture)
    canonical = _pipeline(_Conn(), _Store())
    canonical.config.sterilize_enabled = False
    custom = _pipeline(_Conn(), _Store())
    custom.config.sterilize_enabled = False
    custom.raw_collection = "vo_rag_reddit_v1"
    canonical_ctx = _context()
    canonical_ctx.config.sterilize_enabled = False
    custom_ctx = _context()
    custom_ctx.config.sterilize_enabled = False

    await canonical._start_generation(canonical_ctx)
    await custom._start_generation(custom_ctx)

    assert {generation.raw_collection for generation in created} == {
        "vo_rag_books_raw_v1",
        "vo_rag_reddit_v1",
    }
    assert canonical_ctx.generation_id != custom_ctx.generation_id


@pytest.mark.asyncio
async def test_active_create_conflict_stops_before_any_qdrant_or_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    envelope = _context().envelope
    artifact_writes: list[object] = []
    existing_generation_id = ""

    async def run_step(
        ctx: BookRagContext,
        step_name: str,
        _step: object,
    ) -> None:
        if step_name == "register_source":
            ctx.source_id = "source-1"
        elif step_name == "register_artifact":
            artifact_writes.append(ctx)

    active_reads = 0

    async def current_active(
        _pool: object,
        _source_id: str,
        _raw_collection: str,
    ) -> SourceIngestGeneration | None:
        nonlocal active_reads
        active_reads += 1
        if active_reads == 1:
            return None
        return await existing(_pool, existing_generation_id)

    async def lost_insert(
        _pool: object,
        _generation: SourceIngestGeneration,
    ) -> bool:
        return False

    async def existing(
        _pool: object,
        generation_id: str,
    ) -> SourceIngestGeneration:
        nonlocal existing_generation_id
        existing_generation_id = generation_id
        store.actual_states[("vo_rag_books_raw_v1", generation_id)] = "active"
        store.actual_states[("vo_rag_books_sterile_v1", generation_id)] = "active"
        return SourceIngestGeneration(
            generation_id=generation_id,
            source_id="source-1",
            raw_collection="vo_rag_books_raw_v1",
            envelope_id=str(envelope.id),
            schema_version="book-rag-points-v1",
            chunking_version="default",
            sterilize_enabled=True,
            state=IngestGenerationState.ACTIVE,
            expected_raw_points=2,
            actual_raw_points=2,
            expected_sterile_points=2,
            actual_sterile_points=2,
            expected_artifacts=1,
            actual_artifacts=1,
        )

    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setenv("VORAG_BOOK_RAG_FORCE_REINGEST", "1")
    monkeypatch.setattr(pipeline, "_run_ingest_step", run_step)
    monkeypatch.setattr(workflow_module, "get_active_generation", current_active)
    monkeypatch.setattr(workflow_module, "create_generation", lost_insert)
    monkeypatch.setattr(workflow_module, "get_generation", existing)
    monkeypatch.setattr(
        workflow_module,
        "count_active_generation_artifacts",
        lambda *_args: _async_value(1),
    )

    output = await pipeline.ingest(envelope)

    assert output.idempotent_noop is True
    assert output.generation_state == IngestGenerationState.ACTIVE
    assert store.state_calls == []
    assert store.delete_calls == []
    assert artifact_writes == []


@pytest.mark.asyncio
async def test_cancelled_owner_releases_staging_lease_for_immediate_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_owner: str | None = None

    async def no_active(
        _pool: object,
        _source_id: str,
        _raw_collection: str,
    ) -> None:
        return None

    async def claim(
        _pool: object,
        generation: SourceIngestGeneration,
    ) -> bool:
        nonlocal claim_owner
        if claim_owner is not None:
            return False
        claim_owner = generation.owner_token
        return True

    async def release(
        _pool: object,
        _generation_id: str,
        owner_token: str,
    ) -> bool:
        nonlocal claim_owner
        if claim_owner != owner_token:
            return False
        claim_owner = None
        return True

    pipeline = _pipeline(_Conn(), _Store())
    envelope = _context().envelope

    async def cancel_during_parse(
        ctx: BookRagContext,
        step_name: str,
        _step: object,
    ) -> None:
        if step_name == "register_source":
            ctx.source_id = "source-1"
        elif step_name == "parse":
            raise asyncio.CancelledError

    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setattr(pipeline, "_run_ingest_step", cancel_during_parse)
    monkeypatch.setattr(workflow_module, "get_active_generation", no_active)
    monkeypatch.setattr(workflow_module, "create_generation", claim)
    monkeypatch.setattr(workflow_module, "release_generation_claim", release)

    with pytest.raises(asyncio.CancelledError):
        await pipeline.ingest(envelope)

    assert claim_owner is None
    retry_ctx = _context()
    retry_ctx.source_id = "source-1"
    await pipeline._start_generation(retry_ctx)
    assert claim_owner == retry_ctx.generation_owner_token


@pytest.mark.asyncio
async def test_expired_takeover_fences_still_running_old_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(_Conn(), _Store())
    old_ctx = _context()
    old_ctx.generation_owner_token = "owner-old"
    updates: list[str] = []

    async def current(
        _pool: object,
        _generation_id: str,
        owner_token: str,
    ) -> bool:
        return owner_token == "owner-new"

    async def update(*_args: object, **_kwargs: object) -> bool:
        updates.append("updated")
        return True

    monkeypatch.setattr(workflow_module, "generation_claim_is_current", current)
    monkeypatch.setattr(workflow_module, "update_generation_counts", update)

    with pytest.raises(IngestGenerationClaimed, match="owner lease was lost"):
        await pipeline._record_generation_counts(
            old_ctx,
            expected_raw_points=2,
            actual_raw_points=2,
        )

    assert updates == []
    assert old_ctx.generation_lease_lost is True


@pytest.mark.asyncio
async def test_refresh_rejection_cancels_old_owner_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(_Conn(), _Store())
    ctx = _context()

    async def refresh_rejected(*_args: object) -> bool:
        return False

    async def old_owner() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(workflow_module, "GENERATION_CLAIM_HEARTBEAT_SECONDS", 0)
    monkeypatch.setattr(workflow_module, "refresh_generation_claim", refresh_rejected)
    owner_task = asyncio.create_task(old_owner())

    await pipeline._maintain_generation_lease(ctx, owner_task=owner_task)
    with pytest.raises(asyncio.CancelledError):
        await owner_task

    assert ctx.generation_lease_lost is True


@pytest.mark.asyncio
async def test_active_create_conflict_rejects_unreconciled_qdrant_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MismatchedStore(_Store):
        async def count_source(
            self,
            collection: str,
            source_id: str,
            **kwargs: Any,
        ) -> int:
            await super().count_source(collection, source_id, **kwargs)
            return 1 if collection == "vo_rag_books_raw_v1" else 2

    store = _MismatchedStore()
    pipeline = _pipeline(_Conn(), store)
    ctx = _context()
    active = SourceIngestGeneration(
        generation_id="gen-active",
        source_id="source-1",
        raw_collection="vo_rag_books_raw_v1",
        envelope_id=str(ctx.envelope.id),
        schema_version="book-rag-points-v1",
        chunking_version="default",
        sterilize_enabled=True,
        state=IngestGenerationState.ACTIVE,
        expected_raw_points=2,
        actual_raw_points=2,
        expected_sterile_points=2,
        actual_sterile_points=2,
        expected_artifacts=1,
        actual_artifacts=1,
    )

    async def current(
        _pool: object,
        _source_id: str,
        _raw_collection: str,
    ) -> SourceIngestGeneration:
        return active

    async def lost_insert(
        _pool: object,
        _generation: SourceIngestGeneration,
    ) -> bool:
        return False

    async def existing(
        _pool: object,
        _generation_id: str,
    ) -> SourceIngestGeneration:
        return active

    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setattr(workflow_module, "get_active_generation", current)
    monkeypatch.setattr(workflow_module, "create_generation", lost_insert)
    monkeypatch.setattr(workflow_module, "get_generation", existing)
    monkeypatch.setattr(
        workflow_module,
        "count_active_generation_artifacts",
        lambda *_args: _async_value(1),
    )
    ctx.source_id = "source-1"

    with pytest.raises(RuntimeError, match="not fully reconciled"):
        await pipeline._start_generation(ctx)

    assert store.state_calls == []
    assert store.delete_calls == []


@pytest.mark.asyncio
async def test_active_create_conflict_rejects_nonactive_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(_Conn(), _Store())
    ctx = _context()
    active = SourceIngestGeneration(
        generation_id="gen-active",
        source_id="source-1",
        raw_collection="vo_rag_books_raw_v1",
        envelope_id=str(ctx.envelope.id),
        schema_version="book-rag-points-v1",
        chunking_version="default",
        sterilize_enabled=True,
        state=IngestGenerationState.ACTIVE,
        expected_raw_points=2,
        actual_raw_points=2,
        expected_sterile_points=2,
        actual_sterile_points=2,
        expected_artifacts=1,
        actual_artifacts=1,
    )

    async def current(*_args: object) -> SourceIngestGeneration:
        return active

    async def lost_insert(*_args: object) -> bool:
        return False

    async def no_active_artifact(*_args: object) -> int:
        return 0

    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setattr(workflow_module, "get_active_generation", current)
    monkeypatch.setattr(workflow_module, "create_generation", lost_insert)
    monkeypatch.setattr(workflow_module, "get_generation", current)
    monkeypatch.setattr(
        workflow_module,
        "count_active_generation_artifacts",
        no_active_artifact,
    )
    ctx.source_id = "source-1"

    with pytest.raises(RuntimeError, match="not fully reconciled"):
        await pipeline._start_generation(ctx)

    assert pipeline.qdrant_store.count_calls == []


@pytest.mark.asyncio
async def test_normal_duplicate_reuse_rejects_nonactive_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(_Conn(), _Store())
    active = SourceIngestGeneration(
        generation_id="gen-active",
        source_id="source-1",
        raw_collection="vo_rag_books_raw_v1",
        envelope_id="env-active",
        schema_version="book-rag-points-v1",
        chunking_version="default",
        sterilize_enabled=True,
        state=IngestGenerationState.ACTIVE,
        expected_raw_points=2,
        actual_raw_points=2,
        expected_sterile_points=2,
        actual_sterile_points=2,
        expected_artifacts=1,
        actual_artifacts=1,
    )

    async def current(*_args: object) -> SourceIngestGeneration:
        return active

    async def no_active_artifact(*_args: object) -> int:
        return 0

    monkeypatch.setenv("VORAG_INGEST_GENERATIONS", "1")
    monkeypatch.setattr(workflow_module, "get_active_generation", current)
    monkeypatch.setattr(
        workflow_module,
        "count_active_generation_artifacts",
        no_active_artifact,
    )

    assert await pipeline._source_already_ingested("source-1") is False
    assert pipeline.qdrant_store.count_calls == []


@pytest.mark.asyncio
async def test_failed_transition_cannot_hide_durably_active_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    reconciled: list[str] = []

    async def transition_lost(*_args: object, **_kwargs: object) -> bool:
        return False

    async def reconcile(ctx: BookRagContext) -> bool:
        reconciled.append(str(ctx.generation_id))
        return True

    async def active(*_args: object) -> SourceIngestGeneration:
        return SourceIngestGeneration(
            generation_id="gen-new",
            source_id="source-1",
            raw_collection="vo_rag_books_raw_v1",
            envelope_id=str(_context().envelope.id),
            schema_version="book-rag-points-v1",
            chunking_version="default",
            state=IngestGenerationState.ACTIVE,
        )

    monkeypatch.setattr(workflow_module, "transition_generation", transition_lost)
    monkeypatch.setattr(workflow_module, "get_generation", active)
    monkeypatch.setattr(pipeline, "_reconcile_activation_after_transaction_error", reconcile)

    await pipeline._fail_generation(_context(), "ambiguous activation exit")

    assert reconciled == ["gen-new"]
    assert not any(call[3] == "failed" for call in store.state_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate_state",
    [IngestGenerationState.FAILED, IngestGenerationState.QUARANTINED],
)
async def test_gate_rejection_losing_pg_race_reconciles_without_demoting_qdrant(
    monkeypatch: pytest.MonkeyPatch,
    gate_state: IngestGenerationState,
) -> None:
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    reconciled: list[str] = []

    async def transition_lost(*_args: object, **_kwargs: object) -> bool:
        return False

    async def reconcile(ctx: BookRagContext) -> bool:
        reconciled.append(str(ctx.generation_id))
        return True

    async def active(*_args: object) -> SourceIngestGeneration:
        return SourceIngestGeneration(
            generation_id="gen-new",
            source_id="source-1",
            raw_collection="vo_rag_books_raw_v1",
            envelope_id=str(_context().envelope.id),
            schema_version="book-rag-points-v1",
            chunking_version="default",
            state=IngestGenerationState.ACTIVE,
        )

    monkeypatch.setattr(workflow_module, "transition_generation", transition_lost)
    monkeypatch.setattr(workflow_module, "get_generation", active)
    monkeypatch.setattr(pipeline, "_reconcile_activation_after_transaction_error", reconcile)

    changed = await pipeline._reject_generation(
        _context(),
        gate_state,
        failure_reason="gate rejected",
    )

    assert changed is False
    assert reconciled == ["gen-new"]
    assert not any(call[3] == str(gate_state) for call in store.state_calls)


def test_legacy_canonical_artifact_id_remains_backward_compatible() -> None:
    expected = deterministic_hex(
        "source-1",
        "book_rag",
        "extraction",
        length=26,
    ).upper()

    assert (
        workflow_module._artifact_id(
            "source-1",
            "vo_rag_books_raw_v1",
        )
        == expected
    )
    assert (
        workflow_module._artifact_id(
            "source-1",
            "vo_rag_reddit_v1",
        )
        != expected
    )


@pytest.mark.asyncio
async def test_failed_postgres_activation_restores_prior_qdrant_view() -> None:
    store = _Store()

    with pytest.raises(RuntimeError, match="lost its validated state"):
        await _pipeline(_Conn(fail_activation=True), store)._activate_generation(_context())

    assert ("vo_rag_books_raw_v1", "source-1", "gen-new", "staging") in store.state_calls
    assert ("vo_rag_books_raw_v1", "source-1", "gen-old", "active") in store.state_calls
    assert store.legacy_state_calls == []
    assert store.delete_calls == []


@pytest.mark.asyncio
async def test_failed_first_migration_restores_legacy_view() -> None:
    store = _Store()

    with pytest.raises(RuntimeError, match="lost its validated state"):
        await _pipeline(
            _Conn(old_generation=None, fail_activation=True),
            store,
        )._activate_generation(_context())

    assert (
        "vo_rag_books_raw_v1",
        "source-1",
        "superseded",
    ) in store.legacy_state_calls
    assert store.legacy_state_calls[-2:] == [
        ("vo_rag_books_raw_v1", "source-1", "active"),
        ("vo_rag_books_sterile_v1", "source-1", "active"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_on_state_call", [1, 2, 3, 4])
async def test_cancellation_at_each_generation_switch_boundary_rolls_back(
    cancel_on_state_call: int,
) -> None:
    store = _Store(cancel_on_state_call=cancel_on_state_call)

    with pytest.raises(asyncio.CancelledError):
        await _pipeline(_Conn(), store)._activate_generation(_context())

    for collection in ("vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"):
        assert store.actual_states[(collection, "gen-old")] == "active"
        assert store.actual_states[(collection, "gen-new")] == "staging"
    assert store.legacy_state_calls == []
    assert store.delete_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_on_delete_call", [1, 2, 3, 4])
async def test_cancellation_at_each_post_activation_cleanup_boundary_is_recorded(
    cancel_on_delete_call: int,
) -> None:
    store = _Store(cancel_on_delete_call=cancel_on_delete_call)
    pipeline = _pipeline(_Conn(), store)

    with pytest.raises(asyncio.CancelledError):
        await pipeline._activate_generation(_context())

    for collection in ("vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"):
        assert store.actual_states[(collection, "gen-new")] == "active"
    generation_id, cleanup_required, cleanup_error = _generation_pool(pipeline).calls[-1][1]
    assert generation_id == "gen-new"
    assert cleanup_required is True
    assert "CancelledError" in cleanup_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_error",
    [OSError("commit acknowledgement lost"), asyncio.CancelledError()],
)
async def test_transaction_exit_error_with_durable_new_commit_continues_successfully(
    exit_error: BaseException,
) -> None:
    conn = _Conn(
        exit_error=exit_error,
        durable_after_exit="gen-new",
    )
    store = _Store()

    await _pipeline(conn, store)._activate_generation(_context())

    for collection in ("vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"):
        assert store.actual_states[(collection, "gen-old")] == "superseded"
        assert store.actual_states[(collection, "gen-new")] == "active"
    assert store.delete_calls == [
        ("vo_rag_books_raw_v1", "source-1", "gen-old"),
        ("vo_rag_books_sterile_v1", "source-1", "gen-old"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_error",
    [OSError("commit failed"), asyncio.CancelledError()],
)
async def test_transaction_exit_error_with_durable_old_commit_rolls_qdrant_back(
    exit_error: BaseException,
) -> None:
    conn = _Conn(
        exit_error=exit_error,
        durable_after_exit="gen-old",
    )
    store = _Store()

    with pytest.raises(type(exit_error)):
        await _pipeline(conn, store)._activate_generation(_context())

    for collection in ("vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"):
        assert store.actual_states[(collection, "gen-old")] == "active"
        assert store.actual_states[(collection, "gen-new")] == "staging"
    assert store.delete_calls == []


@pytest.mark.asyncio
async def test_reconciliation_exit_error_cannot_fail_durably_active_generation() -> None:
    conn = _Conn(
        exit_error=OSError("activation commit acknowledgement lost"),
        reconciliation_exit_error=OSError("reconciliation commit acknowledgement lost"),
        durable_after_exit="gen-new",
    )
    store = _Store()

    await _pipeline(conn, store)._activate_generation(_context())

    for collection in ("vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"):
        assert store.actual_states[(collection, "gen-new")] == "active"
    assert store.delete_calls


@pytest.mark.asyncio
async def test_failed_validation_warns_without_rejecting_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RejectingValidator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def validate(self, **_kwargs: object) -> ValidationReport:
            return ValidationReport(
                document_id="doc-1",
                source_id="source-1",
                checks=[],
                passed=False,
            )

    monkeypatch.setattr(workflow_module, "IngestionValidator", _RejectingValidator)
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    ctx = _context()
    ctx.parse_meta = {"document_id": "doc-1"}

    await pipeline._validate(ctx)

    assert not store.state_calls
    assert ctx.quality_warnings[0]["code"] == "audit_failed"
    query, args = _generation_pool(pipeline).calls[-1]
    assert "state = ANY($6::text[])" in query
    assert args[1] == "staging"
    assert args[2]["passed"] is False
    assert args[2]["quality_warnings"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_state"),
    [
        (smoke.GateDecision.QUARANTINE, IngestGenerationState.QUARANTINED),
        (smoke.GateDecision.FAIL, IngestGenerationState.FAILED),
    ],
)
async def test_failed_smoke_warns_without_rejecting_generation(
    monkeypatch: pytest.MonkeyPatch,
    decision: smoke.GateDecision,
    expected_state: IngestGenerationState,
) -> None:
    async def generate_self_queries(**_kwargs: object) -> list[object]:
        return []

    async def score_self_retrieval(**_kwargs: object) -> object:
        return object()

    def gate_decision(**_kwargs: object) -> smoke.GateResult:
        return smoke.GateResult(
            decision=decision,
            reason="gate rejected generation",
            score=0.0,
            n_scored=1,
        )

    monkeypatch.setattr(smoke, "generate_self_queries", generate_self_queries)
    monkeypatch.setattr(smoke, "score_self_retrieval", score_self_retrieval)
    monkeypatch.setattr(smoke, "gate_decision", gate_decision)
    store = _Store()
    pipeline = _pipeline(_Conn(), store)
    pipeline.llm_client = SimpleNamespace()
    pipeline.config.retrieval_smoke_enabled = True
    ctx = _context()

    await pipeline._retrieval_smoke(ctx)

    assert not store.state_calls
    assert ctx.quality_warnings[0]["code"] == str(decision)
    query, args = _generation_pool(pipeline).calls[-1]
    assert "state = ANY($6::text[])" in query
    assert args[1] == "validated"
