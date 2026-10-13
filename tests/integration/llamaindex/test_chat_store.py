"""Tests for VoRagChatStore — RED-first, fakes at the DB seam, no live DB.

CI job note
-----------
This module imports ``llama_index.*`` at module top (via
``vo_rag.integrations.llamaindex.chat_store``).  It is collected and run
only in the ``test-llamaindex`` adapter job
(``uv sync --extra dev --extra llamaindex``).  The light unit job
(``uv sync --extra dev``) drops it automatically via the
``collect_ignore`` in ``tests/integration/llamaindex/conftest.py``.
The defensive ``pytest.importorskip("llama_index")`` below is present
for belt-and-suspenders; do NOT remove it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

import asyncpg  # type: ignore[import-untyped]
import pytest

pytest.importorskip("llama_index")

from llama_index.core.llms import ChatMessage

from vo_rag.core.observability.redaction import RedactConfig, RedactionLevel, Redactor
from vo_rag.integrations.llamaindex.chat_store import (
    _TRUST_KEY,
    CREATE_CHAT_MEMORY_TABLE,
    TrustLevel,
    VoRagChatStore,
)

# --------------------------------------------------------------------------- #
# Fake at the DB seam (asyncpg.Pool shaped)
# --------------------------------------------------------------------------- #


@dataclass
class _FakePool:
    """In-memory asyncpg-shaped pool.  Parses ONLY the SQL emitted by
    :class:`VoRagChatStore` — not a general SQL engine."""

    _rows: list[dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        if "INSERT INTO chat_memory" in sql:
            data_val = args[3]
            row: dict[str, Any] = {
                "id": self._next_id,
                "key": args[0],
                "sequence_num": args[1],
                "role": args[2],
                "data": data_val if isinstance(data_val, dict) else json.loads(data_val),
                "trust": args[4],
            }
            self._rows.append(row)
            self._next_id += 1
            return "INSERT 1"
        if "DELETE FROM chat_memory WHERE key = $1" in sql and "id = ANY" not in sql:
            before = len(self._rows)
            self._rows = [r for r in self._rows if r.get("key") != args[0]]
            return f"DELETE {before - len(self._rows)}"
        if "DELETE FROM chat_memory WHERE id = ANY" in sql:
            ids = list(args[0])
            before = len(self._rows)
            self._rows = [r for r in self._rows if r["id"] not in ids]
            return f"DELETE {before - len(self._rows)}"
        if "DELETE FROM chat_memory WHERE id = $1" in sql:
            before = len(self._rows)
            self._rows = [r for r in self._rows if r["id"] != args[0]]
            return f"DELETE {before - len(self._rows)}"
        return "OK"

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, args))
        if "SELECT DISTINCT key FROM chat_memory" in sql:
            keys = sorted({r["key"] for r in self._rows})
            return [{"key": k} for k in keys]
        if "COALESCE(MAX(sequence_num), 0)" in sql:
            key = args[0] if args else None
            rows_for_key = [r for r in self._rows if r.get("key") == key]
            max_seq = max((r["sequence_num"] for r in rows_for_key), default=0)
            return [{"max_seq": max_seq}]
        key = args[0] if args else None
        rows = [r for r in self._rows if r.get("key") == key]
        if "ORDER BY sequence_num ASC" in sql:
            rows = sorted(rows, key=lambda r: r["sequence_num"])
        elif "ORDER BY sequence_num DESC" in sql:
            rows = sorted(rows, key=lambda r: r["sequence_num"], reverse=True)
        if "LIMIT 1" in sql:
            rows = rows[:1]
        return rows

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_pool() -> _FakePool:
    return _FakePool()


@pytest.fixture
def store(fake_pool: _FakePool) -> VoRagChatStore:
    return VoRagChatStore(pool=cast(asyncpg.Pool, fake_pool))


# --------------------------------------------------------------------------- #
# RED tests
# --------------------------------------------------------------------------- #


async def test_roundtrip_persists_order(fake_pool: _FakePool, store: VoRagChatStore) -> None:
    """Write three messages and read them back; order and content match."""
    messages = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="second"),
        ChatMessage(role="user", content="third"),
    ]
    for msg in messages:
        await store.async_add_message("conv-1", msg)

    back = await store.aget_messages("conv-1")
    assert len(back) == 3
    assert [m.role for m in back] == ["user", "assistant", "user"]
    assert [m.content for m in back] == ["first", "second", "third"]


async def test_redaction_on_write_scrubs_secret(fake_pool: _FakePool) -> None:
    """A block containing an api_key pattern is scrubbed before the persist
    call — the stored payload has no secret.
    """
    redactor = Redactor(RedactConfig(level=RedactionLevel.STANDARD))
    store = VoRagChatStore(pool=cast(asyncpg.Pool, fake_pool), redactor=redactor)
    secret = "sk-ant-api03-123456789012345678901234567890"  # pragma: allowlist secret
    msg = ChatMessage(role="user", content=f"my key is {secret}")

    await store.async_add_message("conv-1", msg)

    assert len(fake_pool._rows) == 1
    data = fake_pool._rows[0]["data"]
    content = str(ChatMessage.model_validate(data).content or "")
    assert secret not in content
    assert "<API_KEY>" in content


async def test_taint_preservation_untrusted_roundtrips(
    fake_pool: _FakePool, store: VoRagChatStore
) -> None:
    """An untrusted block round-trips with its trust tag intact so a future
    agent can tell it's untrusted.
    """
    msg = ChatMessage(
        role="assistant",
        content="result from web search",
        additional_kwargs={_TRUST_KEY: TrustLevel.UNTRUSTED},
    )
    await store.async_add_message("conv-1", msg)

    back = await store.aget_messages("conv-1")
    assert len(back) == 1
    assert back[0].additional_kwargs.get(_TRUST_KEY) == TrustLevel.UNTRUSTED


async def test_budget_cap_evicts_oldest(fake_pool: _FakePool) -> None:
    """Writing past the cap evicts oldest; replay stays under budget."""
    # Each content is ~10 bytes; budget 25 bytes means 2 fit, 1 evicted.
    store = VoRagChatStore(pool=cast(asyncpg.Pool, fake_pool), max_bytes_budget=25)
    await store.async_add_message("conv-1", ChatMessage(role="user", content="msg-one-00"))
    await store.async_add_message("conv-1", ChatMessage(role="user", content="msg-two-00"))
    await store.async_add_message("conv-1", ChatMessage(role="user", content="msg-three"))

    back = await store.aget_messages("conv-1")
    contents = [m.content for m in back]
    assert "msg-one-00" not in contents  # oldest evicted
    assert "msg-two-00" in contents
    assert "msg-three" in contents

    # Verify total bytes stayed under budget.
    total = sum(len((m.content or "").encode("utf-8")) for m in back)
    assert total <= 25


async def test_set_messages_replaces_and_respects_budget(fake_pool: _FakePool) -> None:
    """set_messages wipes old state and applies the budget cap."""
    store = VoRagChatStore(pool=cast(asyncpg.Pool, fake_pool), max_bytes_budget=20)
    await store.async_add_message("conv-1", ChatMessage(role="user", content="old-stale"))

    await store.aset_messages(
        "conv-1",
        [
            ChatMessage(role="user", content="aaa"),
            ChatMessage(role="assistant", content="bbb"),
            ChatMessage(role="user", content="ccc-long"),
        ],
    )

    back = await store.aget_messages("conv-1")
    contents = [m.content for m in back]
    assert "old-stale" not in contents
    # Oldest of the new batch may have been evicted if over budget.
    total = sum(len((m.content or "").encode("utf-8")) for m in back)
    assert total <= 20


async def test_delete_message_by_index(fake_pool: _FakePool, store: VoRagChatStore) -> None:
    """delete_message removes the Nth message in sequence order."""
    await store.async_add_message("conv-1", ChatMessage(role="user", content="zero"))
    await store.async_add_message("conv-1", ChatMessage(role="user", content="one"))
    await store.async_add_message("conv-1", ChatMessage(role="user", content="two"))

    deleted = await store.adelete_message("conv-1", 1)
    assert deleted is not None
    assert deleted.content == "one"

    back = await store.aget_messages("conv-1")
    assert [m.content for m in back] == ["zero", "two"]


async def test_delete_last_message(fake_pool: _FakePool, store: VoRagChatStore) -> None:
    """delete_last_message removes the most recent message."""
    await store.async_add_message("conv-1", ChatMessage(role="user", content="first"))
    await store.async_add_message("conv-1", ChatMessage(role="user", content="last"))

    deleted = await store.adelete_last_message("conv-1")
    assert deleted is not None
    assert deleted.content == "last"

    back = await store.aget_messages("conv-1")
    assert len(back) == 1
    assert back[0].content == "first"


async def test_delete_messages_returns_all_and_clears(
    fake_pool: _FakePool, store: VoRagChatStore
) -> None:
    """delete_messages returns the full list and empties the key."""
    await store.async_add_message("conv-1", ChatMessage(role="user", content="a"))
    await store.async_add_message("conv-1", ChatMessage(role="user", content="b"))

    deleted = await store.adelete_messages("conv-1")
    assert deleted is not None
    assert [m.content for m in deleted] == ["a", "b"]

    back = await store.aget_messages("conv-1")
    assert back == []


async def test_get_keys(fake_pool: _FakePool, store: VoRagChatStore) -> None:
    """get_keys returns distinct conversation keys."""
    await store.async_add_message("conv-a", ChatMessage(role="user", content="x"))
    await store.async_add_message("conv-b", ChatMessage(role="user", content="y"))

    keys = await store.aget_keys()
    assert sorted(keys) == ["conv-a", "conv-b"]


async def test_trusted_default_when_no_tag(fake_pool: _FakePool, store: VoRagChatStore) -> None:
    """Messages without an explicit trust tag default to TRUSTED."""
    msg = ChatMessage(role="user", content="hello")
    await store.async_add_message("conv-1", msg)

    back = await store.aget_messages("conv-1")
    assert len(back) == 1
    assert back[0].additional_kwargs.get(_TRUST_KEY) == TrustLevel.TRUSTED


async def test_create_table_ddl_is_documented() -> None:
    """The DDL string contains the expected schema elements."""
    assert "CREATE TABLE IF NOT EXISTS chat_memory" in CREATE_CHAT_MEMORY_TABLE
    assert "trust TEXT NOT NULL DEFAULT 'trusted'" in CREATE_CHAT_MEMORY_TABLE
    assert "data JSONB NOT NULL" in CREATE_CHAT_MEMORY_TABLE
    assert "CREATE INDEX IF NOT EXISTS idx_chat_memory_key" in CREATE_CHAT_MEMORY_TABLE
