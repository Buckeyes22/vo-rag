"""VoRagChatStore — Postgres-backed, taint-aware chat memory store.

SECURITY CONTRACT
-----------------
1. TAINT-AWARE: Every memory block carries a ``trust`` field.
   Blocks derived from corpus/web tool output are UNTRUSTED.
   When replayed into a prompt, untrusted blocks MUST re-enter through
   the same protected packaging used at first contact. The store records
   ``trust`` on write and returns it in ``additional_kwargs["vo_rag_trust"]``
   on read so the caller (a future agent) can route untrusted content
   correctly. Untrusted blocks must NEVER be laundered into trusted context.

2. REDACTED ON WRITE: Every block is passed through ``redact_mapping``
   (via ``Redactor``) before persistence. No secrets land in the
   ``chat_memory`` table.

3. BUDGET CAP: A configurable max-byte budget limits replayed memory.
   On write, oldest blocks are physically evicted when the cap is exceeded.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any, Final

import asyncpg  # type: ignore[import-untyped]
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import BaseChatStore

from vo_rag.core.observability.redaction import Redactor, get_default_redactor


class TrustLevel(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


_DEFAULT_TRUST: Final = TrustLevel.TRUSTED
_TRUST_KEY: Final = "vo_rag_trust"


CREATE_CHAT_MEMORY_TABLE: Final = """
CREATE TABLE IF NOT EXISTS chat_memory (
    id SERIAL PRIMARY KEY,
    key TEXT NOT NULL,
    sequence_num BIGINT NOT NULL,
    role TEXT NOT NULL,
    data JSONB NOT NULL,
    trust TEXT NOT NULL DEFAULT 'trusted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_memory_key ON chat_memory(key);
CREATE INDEX IF NOT EXISTS idx_chat_memory_key_seq ON chat_memory(key, sequence_num);
"""


class VoRagChatStore(BaseChatStore):
    """Postgres-backed chat store with taint tracking and redaction.

    Conforms to LlamaIndex ``BaseChatStore`` interface (v0.12.52.post1).
    Async methods are native async over asyncpg; sync methods delegate
    through ``asyncio.run`` and must not be called from inside an
    active async event loop.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        redactor: Redactor | None = None,
        max_bytes_budget: int = 0,
    ) -> None:
        super().__init__()
        self._pool = pool
        self._redactor = redactor or get_default_redactor()
        self._max_bytes_budget = max_bytes_budget

    @classmethod
    def class_name(cls) -> str:
        return "VoRagChatStore"

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _redact_message(self, message: ChatMessage) -> ChatMessage:
        """Serialize → redact → deserialize.  Idempotent and safe."""
        raw: dict[str, Any] = message.model_dump(mode="json")
        redacted = self._redactor.redact_mapping(raw, pipeline="chat_memory")
        return ChatMessage.model_validate(redacted)

    @staticmethod
    def _extract_trust(message: ChatMessage) -> TrustLevel:
        trust_value = message.additional_kwargs.pop(_TRUST_KEY, _DEFAULT_TRUST)
        if isinstance(trust_value, TrustLevel):
            return trust_value
        return TrustLevel(str(trust_value))

    @staticmethod
    def _inject_trust(record: asyncpg.Record) -> ChatMessage:
        data: dict[str, Any] = dict(record["data"])
        additional_kwargs: dict[str, Any] = data.setdefault("additional_kwargs", {})
        additional_kwargs[_TRUST_KEY] = record["trust"]
        return ChatMessage.model_validate(data)

    @staticmethod
    def _message_bytes(message: ChatMessage) -> int:
        content = message.content or ""
        return len(content.encode("utf-8"))

    async def _evict_if_over_budget(self, key: str) -> None:
        if self._max_bytes_budget <= 0:
            return

        rows = await self._pool.fetch(
            "SELECT id, data, trust FROM chat_memory WHERE key = $1 ORDER BY sequence_num ASC",
            key,
        )
        total = 0
        for row in rows:
            msg = self._inject_trust(row)
            total += self._message_bytes(msg)

        ids_to_delete: list[int] = []
        for row in rows:
            if total <= self._max_bytes_budget:
                break
            msg = self._inject_trust(row)
            total -= self._message_bytes(msg)
            ids_to_delete.append(row["id"])

        if ids_to_delete:
            await self._pool.execute(
                "DELETE FROM chat_memory WHERE id = ANY($1::int[])",
                ids_to_delete,
            )

    # ------------------------------------------------------------------ #
    # Async API (native, preferred)
    # ------------------------------------------------------------------ #

    async def aget_messages(self, key: str) -> list[ChatMessage]:
        rows = await self._pool.fetch(
            "SELECT data, trust FROM chat_memory WHERE key = $1 ORDER BY sequence_num ASC",
            key,
        )
        return [self._inject_trust(row) for row in rows]

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        redacted = self._redact_message(message)
        trust = self._extract_trust(redacted)
        data = redacted.model_dump(mode="json")

        seq_row = await self._pool.fetchrow(
            "SELECT COALESCE(MAX(sequence_num), 0) AS max_seq FROM chat_memory WHERE key = $1",
            key,
        )
        next_seq = (seq_row["max_seq"] if seq_row is not None else 0) + 1

        await self._pool.execute(
            """
            INSERT INTO chat_memory (key, sequence_num, role, data, trust)
            VALUES ($1, $2, $3, $4, $5)
            """,
            key,
            next_seq,
            message.role,
            data,
            trust.value,
        )
        await self._evict_if_over_budget(key)

    async def aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
        await self._pool.execute("DELETE FROM chat_memory WHERE key = $1", key)

        for i, message in enumerate(messages):
            redacted = self._redact_message(message)
            trust = self._extract_trust(redacted)
            data = redacted.model_dump(mode="json")
            await self._pool.execute(
                """
                INSERT INTO chat_memory (key, sequence_num, role, data, trust)
                VALUES ($1, $2, $3, $4, $5)
                """,
                key,
                i + 1,
                message.role,
                data,
                trust.value,
            )

        await self._evict_if_over_budget(key)

    async def adelete_messages(self, key: str) -> list[ChatMessage] | None:
        rows = await self._pool.fetch(
            "SELECT data, trust FROM chat_memory WHERE key = $1 ORDER BY sequence_num ASC",
            key,
        )
        if not rows:
            return None
        messages = [self._inject_trust(row) for row in rows]
        await self._pool.execute("DELETE FROM chat_memory WHERE key = $1", key)
        return messages

    async def adelete_message(self, key: str, idx: int) -> ChatMessage | None:
        rows = await self._pool.fetch(
            "SELECT id, data, trust FROM chat_memory WHERE key = $1 ORDER BY sequence_num ASC",
            key,
        )
        if idx < 0 or idx >= len(rows):
            return None
        target = rows[idx]
        await self._pool.execute("DELETE FROM chat_memory WHERE id = $1", target["id"])
        return self._inject_trust(target)

    async def adelete_last_message(self, key: str) -> ChatMessage | None:
        row = await self._pool.fetchrow(
            "SELECT id, data, trust FROM chat_memory WHERE key = $1 "
            "ORDER BY sequence_num DESC LIMIT 1",
            key,
        )
        if row is None:
            return None
        await self._pool.execute("DELETE FROM chat_memory WHERE id = $1", row["id"])
        return self._inject_trust(row)

    async def aget_keys(self) -> list[str]:
        rows = await self._pool.fetch("SELECT DISTINCT key FROM chat_memory")
        return [str(row["key"]) for row in rows]

    # ------------------------------------------------------------------ #
    # Sync API (delegates through asyncio.run — do NOT call from async)
    # ------------------------------------------------------------------ #

    def get_messages(self, key: str) -> list[ChatMessage]:
        return asyncio.run(self.aget_messages(key))

    def add_message(self, key: str, message: ChatMessage) -> None:
        return asyncio.run(self.async_add_message(key, message))

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        return asyncio.run(self.aset_messages(key, messages))

    def delete_messages(self, key: str) -> list[ChatMessage] | None:
        return asyncio.run(self.adelete_messages(key))

    def delete_message(self, key: str, idx: int) -> ChatMessage | None:
        return asyncio.run(self.adelete_message(key, idx))

    def delete_last_message(self, key: str) -> ChatMessage | None:
        return asyncio.run(self.adelete_last_message(key))

    def get_keys(self) -> list[str]:
        return asyncio.run(self.aget_keys())
