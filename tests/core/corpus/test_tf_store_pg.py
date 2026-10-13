from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.corpus import tf_store
from vo_rag.core.corpus.tf_schema import TF_TAG_VERSION, validate_batch, validate_book

BOOK = validate_book(
    {
        "tf_source_type": "textbook",
        "tf_evidence_type": "empirical_research",
        "tf_pub_year": 2011,
        "tf_book_athlete_level": ["elite"],
        "tf_book_age_group": ["adult"],
        "tf_book_sex": "mixed",
    }
)
CHUNK = validate_batch(
    {
        "c": {
            "tf_event_family": ["throws"],
            "tf_event": ["discus"],
            "tf_quality": ["technique"],
            "tf_discipline": ["biomechanics"],
            "tf_training_phase": ["not_stated"],
            "tf_athlete_level": ["not_stated"],
            "tf_age_group": ["not_stated"],
            "tf_sex": "not_stated",
            "tf_intervention": ["technique_cue"],
            "tf_claim_basis": "textbook_principle",
        }
    },
    ["c"],
)["c"]


class _Ctx:
    """An async context manager yielding the fake itself (acquire/transaction)."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def __aenter__(self) -> Any:
        return self.pool

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakePool:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.rows: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.command_tag = "INSERT 0 1"

    async def execute(self, q: str, *a: Any) -> str:
        self.calls.append("execute")
        self.executed.append((q, a))
        return self.command_tag

    async def executemany(self, q: str, args: list[tuple[Any, ...]]) -> None:
        self.calls.append("executemany")
        self.executed.extend((q, a) for a in args)

    def acquire(self) -> Any:
        return _Ctx(self)

    def transaction(self) -> Any:
        self.calls.append("transaction")
        return _Ctx(self)

    async def fetch(self, q: str, *a: Any) -> list[dict[str, Any]]:
        self.calls.append("fetch")
        self.executed.append((q, a))
        return self.rows


async def test_upsert_book_stores_dict_not_json_string() -> None:
    pool = FakePool()
    await tf_store.upsert_book(pool, "s1", BOOK, reviewed=True)
    q, args = pool.executed[0]
    assert "ON CONFLICT (source_id) DO UPDATE" in q
    assert "WHERE NOT tf_book_tags.reviewed OR EXCLUDED.reviewed" in q
    assert (
        args[0] == "s1"
        and isinstance(args[1], dict)
        and args[2] is True
        and args[3] == TF_TAG_VERSION
    )


async def test_upsert_book_reports_whether_a_row_was_written() -> None:
    pool = FakePool()
    assert await tf_store.upsert_book(pool, "s1", BOOK, reviewed=True) is True
    # A draft that loses the ON CONFLICT guard updates nothing.
    pool.command_tag = "INSERT 0 0"
    assert await tf_store.upsert_book(pool, "s1", BOOK, reviewed=False) is False


async def test_upsert_book_reports_no_write_for_an_unreadable_command_tag() -> None:
    pool = FakePool()
    pool.command_tag = ""
    assert await tf_store.upsert_book(pool, "s1", BOOK, reviewed=False) is False


@pytest.mark.parametrize("source_id", ["", "   "], ids=["blank", "whitespace_only"])
async def test_upsert_book_rejects_a_blank_source_id(source_id: str) -> None:
    pool = FakePool()
    with pytest.raises(ValueError):
        await tf_store.upsert_book(pool, source_id, BOOK, reviewed=True)
    assert pool.calls == []


async def test_book_records_returns_every_record_with_its_flag() -> None:
    pool = FakePool()
    pool.rows = [{"source_id": "s1", "tags": BOOK.model_dump(), "reviewed": False}]
    books = await tf_store.book_records(pool)
    assert "tag_version" not in pool.executed[0][0] and books["s1"] == (BOOK, False)


async def test_chunk_tags_by_hash_filters_current_version() -> None:
    pool = FakePool()
    pool.rows = [{"content_sha256": "h1", "tags": CHUNK.model_dump(), "tagger_model": "m-x"}]
    got = await tf_store.chunk_tags_by_hash(pool, "s1")
    assert "tag_version = $2" in pool.executed[0][0]
    assert "tagger_model" in pool.executed[0][0]
    assert pool.executed[0][1] == ("s1", TF_TAG_VERSION)
    # The stored tagger model rides along with the cached tags so a restamp
    # can stamp the model that actually produced them, not today's default.
    assert got == {"h1": (CHUNK, "m-x", {})}


async def test_upsert_chunk_tags_writes_one_row_per_hash() -> None:
    pool = FakePool()
    n = await tf_store.upsert_chunk_tags(pool, "s1", [("h1", CHUNK), ("h2", CHUNK)], model="m")
    assert n == 2 and len(pool.executed) == 2
    args = pool.executed[0][1]
    assert args[:2] == ("s1", "h1")
    assert isinstance(args[2], dict) and args[3] == TF_TAG_VERSION and args[4] == "m"
    assert args[2]["_tag_provenance"] == {}


async def test_upsert_chunk_tags_with_no_rows_skips_the_pool() -> None:
    pool = FakePool()
    n = await tf_store.upsert_chunk_tags(pool, "s1", [], model="m")
    assert n == 0 and pool.calls == []


@pytest.mark.parametrize("source_id", ["", "   "], ids=["blank", "whitespace_only"])
async def test_upsert_chunk_tags_rejects_a_blank_source_id(source_id: str) -> None:
    pool = FakePool()
    with pytest.raises(ValueError):
        await tf_store.upsert_chunk_tags(pool, source_id, [("h1", CHUNK)], model="m")
    assert pool.calls == []


async def test_rekey_moves_each_legacy_row_to_its_new_key_without_clobbering() -> None:
    """Copy under the new key (never overwriting a row already there), then drop the
    legacy row, so a re-run is a no-op (VOR-1450)."""
    pool = FakePool()
    pool.rows = [{"content_sha256": "old1"}]
    n = await tf_store.rekey_chunk_tags(
        pool, "s1", [("old1", "new1"), ("same", "same"), ("gone", "new3")]
    )
    # "gone" has no stored row, so it is not a move and not counted.
    assert n == 1
    lookup, insert, delete = pool.executed
    assert "ANY($2::text[])" in lookup[0] and lookup[1] == ("s1", ["gone", "old1"])
    assert "INSERT INTO tf_chunk_tags" in insert[0] and "ON CONFLICT" in insert[0]
    assert "DO NOTHING" in insert[0] and insert[1] == ("s1", "old1", "new1")
    assert "DELETE FROM tf_chunk_tags" in delete[0] and delete[1] == ("s1", "old1")
    assert "transaction" in pool.calls


async def test_rekey_rejects_a_blank_source_id() -> None:
    with pytest.raises(ValueError):
        await tf_store.rekey_chunk_tags(FakePool(), " ", [("a", "b")])


async def test_rekey_moves_nothing_when_no_legacy_row_is_stored() -> None:
    pool = FakePool()
    assert await tf_store.rekey_chunk_tags(pool, "s1", [("old1", "new1")]) == 0
    assert len(pool.executed) == 1  # the lookup only
