from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from qdrant_client.models import (
    DeletePayloadOperation,
    FieldCondition,
    Filter,
    MatchValue,
    SetPayloadOperation,
)

from vo_rag.core.corpus import health, tf_store
from vo_rag.core.corpus.tf_schema import content_sha256, validate_batch, validate_book

BOOK = validate_book(
    {
        "tf_source_type": "coaching_manual",
        "tf_evidence_type": "coaching_practice",
        "tf_pub_year": 1987,
        "tf_book_athlete_level": ["general"],
        "tf_book_age_group": ["not_stated"],
        "tf_book_sex": "not_stated",
    }
)
TAGS = validate_batch(
    {
        "c": {
            "tf_event_family": ["relay"],
            "tf_event": ["4x100"],
            "tf_quality": ["technique"],
            "tf_discipline": ["training_methodology"],
            "tf_training_phase": ["competition"],
            "tf_athlete_level": ["not_stated"],
            "tf_age_group": ["not_stated"],
            "tf_sex": "not_stated",
            "tf_intervention": ["drill"],
            "tf_claim_basis": "coach_experience",
        }
    },
    ["c"],
)["c"]


class FakeClient:
    def __init__(self, points: list[Any]) -> None:
        self.points = points
        self.ops: list[Any] = []
        self.facet_calls: list[dict[str, Any]] = []
        self.scroll_calls: list[dict[str, Any]] = []

    async def scroll(self, **kw: Any) -> tuple[list[Any], None]:
        self.scroll_calls.append(kw)
        return self.points, None

    async def batch_update_points(
        self, collection_name: str, update_operations: list[Any], wait: bool
    ) -> None:
        self.ops.append(update_operations)

    async def facet(self, **kw: Any) -> Any:
        self.facet_calls.append(kw)
        return SimpleNamespace(
            hits=[SimpleNamespace(value="s1", count=3), SimpleNamespace(value="s2", count=1)]
        )


def _pt(pid: int, text: str, page_number: int | None = None) -> Any:
    payload: dict[str, Any] = {"chunk_id": f"c{pid}", "section_path": "S", "content": text}
    if page_number is not None:
        payload["page_number"] = page_number
    return SimpleNamespace(id=pid, payload=payload)


async def test_tf_source_ids_facets_on_the_domain_tag() -> None:
    client = FakeClient([])

    assert await tf_store.tf_source_ids(client) == ["s1", "s2"]

    call = client.facet_calls[0]
    assert call["collection_name"] == health.BOOKS_RAW_COLLECTION
    assert call["key"] == "source_id"
    assert call["facet_filter"] == Filter(
        must=[FieldCondition(key="domain_tags", match=MatchValue(value="track-and-field"))]
    )
    assert call["exact"] is False


async def test_book_chunks_defaults_a_bare_or_missing_payload() -> None:
    client = FakeClient(
        [SimpleNamespace(id=7, payload={"content": "x"}), SimpleNamespace(id=8, payload=None)]
    )

    chunks = await tf_store.book_chunks(client, "s1")

    bare, missing = chunks
    assert bare == {
        "point_id": 7,
        "chunk_id": "7",
        "section_path": "",
        "content": "x",
        "content_sha256": content_sha256("x"),
        "tf_tag_version": None,
        "page_number": None,
    }
    assert missing["point_id"] == 8
    assert missing["chunk_id"] == "8"
    assert missing["section_path"] == ""
    assert missing["content"] == ""
    assert missing["content_sha256"] == content_sha256("")
    assert missing["tf_tag_version"] is None
    assert missing["page_number"] is None


async def test_book_chunks_orders_by_page_number_then_chunk_id(monkeypatch: Any) -> None:
    """Scrolled points come back in whatever order Qdrant's scroll cursor holds
    (shuffled here); `book_chunks` restores document order — ascending page
    number, points with no page number last, tied by chunk_id."""
    points = [
        SimpleNamespace(id=1, payload={"chunk_id": "b", "content": "x", "page_number": 5}),
        SimpleNamespace(id=2, payload={"chunk_id": "a", "content": "x", "page_number": 1}),
        SimpleNamespace(id=3, payload={"chunk_id": "z", "content": "x", "page_number": None}),
        SimpleNamespace(id=4, payload={"chunk_id": "y", "content": "x"}),
        SimpleNamespace(id=5, payload={"chunk_id": "c", "content": "x", "page_number": 3}),
    ]
    client = FakeClient(points)

    chunks = await tf_store.book_chunks(client, "s1")

    assert [c["chunk_id"] for c in chunks] == ["a", "c", "b", "y", "z"]
    assert [c["page_number"] for c in chunks] == [1, 3, 5, None, None]


async def test_apply_payloads_fetches_page_number_in_the_scroll() -> None:
    client = FakeClient([])
    await tf_store.book_chunks(client, "s1")
    assert client.scroll_calls[0]["with_payload"] == [
        "chunk_id",
        "section_path",
        "content",
        "tf_tag_version",
        "page_number",
    ]


async def test_apply_payloads_batches_by_256() -> None:
    client = FakeClient([])
    # tf_pub_year present, so no DeletePayloadOperation is added per item -- this
    # test is only about the 256-item batching, not the year-clearing behavior.
    items = [(i, {"tf_tag_version": 1, "tf_pub_year": 1990}) for i in range(600)]
    n = await tf_store.apply_payloads(client, items)
    assert n == 600 and [len(b) for b in client.ops] == [256, 256, 88]


async def test_publication_tag_update_is_one_coalesced_source_mutation() -> None:
    from vo_rag.core.storage.publication_client import PublicationClient

    mutations: list[tuple[str, str, dict[str, Any]]] = []

    async def pointers(_corpus: str, _collection: str) -> dict[str, int | None]:
        return {}

    async def mutate(operation: str, collection: str, kwargs: dict[str, Any]) -> Any:
        mutations.append((operation, collection, kwargs))
        return None

    client = PublicationClient(FakeClient([]), pointers, mutate)
    items = [
        (i, {"tf_tag_source_id": "book", "tf_tag_content_sha256": "a" * 64, "tf_pub_year": 1990})
        for i in range(600)
    ]

    assert await tf_store.apply_payloads(client, items) == 600
    assert len(mutations) == 1
    operation, collection, kwargs = mutations[0]
    assert operation == "batch_update_points" and collection == health.BOOKS_RAW_COLLECTION
    assert len(kwargs["update_operations"]) == 600


async def test_apply_payloads_clears_a_stale_year_when_the_payload_omits_it() -> None:
    """A payload missing tf_pub_year (the book's year was cleared to None) must
    delete any stale tf_pub_year/tf_pub_decade already stamped on that point --
    SetPayload only ever adds/overwrites keys, never removes one."""
    client = FakeClient([])
    await tf_store.apply_payloads(client, [(42, {"tf_tag_version": 1})])
    ops = client.ops[0]
    sets = [op for op in ops if isinstance(op, SetPayloadOperation)]
    deletes = [op for op in ops if isinstance(op, DeletePayloadOperation)]
    assert len(sets) == 1 and sets[0].set_payload.points == [42]
    assert len(deletes) == 1
    assert deletes[0].delete_payload.points == [42]
    assert set(deletes[0].delete_payload.keys) == {"tf_pub_year", "tf_pub_decade"}


async def test_apply_payloads_does_not_delete_when_the_payload_carries_a_year() -> None:
    client = FakeClient([])
    await tf_store.apply_payloads(client, [(42, {"tf_pub_year": 1990, "tf_pub_decade": "1990s"})])
    assert not any(isinstance(op, DeletePayloadOperation) for op in client.ops[0])


async def test_restamp_uses_the_hash_cache_and_counts_the_rest(monkeypatch: Any) -> None:
    client = FakeClient([_pt(1, "Baton exchange zones."), _pt(2, "A changed paragraph.")])

    async def fake_cache(pool: Any, source_id: str) -> dict[str, Any]:
        return {
            content_sha256("Baton exchange zones."): (
                TAGS,
                "opencode/other-model",
                {"tf_tag_identity_status": "unknown_runtime_revision"},
            )
        }

    monkeypatch.setattr(tf_store, "chunk_tags_by_hash", fake_cache)
    stamped, untagged = await tf_store.restamp_source(object(), client, "s1", BOOK, reviewed=True)
    assert (stamped, untagged) == (1, 1)
    op = next(op for op in client.ops[0] if hasattr(op, "set_payload"))
    assert op.set_payload.points == [1]
    assert op.set_payload.payload["tf_event"] == ["4x100"]
    assert op.set_payload.payload["tf_book_reviewed"] is True
    # The tagger model that actually produced the cached tags reaches the
    # payload, not the module default -- provenance survives a restamp.
    assert op.set_payload.payload["tf_tagger_model"] == "opencode/other-model"
    assert op.set_payload.payload["tf_tag_identity_status"] == "unknown_runtime_revision"
    assert op.set_payload.payload["tf_tag_source_id"] == "s1"
    assert op.set_payload.payload["tf_tag_content_sha256"] == content_sha256(
        "Baton exchange zones."
    )
