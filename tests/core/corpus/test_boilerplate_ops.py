"""`boilerplate_ops` over a fake Qdrant client (VOR-1427).

Boilerplate delete must reach the collections purge also clears --
``vo_rag_books_sterile_v1`` (publication/partner queries read it) and
``vo_rag_books_contextualized_v1`` -- with the same per-``source_id``
matcher, and must tolerate a collection that doesn't exist.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vo_rag.core.corpus import boilerplate_ops

_RAW = "vo_rag_books_raw_v1"
_STERILE = "vo_rag_books_sterile_v1"
_CONTEXTUALIZED = "vo_rag_books_contextualized_v1"
_BLANK = "This page intentionally left blank"


class _FakeClient:
    def __init__(self, collections: dict[str, dict[str, list[tuple[str, str]]]]) -> None:
        #: collection -> source_id -> [(point id, content)]
        self.collections = collections
        self.scrolled: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, list[str]]] = []

    async def get_collections(self) -> Any:
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.collections])

    async def facet(self, collection_name: str, key: str, *, limit: int, exact: bool) -> Any:
        ids = sorted(self.collections[collection_name])
        return SimpleNamespace(hits=[SimpleNamespace(value=v, count=1) for v in ids])

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any,
        with_payload: Any,
        with_vectors: bool,
        limit: int,
        offset: Any,
    ) -> tuple[list[Any], Any]:
        source_id = scroll_filter.must[0].match.value
        self.scrolled.append((collection_name, source_id))
        points = [
            SimpleNamespace(id=point_id, payload={"content": content})
            for point_id, content in self.collections[collection_name].get(source_id, [])
        ]
        return points, None

    async def delete(self, collection_name: str, points_selector: Any, wait: bool) -> None:
        self.deleted.append((collection_name, list(points_selector)))


def _store(client: Any) -> Any:
    return SimpleNamespace(_client=client)


async def test_derived_scan_matches_per_source_id_and_reports_an_absent_collection() -> None:
    client = _FakeClient(
        {
            _RAW: {"a": [("r1", _BLANK)], "b": [("r2", "real text about gearing")]},
            _STERILE: {
                "a": [("s1", _BLANK), ("s2", "real text about brakes")],
                "b": [("s3", _BLANK)],
            },
        }
    )

    result = await boilerplate_ops.derived_boilerplate_point_ids(_store(client), source_ids=None)

    # Every raw book is scanned in sterile, not only books with raw hits.
    assert result == {_STERILE: {"a": ["s1"], "b": ["s3"]}, _CONTEXTUALIZED: None}
    assert (_STERILE, "b") in client.scrolled
    assert not [c for c, _ in client.scrolled if c == _CONTEXTUALIZED]


async def test_delete_targets_the_named_collection() -> None:
    client = _FakeClient({_RAW: {}, _STERILE: {}})

    deleted = await boilerplate_ops.delete_boilerplate_points(
        _store(client), ["s1", "s3"], collection=_STERILE
    )

    assert deleted == 2
    assert client.deleted == [(_STERILE, ["s1", "s3"])]


async def test_publication_delete_coalesces_ids_into_one_source_mutation() -> None:
    from vo_rag.core.storage.publication_client import PublicationClient

    mutations: list[tuple[str, str, dict[str, Any]]] = []

    async def pointers(_corpus: str, _collection: str) -> dict[str, int | None]:
        return {}

    async def mutate(operation: str, collection: str, kwargs: dict[str, Any]) -> Any:
        mutations.append((operation, collection, kwargs))
        return object()

    wrapped = PublicationClient(_FakeClient({_RAW: {}}), pointers, mutate)
    deleted = await boilerplate_ops.delete_boilerplate_points(
        _store(wrapped), [f"point-{i}" for i in range(700)], collection=_RAW
    )

    assert deleted == 700
    assert len(mutations) == 1
    operation, collection, kwargs = mutations[0]
    assert operation == "delete" and collection == _RAW
    assert len(kwargs["points_selector"]) == 700


async def test_publication_delete_rejects_over_limit_before_mutation() -> None:
    import pytest

    from vo_rag.core.storage.publication_client import PublicationClient

    mutations: list[Any] = []

    async def pointers(_corpus: str, _collection: str) -> dict[str, int | None]:
        return {}

    async def mutate(*args: Any) -> Any:
        mutations.append(args)
        return object()

    wrapped = PublicationClient(_FakeClient({_RAW: {}}), pointers, mutate)
    with pytest.raises(ValueError, match="10000"):
        await boilerplate_ops.delete_boilerplate_points(
            _store(wrapped), [f"point-{i}" for i in range(10_001)], collection=_RAW
        )
    assert mutations == []
