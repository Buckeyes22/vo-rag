"""Envelopes, artifacts, and collection-stats read routes for the portal."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from tests.api.conftest import (
    FakePool,
    FakeQdrantClient,
    FakeQdrantStore,
    auth_header,
    make_client,
    make_services,
)

ARTIFACT_ID = "01J00000000000000000000001"
ENVELOPE_ID = "01J00000000000000000000000"


class FakeRetrieveQdrantClient(FakeQdrantClient):
    async def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                id="point-1",
                payload={
                    "image_ref": "ab" * 32 + "/p001_img000.png",
                    "image_id": "img-1",
                    "vlm_description": "a brake diagram",
                },
            ),
            SimpleNamespace(id="point-2", payload={"text": "no image here"}),
        ]


class FakeFetchrowPool(FakePool):
    async def fetchrow(self, _query: str, *_args: object) -> object | None:
        if not self.fetch_rows:
            return None
        row: object = self.fetch_rows[0]
        return row


def _artifact_row(**overrides):
    row = {
        "id": ARTIFACT_ID,
        "envelope_id": ENVELOPE_ID,
        "pipeline": "book_rag",
        "artifact_type": "summary",
        "title": "Brake system notes",
        "content": {"summary": "ABS and hydraulic braking overview"},
        "tags": ["brakes"],
        "priority": "medium",
        "status": "active",
        "license_class": "unknown",
        "qdrant_collection": None,
        "qdrant_point_ids": [],
        "source_provenance": [],
        "created_at": datetime(2026, 6, 11, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 11, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def test_envelopes_list(monkeypatch) -> None:
    row = {
        "id": "01J00000000000000000000000",
        "source_type": "file_drop",
        "source_data": {},
        "raw_content": {"garage_key": "test.pdf"},
        "metadata": {},
        "triage_result": None,
        "status": "completed",
        "target_pipeline": "book_rag",
        "priority": "medium",
        "sovereignty": "cloud_ok",
        "error": None,
        "created_at": datetime(2026, 6, 11, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 11, tzinfo=UTC),
    }
    client = make_client(
        make_services(db_pool=FakePool(fetch_rows=[row])),
        monkeypatch=monkeypatch,
    )
    resp = client.get("/api/v1/envelopes?limit=10", headers=auth_header())
    assert resp.status_code == 200
    assert resp.json()["items"][0]["status"] == "completed"


def test_collections_stats(monkeypatch) -> None:
    services = make_services(qdrant=FakeQdrantStore(_client=FakeQdrantClient()))
    client = make_client(services, monkeypatch=monkeypatch)
    resp = client.get("/api/v1/collections/stats", headers=auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["collections"]["vo_rag_books_raw_v1"]["points"] == 100


def test_envelopes_503_without_pool(monkeypatch) -> None:
    client = make_client(make_services(), monkeypatch=monkeypatch)
    assert client.get("/api/v1/envelopes", headers=auth_header()).status_code == 503


def test_artifacts_list_503_without_pool(monkeypatch) -> None:
    client = make_client(make_services(), monkeypatch=monkeypatch)
    assert client.get("/api/v1/artifacts", headers=auth_header()).status_code == 503


def test_artifacts_list(monkeypatch) -> None:
    client = make_client(
        make_services(db_pool=FakePool(fetch_rows=[_artifact_row()])),
        monkeypatch=monkeypatch,
    )

    resp = client.get("/api/v1/artifacts?limit=10", headers=auth_header())

    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["id"] == ARTIFACT_ID
    assert body["items"][0]["title"] == "Brake system notes"
    assert body["items"][0]["pipeline"] == "book_rag"
    assert body["next_cursor"] is None


def test_artifact_images_maps_image_ref_payloads(monkeypatch) -> None:
    ref = "ab" * 32 + "/p001_img000.png"
    client = make_client(
        make_services(
            db_pool=FakeFetchrowPool(
                fetch_rows=[
                    _artifact_row(
                        qdrant_collection="vo_rag_books_raw_v1",
                        qdrant_point_ids=["point-1", "point-2"],
                    )
                ]
            ),
            qdrant=FakeQdrantStore(_client=FakeRetrieveQdrantClient()),
        ),
        monkeypatch=monkeypatch,
    )

    resp = client.get(f"/api/v1/artifacts/{ARTIFACT_ID}/images", headers=auth_header())

    assert resp.status_code == 200
    assert resp.json() == {
        "items": [
            {
                "id": "img-1",
                "url": "/api/v1/figures/" + ref,
                "caption": "a brake diagram",
            }
        ]
    }
