"""Serving-route tests: GET /api/v1/figures/{src}/{filename}.

Contract: valid key -> 200 + bytes + content-type + immutable cache
header; malformed/traversal -> 400 with no filesystem access; missing
file -> 404; auth follows the app-wide ApiKeyMiddleware posture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.api.conftest import auth_header, fake_settings, make_client, make_services
from vo_rag.core.storage.figures import FigureStore

SRC = "ab" * 32


@pytest.fixture
def figures_client(tmp_path, monkeypatch):
    settings = fake_settings(figures=SimpleNamespace(root=str(tmp_path)))
    client = make_client(make_services(settings=settings), monkeypatch=monkeypatch)
    return client, FigureStore(tmp_path)


def test_valid_key_streams_bytes(figures_client) -> None:
    client, store = figures_client
    ref = store.write(SRC, 7, 2, b"\x89PNG-real-bytes", ext="png")
    resp = client.get(f"/api/v1/figures/{ref}", headers=auth_header())
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG-real-bytes"
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert resp.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("kind", ["equation", "table", "page"])
def test_source_crop_key_streams_through_existing_authenticated_route(figures_client, kind) -> None:
    client, store = figures_client
    raw = b"\x89PNG-equation-original-pixels"
    writer = {
        "table": store.write_table,
        "page": store.write_page,
        "equation": store.write_equation,
    }[kind]
    ref = writer(SRC, 12, raw)
    response = client.get(f"/api/v1/figures/{ref}", headers=auth_header())
    assert response.status_code == 200
    assert response.content == raw
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_jpg_content_type(figures_client) -> None:
    client, store = figures_client
    ref = store.write(SRC, 1, 0, b"\xff\xd8jpeg", ext="jpg")
    resp = client.get(f"/api/v1/figures/{ref}", headers=auth_header())
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_missing_file_404(figures_client) -> None:
    client, _store = figures_client
    resp = client.get(f"/api/v1/figures/{SRC}/p001_img000.png", headers=auth_header())
    assert resp.status_code == 404
    assert resp.json()["detail"] == "figure_not_found"


@pytest.mark.parametrize(
    "src,filename",
    [
        ("not-hex", "p001_img001.png"),  # malformed src
        ("ZZ" * 32, "p001_img001.png"),  # non-hex src, right length
        (SRC, "evil.txt"),  # non-figure filename
        (SRC, "p001_img001.svg"),  # svg refused
        (SRC, "p001_img001.png.tmp"),  # in-progress suffix
        (SRC, "..png"),  # dot-shenanigans
        (SRC, "page001_img001.png"),  # garage-era shape
    ],
)
def test_malformed_key_400(figures_client, src: str, filename: str) -> None:
    client, _store = figures_client
    resp = client.get(f"/api/v1/figures/{src}/{filename}", headers=auth_header())
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid_figure_ref"


def test_unauthenticated_requests_rejected(figures_client) -> None:
    client, store = figures_client
    ref = store.write(SRC, 1, 0, b"x", ext="png")
    resp = client.get(f"/api/v1/figures/{ref}")  # no X-API-Key
    assert resp.status_code == 401
