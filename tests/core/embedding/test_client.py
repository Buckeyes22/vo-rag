"""Contract tests for vo_rag.core.embedding.client.

Verifies:
- EmbeddingClient posts to /embed (and normalises URLs that already include
  /embed so the request doesn't end up at .../embed/embed)
- Authorization header is only sent when api_key is set
- 30 MB body guard rejects oversize payloads before any HTTP fires
- 502/503/504 retry with exponential backoff (4 attempts, 2s base) — uses
  respx to mock the BGE-M3 server
- Gives up after retry_attempts is exhausted
- Constructor validates retry_attempts >= 1 and retry_backoff_s >= 0
"""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
import respx

from vo_rag.core.embedding.client import EmbeddingClient

_OK_BODY: dict[str, object] = {"dense": [[0.0]], "sparse": [{}], "model": "test"}


def _make_client(
    url: str = "http://embed.local",
    **kwargs: object,
) -> EmbeddingClient:
    # retry_backoff_s=0 keeps the test suite fast.
    return EmbeddingClient(url, retry_backoff_s=0.0, **kwargs)  # type: ignore[arg-type]


def _request_json(request: httpx.Request) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(request.content.decode("utf-8")))


# ---------------------------------------------------------------------------
# Batch sizing and adaptive 413 behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_client_batches_large_inputs_by_env_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MAX_BATCH_TEXTS", "64")
    texts = [f"text-{i}" for i in range(200)]
    seen_batches: list[list[str]] = []

    def response_fn(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        batch_texts = cast(list[str], payload["texts"])
        seen_batches.append(batch_texts)
        dense = [[float(text.removeprefix("text-"))] for text in batch_texts]
        extras = [f"extra-{text}" for text in batch_texts]
        return httpx.Response(
            200,
            json={
                "dense": dense,
                "extra_per_text": extras,
                "model": f"model-from-{batch_texts[0]}",
                "dim": 1,
            },
        )

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(side_effect=response_fn)
        client = _make_client()
        try:
            result = await client.embed(texts)
        finally:
            await client.aclose()

    assert route.call_count == 7
    assert [len(batch) for batch in seen_batches] == [32, 32, 32, 32, 32, 32, 8]
    assert all(len(batch) <= 32 for batch in seen_batches)
    assert [text for batch in seen_batches for text in batch] == texts
    assert result["dense"] == [[float(i)] for i in range(200)]
    assert result["extra_per_text"] == [f"extra-text-{i}" for i in range(200)]
    assert result["model"] == "model-from-text-0"
    assert result["dim"] == 1


@pytest.mark.asyncio
async def test_adaptive_batches_do_not_summarize_partial_identity_as_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MAX_BATCH_TEXTS", "32")
    observed = {"model_id": "model-a", "identity_status": "observed_config"}

    def response_fn(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        count = len(payload["texts"])
        response: dict[str, Any] = {"dense": [[0.1] for _ in range(count)]}
        if payload["texts"][0] == "text-0":
            response["runtime_identity"] = observed
        return httpx.Response(200, json=response)

    async with respx.mock(assert_all_called=True) as mock:
        mock.post("http://embed.local/embed").mock(side_effect=response_fn)
        client = _make_client()
        try:
            result = await client.embed([f"text-{index}" for index in range(33)])
        finally:
            await client.aclose()

    assert result["runtime_identity"] is None
    assert result["runtime_identity_coverage"] == "partial"
    assert result["runtime_identity_per_text"] == [observed] * 32 + [None]


@pytest.mark.asyncio
async def test_embedding_client_preserves_single_request_body_when_batch_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MAX_BATCH_TEXTS", "64")
    response_body = {
        "dense": [[1.0], [2.0]],
        "sparse": [{"indices": [1], "values": [0.5]}, {"indices": [2], "values": [0.7]}],
        "colbert": [[[1.0]], [[2.0]]],
        "model": "test",
        "latency_ms": 1.25,
    }
    expected_payload = {
        "texts": ["alpha", "beta"],
        "return_dense": False,
        "return_sparse": True,
        "return_colbert": True,
        "max_length": 8192,
    }
    expected_body = json.dumps(
        expected_payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    seen_bodies: list[bytes] = []

    def response_fn(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(request.content)
        return httpx.Response(200, json=response_body)

    async with respx.mock(assert_all_called=True) as mock:
        route = mock.post("http://embed.local/embed").mock(side_effect=response_fn)
        client = _make_client()
        try:
            result = await client.embed(
                ["alpha", "beta"],
                return_dense=False,
                return_sparse=True,
                return_colbert=True,
            )
        finally:
            await client.aclose()

    assert route.call_count == 1
    assert seen_bodies == [expected_body]
    assert {
        key: value
        for key, value in result.items()
        if not key.endswith("_per_text")
        and key not in {"runtime_identity", "runtime_identity_coverage", "runtime_identity_mixed"}
    } == response_body
    assert result["runtime_identity_per_text"] == [None, None]
    assert result["input_token_counts_per_text"] == [None, None]
    assert result["input_truncated_per_text"] == [None, None]


@pytest.mark.asyncio
async def test_embedding_client_splits_413_batches_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MAX_BATCH_TEXTS", "4")
    texts = [f"text-{i}" for i in range(4)]
    seen_batches: list[list[str]] = []

    def response_fn(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        batch_texts = cast(list[str], payload["texts"])
        seen_batches.append(batch_texts)
        if len(batch_texts) > 2:
            return httpx.Response(413, text="Request Entity Too Large")
        return httpx.Response(
            200,
            json={"dense": [[float(text.removeprefix("text-"))] for text in batch_texts]},
        )

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(side_effect=response_fn)
        client = _make_client(retry_attempts=1)
        try:
            result = await client.embed(texts)
        finally:
            await client.aclose()

    assert route.call_count == 3
    assert seen_batches == [texts, texts[:2], texts[2:]]
    assert result["dense"] == [[0.0], [1.0], [2.0], [3.0]]


@pytest.mark.asyncio
async def test_embedding_client_raises_clear_error_when_single_text_exceeds_service_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MAX_BATCH_TEXTS", "4")

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(
            return_value=httpx.Response(413, text="Request Entity Too Large"),
        )
        client = _make_client(retry_attempts=1)
        try:
            with pytest.raises(RuntimeError, match="single text exceeds embed service limit"):
                await client.embed(["one-huge-text"])
        finally:
            await client.aclose()

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_embedding_client_forwards_vector_flags_on_every_sub_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MAX_BATCH_TEXTS", "2")
    texts = ["text-0", "text-1", "text-2"]
    seen_flags: list[tuple[bool, bool, bool]] = []

    def response_fn(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        batch_texts = cast(list[str], payload["texts"])
        seen_flags.append(
            (
                bool(payload["return_dense"]),
                bool(payload["return_sparse"]),
                bool(payload["return_colbert"]),
            )
        )
        return httpx.Response(
            200,
            json={
                "colbert": [[[float(text.removeprefix("text-"))]] for text in batch_texts],
            },
        )

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(side_effect=response_fn)
        client = _make_client()
        try:
            result = await client.embed(
                texts,
                return_dense=False,
                return_sparse=False,
                return_colbert=True,
            )
        finally:
            await client.aclose()

    assert route.call_count == 2
    assert seen_flags == [(False, False, True), (False, False, True)]
    assert result["colbert"] == [[[0.0]], [[1.0]], [[2.0]]]


@pytest.mark.asyncio
async def test_embedding_client_returns_empty_vectors_without_request_for_empty_texts() -> None:
    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(
            return_value=httpx.Response(200, json=_OK_BODY),
        )
        client = _make_client()
        try:
            result = await client.embed(
                [],
                return_dense=True,
                return_sparse=False,
                return_colbert=True,
            )
        finally:
            await client.aclose()

    assert not route.called
    assert result == {"dense": [], "colbert": []}


# ---------------------------------------------------------------------------
# Authorization header behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_client_does_not_send_authorization_without_api_key() -> None:
    async with respx.mock(assert_all_called=True) as mock:
        route = mock.post("http://embed.local/embed").mock(
            return_value=httpx.Response(200, json=_OK_BODY),
        )
        client = _make_client()
        try:
            await client.embed(["hello"])
        finally:
            await client.aclose()

        assert route.called
        sent = route.calls.last.request
        assert sent.headers.get("authorization") is None


@pytest.mark.asyncio
async def test_embedding_client_sends_bearer_authorization_when_api_key_set() -> None:
    async with respx.mock(assert_all_called=True) as mock:
        route = mock.post("http://embed.local/embed").mock(
            return_value=httpx.Response(200, json=_OK_BODY),
        )
        client = _make_client(api_key="x")
        try:
            await client.embed(["hello"])
        finally:
            await client.aclose()

        assert route.called
        sent = route.calls.last.request
        assert sent.headers.get("authorization") == "Bearer x"


# ---------------------------------------------------------------------------
# 30 MB body guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_client_rejects_payloads_over_runpod_limit_before_http() -> None:
    async with respx.mock(assert_all_called=False) as mock:
        # If the body guard works, the route is never hit.
        route = mock.post("http://embed.local/embed").mock(
            return_value=httpx.Response(200, json=_OK_BODY),
        )
        client = _make_client()
        try:
            with pytest.raises(ValueError, match="30 MB.*chunk smaller"):
                await client.embed(["x" * 30_000_000])
        finally:
            await client.aclose()

        assert not route.called


# ---------------------------------------------------------------------------
# 5xx retry behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_client_retries_on_502_then_succeeds() -> None:
    call_count = 0

    def response_fn(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, json=_OK_BODY)

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(
            side_effect=response_fn,
        )
        client = _make_client()
        try:
            result = await client.embed(["hello"])
        finally:
            await client.aclose()

        assert call_count == 3
        assert route.call_count == 3
        assert result["model"] == "test"


@pytest.mark.asyncio
async def test_embedding_client_gives_up_after_exhausting_retries() -> None:
    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(
            return_value=httpx.Response(503, text="Service Unavailable"),
        )
        client = EmbeddingClient("http://embed.local", retry_attempts=3, retry_backoff_s=0.0)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.embed(["hello"])
        finally:
            await client.aclose()

        assert route.call_count == 3


@pytest.mark.asyncio
async def test_embedding_client_waits_out_overload_beyond_the_error_budget() -> None:
    """429 is the worker's admission backpressure, not an error: an ingest
    must wait it out rather than fail the book after 4 tries (VOR-1427)."""
    call_count = 0

    def response_fn(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 8:
            return httpx.Response(429, text="Inference worker is at its concurrency limit")
        return httpx.Response(200, json=_OK_BODY)

    async with respx.mock(assert_all_called=False) as mock:
        mock.post("http://embed.local/embed").mock(side_effect=response_fn)
        client = _make_client(retry_attempts=4)
        try:
            result = await client.embed(["hello"])
        finally:
            await client.aclose()

        assert call_count == 9
        assert result["model"] == "test"


@pytest.mark.asyncio
async def test_embedding_client_overload_budget_is_bounded() -> None:
    from vo_rag.core.embedding.client import OVERLOAD_RETRY_ATTEMPTS

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://embed.local/embed").mock(
            return_value=httpx.Response(429, text="busy"),
        )
        client = _make_client(retry_attempts=4)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.embed(["hello"])
        finally:
            await client.aclose()

        assert route.call_count == OVERLOAD_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_overload_waits_are_jittered_and_outlast_a_saturation_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1440: 12 lockstep waits (~2.2 min) lost books when 10 lanes kept
    the embedder saturated for longer; every lane retried on the same
    schedule. Waits are now jittered and the budget spans several minutes."""
    from vo_rag.core.embedding import client as client_mod

    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("vo_rag.core.embedding.client.asyncio.sleep", fake_sleep)
    async with respx.mock(assert_all_called=False) as mock:
        mock.post("http://embed.local/embed").mock(return_value=httpx.Response(429, text="busy"))
        client = EmbeddingClient("http://embed.local", retry_backoff_s=2.0)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.embed(["hello"])
        finally:
            await client.aclose()

    assert len(waits) == client_mod.OVERLOAD_RETRY_ATTEMPTS - 1
    for n, wait in enumerate(waits, start=1):
        ceiling = min(2.0 * 2 ** (n - 1), client_mod._OVERLOAD_MAX_WAIT_S)
        assert ceiling / 2 <= wait <= ceiling
    assert len(set(waits)) > 1  # not a lockstep schedule
    assert sum(waits) >= 5 * 60  # even the shortest jittered budget spans 5 min


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_client_strips_trailing_embed_from_url() -> None:
    seen_urls: list[str] = []

    def response_fn(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=_OK_BODY)

    async with respx.mock(assert_all_called=False) as mock:
        for url in (
            "http://embed.local",
            "http://embed.local/",
            "http://embed.local/embed",
            "http://embed.local/embed/",
        ):
            mock.post("http://embed.local/embed").mock(side_effect=response_fn)
            client = EmbeddingClient(url, retry_backoff_s=0.0)
            try:
                await client.embed(["hi"])
            finally:
                await client.aclose()
            mock.reset()

    assert len(seen_urls) == 4
    for u in seen_urls:
        assert u == "http://embed.local/embed", f"expected …/embed (single), got {u}"


# ---------------------------------------------------------------------------
# Constructor argument validation
# ---------------------------------------------------------------------------


def test_embedding_client_rejects_retry_attempts_below_1() -> None:
    with pytest.raises(ValueError, match="retry_attempts must be >= 1"):
        EmbeddingClient("http://embed.local", retry_attempts=0)
    with pytest.raises(ValueError, match="retry_attempts must be >= 1"):
        EmbeddingClient("http://embed.local", retry_attempts=-3)


def test_embedding_client_rejects_negative_retry_backoff() -> None:
    with pytest.raises(ValueError, match="retry_backoff_s must be >= 0"):
        EmbeddingClient("http://embed.local", retry_backoff_s=-1.0)


def test_embedding_client_rejects_max_length_outside_wire_contract() -> None:
    with pytest.raises(ValueError, match="max_length must be >= 1"):
        EmbeddingClient("http://embed.local", max_length=0)
    with pytest.raises(ValueError, match="max_length must be <= 8192"):
        EmbeddingClient("http://embed.local", max_length=8193)


# ---------------------------------------------------------------------------
# Per-envelope ingest metrics (VOR-1434)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_call_and_its_429_retries_are_recorded_on_the_ingest() -> None:
    from vo_rag.core.observability.metrics import ingest_recording

    call_count = 0

    def response_fn(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return httpx.Response(429, text="busy")
        return httpx.Response(200, json=_OK_BODY)

    async with respx.mock(assert_all_called=False) as mock:
        mock.post("http://embed.local/embed").mock(side_effect=response_fn)
        client = _make_client()
        try:
            with ingest_recording() as recorder:
                await client.embed(["hello"])
                await client.embed(["again"])
        finally:
            await client.aclose()

    assert recorder.embed_calls == 2
    assert recorder.embed_retries_429 == 3


@pytest.mark.asyncio
async def test_an_exhausted_overload_budget_is_still_recorded() -> None:
    from vo_rag.core.embedding.client import OVERLOAD_RETRY_ATTEMPTS
    from vo_rag.core.observability.metrics import ingest_recording

    async with respx.mock(assert_all_called=False) as mock:
        mock.post("http://embed.local/embed").mock(return_value=httpx.Response(429, text="busy"))
        client = _make_client()
        try:
            with ingest_recording() as recorder, pytest.raises(httpx.HTTPStatusError):
                await client.embed(["hello"])
        finally:
            await client.aclose()

    assert recorder.embed_calls == 1
    assert recorder.embed_retries_429 == OVERLOAD_RETRY_ATTEMPTS
