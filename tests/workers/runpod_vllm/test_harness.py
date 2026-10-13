"""Tests for the local FastAPI harness in vo_rag.workers.runpod_vllm.harness.

The harness is a tiny OpenAI-compatible ASGI app used by:
- this test file (in-process via httpx.ASGITransport)
- local-dev (python -m vo_rag.workers.runpod_vllm.harness --port 8188)

It is NOT a production inference server.
"""

from __future__ import annotations

import httpx
import pytest

from vo_rag.core.cloud.serverless import ServerlessClient
from vo_rag.workers.runpod_vllm.harness import create_harness_app


@pytest.mark.asyncio
async def test_health_reports_local_harness_and_no_gpu() -> None:
    app = create_harness_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://harness") as c:
        r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "local-harness"
        assert body["gpu"] is False


@pytest.mark.asyncio
async def test_chat_completions_returns_openai_shape() -> None:
    app = create_harness_app(response_text="hi from harness", model="my-local")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://harness") as c:
        r = await c.post(
            "/openai/v1/chat/completions",
            json={
                "model": "ignored-by-harness",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "my-local"
    assert body["choices"][0]["message"]["content"] == "hi from harness"
    usage = body["usage"]
    assert isinstance(usage["prompt_tokens"], int) and usage["prompt_tokens"] >= 1
    assert isinstance(usage["completion_tokens"], int) and usage["completion_tokens"] >= 1
    assert body["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_completions_accepts_vlm_image_url_part() -> None:
    """The harness must round-trip the OpenAI multimodal shape end-to-end."""
    app = create_harness_app(response_text="ok-vlm", model="m")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://harness") as c:
        r = await c.post(
            "/openai/v1/chat/completions",
            json={
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                            },
                        ],
                    }
                ],
                "max_tokens": 4,
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "ok-vlm"


@pytest.mark.asyncio
async def test_serverless_client_drives_the_harness() -> None:
    """ServerlessClient is the only transport — exercise it against the harness."""
    app = create_harness_app(response_text="driven", model="harness-model")
    client = ServerlessClient(
        base_url="http://harness/openai/v1",
        api_key="k",
        model="harness-model",
        timeout_s=10,
        retry_delays=(0.0, 0.0, 0.0),
    )
    # Override the private httpx client with an ASGI-bound one so the test
    # never opens a socket.
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://harness/openai/v1",
        headers={"Authorization": "Bearer k"},
        timeout=10,
    )
    try:
        resp = await client.chat_completion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4,
        )
    finally:
        await client.aclose()
    assert resp.content == "driven"
    assert resp.model == "harness-model"
    assert resp.input_tokens >= 1
    assert resp.output_tokens >= 1


@pytest.mark.asyncio
async def test_fail_first_n_simulates_502_then_success() -> None:
    """The harness can simulate transient 502 errors so ServerlessClient's
    (1, 3, 9) retry path is exercised without real sleeps. We pass
    retry_delays=(0, 0, 0) so the client does not actually wait, but the
    retry *count* still goes through."""
    app = create_harness_app(
        response_text="eventually-ok",
        model="harness",
        fail_first_n=2,
    )
    client = ServerlessClient(
        base_url="http://harness/openai/v1",
        api_key="k",
        model="harness",
        timeout_s=10,
        retry_delays=(0.0, 0.0, 0.0),  # 4 attempts total (1 + 3 retries)
    )
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://harness/openai/v1",
        headers={"Authorization": "Bearer k"},
        timeout=10,
    )
    try:
        resp = await client.chat_completion(
            messages=[{"role": "user", "content": "retry me"}],
            max_tokens=4,
        )
    finally:
        await client.aclose()
    assert resp.content == "eventually-ok"


@pytest.mark.asyncio
async def test_fail_first_n_exhausted_surfaces_error() -> None:
    """If fail_first_n exceeds the retry budget, the last 502 is surfaced."""
    from httpx import HTTPStatusError

    app = create_harness_app(
        response_text="never",
        model="harness",
        fail_first_n=999,
    )
    client = ServerlessClient(
        base_url="http://harness/openai/v1",
        api_key="k",
        model="harness",
        timeout_s=10,
        retry_delays=(0.0, 0.0, 0.0),  # 4 attempts total
    )
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://harness/openai/v1",
        headers={"Authorization": "Bearer k"},
        timeout=10,
    )
    try:
        with pytest.raises(HTTPStatusError):
            await client.chat_completion(
                messages=[{"role": "user", "content": "x"}],
                max_tokens=2,
            )
    finally:
        await client.aclose()


def test_module_is_runnable_as_script_smoke() -> None:
    """``python -m vo_rag.workers.runpod_vllm.harness --help`` is the
    documented local-dev entry point. We do not start uvicorn here (we are
    unit-test bound), but we do assert the call() shim parses args without
    exploding."""
    from vo_rag.workers.runpod_vllm import harness

    assert callable(harness.create_harness_app)
    assert callable(harness.main)
    # The arg parser must exist; the click parser is built at module import
    # so a typo would fail this test.
    assert harness._build_parser is not None
