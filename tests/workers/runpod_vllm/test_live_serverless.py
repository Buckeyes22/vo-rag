"""Live RunPod serverless smoke test — opt-in only.

Skip unless ``VORAG_RUN_LIVE=1`` is set AND ``RUNPOD_API_KEY`` is
present AND the endpoint env vars for at least one role are present. When
enabled, this is a 1-token probe — no throughput, no real load.

Run with:

    VORAG_RUN_LIVE=1 \\
    RUNPOD_API_KEY=... \\
    SERVERLESS_TEXT_ENDPOINT_ID=... \\
    uv run pytest tests/workers/runpod_vllm/test_live_serverless.py -v -m live
"""

from __future__ import annotations

import os

import pytest

from vo_rag.core.cloud.serverless import ServerlessClient, ServerlessResponse
from vo_rag.core.config.settings import get_settings, reset_settings_cache

pytestmark = pytest.mark.live


def _live_env_ok() -> bool:
    if os.environ.get("VORAG_RUN_LIVE") != "1":
        return False
    if not os.environ.get("RUNPOD_API_KEY"):
        return False
    return bool(
        os.environ.get("SERVERLESS_TEXT_ENDPOINT_ID")
        or os.environ.get("SERVERLESS_VLM_ENDPOINT_ID")
        or os.environ.get("SERVERLESS_DSI_ENDPOINT_ID")
    )


@pytest.mark.skipif(
    not _live_env_ok(),
    reason="live serverless tests require VORAG_RUN_LIVE=1 + RUNPOD_API_KEY + endpoint env",
)
def test_text_gen_one_token_smoke() -> None:
    """One 1-token completion against the text-gen endpoint, if configured."""
    import asyncio

    text_ep = os.environ.get("SERVERLESS_TEXT_ENDPOINT_ID")
    if not text_ep:
        pytest.skip("SERVERLESS_TEXT_ENDPOINT_ID not set — text-gen smoke skipped")
    reset_settings_cache()
    s = get_settings().serverless
    client = ServerlessClient(
        base_url=s.base_url("text"),
        api_key=s.api_key,
        model=s.model_name("text"),
        timeout_s=30,
        retry_delays=(0.0, 0.0, 0.0),  # smoke = no real retry waits
    )

    async def _ping() -> ServerlessResponse:
        return await client.chat_completion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )

    try:
        resp = asyncio.get_event_loop().run_until_complete(_ping())
    finally:
        asyncio.get_event_loop().run_until_complete(client.aclose())
    assert resp.content
    assert resp.input_tokens >= 0
    assert resp.output_tokens >= 0
