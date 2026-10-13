"""Tests for the OpenAI-compatible serverless client."""

from __future__ import annotations

import pytest

from vo_rag.core.cloud.serverless import ServerlessClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://lin:8000/v1",
        "https://api.runpod.ai/v2/text-endpoint/openai/v1",
    ],
)
async def test_serverless_client_accepts_openai_compatible_base_url_suffixes(
    base_url: str,
) -> None:
    client = ServerlessClient(
        base_url=base_url,
        api_key="test-key",  # pragma: allowlist secret
        model="served-model",
    )
    try:
        assert client.model == "served-model"
    finally:
        await client.aclose()


def test_serverless_client_rejects_unknown_base_url_suffix() -> None:
    with pytest.raises(ValueError, match="/openai/v1 or /v1"):
        ServerlessClient(
            base_url="http://lin:8000/foo",
            api_key="test-key",  # pragma: allowlist secret
            model="served-model",
        )
