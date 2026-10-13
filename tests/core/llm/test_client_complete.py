"""Tests for LLMClient complete/complete_json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.prompts import PromptRegistry
from vo_rag.core.llm.response import LLMResponse


class _Answer(BaseModel):
    value: int


class _ScriptedProvider:
    name = "fake"
    model = "fake-x"

    def __init__(self, text: str = '{"value": 1}') -> None:
        self.text = text
        self.calls = 0
        self.last_system: str | None = None
        self.last_messages: list[dict[str, Any]] | None = None

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.last_messages = list(messages)
        self.last_system = system if isinstance(system, str) else None
        return LLMResponse(
            text=self.text,
            model=self.model,
            input_tokens=3,
            output_tokens=2,
            latency_ms=7.0,
            provider=self.name,
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def registry(tmp_path: Path) -> PromptRegistry:
    d = tmp_path / "prompts_fallback"
    d.mkdir()
    (d / "greeter_v1.md").write_text(
        "---\nname: greeter_v1\nversion: 1\nvariables: [target]\n---\nGreet the {{ target }}.\n",
        encoding="utf-8",
    )
    (d / "greeter_v1_canary.md").write_text(
        "---\n"
        "name: greeter_v1_canary\n"
        "version: 1\n"
        "variables: [target]\n"
        "---\n"
        "CANARY greet {{ target }}.\n",
        encoding="utf-8",
    )
    return PromptRegistry(fallback_dir=d)


@pytest.mark.asyncio
async def test_complete_with_explicit_system() -> None:
    provider = _ScriptedProvider()
    client = LLMClient(provider=provider)

    resp = await client.complete(system="be helpful", messages=[{"role": "user", "content": "hi"}])

    assert resp.text == '{"value": 1}'
    assert resp.provider == "fake"
    assert resp.prompt_name is None
    assert provider.last_system == "be helpful"


@pytest.mark.asyncio
async def test_complete_renders_registry_prompt(registry: PromptRegistry) -> None:
    provider = _ScriptedProvider()
    client = LLMClient(provider=provider, prompt_registry=registry)

    resp = await client.complete(
        prompt_name="greeter_v1",
        prompt_variables={"target": "world"},
        messages=[],
    )

    assert resp.prompt_name == "greeter_v1"
    assert provider.last_system is not None
    assert "Greet the world" in provider.last_system


@pytest.mark.asyncio
async def test_complete_variant_weights_pick_sticky(registry: PromptRegistry) -> None:
    provider = _ScriptedProvider()
    client = LLMClient(provider=provider, prompt_registry=registry)

    resp1 = await client.complete(
        prompt_name="greeter_v1",
        prompt_variables={"target": "A"},
        variant_weights={"greeter_v1": 0.5, "greeter_v1_canary": 0.5},
        envelope_id="env-1",
    )
    resp2 = await client.complete(
        prompt_name="greeter_v1",
        prompt_variables={"target": "A"},
        variant_weights={"greeter_v1": 0.5, "greeter_v1_canary": 0.5},
        envelope_id="env-1",
    )

    assert resp1.variant is not None
    assert resp1.variant == resp2.variant
    assert resp1.variant in {"greeter_v1", "greeter_v1_canary"}


@pytest.mark.asyncio
async def test_complete_rejects_both_system_and_prompt_name(registry: PromptRegistry) -> None:
    client = LLMClient(provider=_ScriptedProvider(), prompt_registry=registry)

    with pytest.raises(ValueError):
        await client.complete(
            prompt_name="greeter_v1",
            prompt_variables={"target": "x"},
            system="also",
        )


@pytest.mark.asyncio
async def test_complete_json_parses_schema() -> None:
    provider = _ScriptedProvider('{"value": 42}')
    client = LLMClient(provider=provider)

    parsed, resp = await client.complete_json(
        schema=_Answer,
        system="you",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert parsed.value == 42
    assert resp.attempts == 1


@pytest.mark.asyncio
async def test_cost_accumulator_hook_fires() -> None:
    calls: list[tuple[str, LLMResponse]] = []

    class _Accumulator:
        def add(self, step: str, response: LLMResponse) -> None:
            calls.append((step, response))

    provider = _ScriptedProvider()
    client = LLMClient(
        provider=provider,
        cost_accumulator=_Accumulator(),
    )

    await client.complete(system="s", messages=[])
    assert len(calls) == 1
    assert calls[0][0] == "llm"
