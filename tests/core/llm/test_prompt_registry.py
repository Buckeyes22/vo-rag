"""Tests for prompt registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.llm.prompts import (
    PromptNotFound,
    PromptRegistry,
    PromptRenderError,
)


@pytest.fixture
def fallback_dir(tmp_path: Path) -> Path:
    """Fresh fallback dir per test with one sample prompt."""
    d = tmp_path / "prompts_fallback"
    d.mkdir()
    (d / "hello_v1.md").write_text(
        "---\n"
        "name: hello_v1\n"
        "version: 3\n"
        "cache_stable: true\n"
        "variables:\n"
        "  - name\n"
        "---\n"
        "Hello, {{ name }}.\n",
        encoding="utf-8",
    )
    return d


class _FakeLangfusePrompt:
    def __init__(self, *, text: str, version: int, config: dict[str, Any]) -> None:
        self.name = "hello_v1"
        self.prompt = text
        self.version = version
        self.config = config


class _FakeLangfuseHit:
    def __init__(self, text: str, version: int = 42) -> None:
        self._text = text
        self._version = version

    def get_prompt(self, name: str, *, label: str) -> _FakeLangfusePrompt:
        return _FakeLangfusePrompt(
            text=self._text,
            version=self._version,
            config={"cache_stable": True, "variables": ["name"]},
        )


class _FakeLangfuseMiss:
    def get_prompt(self, name: str, *, label: str) -> _FakeLangfusePrompt:
        raise RuntimeError("langfuse unreachable")


@pytest.mark.asyncio
async def test_langfuse_hit_returns_langfuse_source(fallback_dir: Path) -> None:
    lf = _FakeLangfuseHit("Hi from Langfuse, {{ name }}.", version=42)
    reg = PromptRegistry(fallback_dir=fallback_dir, langfuse_client=lf)

    prompt = await reg.get("hello_v1")

    assert prompt.metadata["prompt_source"] == "langfuse"
    assert prompt.version == 42
    assert "Langfuse" in prompt.template
    assert prompt.cache_stable is True


@pytest.mark.asyncio
async def test_langfuse_down_falls_back_to_file(fallback_dir: Path) -> None:
    reg = PromptRegistry(fallback_dir=fallback_dir, langfuse_client=_FakeLangfuseMiss())

    prompt = await reg.get("hello_v1")

    assert prompt.metadata["prompt_source"] == "fallback"
    assert prompt.version == 3
    assert prompt.cache_stable is True
    assert "{{ name }}" in prompt.template


@pytest.mark.asyncio
async def test_no_langfuse_client_uses_fallback(fallback_dir: Path) -> None:
    reg = PromptRegistry(fallback_dir=fallback_dir)

    prompt = await reg.get("hello_v1")

    assert prompt.metadata["prompt_source"] == "fallback"


@pytest.mark.asyncio
async def test_render_happy_path(fallback_dir: Path) -> None:
    reg = PromptRegistry(fallback_dir=fallback_dir)

    rendered = await reg.render("hello_v1", {"name": "the operator"})

    assert rendered.strip() == "Hello, the operator."


@pytest.mark.asyncio
async def test_render_missing_variable_raises(fallback_dir: Path) -> None:
    reg = PromptRegistry(fallback_dir=fallback_dir)

    with pytest.raises(PromptRenderError) as exc_info:
        await reg.render("hello_v1", {})

    assert "name" in str(exc_info.value)
    assert "hello_v1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unknown_prompt_raises_not_found(fallback_dir: Path) -> None:
    reg = PromptRegistry(fallback_dir=fallback_dir)

    with pytest.raises(PromptNotFound):
        await reg.get("nonexistent_v1")


@pytest.mark.asyncio
async def test_cache_hit_skips_fetch(fallback_dir: Path) -> None:
    lf = _FakeLangfuseHit("Hi from Langfuse, {{ name }}.")
    calls = {"count": 0}

    original = lf.get_prompt

    def _spy(name: str, *, label: str) -> _FakeLangfusePrompt:
        calls["count"] += 1
        return original(name, label=label)

    lf.get_prompt = _spy  # type: ignore[method-assign]
    reg = PromptRegistry(fallback_dir=fallback_dir, langfuse_client=lf)

    await reg.get("hello_v1")
    await reg.get("hello_v1")

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_committed_fallbacks_loadable() -> None:
    """Every prompts_fallback/*.md file in the repo must load cleanly."""
    repo_dir = Path(__file__).resolve().parents[2] / "prompts_fallback"
    if not repo_dir.exists():
        pytest.skip("prompts_fallback/ not present in test environment")
    reg = PromptRegistry(fallback_dir=repo_dir)
    for md in sorted(repo_dir.glob("*.md")):
        name = md.stem
        prompt = await reg.get(name)
        assert prompt.template
        assert prompt.version >= 1
