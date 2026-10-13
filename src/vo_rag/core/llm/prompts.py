"""Prompt registry with Langfuse primary + local-file fallback.

Prompts are first-class versioned resources. Production reads from Langfuse
by label (`production`, `canary`, `experiment`). If Langfuse is unreachable
or the prompt is missing, we fall back to a git-tracked `prompts_fallback/`
markdown file with yaml frontmatter so pipelines never break.

Fallback file format:
    ---
    name: sterilizer_v1
    version: 1
    cache_stable: true
    variables: [var_a, var_b]
    ---
    <jinja2 template body>

Rendering is strict: every declared variable must be provided or
PromptRenderError is raised before the template runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
import yaml
from jinja2 import StrictUndefined, TemplateError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from vo_rag.core.llm.errors import LLMError
from vo_rag.core.models.common import VoBaseModel

_log = structlog.get_logger(__name__)

_SANDBOX = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)


class PromptError(LLMError):
    """Base class for prompt-registry errors."""


class PromptNotFound(PromptError):
    """Prompt not found in Langfuse or fallback dir."""


class PromptRenderError(PromptError):
    """Template rendering failed (missing variable or jinja2 error)."""


class PromptParseError(PromptError):
    """Fallback file is malformed."""


class Prompt(VoBaseModel):
    name: str
    version: int
    label: str
    template: str
    cache_stable: bool = False
    metadata: dict[str, Any] = {}


class PromptRegistry:
    """Fetch-and-render prompts with Langfuse primary + file fallback."""

    def __init__(
        self,
        fallback_dir: Path | str = Path("prompts_fallback"),
        *,
        langfuse_client: Any | None = None,
    ) -> None:
        self._fallback_dir = Path(fallback_dir)
        self._lf = langfuse_client
        self._cache: dict[tuple[str, str], Prompt] = {}

    def reset_cache(self) -> None:
        """Clear the in-process prompt cache (tests, hot reload)."""
        self._cache.clear()

    async def get(self, name: str, *, label: str = "production") -> Prompt:
        key = (name, label)
        if key in self._cache:
            return self._cache[key]
        prompt = await self._fetch(name, label)
        self._cache[key] = prompt
        return prompt

    async def render(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
        *,
        label: str = "production",
    ) -> str:
        prompt = await self.get(name, label=label)
        return self._render_template(prompt, variables or {})

    async def _fetch(self, name: str, label: str) -> Prompt:
        if self._lf is not None:
            try:
                lf_prompt = await asyncio.to_thread(self._lf.get_prompt, name, label=label)
                return _from_langfuse(lf_prompt, name=name, label=label)
            except Exception as exc:
                _log.warning(
                    "prompt_langfuse_miss",
                    name=name,
                    label=label,
                    error=str(exc),
                )
        return self._load_fallback(name, label)

    def _load_fallback(self, name: str, label: str) -> Prompt:
        path = self._fallback_dir / f"{name}.md"
        if not path.exists():
            raise PromptNotFound(f"no fallback for prompt '{name}' at {path}")
        text = path.read_text(encoding="utf-8")
        front, body = _parse_fallback_file(text, path=path)
        return Prompt(
            name=str(front.get("name", name)),
            version=int(front.get("version", 1)),
            label=label,
            template=body,
            cache_stable=bool(front.get("cache_stable", False)),
            metadata={
                "prompt_source": "fallback",
                "variables": list(front.get("variables", []) or []),
            },
        )

    def _render_template(self, prompt: Prompt, variables: dict[str, Any]) -> str:
        expected = list(prompt.metadata.get("variables", []) or [])
        missing = [v for v in expected if v not in variables]
        if missing:
            raise PromptRenderError(f"prompt '{prompt.name}' missing variables: {missing}")
        try:
            return _SANDBOX.from_string(prompt.template).render(**variables)
        except UndefinedError as exc:
            raise PromptRenderError(
                f"prompt '{prompt.name}' references undefined variable: {exc}"
            ) from exc
        except TemplateError as exc:
            raise PromptRenderError(f"prompt '{prompt.name}' template error: {exc}") from exc


def _parse_fallback_file(text: str, *, path: Path) -> tuple[dict[str, Any], str]:
    """Split yaml frontmatter from body. Expects leading '---\\n' fence."""
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        raise PromptParseError(f"{path}: missing yaml frontmatter")
    stripped = text.replace("\r\n", "\n")
    parts = stripped.split("---\n", 2)
    if len(parts) < 3:
        raise PromptParseError(f"{path}: unterminated frontmatter")
    _, front_yaml, body = parts
    front = yaml.safe_load(front_yaml) or {}
    if not isinstance(front, dict):
        raise PromptParseError(f"{path}: frontmatter must be a mapping")
    return front, body.strip()


def _from_langfuse(lf_prompt: Any, *, name: str, label: str) -> Prompt:
    template = _prompt_to_string(lf_prompt)
    config = getattr(lf_prompt, "config", None) or {}
    if not isinstance(config, dict):
        config = {}
    return Prompt(
        name=str(getattr(lf_prompt, "name", name)),
        version=int(getattr(lf_prompt, "version", 1) or 1),
        label=label,
        template=template,
        cache_stable=bool(config.get("cache_stable", False)),
        metadata={
            "prompt_source": "langfuse",
            "variables": list(config.get("variables", []) or []),
        },
    )


def _prompt_to_string(lf_prompt: Any) -> str:
    raw = getattr(lf_prompt, "prompt", None) or getattr(lf_prompt, "template", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n\n".join(
            f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in raw if isinstance(m, dict)
        )
    return str(raw)


__all__ = [
    "Prompt",
    "PromptError",
    "PromptNotFound",
    "PromptParseError",
    "PromptRegistry",
    "PromptRenderError",
]
