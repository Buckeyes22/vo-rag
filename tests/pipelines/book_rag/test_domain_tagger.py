"""Contract tests for the book_rag domain vocabulary shape and the
DomainTagger leaf.

Behavior pinned here:

- `DomainTagger` takes an injected `vo_rag.core.llm.client.LLMClient`. The
  test wires a scripted provider the same way
  `tests/core/llm/test_protected.py` does.
- Vocabulary and keyword packs are overridable at construction
  (`vocabulary=`, `keywords=`) so operators can plug a non-motorsport
  pack in without forking the leaf. The shipped default IS the
  motorsport pack because the book_rag eval baselines are motorsport.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.response import LLMResponse
from vo_rag.pipelines.book_rag.domain_tagger import (
    DOMAIN_KEYWORDS,
    DOMAIN_VOCABULARY,
    DomainTagger,
    heuristic_domain_tags,
)
from vo_rag.pipelines.book_rag.domain_vocab import (
    DOMAIN_KEYWORDS as VOCAB_DOMAIN_KEYWORDS,
)
from vo_rag.pipelines.book_rag.domain_vocab import (
    DOMAIN_VOCABULARY as VOCAB_DOMAIN_VOCABULARY,
)

# ---------------------------------------------------------------------------
# Vocabulary / keyword pack — shape contract
# ---------------------------------------------------------------------------


def test_domain_vocabulary_is_list_of_str() -> None:
    assert isinstance(VOCAB_DOMAIN_VOCABULARY, list)
    assert VOCAB_DOMAIN_VOCABULARY, "vocab must not be empty"
    assert all(isinstance(tag, str) and tag for tag in VOCAB_DOMAIN_VOCABULARY)


def test_domain_keywords_is_dict_of_tuples() -> None:
    assert isinstance(VOCAB_DOMAIN_KEYWORDS, dict)
    assert VOCAB_DOMAIN_KEYWORDS, "keyword pack must not be empty"
    for tag, kws in VOCAB_DOMAIN_KEYWORDS.items():
        assert isinstance(tag, str) and tag
        assert isinstance(kws, tuple)
        assert all(isinstance(k, str) and k for k in kws)


def test_domain_vocab_and_keywords_keys_match() -> None:
    """Every keyword pack key must be in the controlled vocabulary; orphan
    keywords would be silently unreachable from the LLM tag path."""
    assert set(VOCAB_DOMAIN_KEYWORDS.keys()) <= set(VOCAB_DOMAIN_VOCABULARY)


def test_default_pack_is_motorsport_relevant() -> None:
    """The shipped default pack IS motorsport — book_rag eval baselines are
    motorsport per the user directive. Operators can override via the
    `vocabulary=` / `keywords=` ctor args; the default just needs to
    cover the canonical motorsport tags."""
    expected_anchors = {
        "braking",
        "tire_mechanics",
        "telemetry",
        "vehicle_dynamics",
        "powertrain",
    }
    assert expected_anchors <= set(VOCAB_DOMAIN_VOCABULARY)


def test_domain_tagger_re_exports_vocab_default() -> None:
    """The `domain_tagger` module re-exposes the default pack so callers
    that import only `DomainTagger` don't have to reach into `domain_vocab`."""
    assert DOMAIN_VOCABULARY is VOCAB_DOMAIN_VOCABULARY
    assert DOMAIN_KEYWORDS is VOCAB_DOMAIN_KEYWORDS


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str = "Trail braking and rotation") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="doc-1",
        source_id="src-1",
        page_number=1,
        page_range="1",
        section_path="Book > Chapter",
        section_title="Chapter",
        content=text,
        embedding_text=text,
        token_count=len(text) // 4,
    )


class _ScriptedProvider:
    """Same shape as the one in tests/core/llm/test_protected.py so
    `LLMClient(provider=scripted)` works without touching the network."""

    name = "scripted"
    model = "claude-sonnet-4-6"

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_system: str | None = None
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        self.last_system = system if isinstance(system, str) else None
        self.last_messages = list(messages)
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        return LLMResponse(
            text=self._text,
            model=self.model,
            input_tokens=10,
            output_tokens=5,
            provider=self.name,
        )

    async def aclose(self) -> None:
        return None


def _llm_client_with_payload(payload: dict[str, Any] | str) -> tuple[LLMClient, _ScriptedProvider]:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    provider = _ScriptedProvider(text)
    return LLMClient(provider=provider), provider


# ---------------------------------------------------------------------------
# heuristic_domain_tags
# ---------------------------------------------------------------------------


def test_heuristic_domain_tags_uses_controlled_vocabulary() -> None:
    tags = heuristic_domain_tags(
        "Trail braking rotates the car at turn-in while tire slip angle and "
        "telemetry traces change.",
        max_tags=4,
    )

    assert tags
    assert set(tags) <= set(VOCAB_DOMAIN_VOCABULARY)
    assert "braking" in tags
    assert "tire_mechanics" in tags
    assert "telemetry" in tags


def test_heuristic_domain_tags_respects_max_tags() -> None:
    text = (
        "Suspension geometry, spring rate, damper tuning, brake bias, downforce balance, "
        "tire slip, telemetry traces, driver input, and race strategy combine on track."
    )
    tags = heuristic_domain_tags(text, max_tags=3)
    assert len(tags) <= 3


def test_heuristic_domain_tags_empty_when_no_match() -> None:
    assert heuristic_domain_tags("Lorem ipsum dolor sit amet.", max_tags=6) == []


# ---------------------------------------------------------------------------
# DomainTagger.tag — single chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_filters_to_vocab_only() -> None:
    client, _ = _llm_client_with_payload({"tags": ["braking", "made_up", "driver_technique"]})

    tagger = DomainTagger(client)
    tags = await tagger.tag(_chunk("c-1"))

    assert tags == ["braking", "driver_technique"]


@pytest.mark.asyncio
async def test_tag_enforces_max_tags_per_chunk() -> None:
    client, _ = _llm_client_with_payload({"tags": VOCAB_DOMAIN_VOCABULARY[:10]})

    tagger = DomainTagger(client, max_tags_per_chunk=6)
    tags = await tagger.tag(_chunk("c-2"))

    assert len(tags) == 6
    assert tags == VOCAB_DOMAIN_VOCABULARY[:6]


@pytest.mark.asyncio
async def test_tag_returns_empty_on_invalid_json() -> None:
    """When the LLM returns a payload that can't be parsed into the schema
    (after the structured-output retry budget is exhausted) the tagger must
    return `[]` rather than raising — the LLM path is best-effort."""
    client, _ = _llm_client_with_payload("not-json-at-all")

    tagger = DomainTagger(client)
    tags = await tagger.tag(_chunk("c-bad"))

    assert tags == []


# ---------------------------------------------------------------------------
# DomainTagger.tag_batch — multi-chunk concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_batch_preserves_chunk_id_mapping() -> None:
    client, _ = _llm_client_with_payload({"tags": ["telemetry"]})

    chunks = [_chunk("c-1"), _chunk("c-2"), _chunk("c-3")]
    out = await DomainTagger(client).tag_batch(chunks)

    assert out == {"c-1": ["telemetry"], "c-2": ["telemetry"], "c-3": ["telemetry"]}


@pytest.mark.asyncio
async def test_tag_batch_concurrency_bounded_by_semaphore() -> None:
    class _SlowProvider(_ScriptedProvider):
        async def complete_raw(
            self,
            *,
            messages: list[dict[str, Any]],
            system: str | list[dict[str, Any]] | None = None,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            extra: dict[str, Any] | None = None,  # noqa: ARG002
        ) -> LLMResponse:
            await asyncio.sleep(0.1)
            return await super().complete_raw(
                messages=messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )

    provider = _SlowProvider(json.dumps({"tags": ["braking"]}))
    client = LLMClient(provider=provider)

    chunks = [_chunk(f"c-{i}") for i in range(32)]
    tagger = DomainTagger(client, concurrency=8)

    t0 = time.perf_counter()
    out = await tagger.tag_batch(chunks)
    elapsed = time.perf_counter() - t0

    assert len(out) == 32
    # 32 chunks at 0.1s each, 8 in parallel → ~0.4s floor. Allow a small margin.
    assert elapsed >= 0.4


# ---------------------------------------------------------------------------
# DomainTagger — vocabulary / keyword override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_tagger_accepts_custom_vocabulary() -> None:
    """Operators can plug a non-motorsport pack via `vocabulary=` /
    `keywords=`. The default pack is motorsport, but the override path
    must be first-class (not a fork)."""
    client, _ = _llm_client_with_payload(
        {"tags": ["marine_navigation", "made_up", "celestial_sight"]}
    )
    custom_vocab = ["marine_navigation", "celestial_sight", "tide_tables"]
    custom_kws: dict[str, tuple[str, ...]] = {
        "marine_navigation": ("compass", "bearing"),
        "celestial_sight": ("star", "sextant"),
        "tide_tables": ("tide",),
    }

    tagger = DomainTagger(client, vocabulary=custom_vocab, keywords=custom_kws)
    tags = await tagger.tag(_chunk("c-vocab"))

    assert tags == ["marine_navigation", "celestial_sight"]


def test_heuristic_domain_tags_uses_custom_keywords_when_provided() -> None:
    """Constructor-time override feeds the heuristic path too — the override
    is pack-wide, not LLM-only."""
    custom_vocab = ["compass", "tide"]
    custom_kws: dict[str, tuple[str, ...]] = {
        "compass": ("compass", "bearing", "navigate"),
        "tide": ("tide", "current"),
    }
    client, _ = _llm_client_with_payload({"tags": []})
    tagger = DomainTagger(client, vocabulary=custom_vocab, keywords=custom_kws)

    # The custom-pack heuristic should match "compass" and "tide" against
    # the operator-supplied keywords, not the default motorsport pack.
    tags = tagger.heuristic_tags(
        "Use a compass to navigate by bearing and read the tide tables.",
        max_tags=5,
    )

    assert set(tags) <= set(custom_vocab)
    # At least one of the custom tags must have matched — the default
    # motorsport pack would have produced an empty result here.
    assert tags
