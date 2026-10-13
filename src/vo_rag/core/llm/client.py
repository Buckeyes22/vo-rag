"""LLMClient — the single seam through which every LLM call flows.

Applies retry discipline (tenacity), structured-output with retry-on-
validation, prompt-registry fetch with optional variant weighting, and
Prometheus observability. Cost is routed through the ambient
`CostAccumulator` installed by `vo_rag.core.observability.cost_context`.

OpenTelemetry gen_ai spans are emitted at this seam as one span per logical
LLM call, not one span per retry attempt. The span name follows the gen_ai
semantic convention shape `"{operation} {model}"` with operation `chat`.
`gen_ai.system` records the API protocol family: Anthropic is `anthropic`,
while OpenAI-compatible providers use `openai`; the concrete model remains in
`gen_ai.request.model`. Prompt, message, and response text must never be added
to span attributes.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import httpx
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from pydantic import BaseModel

from vo_rag.core.llm.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError
from vo_rag.core.llm.prompts import PromptRegistry
from vo_rag.core.llm.providers.anthropic_provider import AnthropicProvider
from vo_rag.core.llm.providers.base import Provider
from vo_rag.core.llm.providers.kimi_provider import KimiProvider
from vo_rag.core.llm.providers.llama_cpp_provider import LlamaCppProvider
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.llm.retry import with_retry
from vo_rag.core.llm.structured import complete_with_schema
from vo_rag.core.llm.variants import select_variant
from vo_rag.core.observability.cost import current_cost_accumulator
from vo_rag.core.observability.metrics import llm_calls_total, llm_latency_seconds
from vo_rag.core.observability.otel import get_tracer

T = TypeVar("T", bound=BaseModel)

_ProviderSpec = Provider | Literal["anthropic", "llama_cpp", "kimi", "runpod_serverless"]
_GEN_AI_SYSTEM_BY_PROVIDER = {"anthropic": "anthropic"}


@dataclass(frozen=True)
class LLMStreamEvent:
    """One text delta emitted by an OpenAI-compatible streaming response."""

    delta: str
    text: str
    model: str
    provider: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class _OpenAIStreamRequest:
    client: httpx.AsyncClient
    path: str
    body: dict[str, Any]


def _compute_cost(resp: LLMResponse) -> float:
    """Compute USD cost from the response.

    Currently returns 0.0 because the rate registry (budget_config) is not
    yet built. When rates are available this will look up per-model pricing.
    """
    return 0.0


def _gen_ai_system(provider_name: str) -> str:
    return _GEN_AI_SYSTEM_BY_PROVIDER.get(provider_name, "openai")


def _gen_ai_span_name(model: str) -> str:
    return f"chat {model}"


def _gen_ai_tracer() -> Tracer:
    return get_tracer("vo_rag.core.llm")


def _set_gen_ai_start_attributes(span: Span, *, provider_name: str, model: str) -> None:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.system", _gen_ai_system(provider_name))
    span.set_attribute("gen_ai.request.model", model)


def _set_gen_ai_usage_attributes(span: Span, resp: LLMResponse) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", str(resp.input_tokens))
    span.set_attribute("gen_ai.usage.output_tokens", str(resp.output_tokens))


def _record_span_exception(span: Span, exc: BaseException) -> None:
    # Record ONLY the exception type + ERROR status. The exception message and
    # stacktrace must NEVER be emitted on an LLM span: StructuredOutputError
    # (and other LLM errors) embed the raw model response in their message, so
    # span.record_exception(exc) would leak response text into the span's
    # exception event (exception.message / exception.stacktrace). ERROR status
    # alone is what drives the gateway's error tail-sampling retention.
    span.set_status(Status(StatusCode.ERROR, type(exc).__qualname__))
    span.add_event("exception", {"exception.type": type(exc).__qualname__})


def _estimate_input_tokens(system: Any, messages: list[dict[str, Any]]) -> int:
    """Rough 4-chars-per-token estimate. Err high (conservative) on purpose."""
    total_chars = 0
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total_chars += len(str(block.get("text") or block.get("content") or ""))
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += sum(
                len(str(b.get("text") or b.get("content") or ""))
                for b in content
                if isinstance(b, dict)
            )
    return max(1, total_chars // 4)


class LLMClient:
    def __init__(
        self,
        *,
        provider: _ProviderSpec = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        role: str = "text",
        budget: Any | None = None,
        cost_accumulator: Any | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._provider = _build_provider(
            provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            role=role,
        )
        self.budget = budget
        self.cost_accumulator = cost_accumulator
        self.prompt_registry = prompt_registry

    @property
    def provider(self) -> Provider:
        return self._provider

    async def complete(
        self,
        *,
        prompt_name: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        label: str = "production",
        variant_weights: dict[str, float] | None = None,
        envelope_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        resolved_system, used_variant = await self._resolve_system(
            prompt_name=prompt_name,
            prompt_variables=prompt_variables,
            explicit_system=system,
            label=label,
            variant_weights=variant_weights,
            envelope_id=envelope_id,
        )
        resolved_messages: list[dict[str, Any]] = list(messages or [])

        await self._maybe_charge_budget(resolved_system, resolved_messages, max_tokens)

        async def _call() -> LLMResponse:
            return await self._provider.complete_raw(
                messages=resolved_messages,
                system=resolved_system,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )

        model = self._provider.model
        with _gen_ai_tracer().start_as_current_span(
            _gen_ai_span_name(model),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            _set_gen_ai_start_attributes(span, provider_name=self._provider.name, model=model)
            try:
                resp = await with_retry(self._provider.name, model, _call)
                resp = resp.model_copy(
                    update={
                        "prompt_name": prompt_name,
                        "variant": used_variant,
                        "cost_usd": _compute_cost(resp),
                    }
                )
                _set_gen_ai_usage_attributes(span, resp)
                self._emit_success(prompt_name, resp)
                self._maybe_record_cost(prompt_name, resp)
                return resp
            except BaseException as exc:
                _record_span_exception(span, exc)
                raise

    async def complete_json(
        self,
        *,
        schema: type[T],
        prompt_name: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        label: str = "production",
        variant_weights: dict[str, float] | None = None,
        envelope_id: str | None = None,
        max_retries: int = 2,
    ) -> tuple[T, LLMResponse]:
        resolved_system, used_variant = await self._resolve_system(
            prompt_name=prompt_name,
            prompt_variables=prompt_variables,
            explicit_system=system,
            label=label,
            variant_weights=variant_weights,
            envelope_id=envelope_id,
        )
        resolved_messages: list[dict[str, Any]] = list(messages or [])

        await self._maybe_charge_budget(resolved_system, resolved_messages, max_tokens)

        model = self._provider.model
        with _gen_ai_tracer().start_as_current_span(
            _gen_ai_span_name(model),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            _set_gen_ai_start_attributes(span, provider_name=self._provider.name, model=model)
            try:
                parsed, resp = await complete_with_schema(
                    self._provider,
                    schema=schema,
                    messages=resolved_messages,
                    system=resolved_system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    prompt_name=prompt_name,
                )
                resp = resp.model_copy(
                    update={
                        "prompt_name": prompt_name,
                        "variant": used_variant,
                        "cost_usd": _compute_cost(resp),
                    }
                )
                _set_gen_ai_usage_attributes(span, resp)
                self._emit_success(prompt_name, resp)
                self._maybe_record_cost(prompt_name, resp)
                return parsed, resp
            except BaseException as exc:
                _record_span_exception(span, exc)
                raise

    async def stream_complete(
        self,
        *,
        prompt_name: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        label: str = "production",
        variant_weights: dict[str, float] | None = None,
        envelope_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        resolved_system, used_variant = await self._resolve_system(
            prompt_name=prompt_name,
            prompt_variables=prompt_variables,
            explicit_system=system,
            label=label,
            variant_weights=variant_weights,
            envelope_id=envelope_id,
        )
        resolved_messages: list[dict[str, Any]] = list(messages or [])

        await self._maybe_charge_budget(resolved_system, resolved_messages, max_tokens)

        stream_request = self._openai_stream_request(
            system=resolved_system,
            messages=resolved_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

        model = self._provider.model
        span = _gen_ai_tracer().start_span(_gen_ai_span_name(model))
        _set_gen_ai_start_attributes(span, provider_name=self._provider.name, model=model)
        try:
            t0 = time.perf_counter()
            text = ""
            usage: dict[str, Any] | None = None
            response_model = model
            try:
                async with stream_request.client.stream(
                    "POST",
                    stream_request.path,
                    json=stream_request.body,
                ) as response:
                    _raise_for_stream_status(response)
                    async for line in response.aiter_lines():
                        payload = _decode_sse_line(line)
                        if payload is None:
                            continue
                        if isinstance(payload, str):
                            if payload == "[DONE]":
                                break
                            continue
                        response_model = str(payload.get("model") or response_model)
                        chunk_usage = _as_mapping(payload.get("usage"))
                        if chunk_usage:
                            usage = dict(chunk_usage)
                        for delta in _iter_choice_deltas(payload):
                            text += delta
                            yield LLMStreamEvent(
                                delta=delta,
                                text=text,
                                model=response_model,
                                provider=self._provider.name,
                                raw=payload,
                            )
            except httpx.HTTPStatusError:
                raise
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as exc:
                raise LLMServerError(str(exc)) from exc

            if usage is None:
                raise RuntimeError("streaming provider completed without a final usage chunk")

            resp = LLMResponse(
                text=text,
                model=response_model,
                input_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
                output_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                prompt_name=prompt_name,
                variant=used_variant,
                cost_usd=0.0,
                provider=self._provider.name,
            )
            resp = resp.model_copy(update={"cost_usd": _compute_cost(resp)})
            _set_gen_ai_usage_attributes(span, resp)
            self._emit_success(prompt_name, resp)
            self._maybe_record_cost(prompt_name, resp)
        except BaseException as exc:
            _record_span_exception(span, exc)
            raise
        finally:
            span.end()

    async def aclose(self) -> None:
        await self._provider.aclose()

    def _openai_stream_request(
        self,
        *,
        system: str | list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra: dict[str, Any] | None,
    ) -> _OpenAIStreamRequest:
        provider_name = self._provider.name
        if provider_name == "runpod_serverless":
            serverless_client = getattr(self._provider, "_client", None)
            http_client = _httpx_client_from(serverless_client)
            body: dict[str, Any] = {
                "model": self._provider.model,
                "messages": _prepare_openai_messages(system, messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            _merge_extra(body, extra)
            _force_stream_options(body)
            return _OpenAIStreamRequest(client=http_client, path="/chat/completions", body=body)

        if provider_name == "kimi":
            http_client = _httpx_client_from(self._provider)
            body = {
                "model": self._provider.model,
                "messages": _prepare_openai_messages(system, messages),
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            if self._provider.model.startswith("kimi-k2."):
                thinking_enabled = bool(getattr(self._provider, "_thinking_enabled", True))
                body["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
            else:
                body["temperature"] = temperature
            _merge_extra(body, extra)
            _force_stream_options(body)
            return _OpenAIStreamRequest(client=http_client, path="/chat/completions", body=body)

        raise NotImplementedError(
            f"provider {provider_name!r} does not support real token streaming"
        )

    async def _resolve_system(
        self,
        *,
        prompt_name: str | None,
        prompt_variables: dict[str, Any] | None,
        explicit_system: str | None,
        label: str,
        variant_weights: dict[str, float] | None,
        envelope_id: str | None,
    ) -> tuple[str | None, str | None]:
        if explicit_system is not None and prompt_name is not None:
            raise ValueError("pass either prompt_name or system, not both")
        if explicit_system is not None:
            return explicit_system, None
        if prompt_name is None:
            return None, None
        if self.prompt_registry is None:
            raise ValueError("prompt_name requires a prompt_registry on LLMClient")

        used_variant: str | None = None
        fetch_name = prompt_name
        if variant_weights:
            used_variant = select_variant(prompt_name, variant_weights, envelope_id=envelope_id)
            fetch_name = used_variant

        system = await self.prompt_registry.render(fetch_name, prompt_variables or {}, label=label)
        return system, used_variant

    def _emit_success(self, prompt_name: str | None, resp: LLMResponse) -> None:
        llm_calls_total.labels(
            provider=self._provider.name,
            model=self._provider.model,
            prompt_name=prompt_name or "unknown",
            result="success",
        ).inc()
        llm_latency_seconds.labels(
            provider=self._provider.name,
            model=self._provider.model,
        ).observe(resp.latency_ms / 1000.0)

    def _maybe_record_cost(self, prompt_name: str | None, resp: LLMResponse) -> None:
        """Record the call against the active CostAccumulator.

        Preference order: explicit DI > ambient contextvar. An absent
        accumulator is a no-op. Failures swallow — cost accounting must
        never crash an LLM call path.
        """
        acc = self.cost_accumulator or current_cost_accumulator()
        if acc is None:
            return
        add = getattr(acc, "add", None)
        if add is None:
            return
        with contextlib.suppress(Exception):
            add(prompt_name or "llm", resp)

    async def _maybe_charge_budget(
        self,
        resolved_system: Any,
        resolved_messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> None:
        """Pre-call budget protection. If self.budget is set, charge the
        estimated cost of this call — if it would exceed the cap, BudgetExceeded
        propagates and no network call is made."""
        budget = self.budget
        if budget is None:
            return
        charge_estimated = getattr(budget, "charge_estimated", None)
        if charge_estimated is None:
            return
        in_tokens = _estimate_input_tokens(resolved_system, resolved_messages)
        out_tokens = max_tokens
        await charge_estimated(in_tokens, out_tokens, self._provider.model)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _flatten_system(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    return "\n".join(
        str(block.get("text") or block.get("content") or "")
        for block in system
        if isinstance(block, dict)
    )


def _prepare_openai_messages(
    system: str | list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if system is None:
        return list(messages)
    return [{"role": "system", "content": _flatten_system(system)}, *messages]


def _httpx_client_from(owner: Any) -> httpx.AsyncClient:
    http_client = getattr(owner, "_client", None)
    if isinstance(http_client, httpx.AsyncClient):
        return http_client
    raise NotImplementedError("streaming provider does not expose an httpx AsyncClient")


def _merge_extra(body: dict[str, Any], extra: dict[str, Any] | None) -> None:
    if extra:
        body.update(extra)


def _force_stream_options(body: dict[str, Any]) -> None:
    stream_options = body.get("stream_options")
    if isinstance(stream_options, dict):
        body["stream_options"] = {**stream_options, "include_usage": True}
    else:
        body["stream_options"] = {"include_usage": True}
    body["stream"] = True


def _decode_sse_line(line: str) -> dict[str, Any] | str | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    data = stripped.removeprefix("data:").strip()
    if data == "[DONE]":
        return data
    decoded = json.loads(data)
    if not isinstance(decoded, dict):
        return None
    return decoded


def _iter_choice_deltas(payload: Mapping[str, Any]) -> list[str]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    deltas: list[str] = []
    for choice in choices:
        choice_mapping = _as_mapping(choice)
        delta_mapping = _as_mapping(choice_mapping.get("delta"))
        content = delta_mapping.get("content")
        if isinstance(content, str) and content:
            deltas.append(content)
    return deltas


def _usage_int(usage: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _raise_for_stream_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            raise LLMRateLimitError(str(exc)) from exc
        if 500 <= status < 600:
            raise LLMServerError(str(exc)) from exc
        raise LLMBadRequestError(str(exc)) from exc


def _build_provider(
    spec: _ProviderSpec,
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None = None,
    role: str = "text",
) -> Provider:
    if not isinstance(spec, str):
        return spec
    if spec == "anthropic":
        if api_key is None or model is None:
            raise ValueError("anthropic provider requires api_key and model")
        return AnthropicProvider(api_key=api_key, model=model)
    if spec == "llama_cpp":
        if base_url is None or model is None:
            raise ValueError("llama_cpp provider requires base_url and model")
        return LlamaCppProvider(
            base_url=base_url,
            model=model,
            timeout_s=float(os.environ.get("LOCAL_LLM_TIMEOUT_S", "60")),
        )
    if spec == "kimi":
        if api_key is None or model is None:
            raise ValueError("kimi provider requires api_key and model")
        return KimiProvider(
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.moonshot.ai/v1",
            timeout_s=float(os.environ.get("KIMI_TIMEOUT_S", "90")),
            thinking_enabled=True,
        )
    if spec == "runpod_serverless":
        from vo_rag.core.cloud.serverless import ServerlessClient
        from vo_rag.core.config import get_settings
        from vo_rag.core.llm.providers.runpod_serverless_provider import (
            RunpodServerlessProvider,
        )

        settings = get_settings().serverless
        client = ServerlessClient(
            base_url=settings.base_url(role),
            api_key=settings.api_key,
            model=model or settings.model_name(role),
            timeout_s=settings.request_timeout_s,
        )
        return RunpodServerlessProvider(client=client)
    raise ValueError(f"unknown provider spec: {spec!r}")


__all__ = ["LLMClient", "LLMStreamEvent"]
