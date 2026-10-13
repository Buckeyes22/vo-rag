"""VoRagLLM — LlamaIndex CustomLLM over vo-rag's LLMClient.

SECURITY: This is the UNPROTECTED, trusted-only seam. It wraps
LLMClient.complete, which applies NO injection defense. Do not pass
untrusted retrieved context through it; use complete_json_protected
(the query engine does) for that.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator, Sequence
from typing import Any

from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.response import LLMResponse
from vo_rag.integrations.llamaindex._async import run_coro_sync

logger = logging.getLogger(__name__)


def _messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(f"{message.role}: {message.content or ''}" for message in messages)


class VoRagLLM(CustomLLM):
    """LlamaIndex LLM over vo-rag's LLMClient.

    TRUSTED SEAM ONLY. This wraps LLMClient.complete, which applies NO
    injection defense. Do not pass untrusted retrieved context through it;
    use complete_json_protected (the query engine does) for that.
    """

    model_name: str = "vo-rag"
    context_window: int = 8192
    num_output: int = 1024

    _client: LLMClient

    def __init__(
        self,
        *,
        client: LLMClient,
        context_window: int = 8192,
        num_output: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            context_window=context_window,
            num_output=num_output,
            model_name=client.provider.model,
            **kwargs,
        )
        object.__setattr__(self, "_client", client)

    @property
    def supports_guided_decoding(self) -> bool:
        return self._client.provider.name in {"runpod_serverless", "kimi"}

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return run_coro_sync(
            lambda: self._achat_inner(messages, **kwargs),
            component="VoRagLLM.chat",
        )

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return await self._achat_inner(messages, **kwargs)

    async def _achat_inner(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = _messages_to_prompt(messages)
        completion = await self.acomplete(prompt, **kwargs)
        return self._to_chat(completion)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        prompt = _messages_to_prompt(messages)
        completion_gen = self.stream_complete(prompt, **kwargs)

        def _gen() -> ChatResponseGen:
            for completion in completion_gen:
                yield self._to_chat(completion, delta=completion.delta)

        return _gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def _gen() -> ChatResponseAsyncGen:
            async for completion in self._astream_complete_inner(
                _messages_to_prompt(messages), **kwargs
            ):
                yield self._to_chat(completion, delta=completion.delta)

        return _gen()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        resp = run_coro_sync(
            lambda: self._acomplete_inner(prompt, **kwargs),
            component="VoRagLLM.complete",
        )
        return self._to_completion(resp)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        resp = await self._acomplete_inner(prompt, **kwargs)
        return self._to_completion(resp)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return _sync_completion_gen(
            self._astream_complete_inner(prompt, **kwargs),
            component="VoRagLLM.stream_complete",
        )

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return self._astream_complete_inner(prompt, **kwargs)

    async def _acomplete_inner(self, prompt: str, **kwargs: Any) -> LLMResponse:
        max_tokens, temperature, system = _completion_options(self.num_output, kwargs)
        return await self._client.complete(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _astream_complete_inner(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        max_tokens, temperature, system = _completion_options(self.num_output, kwargs)
        async for event in self._client.stream_complete(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield CompletionResponse(text=event.text, delta=event.delta, raw=event.raw)

    @staticmethod
    def _to_completion(resp: LLMResponse) -> CompletionResponse:
        return CompletionResponse(text=resp.text, raw=resp.model_dump())

    @staticmethod
    def _to_chat(completion: CompletionResponse, *, delta: str | None = None) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=completion.text),
            raw=completion.raw,
            delta=delta,
        )


def _completion_options(
    default_max_tokens: int,
    kwargs: dict[str, Any],
) -> tuple[int, float, str | None]:
    max_tokens = int(kwargs.pop("max_tokens", default_max_tokens))
    temperature = float(kwargs.pop("temperature", 0.0))
    system = kwargs.pop("system", None)
    if kwargs:
        logger.warning(
            "VoRagLLM ignored unsupported LlamaIndex kwargs: %s",
            ", ".join(sorted(kwargs)),
        )
    return max_tokens, temperature, system if isinstance(system, str) else None


def _sync_completion_gen(
    async_gen: AsyncGenerator[CompletionResponse, None],
    *,
    component: str,
) -> CompletionResponseGen:
    def _gen() -> Generator[CompletionResponse, None, None]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                f"{component} cannot use its synchronous LlamaIndex API inside a running "
                "event loop. Use the adapter's async API instead."
            )

        async def _next() -> CompletionResponse:
            return await anext(async_gen)

        with asyncio.Runner() as runner:
            try:
                while True:
                    try:
                        yield runner.run(_next())
                    except StopAsyncIteration:
                        break
            finally:
                runner.run(async_gen.aclose())

    return _gen()
