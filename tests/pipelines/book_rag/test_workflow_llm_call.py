from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.llm.response import LLMResponse
from vo_rag.pipelines.book_rag.workflow import BookRagPipeline


def _pipeline(**overrides: Any) -> BookRagPipeline:
    base: dict[str, Any] = {
        "garage": SimpleNamespace(),
        "qdrant_store": SimpleNamespace(),
        "embedding_client": SimpleNamespace(),
        "llm_client": SimpleNamespace(),
        "retriever": SimpleNamespace(),
        "parser_router": SimpleNamespace(),
    }
    base.update(overrides)
    return BookRagPipeline(**base)


@pytest.mark.asyncio
async def test_llm_call_disables_thinking_and_strips_reasoning() -> None:
    class _CapturingLLM:
        def __init__(self) -> None:
            self.last_kwargs: dict[str, Any] = {}

        async def complete(self, **kwargs: Any) -> LLMResponse:
            self.last_kwargs = kwargs
            return LLMResponse(
                text='<think>internal scratchpad</think>\n{"queries": []}',
                model="fake-model",
                input_tokens=3,
                output_tokens=4,
            )

    llm = _CapturingLLM()
    pipeline = _pipeline(llm_client=llm)

    resp = await pipeline._llm_call("system", "user", 128, 0.1)

    assert llm.last_kwargs["extra"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm.last_kwargs["messages"] == [{"role": "user", "content": "user"}]
    assert resp.text == '{"queries": []}'
    assert resp.input_tokens == 3
