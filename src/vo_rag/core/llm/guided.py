"""Structured-output ``extra`` settings for advanced-layer extraction.

The advanced-layer builders (graph/DSI/summary/RAPTOR) take a 4-arg ``llm_call``
with no ``extra`` slot, so the structured-output settings and
``enable_thinking=False`` are baked into the call adapter via this helper. It is
the single source of truth shared by the CLI build path
(:mod:`vo_rag.cli.build`) and the LlamaIndex AdvancedBuildWorkflow
(:mod:`vo_rag.integrations.llamaindex.build_workflow`).
"""

from __future__ import annotations

import os
from typing import Any


def structured_output_extra(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Build LLM ``extra`` settings for advanced-layer structured output.

    ``VORAG_GUIDED_DECODING_MODE`` selects how a supplied JSON Schema is sent:

    - ``json_schema`` (default) — OpenAI-compatible ``response_format``. The
      portable path for stock vLLM/OpenAI servers, which ignore a top-level
      ``guided_json`` field.
    - ``guided_json`` — legacy worker-vllm xgrammar top-level field.
    - ``native_json_schema`` — schema as a top-level ``json_schema`` field, for
      llama.cpp servers whose OpenAI ``response_format: json_schema`` shim does
      not constrain decoding (it 500s on the strict parse of free text) while the
      native field enforces the grammar.
    - ``json_object`` — valid JSON only, no schema enforcement.

    Unknown values fall back to ``json_schema``. When no schema is supplied the
    builder always uses ``json_object``.
    """
    extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
    if schema is None:
        extra["response_format"] = {"type": "json_object"}
        return extra

    mode = os.environ.get("VORAG_GUIDED_DECODING_MODE", "json_schema").strip().lower()
    if mode == "guided_json":
        extra["guided_json"] = schema
    elif mode == "native_json_schema":
        extra["json_schema"] = schema
    elif mode == "json_object":
        extra["response_format"] = {"type": "json_object"}
    else:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": schema},
        }
    return extra
