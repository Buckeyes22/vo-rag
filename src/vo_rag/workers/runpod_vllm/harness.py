"""Local FastAPI harness for the RunPod vLLM worker surface.

This app is a *test + local-dev* double for the real RunPod serverless
endpoints. It is deliberately small:

- POST ``/openai/v1/chat/completions`` returns an OpenAI-shaped JSON
  body (choices[0].message.content, usage.prompt_tokens,
  usage.completion_tokens, model) so ``ServerlessClient`` can drive it
  end-to-end with no GPU and no real RunPod bill.
- GET ``/health`` reports ``{"mode": "local-harness", "gpu": false}``
  so callers and tests can tell at a glance that they are pointed at
  the harness, not production.

The harness is NOT a production inference server and is NOT a RunPod
transport. It must never call ``api.runpod.ai`` and must never import
``vo_rag.core.cloud.serverless`` directly — it is the *target* of that
client, not a peer of it.

Run locally:

    python -m vo_rag.workers.runpod_vllm.harness --port 8188
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_harness_app(
    *,
    response_text: str = "ok",
    model: str = "local-harness-model",
    fail_first_n: int = 0,
) -> FastAPI:
    """Build a FastAPI app that imitates a RunPod vLLM worker.

    Args:
        response_text: literal string returned in ``choices[0].message.content``.
        model: string returned in the response ``model`` field.
        fail_first_n: number of initial POSTs to ``/openai/v1/chat/completions``
            that should return HTTP 502 before the harness starts
            returning 200. This exercises ``ServerlessClient``'s
            ``(1, 3, 9)`` retry loop *without* requiring real sleeps —
            the test typically overrides the retry delays to ``(0, 0, 0)``
            so the harness's 502→200 sequence completes in milliseconds
            while the retry *count* still goes through.

    Returns:
        A fresh ``FastAPI`` instance. Each call to this factory returns a
        new app with its own counter (counters are app-scoped, not
        process-scoped), so test isolation is trivial.
    """
    call_lock = threading.Lock()
    call_count = 0

    app = FastAPI(title="vo-rag runpod-vllm local harness", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"mode": "local-harness", "gpu": False}

    @app.post("/openai/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1

        if call_index < fail_first_n:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "simulated 502", "type": "harness"}},
            )

        messages = body.get("messages") or []
        # Naive token count: characters/4 — fine for harness smoke tests.
        prompt_text = _flatten_messages(messages)
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = max(1, len(response_text) // 4)

        # `created` must be an int; real OpenAI uses epoch seconds.
        return JSONResponse(
            status_code=200,
            content={
                "id": f"harness-{call_index}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    return app


def _flatten_messages(messages: Any) -> str:
    """Concatenate message content for the harness's prompt-token estimate."""
    if not isinstance(messages, list):
        return ""
    chunks: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    return "\n".join(chunks)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vo_rag.workers.runpod_vllm.harness",
        description=(
            "Local OpenAI-compatible harness for the RunPod vLLM worker "
            "surface. Test + local-dev only. NOT a production inference server."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default 127.0.0.1; do not expose publicly).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8188,
        help="Bind port (default 8188).",
    )
    parser.add_argument(
        "--response-text",
        default="ok",
        help="Literal text returned by the harness for every request.",
    )
    parser.add_argument(
        "--model",
        default="local-harness-model",
        help="Model name returned in the response `model` field.",
    )
    parser.add_argument(
        "--fail-first-n",
        type=int,
        default=0,
        help="Number of initial requests that return HTTP 502 before success.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    app = create_harness_app(
        response_text=args.response_text,
        model=args.model,
        fail_first_n=args.fail_first_n,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


__all__ = ["create_harness_app", "main"]


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m ...`
    main()
