"""Contract tests for src/vo_rag/workers/runpod_vllm/{schemas,contracts}.py.

These tests pin the wire format and the no-bypass invariant — every workload
goes through ``vo_rag.core.cloud.serverless.ServerlessClient`` and never through
raw httpx / a RunPod SDK call to ``api.runpod.ai``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from vo_rag.workers.runpod_vllm.contracts import (
    build_chat_payload,
    client_for_workload,
    role_for_workload,
)
from vo_rag.workers.runpod_vllm.schemas import (
    ChatContentPart,
    ChatMessage,
    RunpodWorkload,
    ServerlessRole,
    ServerlessWorkloadRequest,
    ServerlessWorkloadResult,
)

# --------------------------------------------------------------------------- #
# Workload + role enum surfaces
# --------------------------------------------------------------------------- #


def test_workload_literal_exactly_three_values() -> None:
    """The workload Literal is exactly text-gen, vlm, dsi — nothing else."""
    assert RunpodWorkload.__args__ == ("text-gen", "vlm", "dsi")  # type: ignore[attr-defined]


def test_serverless_role_literal_exactly_three_values() -> None:
    """The role Literal is exactly text, vlm, dsi — not the legacy 'exp'."""
    assert ServerlessRole.__args__ == ("text", "vlm", "dsi")  # type: ignore[attr-defined]


def test_role_for_workload_is_explicit_mapping() -> None:
    assert role_for_workload("text-gen") == "text"
    assert role_for_workload("vlm") == "vlm"
    assert role_for_workload("dsi") == "dsi"


def test_role_for_workload_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown workload"):
        role_for_workload("bogus-workload")
    with pytest.raises(ValueError):
        role_for_workload("embedding")


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #


def test_chat_content_part_text_round_trip() -> None:
    part = ChatContentPart(type="text", text="hello")
    assert part.type == "text"
    assert part.text == "hello"


def test_chat_content_part_image_url_round_trip() -> None:
    part = ChatContentPart(
        type="image_url",
        image_url={"url": "data:image/png;base64,iVBORw0KGgo="},
    )
    assert part.type == "image_url"
    assert part.image_url is not None
    assert part.image_url["url"].startswith("data:image/png;base64,")


def test_chat_content_part_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ChatContentPart(type="text", text="x", not_a_field="nope")  # type: ignore[call-arg]


def test_chat_message_text_only_role_user() -> None:
    msg = ChatMessage(role="user", content="hi")
    assert msg.role == "user"
    assert msg.content == "hi"


def test_chat_message_multimodal_with_image_url_part() -> None:
    msg = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ],
    )
    parts = [ChatContentPart.model_validate(p) for p in msg.content]
    assert parts[0].type == "text"
    assert parts[1].type == "image_url"


def test_chat_message_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="hi", extra_field="x")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# ServerlessWorkloadRequest: workload + content rules
# --------------------------------------------------------------------------- #


def _text_only_request(workload: str) -> ServerlessWorkloadRequest:
    return ServerlessWorkloadRequest(
        workload=workload,  # type: ignore[arg-type]
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=64,
        temperature=0.0,
    )


def _vlm_multimodal_request() -> ServerlessWorkloadRequest:
    return ServerlessWorkloadRequest(
        workload="vlm",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            )
        ],
        max_tokens=64,
        temperature=0.0,
    )


def test_text_gen_request_accepts_text_only() -> None:
    req = _text_only_request("text-gen")
    assert req.workload == "text-gen"
    assert req.messages[0].content == "hi"


def test_dsi_request_accepts_text_only() -> None:
    req = _text_only_request("dsi")
    assert req.workload == "dsi"


def test_vlm_request_accepts_text_only_too() -> None:
    """VLM is a superset — it can still answer text-only queries."""
    req = _text_only_request("vlm")
    assert req.workload == "vlm"


def test_vlm_request_accepts_image_url_content_part() -> None:
    req = _vlm_multimodal_request()
    assert req.workload == "vlm"
    parts = req.messages[0].content
    assert isinstance(parts, list)
    assert len(parts) == 2
    # parts may be ChatContentPart or raw dicts (Pydantic coerces);
    # check the type string from each regardless of representation.
    assert _part_type(parts[0]) == "text"
    assert _part_type(parts[1]) == "image_url"


def _part_type(part: object) -> str | None:
    t = getattr(part, "type", None)
    if isinstance(t, str):
        return t
    if isinstance(part, dict):
        raw = part.get("type")
        return raw if isinstance(raw, str) else None
    return None


def test_text_gen_request_rejects_image_url_content_part() -> None:
    with pytest.raises(ValidationError, match="image_url"):
        ServerlessWorkloadRequest(
            workload="text-gen",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "no image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                    ],
                )
            ],
            max_tokens=8,
        )


def test_dsi_request_rejects_image_url_content_part() -> None:
    with pytest.raises(ValidationError, match="image_url"):
        ServerlessWorkloadRequest(
            workload="dsi",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        }
                    ],
                )
            ],
            max_tokens=8,
        )


def test_serverless_workload_request_rejects_unknown_workload() -> None:
    with pytest.raises(ValidationError):
        ServerlessWorkloadRequest(
            workload="bogus",  # type: ignore[arg-type]
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=8,
        )


def test_serverless_workload_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ServerlessWorkloadRequest(
            workload="text-gen",
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=8,
            surprise="nope",  # type: ignore[call-arg]
        )


def test_serverless_workload_request_optional_extra() -> None:
    """`extra` is a typed dict escape hatch for OpenAI extras like top_p."""
    req = ServerlessWorkloadRequest(
        workload="vlm",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=8,
        extra={"top_p": 0.9, "frequency_penalty": 0.1},
    )
    assert req.extra == {"top_p": 0.9, "frequency_penalty": 0.1}


# --------------------------------------------------------------------------- #
# ServerlessWorkloadResult
# --------------------------------------------------------------------------- #


def test_serverless_workload_result_round_trip() -> None:
    res = ServerlessWorkloadResult(
        workload="vlm",
        content="the image shows a cat",
        model="qwen3-vl-32b",
        input_tokens=12,
        output_tokens=4,
        finish_reason="stop",
        duration_ms=240,
    )
    assert res.workload == "vlm"
    assert res.content == "the image shows a cat"
    assert res.input_tokens == 12
    assert res.output_tokens == 4


def test_serverless_workload_result_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ServerlessWorkloadResult(
            workload="vlm",
            content="x",
            model="m",
            input_tokens=1,
            output_tokens=1,
            finish_reason="stop",
            duration_ms=1,
            extra_field="nope",  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------- #
# build_chat_payload
# --------------------------------------------------------------------------- #


def test_build_chat_payload_text_gen_shape() -> None:
    req = ServerlessWorkloadRequest(
        workload="text-gen",
        messages=[ChatMessage(role="user", content="ping")],
        max_tokens=8,
        temperature=0.0,
    )
    payload = build_chat_payload(req)
    assert payload["messages"] == [{"role": "user", "content": "ping"}]
    assert payload["max_tokens"] == 8
    assert payload["temperature"] == 0.0
    assert payload["model"] == ""  # model filled by client_for_workload / ServerlessClient


def test_build_chat_payload_vlm_keeps_multimodal_list() -> None:
    req = ServerlessWorkloadRequest(
        workload="vlm",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "what?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            )
        ],
        max_tokens=8,
    )
    payload = build_chat_payload(req)
    assert payload["messages"][0]["role"] == "user"
    assert isinstance(payload["messages"][0]["content"], list)
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert payload["messages"][0]["content"][1]["type"] == "image_url"


def test_build_chat_payload_merges_extra() -> None:
    req = ServerlessWorkloadRequest(
        workload="vlm",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=8,
        extra={"top_p": 0.9},
    )
    payload = build_chat_payload(req)
    assert payload["top_p"] == 0.9


# --------------------------------------------------------------------------- #
# client_for_workload: settings wiring
# --------------------------------------------------------------------------- #


def _make_settings() -> Any:
    """A minimal stand-in for vo_rag.core.config.settings.ServerlessSettings.

    Returns the *ServerlessSettings-shaped* object directly, because
    ``client_for_workload`` reads ``base_url``/``model_name``/``api_key``/
    ``request_timeout_s`` off the passed-in settings (per the documented
    contract: ``settings.serverless.base_url(role)`` — the ``.serverless``
    access is the caller's responsibility).
    """
    from types import SimpleNamespace

    base_url_template = "https://api.runpod.ai/v2/{endpoint_id}/openai/v1"

    def _base_url(role: str) -> str:
        endpoint_id = {
            "vlm": "vlm-ep",
            "text": "text-ep",
            "dsi": "dsi-ep",
            "exp": "exp-ep",
        }[role]
        return base_url_template.format(endpoint_id=endpoint_id)

    def _model_name(role: str) -> str:
        return {
            "vlm": "qwen3-vl-32b",
            "text": "qwen3-6-27b",
            "dsi": "qwen3-dsi-7b",
            "exp": "qwen3-exp-9b",
        }[role]

    return SimpleNamespace(
        api_key="test-runpod-key",  # pragma: allowlist secret
        vlm_endpoint_id="vlm-ep",
        vlm_model_name="qwen3-vl-32b",
        text_endpoint_id="text-ep",
        text_model_name="qwen3-6-27b",
        dsi_endpoint_id="dsi-ep",
        dsi_model_name="qwen3-dsi-7b",
        request_timeout_s=600,
        base_url=_base_url,
        model_name=_model_name,
    )


def test_client_for_workload_text_gen_uses_text_role_seam() -> None:
    settings = _make_settings()
    with patch("vo_rag.workers.runpod_vllm.contracts.ServerlessClient") as mock:
        client_for_workload("text-gen", settings=settings)
        mock.assert_called_once_with(
            base_url="https://api.runpod.ai/v2/text-ep/openai/v1",
            api_key="test-runpod-key",  # pragma: allowlist secret
            model="qwen3-6-27b",
            timeout_s=600,
        )


def test_client_for_workload_vlm_uses_vlm_role_seam() -> None:
    settings = _make_settings()
    with patch("vo_rag.workers.runpod_vllm.contracts.ServerlessClient") as mock:
        client_for_workload("vlm", settings=settings)
        mock.assert_called_once_with(
            base_url="https://api.runpod.ai/v2/vlm-ep/openai/v1",
            api_key="test-runpod-key",  # pragma: allowlist secret
            model="qwen3-vl-32b",
            timeout_s=600,
        )


def test_client_for_workload_dsi_uses_dsi_role_seam() -> None:
    settings = _make_settings()
    with patch("vo_rag.workers.runpod_vllm.contracts.ServerlessClient") as mock:
        client_for_workload("dsi", settings=settings)
        mock.assert_called_once_with(
            base_url="https://api.runpod.ai/v2/dsi-ep/openai/v1",
            api_key="test-runpod-key",  # pragma: allowlist secret
            model="qwen3-dsi-7b",
            timeout_s=600,
        )


def test_client_for_workload_returns_real_client_instance() -> None:
    """When not mocked, the real ServerlessClient is returned (no network here)."""
    from vo_rag.core.cloud.serverless import ServerlessClient

    settings = _make_settings()
    client = client_for_workload("text-gen", settings=settings)
    assert isinstance(client, ServerlessClient)
    assert client.model == "qwen3-6-27b"
    # We deliberately do NOT call client.aclose() here: it lazily creates
    # an httpx.AsyncClient whose event loop is bound to whatever loop
    # is active when the property is first read. Closing it under a
    # different loop (which is what happens when pytest-asyncio has
    # already torn one down) raises "loop is closed" warnings. The
    # process exits cleanly without a close; the AsyncClient is a
    # test-local object with no live connections.


# --------------------------------------------------------------------------- #
# No-bypass invariant: workers/runpod_vllm must not talk to api.runpod.ai
# outside vo_rag.core.cloud.serverless.
# --------------------------------------------------------------------------- #


def test_workers_runpod_vllm_has_no_raw_httpx_or_runpod_sdk() -> None:
    """No module under vo_rag.workers.runpod_vllm may import httpx directly or
    use a RunPod SDK. The single permitted transport seam is
    ``vo_rag.core.cloud.serverless.ServerlessClient``.

    Allowed exceptions:
    - The local FastAPI harness (``harness.py``) is *not* a RunPod client
      — it is a tiny ASGI app used by tests and local dev. It is allowed
      to import fastapi + uvicorn (its job is to *be* an HTTP target,
      not a RunPod client). It must not, however, make outgoing HTTP
      calls: there is no `httpx` import and no `api.runpod.ai` URL is
      actually hit at runtime. Docstring mentions of the RunPod host
      describing the seam are tolerated (the harness is a double for
      that endpoint), only live network calls would be a violation.
    """
    pkg = Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "workers" / "runpod_vllm"
    assert pkg.is_dir(), f"package dir missing: {pkg}"

    # Walk every .py file under the package.
    py_files = sorted(pkg.rglob("*.py"))
    assert py_files, "no .py files under workers/runpod_vllm yet"

    forbidden: list[str] = []
    for f in py_files:
        text = f.read_text(encoding="utf-8")
        # The harness is local infra, not a RunPod transport — it is the
        # *target* of ServerlessClient, not a peer. It is allowed to
        # import fastapi / uvicorn. It must not, however, issue any
        # outgoing HTTP to api.runpod.ai.
        if f.name == "harness.py":
            # No httpx import — the harness has nothing to send.
            if "import httpx" in text or "from httpx" in text:
                forbidden.append(f"{f}: harness imported httpx")
            continue
        # Disallow raw httpx imports anywhere in the contracts/transport surface.
        if "import httpx" in text or "from httpx" in text:
            forbidden.append(f"{f}: raw httpx import")
        if "api.runpod.ai" in text:
            forbidden.append(f"{f}: talks to api.runpod.ai directly")

    assert not forbidden, "no-bypass invariant violated:\n" + "\n".join(forbidden)
