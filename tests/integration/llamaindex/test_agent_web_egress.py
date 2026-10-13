"""DNS-rebinding regression tests for the web_search egress path."""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.web.url_safety import ResolvedURL


class _FakeClient:
    prompt_registry = None

    async def complete(self, **kwargs: Any) -> LLMResponse:
        raise AssertionError("bare complete must not receive fetched web text")


class _PinnedTransportSpy:
    def __init__(self, *, pins: dict[tuple[str, int], str]) -> None:
        self.pins = pins


class _PinnedResponse:
    text = "pinned response"

    def __init__(self) -> None:
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True


class _AsyncClientSpy:
    instances: list[_AsyncClientSpy] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.response = _PinnedResponse()
        self.get_calls: list[str] = []
        self.exited = False
        self.__class__.instances.append(self)

    async def __aenter__(self) -> _AsyncClientSpy:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited = True

    async def get(self, url: str) -> _PinnedResponse:
        self.get_calls.append(url)
        return self.response


def _resolved(url: str, *, host: str = "example.com") -> ResolvedURL:
    return ResolvedURL(url=url, scheme="https", host=host, ip="93.184.216.34", port=443)


@pytest.mark.parametrize(
    ("url", "expected_port"),
    [
        ("https://example.com/article", 443),
        ("https://example.com:8443/article", 8443),
    ],
)
@pytest.mark.asyncio
async def test_fetch_url_text_pins_resolved_ip_with_effective_port(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    expected_port: int,
) -> None:
    import vo_rag.integrations.llamaindex.agents.tools.web_search as web_search

    transport_spies: list[_PinnedTransportSpy] = []

    class PinnedTransportSpy(_PinnedTransportSpy):
        def __init__(self, *, pins: dict[tuple[str, int], str]) -> None:
            super().__init__(pins=pins)
            transport_spies.append(self)

    _AsyncClientSpy.instances.clear()
    monkeypatch.setattr(web_search, "PinnedAsyncHTTPTransport", PinnedTransportSpy)
    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClientSpy)

    text = await web_search._fetch_url_text(
        url,
        resolved=_resolved(url),
        http_client=None,
        timeout_s=4.0,
    )

    assert text == "pinned response"
    assert [transport.pins for transport in transport_spies] == [
        {("example.com", expected_port): "93.184.216.34"}
    ]
    assert len(_AsyncClientSpy.instances) == 1
    client = _AsyncClientSpy.instances[0]
    assert client.kwargs["transport"] is transport_spies[0]
    assert client.kwargs["follow_redirects"] is False
    assert client.get_calls == [url]
    assert client.response.raise_for_status_called is True
    assert client.exited is True


@pytest.mark.asyncio
async def test_run_web_search_fetches_original_url_with_validated_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.integrations.llamaindex.agents.tools.web_search as web_search

    target_url = "https://example.com/article"
    resolved = _resolved(target_url)
    fetch_calls: list[dict[str, object]] = []

    async def fake_validate(url: str) -> ResolvedURL:
        assert url == target_url
        return resolved

    async def fake_fetch_url_text(
        url: str,
        *,
        resolved: ResolvedURL,
        http_client: httpx.AsyncClient | None,
        timeout_s: float,
    ) -> str:
        fetch_calls.append(
            {
                "url": url,
                "resolved": resolved,
                "http_client": http_client,
                "timeout_s": timeout_s,
            }
        )
        return "safe fetched page"

    async def protected_spy(
        client: object,
        *,
        schema: type[Any],
        untrusted_content: str,
        pipeline: str,
        **complete_kwargs: object,
    ) -> tuple[Any, LLMResponse, ValidationResult, bool]:
        assert untrusted_content == "safe fetched page"
        parsed = schema(
            title="Pinned fetch",
            summary="safe structured summary",
            key_points=["safe point"],
            injection_suspected=False,
            injection_reason=None,
        )
        return parsed, LLMResponse(text="", model="fake"), ValidationResult(ok=True), False

    monkeypatch.setattr(web_search, "validate_external_url", fake_validate)
    monkeypatch.setattr(web_search, "_fetch_url_text", fake_fetch_url_text)
    monkeypatch.setattr(web_search, "complete_json_protected", protected_spy)

    result = await web_search.run_web_search(
        url=target_url,
        client=cast(LLMClient, _FakeClient()),
        allowed_domains=["example.com"],
        timeout_s=2.5,
    )

    assert result.source_url == target_url
    assert result.summary == "safe structured summary"
    assert fetch_calls == [
        {
            "url": target_url,
            "resolved": resolved,
            "http_client": None,
            "timeout_s": 2.5,
        }
    ]


@pytest.mark.asyncio
async def test_run_web_search_does_not_follow_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.integrations.llamaindex.agents.tools.web_search as web_search

    target_url = "https://example.com/start"
    requests: list[str] = []

    async def fake_validate(url: str) -> ResolvedURL:
        assert url == target_url
        return _resolved(url)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if str(request.url) == target_url:
            return httpx.Response(
                status_code=302,
                headers={"Location": "https://example.com/final"},
            )
        return httpx.Response(status_code=200, text="followed")

    monkeypatch.setattr(web_search, "validate_external_url", fake_validate)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as http_client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await web_search.run_web_search(
                url=target_url,
                client=cast(LLMClient, _FakeClient()),
                allowed_domains=["example.com"],
                http_client=http_client,
            )

    assert exc_info.value.response.status_code == 302
    assert requests == [target_url]
