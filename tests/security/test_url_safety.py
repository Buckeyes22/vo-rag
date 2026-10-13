"""F-014 / F-048 / F-049: SSRF gate (url_safety) — single chokepoint for all
external URL validation. Blocks loopback, RFC1918, link-local, CGNAT/Tailscale
(100.64/10), IPv6 loopback/ULA/multicast, and disallowed schemes. DNS
resolve-then-pin: any blocked resolved address rejects the URL even when other
resolved addresses are public (strict mixed-address rejection).
"""

from __future__ import annotations

import pytest

from vo_rag.web.url_safety import ResolvedURL, SSRFBlocked, validate_external_url

pytestmark = pytest.mark.security


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        "http://10.0.0.5",
        "http://10.255.255.255",
        "http://172.16.0.1",
        "http://172.31.255.255",
        "http://192.168.1.1",
        "http://192.168.0.0",
        "http://169.254.169.254",  # AWS / GCP / Azure metadata
        "http://127.0.0.1",  # CGNAT / Tailscale
        "http://127.0.0.1",
        "http://[::]",
        "http://[::1]",
        "http://[::ffff:127.0.0.1]",
        "http://[::ffff:10.0.0.1]",
        "http://[::ffff:169.254.169.254]",
        "http://[fc00::1]",  # ULA
        "http://[fe80::1]",  # link-local
        "ftp://example.com",  # non-allowlisted scheme
        "ssh://attacker.example.com/foo",
        "file:///etc/passwd",
        "javascript:alert(1)",
    ],
)
@pytest.mark.asyncio
async def test_blocked_url_is_rejected(blocked_url: str) -> None:
    with pytest.raises(SSRFBlocked):
        await validate_external_url(blocked_url)


@pytest.mark.asyncio
async def test_empty_url_is_rejected() -> None:
    with pytest.raises(SSRFBlocked):
        await validate_external_url("")


@pytest.mark.asyncio
async def test_url_without_host_is_rejected() -> None:
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://")


@pytest.mark.asyncio
async def test_dns_resolution_failure_is_rejected() -> None:
    with pytest.raises(SSRFBlocked) as excinfo:
        await validate_external_url("http://this-host-definitely-does-not-exist.invalid")
    assert "DNS" in str(excinfo.value) or "resolution" in str(excinfo.value).lower()


def test_ssmf_blocked_carries_url_and_reason() -> None:
    exc = SSRFBlocked("http://127.0.0.1", "test reason")
    assert exc.url == "http://127.0.0.1"
    assert exc.reason == "test reason"
    assert "127.0.0.1" in str(exc)
    assert "test reason" in str(exc)


@pytest.mark.asyncio
async def test_valid_http_url_resolves_to_pinned_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid public hostname: getaddrinfo runs and ResolvedURL is returned.

    We don't mock getaddrinfo (risky in a test) — we just assert the structural
    shape: a public resolver-only call would do real DNS and that depends on
    the network. The intent is to verify the chokepoint returns a ResolvedURL
    with the expected fields. The path to network has its own live tests.
    """
    # Patch getaddrinfo to a deterministic public answer to keep this hermetic.
    monkeypatch.setattr(
        "vo_rag.web.url_safety.socket.getaddrinfo",
        lambda host, port, proto: [
            (None, None, None, None, ("93.184.216.34", port or 80)),
        ],
    )

    result = await validate_external_url("http://example.com")
    assert isinstance(result, ResolvedURL)
    assert result.scheme == "http"
    assert result.host == "example.com"
    assert result.port == 80
    assert result.ip == "93.184.216.34"
    assert result.url == "http://example.com"


@pytest.mark.asyncio
async def test_https_uses_443_when_no_explicit_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vo_rag.web.url_safety.socket.getaddrinfo",
        lambda host, port, proto: [
            (None, None, None, None, ("93.184.216.34", port or 443)),
        ],
    )
    result = await validate_external_url("https://example.com")
    assert result.scheme == "https"
    assert result.port == 443


@pytest.mark.asyncio
async def test_dns_resolved_blocked_address_rejects_even_if_others_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mixed-address rejection: any blocked resolved IP → reject."""
    monkeypatch.setattr(
        "vo_rag.web.url_safety.socket.getaddrinfo",
        lambda host, port, proto: [
            (None, None, None, None, ("8.8.8.8", port or 80)),
            (None, None, None, None, ("127.0.0.1", port or 80)),
        ],
    )
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://attacker.example.com")


@pytest.mark.asyncio
async def test_dns_resolved_ipv4_mapped_ipv6_private_address_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv4-mapped IPv6 answers must inherit the IPv4 block-list."""
    monkeypatch.setattr(
        "vo_rag.web.url_safety.socket.getaddrinfo",
        lambda host, port, proto: [
            (None, None, None, None, ("::ffff:127.0.0.1", port or 80)),
        ],
    )
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://attacker.example.com")


@pytest.mark.asyncio
async def test_dns_resolved_ipv6_unspecified_address_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS answers resolving to :: must be blocked like 0.0.0.0."""
    monkeypatch.setattr(
        "vo_rag.web.url_safety.socket.getaddrinfo",
        lambda host, port, proto: [
            (None, None, None, None, ("::", port or 80)),
        ],
    )
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://attacker.example.com")


@pytest.mark.asyncio
async def test_literal_private_ip_is_rejected_without_dns() -> None:
    """Literal private IP should be rejected even if DNS would also have rejected it."""
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://10.0.0.5")


@pytest.mark.asyncio
async def test_literal_public_ip_is_accepted() -> None:
    result = await validate_external_url("http://8.8.8.8")
    assert result.ip == "8.8.8.8"
    assert result.host == "8.8.8.8"


@pytest.mark.asyncio
async def test_literal_loopback_ipv6_is_rejected() -> None:
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://[::1]")


@pytest.mark.asyncio
async def test_cgnat_tailscale_range_rejected() -> None:
    # 127.0.0.1/10 covers Tailscale's 127.0.0.1 + 100.64-100.127 range.
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://127.0.0.1")
