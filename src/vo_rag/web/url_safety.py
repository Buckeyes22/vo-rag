"""URL safety gate — blocks SSRF before scrapers / git clone touch a hostile host.

vo-rag's intake API lets attackers submit arbitrary URLs and repo URLs. Without
this gate, a submitted URL like ``http://localhost:6333/collections`` reaches
Qdrant on the Tailscale mesh (F-014); a submitted ``ssh://attacker/foo`` repo
URL lets ``git clone`` ssh-out to anywhere (F-010); a followed redirect can land
on a private IP after the first hop passed validation (F-049); DNS rebinding
can swap a public IP for a private one between validation and connect (F-048).
The shipped fetch tools never follow redirects (``follow_redirects=False``), so
a 3xx response raises instead of dialing a second address.

``validate_external_url`` is the single chokepoint: parse → IP-literal block
list → DNS-resolve → IP block list → return the resolved IP for downstream
pinning. Every scraper client and the repo-analysis cloner calls it before
any network I/O.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Blocked IP ranges per the F-014 plan. Covers loopback, RFC1918, link-local,
# CGNAT / Tailscale (100.64/10), IPv6 loopback, and unique-local IPv6.
_BLOCKED_V4_NETS = tuple(
    ipaddress.IPv4Network(cidr)
    for cidr in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "100.64.0.0/10",
        "0.0.0.0/8",
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved
    )
)
_BLOCKED_V6_NETS = tuple(
    ipaddress.IPv6Network(cidr)
    for cidr in (
        "::1/128",
        "fc00::/7",  # ULA
        "fe80::/10",  # link-local
        "ff00::/8",  # multicast
    )
)


@dataclass(frozen=True)
class ResolvedURL:
    """Parsed + DNS-resolved URL components.

    The ``ip`` field is the first A/AAAA record returned by getaddrinfo and
    is intended to be used by callers for connect-time pinning (mitigates DNS
    rebinding per F-048).
    """

    url: str
    scheme: str
    host: str
    ip: str
    port: int


class SSRFBlocked(Exception):
    """Raised when a URL fails the safety gate."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"SSRF blocked for {url!r}: {reason}")
        self.url = url
        self.reason = reason


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> tuple[bool, str]:
    if ip.is_unspecified:
        return True, "IP address is unspecified"
    if isinstance(ip, ipaddress.IPv4Address):
        for v4_net in _BLOCKED_V4_NETS:
            if ip in v4_net:
                return True, f"IPv4 in blocked network {v4_net}"
    else:
        if ip.ipv4_mapped is not None:
            blocked, reason = _ip_is_blocked(ip.ipv4_mapped)
            if blocked:
                return True, f"IPv4-mapped IPv6 {ip.ipv4_mapped} — {reason}"
        for v6_net in _BLOCKED_V6_NETS:
            if ip in v6_net:
                return True, f"IPv6 in blocked network {v6_net}"
    return False, ""


def _parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return the parsed IP if ``host`` is a literal IP, else None.

    Strips IPv6 brackets ``[::1]`` → ``::1``.
    """
    candidate = host.strip("[]")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def _resolved_addresses(
    host: str, port: int
) -> Iterable[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """All A/AAAA records returned by getaddrinfo for ``host``.

    Wrapped via ``asyncio.to_thread`` at the async entry point; this helper
    runs in a worker thread.
    """
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        addr = str(sockaddr[0])
        if addr in seen:
            continue
        seen.add(addr)
        out.append(ipaddress.ip_address(addr))
    return out


def validate_external_url_sync(url: str) -> ResolvedURL:
    """Validate ``url`` is safe to fetch from a public-internet egress point.

    Sync core of the gate (same stages, same block lists as the async entry
    point), also usable by a synchronous caller that follows redirects itself
    and re-validates every hop. In this tree only ``validate_external_url``
    calls it. Blocking: performs getaddrinfo.

    Raises:
        SSRFBlocked: if the URL fails any of the gates (scheme, literal IP,
            DNS resolution, resolved-IP block list).

    Returns:
        ResolvedURL with the first resolved IP for connect-time pinning.
    """
    if not url or not isinstance(url, str):
        raise SSRFBlocked(str(url), "url must be a non-empty string")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise SSRFBlocked(url, f"scheme {scheme!r} not in {sorted(_ALLOWED_SCHEMES)}")

    host = parsed.hostname or ""
    if not host:
        raise SSRFBlocked(url, "hostname is empty")

    port = parsed.port or (443 if scheme == "https" else 80)

    # Stage 1 — literal IP rejection (skips DNS for attacker who already knows
    # the internal IP).
    literal = _parse_ip_literal(host)
    if literal is not None:
        blocked, reason = _ip_is_blocked(literal)
        if blocked:
            raise SSRFBlocked(url, reason)
        return ResolvedURL(url=url, scheme=scheme, host=host, ip=str(literal), port=port)

    # Stage 2 — DNS resolution.
    try:
        addresses = list(_resolved_addresses(host, port))
    except socket.gaierror as exc:
        raise SSRFBlocked(url, f"DNS resolution failed: {exc}") from exc

    if not addresses:
        raise SSRFBlocked(url, "DNS returned no addresses")

    # Any blocked address → reject. (Strict: a hostname that resolves to BOTH
    # a public and a private IP is rejected, because either can be picked at
    # connect time by the underlying client.)
    for addr in addresses:
        blocked, reason = _ip_is_blocked(addr)
        if blocked:
            raise SSRFBlocked(url, f"host {host!r} resolves to {addr} — {reason}")

    return ResolvedURL(
        url=url,
        scheme=scheme,
        host=host,
        ip=str(addresses[0]),
        port=port,
    )


async def validate_external_url(url: str) -> ResolvedURL:
    """Async twin of ``validate_external_url_sync`` — the same gate run with
    getaddrinfo off the event-loop thread. Behavior is identical by
    construction: this is a thin ``asyncio.to_thread`` delegate.
    """
    return await asyncio.to_thread(validate_external_url_sync, url)


__all__ = [
    "validate_external_url",
    "validate_external_url_sync",
    "ResolvedURL",
    "SSRFBlocked",
]
