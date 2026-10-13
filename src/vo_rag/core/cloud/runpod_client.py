"""Minimal RunPod REST client for kill-switch pod termination.

The full pod lifecycle client (create/list/find/delete) is superseded by the
static workload→location router. Only termination helpers remain for the
emergency kill switch.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

log = logging.getLogger("vo_rag.cloud.runpod_client")


class RunPodError(RuntimeError):
    """Any RunPod API failure."""


class RunPodRestError(RunPodError):
    """RunPod REST API returned a non-2xx response."""

    def __init__(self, method: str, path: str, status_code: int, body: str) -> None:
        super().__init__(f"{method} {path} failed with HTTP {status_code}: {body}")
        self.method = method
        self.path = path
        self.status_code = status_code
        self.body = body


def _require_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunPodError("RUNPOD_API_KEY not set in process env")
    return key


def _rest_base_url() -> str:
    return os.environ.get("RUNPOD_REST_API_URL", "https://rest.runpod.io/v1").rstrip("/")


def _rest_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_require_api_key()}",
        "Content-Type": "application/json",
    }


def _rest_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> Any:
    url = f"{_rest_base_url()}/{path.lstrip('/')}"
    with httpx.Client(timeout=timeout_s) as client:
        response = client.request(
            method,
            url,
            headers=_rest_headers(),
            json=json_body,
            params=params,
        )
    if response.status_code == 204:
        return {}
    if response.status_code >= 400:
        raise RunPodRestError(method, path, response.status_code, response.text)
    if not response.content:
        return {}
    return response.json()


def _normalize_pod(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten the REST response into a stable shape."""
    return {
        "id": raw.get("id"),
        "name": raw.get("name") or raw.get("pod_name"),
        "runtime": raw.get("runtime"),
        "ports": raw.get("ports"),
    }


def get_pods_sync() -> list[dict[str, Any]]:
    """Return all pods owned by the account's RunPod user."""
    result = _rest_request(
        "GET",
        "pods",
        params={"includeMachine": "true", "includeNetworkVolume": "true"},
    )
    if isinstance(result, list):
        return [_normalize_pod(p) for p in result if isinstance(p, dict)]
    if isinstance(result, dict):
        return [_normalize_pod(p) for p in result.values() if isinstance(p, dict)]
    log.warning("get_pods returned unexpected type %s", type(result).__name__)
    return []


async def get_pods() -> list[dict[str, Any]]:
    """Async wrapper for get_pods_sync."""
    return await asyncio.to_thread(get_pods_sync)


def terminate_pod_sync(pod_id: str) -> None:
    """Irrevocably destroy a pod. Idempotent: silent on already-terminated."""
    try:
        _rest_request("DELETE", f"pods/{pod_id}")
    except RunPodRestError as exc:
        if exc.status_code == 404:
            log.info("pod %s already terminated or never existed", pod_id)
            return
        raise RunPodError(f"terminate_pod({pod_id}) failed: {exc}") from exc
    except Exception as exc:
        raise RunPodError(f"terminate_pod({pod_id}) failed: {exc}") from exc


async def terminate_pod(pod_id: str) -> None:
    """Async wrapper for terminate_pod_sync."""
    await asyncio.to_thread(terminate_pod_sync, pod_id)


async def terminate_all_with_tag(name_prefix: str = "vorag-") -> int:
    """Kill-switch helper: terminate every pod whose name starts with `name_prefix`.

    Returns the count terminated.
    """
    pods = await get_pods()
    killed = 0
    for p in pods:
        name = p.get("name") or ""
        pod_id = p.get("id")
        if not pod_id or not name.startswith(name_prefix):
            continue
        log.warning("kill switch: terminating pod %s (%s)", pod_id, name)
        await terminate_pod(pod_id)
        killed += 1
    return killed


__all__ = [
    "RunPodError",
    "RunPodRestError",
    "get_pods",
    "get_pods_sync",
    "terminate_pod",
    "terminate_pod_sync",
    "terminate_all_with_tag",
]
