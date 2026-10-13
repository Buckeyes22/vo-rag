"""Calls the RunPod farm makes through the external ``pitwall`` GPU-broker CLI.

Every call goes through an injected ``Runner`` (``coordinator.subprocess_runner``
in production), always with ``--json``. Mutations (pod create / terminate)
pass an idempotency key and the exact ``--confirm`` value pitwall requires.
Nothing here reads, prints or stores a credential: pod SSH details come
from ``pitwall runpod pods get`` (address and port only).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from vo_rag.ocr_farm.coordinator import Runner

Cloud = Literal["secure", "community"]

#: Seconds allowed for one pitwall call (a create can take a while).
PITWALL_TIMEOUT_S = 180.0


class PitwallError(RuntimeError):
    """A pitwall call failed or returned something unusable. ``code`` is
    pitwall's error code when it gave one (``resource_not_found`` ...)."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PodInfo:
    pod_id: str
    name: str
    status: str
    ssh_host: str | None
    ssh_port: int | None

    @property
    def ssh_ready(self) -> bool:
        return bool(self.ssh_host and self.ssh_port)


@dataclass(frozen=True)
class Budget:
    """``pitwall budget show``: what is left this month and the per-request cap
    (a pod create reserves about TTL x rate against both)."""

    remaining_usd: Decimal
    per_request_max_usd: Decimal


class Pitwall:
    def __init__(self, runner: Runner, binary: str = "pitwall") -> None:
        self._runner = runner
        self._binary = binary

    async def _json(self, *args: str) -> Any:
        argv = [self._binary, *args, "--json"]
        what = " ".join(args[:3])
        try:
            result = await asyncio.wait_for(self._runner(argv), PITWALL_TIMEOUT_S)
        except TimeoutError as exc:
            raise PitwallError(f"pitwall {what}: timed out", "timeout") from exc
        except OSError as exc:
            # A missing or unexecutable binary (FileNotFoundError,
            # PermissionError): pitwall is unreachable, and callers only
            # catch PitwallError.
            raise PitwallError(
                f"pitwall {what}: cannot run {self._binary!r}: {exc}", "unreachable"
            ) from exc
        try:
            body = json.loads(result.stdout) if result.stdout.strip() else None
        except json.JSONDecodeError:
            body = None
        if result.returncode != 0 or body is None or (isinstance(body, dict) and "error" in body):
            code = str(body["error"]) if isinstance(body, dict) and "error" in body else None
            detail = body.get("detail") or body.get("error") if isinstance(body, dict) else None
            tail = (result.stderr.strip().splitlines() or [""])[-1][:300]
            raise PitwallError(
                f"pitwall {what}: rc={result.returncode} {detail or tail}".strip(), code
            )
        return body

    async def budget(self) -> Budget:
        body = await self._json("budget", "show")
        return Budget(
            _decimal(body, "budget_remaining_usd", "pitwall budget show"),
            _decimal(body, "per_request_max_usd", "pitwall budget show"),
        )

    async def budget_remaining_usd(self) -> Decimal:
        return (await self.budget()).remaining_usd

    async def gpu_price_usd_per_hour(self, gpu_type: str, cloud: Cloud) -> Decimal:
        """On-demand price of one ``gpu_type`` GPU for an hour, from the catalogue."""
        body = await self._json("runpod", "catalogue")
        for gpu in body.get("gpus") or []:
            if gpu.get("gpu_type_id") != gpu_type:
                continue
            prices = ((gpu.get("graphql") or {}).get("on_demand_price_usd_per_gpu_hour")) or {}
            if prices.get(cloud) is None:
                break
            return _decimal(prices, cloud, f"catalogue {gpu_type} {cloud} price")
        raise PitwallError(
            f"pitwall catalogue has no {cloud} on-demand price for {gpu_type!r} "
            f"(state={body.get('state')!r}); pass --price-per-gpu-hour"
        )

    async def create_pod(self, request: dict[str, Any], idempotency_key: str) -> str:
        body = await self._json(
            "runpod",
            "pods",
            "create",
            "--request-json",
            json.dumps(request, separators=(",", ":")),
            "--idempotency-key",
            idempotency_key,
            "--confirm",
            str(request["name"]),
        )
        pod_id = body.get("resource_id") if isinstance(body, dict) else None
        if not pod_id:
            raise PitwallError("pitwall pods create: no resource_id in the reply")
        return str(pod_id)

    async def find_pod_by_name(self, name: str) -> str | None:
        """The id of the account pod called ``name``, or ``None``. pitwall
        refuses a create whose name already exists and does not replay an
        idempotency key, so this is how a create that timed out client-side
        but succeeded server-side is found."""
        body = await self._json("runpod", "pods", "list")
        for pod in (body.get("pods") if isinstance(body, dict) else None) or []:
            if isinstance(pod, dict) and pod.get("name") == name and pod.get("id"):
                return str(pod["id"])
        return None

    async def get_pod(self, pod_id: str) -> PodInfo:
        body = await self._json("runpod", "pods", "get", pod_id)
        port = (body.get("port_mappings") or {}).get("22")
        return PodInfo(
            pod_id=str(body.get("id") or pod_id),
            name=str(body.get("name") or pod_id),
            status=str(body.get("status") or "unknown"),
            ssh_host=body.get("public_ip") or None,
            ssh_port=int(port) if port else None,
        )

    async def terminate_pod(self, pod_id: str, idempotency_key: str) -> None:
        await self._json(
            "runpod",
            "pods",
            "terminate",
            pod_id,
            "--idempotency-key",
            idempotency_key,
            "--confirm",
            pod_id,
        )


def _decimal(body: Any, key: str, what: str) -> Decimal:
    raw = body.get(key) if isinstance(body, dict) else None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise PitwallError(f"{what}: {key}={raw!r} is not a number") from exc
