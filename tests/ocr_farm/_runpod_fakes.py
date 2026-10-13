"""Fakes for the RunPod OCR farm tests (not a test module: no ``test_`` prefix).

Nothing here creates, updates or terminates a real pod, or spends money:
``FakeRunner`` answers pitwall / ssh / scp argv from canned replies.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]

from tests.ocr_farm._farm_fakes import FakeGarage
from vo_rag.ocr_farm.coordinator import CommandResult, FarmItem


def pdf_bytes(pages: int) -> bytes:
    with fitz.open() as doc:
        for n in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {n + 1}")
        data: bytes = doc.tobytes()
    return data


def pdf_item(pages: int, name: str = "a", fmt: str = "pdf") -> tuple[FarmItem, bytes]:
    data = pdf_bytes(pages) if fmt == "pdf" else f"epub-{name}".encode()
    sha = hashlib.sha256(data).hexdigest()
    item = FarmItem(f"ENV{name}", f"uploads/{name}.{fmt}", sha, f"Title {name}", pages, fmt)
    return item, data


class FakeCache(FakeGarage):
    """FakeGarage plus ``upload_file``, recording the upload order."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        super().__init__(objects)
        self.uploads: list[str] = []

    async def upload_file(self, bucket: str, key: str, path: Path) -> None:
        self.uploads.append(key)
        self.objects[(bucket, key)] = Path(path).read_bytes()


Handler = Callable[[list[str]], CommandResult | None]


class FakeRunner:
    """Records argv; the first handler that returns a result answers it."""

    def __init__(self, *handlers: Handler) -> None:
        self.calls: list[list[str]] = []
        self.handlers = list(handlers)

    async def __call__(self, argv: Sequence[str], /) -> CommandResult:
        call = list(argv)
        self.calls.append(call)
        for handler in self.handlers:
            result = handler(call)
            if result is not None:
                return result
        return CommandResult(0, "", "")

    def matching(self, *needles: str) -> list[list[str]]:
        """Calls that contain every needle as a whole argv element."""
        return [c for c in self.calls if all(n in c for n in needles)]


def pitwall_replies(
    *,
    remaining: str = "20.00",
    per_request: str = "10.00",
    price: str | None = "0.40",
    gpu: str = "NVIDIA A40",
    pods: dict[str, dict[str, Any]] | None = None,
    fail: dict[str, int] | None = None,
    gone: set[str] | None = None,
    get_errors: set[str] | None = None,
    listed: list[dict[str, Any]] | None = None,
) -> Handler:
    """Canned ``pitwall ... --json`` replies. ``pods`` maps pod id -> get body;
    ``listed`` is what ``runpod pods list`` returns (default: nothing)."""
    created: list[int] = []
    fail = fail or {}

    def handle(argv: list[str]) -> CommandResult | None:
        if not argv or argv[0] != "pitwall":
            return None
        verb = " ".join(argv[1:4])
        for needle, code in fail.items():
            if needle in verb:
                return CommandResult(code, json.dumps({"error": "boom", "detail": needle}), "")
        if argv[1:3] == ["budget", "show"]:
            return _ok({"budget_remaining_usd": remaining, "mtd_spend_usd": "1.00",
                        "per_request_max_usd": per_request})  # fmt: skip
        if argv[1:3] == ["runpod", "catalogue"]:
            prices = {"secure": price, "community": None}
            return _ok(
                {"state": "fresh", "gpus": [{"gpu_type_id": gpu, "graphql": {
                    "on_demand_price_usd_per_gpu_hour": prices}}]}
            )  # fmt: skip
        if verb == "runpod pods create":
            created.append(1)
            return _ok({"resource_id": f"pod{len(created) - 1}", "changed": True})
        if verb == "runpod pods get":
            pod_id = argv[4]
            if pod_id in (get_errors or set()):
                body = {"error": "provider_unavailable", "detail": "RunPod API timed out"}
                return CommandResult(1, json.dumps(body), "")
            if pod_id in (gone or set()):
                body = {"error": "resource_not_found", "detail": "RunPod pod was not found"}
                return CommandResult(1, json.dumps(body), "")
            body = (pods or {}).get(pod_id) or {
                "id": pod_id,
                "name": pod_id,
                "status": "running",
                "public_ip": "203.0.113.7",
                "port_mappings": {"22": 40000 + int(pod_id[-1])},
            }
            return _ok(body)
        if verb == "runpod pods list":
            return _ok({"pods": list(listed or [])})
        if verb.startswith("runpod pods terminate"):
            return _ok({"resource_id": argv[4], "changed": True})
        return None

    return handle


def _ok(body: Any) -> CommandResult:
    return CommandResult(0, json.dumps(body), "")
