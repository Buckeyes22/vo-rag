"""The pitwall CLI wrapper, against canned replies only (VOR-1434)."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from tests.ocr_farm._runpod_fakes import FakeRunner, pitwall_replies
from vo_rag.ocr_farm.coordinator import CommandResult
from vo_rag.ocr_farm.runpod.pitwall import Pitwall, PitwallError


async def test_budget_and_price_parse_as_decimals() -> None:
    runner = FakeRunner(pitwall_replies(remaining="12.34", price="0.44"))
    pitwall = Pitwall(runner)

    assert await pitwall.budget_remaining_usd() == Decimal("12.34")
    assert await pitwall.gpu_price_usd_per_hour("NVIDIA A40", "secure") == Decimal("0.44")
    assert runner.calls[0] == ["pitwall", "budget", "show", "--json"]
    assert runner.calls[1] == ["pitwall", "runpod", "catalogue", "--json"]


async def test_missing_catalogue_price_says_to_pass_one() -> None:
    pitwall = Pitwall(FakeRunner(pitwall_replies(price=None)))

    with pytest.raises(PitwallError, match="--price-per-gpu-hour"):
        await pitwall.gpu_price_usd_per_hour("NVIDIA A40", "secure")
    with pytest.raises(PitwallError, match="community"):
        await pitwall.gpu_price_usd_per_hour("NVIDIA A40", "community")


async def test_error_bodies_and_exit_codes_raise() -> None:
    pitwall = Pitwall(FakeRunner(pitwall_replies(fail={"budget": 1})))
    with pytest.raises(PitwallError, match="budget"):
        await pitwall.budget_remaining_usd()

    def not_json(argv: list[str]) -> CommandResult:
        return CommandResult(0, "Traceback ...", "")

    with pytest.raises(PitwallError):
        await Pitwall(FakeRunner(not_json)).budget_remaining_usd()


async def test_pod_get_reads_only_the_address_and_ssh_port() -> None:
    pods = {"p1": {"id": "p1", "name": "n", "status": "running", "public_ip": "203.0.113.9",
                   "port_mappings": {"22": 41022, "8888": 1}}}  # fmt: skip
    info = await Pitwall(FakeRunner(pitwall_replies(pods=pods))).get_pod("p1")

    assert (info.ssh_host, info.ssh_port, info.ssh_ready) == ("203.0.113.9", 41022, True)
    pending: dict[str, dict[str, Any]] = {
        "p2": {"id": "p2", "status": "created", "public_ip": None, "port_mappings": {}}
    }
    assert not (await Pitwall(FakeRunner(pitwall_replies(pods=pending))).get_pod("p2")).ssh_ready


async def test_mutations_carry_idempotency_key_and_exact_confirm() -> None:
    runner = FakeRunner(pitwall_replies())
    pitwall = Pitwall(runner)

    pod_id = await pitwall.create_pod(
        {"name": "vorag-ocr-j-0", "ttl_minutes": 60}, "vorag-ocr-j-0-c"
    )
    await pitwall.terminate_pod(pod_id, "vorag-ocr-j-0-t")

    create, terminate = runner.calls
    assert json.loads(create[create.index("--request-json") + 1])["ttl_minutes"] == 60
    assert create[create.index("--confirm") + 1] == "vorag-ocr-j-0"
    assert terminate[:5] == ["pitwall", "runpod", "pods", "terminate", pod_id]
    assert terminate[terminate.index("--confirm") + 1] == pod_id
    assert "--dry-run" not in create


async def test_a_missing_pitwall_binary_is_a_pitwall_error() -> None:
    """The runner's FileNotFoundError/OSError becomes a
    PitwallError, so the CLI reports it instead of a raw traceback."""
    from vo_rag.ocr_farm.coordinator import subprocess_runner
    from vo_rag.ocr_farm.runpod.pitwall import PitwallError

    pitwall = Pitwall(subprocess_runner, "/nonexistent/pitwall-binary")
    with pytest.raises(PitwallError, match="cannot run"):
        await pitwall.budget()


async def test_find_pod_by_name_reads_the_pod_list() -> None:
    from tests.ocr_farm._runpod_fakes import FakeRunner, pitwall_replies

    runner = FakeRunner(
        pitwall_replies(listed=[{"id": "podA", "name": "other"}, {"id": "podB", "name": "want"}])
    )
    pitwall = Pitwall(runner)
    assert await pitwall.find_pod_by_name("want") == "podB"
    assert await pitwall.find_pod_by_name("missing") is None
    assert runner.matching("pods", "list")
