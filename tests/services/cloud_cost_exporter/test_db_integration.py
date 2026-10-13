"""Integration test for cloud_cost_exporter with a live Postgres pool."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from services.cloud_cost_exporter.app import _refresh, create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def db_url():
    url = os.environ.get("VORAG_DATABASE_URL", os.environ.get("DATABASE_URL", ""))
    if not url:
        pytest.skip("No DB URL configured")
    return url


@pytest.fixture(scope="module")
async def pool(db_url):
    p = await asyncpg.create_pool(db_url, min_size=1, max_size=2)
    yield p
    await p.close()


@pytest.fixture(autouse=True)
async def clean_cloud_cost(pool):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM cloud_cost WHERE job_id LIKE 'test-exporter-%'")
        await conn.execute("DELETE FROM cloud_kill_log WHERE job_id LIKE 'test-exporter-%'")
    yield
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM cloud_cost WHERE job_id LIKE 'test-exporter-%'")
        await conn.execute("DELETE FROM cloud_kill_log WHERE job_id LIKE 'test-exporter-%'")


def _parse_metrics(client: TestClient):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    out: dict[str, Any] = {}
    for family in text_string_to_metric_families(resp.text):
        out[family.name] = []
        for sample in family.samples:
            out[family.name].append({"labels": sample.labels, "value": sample.value})
    return out


async def _seed(pool):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO cloud_cost (
                job_id, provider, gpu_type, workload, started_at, status, cost_usd
            )
            VALUES
                ('test-exporter-1', 'runpod', 'A100', 'text-gen', $1, 'completed', 5.0),
                ('test-exporter-2', 'runpod', 'A100', 'vlm', $1, 'completed', 3.0),
                ('test-exporter-3', 'runpod', 'A100', 'dsi', $1, 'completed', 2.0),
                ('test-exporter-4', 'runpod', 'A100', 'embedding-batch', $1, 'completed', 1.0)
            ON CONFLICT (job_id) DO NOTHING
            """,
            datetime.now(UTC),
        )
        await conn.execute(
            """
            INSERT INTO cloud_kill_log (job_id, triggered_at, reason)
            VALUES
                ('test-exporter-1', $1, 'budget')
            ON CONFLICT DO NOTHING
            """,
            datetime.now(UTC),
        )


def test_integration_spend_and_kill(pool):
    asyncio.run(_seed(pool))
    app = create_app(pool=pool)
    with TestClient(app) as client:
        asyncio.run(_refresh(app))
        metrics = _parse_metrics(client)

    spend = metrics.get("vo_rag_cloud_spend_month_usd", [])
    assert any(s["labels"]["workload"] == "text-gen" and s["value"] == 5.0 for s in spend)
    assert any(s["labels"]["workload"] == "vlm" and s["value"] == 3.0 for s in spend)
    assert any(s["labels"]["workload"] == "dsi" and s["value"] == 2.0 for s in spend)
    assert any(s["labels"]["workload"] == "embedding-batch" and s["value"] == 1.0 for s in spend)

    ks = metrics.get("vo_rag_cloud_kill_switch_triggers_7d", [])
    assert ks
    assert any(s["value"] >= 1 for s in ks)
