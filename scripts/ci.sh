#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== locked bootstrap (dev + llamaindex) =="
uv sync --locked --extra dev --extra llamaindex
echo "== ruff =="
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
echo "== mypy (strict) =="
uv run --no-sync mypy
echo "== pytest (+coverage) =="
uv run --no-sync pytest -q -m "not integration and not slow and not live" --cov=src/vo_rag --cov-report=term-missing
echo "== bandit (SAST) =="
uv run --no-sync bandit -r src -c pyproject.toml
echo "== detect-secrets =="
git ls-files -z | xargs -0 uv run --no-sync detect-secrets-hook --baseline .secrets.baseline
echo "== docs =="
uv run --no-sync python scripts/check_docs.py
uv run --no-sync python scripts/gen_reference_docs.py --check
# The default gate is hermetic replay. Source publication/repair changes also
# require the isolated infrastructure lane and per-source production receipts.
# Semantic retrieval evaluation remains separately runnable against an explicitly
# selected live snapshot; replay success does not establish semantic accuracy.
echo "== eval gate =="
VO_RAG_EVAL_MODE="${VO_RAG_EVAL_MODE:-replay_smoke}" uv run --no-sync python scripts/run_eval_gate.py
echo "ALL GREEN"
