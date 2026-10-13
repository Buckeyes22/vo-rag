.PHONY: test lint typecheck eval-demo

test:
	uv run --extra dev --extra llamaindex pytest -q -m "not integration and not slow and not live"

lint:
	uv run --extra dev ruff check src tests

typecheck:
	uv run --extra dev --extra llamaindex mypy

eval-demo:
	VO_RAG_EVAL_MODE=replay_smoke uv run python scripts/run_eval_gate.py
