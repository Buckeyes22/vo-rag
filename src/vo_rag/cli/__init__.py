"""vo-rag command-line surface (Click).

Public entry point: :data:`cli` (a :class:`click.Group`).
Sub-commands are registered by :func:`register_query_commands`,
:func:`register_ingest_commands`, :func:`register_eval_commands`,
and :func:`register_admin_commands` and grouped under the
``vo-rag`` console script. The CLI is a thin HTTP client over
:mod:`vo_rag.cli.client.VoRagApiClient` — every command posts a
request to the API and renders the JSON response.

Command roster:

- ``vo-rag version`` — print the package version.
- ``vo-rag pipeline {list,status}`` — derive the pipeline roster
  from the SOT and report status counts.
- ``vo-rag config show`` — display the resolved CLI config with
  the API key masked.
- ``vo-rag query`` — POST /api/v1/query.
- ``vo-rag answer`` — POST /api/v1/answer (preserves
  ``context_scope=sterile`` by default — F-045).
- ``vo-rag ingest {text,url,idea,file-drop,file}`` — POST
  /api/v1/intake with a payload the API normalizes. The URL form
  runs the SSRF gate (F-014) locally before any HTTP call.
- ``vo-rag eval-gate`` — runs :func:`vo_rag.core.eval.gate.run_eval_gate`
  against the SOT roster; exits non-zero on regression.
- ``vo-rag admin status`` — GET /api/v1/admin/status.
- ``vo-rag health`` — GET /health (no auth).
- ``vo-rag status`` — GET /api/v1/admin/status (watchable).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from vo_rag.cli.client import VoRagApiClient, VoRagApiError


def __getattr__(name: str) -> Any:
    if name in {"cli", "main"}:
        main_module = import_module("vo_rag.cli.main")
        return getattr(main_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VoRagApiClient",
    "VoRagApiError",
    "cli",
    "main",
]
