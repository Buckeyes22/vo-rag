"""HTTP route modules for the vo-rag API surface.

Each file in this package owns one FastAPI ``APIRouter`` and a small set
of related routes; ``api/main.py`` includes whichever routers are
present: system/readiness/admin, ingest, and query/answer.
"""

__all__ = ["admin", "answer", "ingest", "query", "system"]
