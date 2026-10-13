"""BGE-M3 embedding worker package.

Exposes the FastAPI ``app`` and the model wrapper so callers can::

    from vo_rag.workers.embedding import server, BgeM3Worker
    from vo_rag.workers.embedding.server import EmbedRequest, EmbedResponse

The HTTP entrypoint is the canonical contract — see
``vo_rag.core.embedding.schemas`` for the wire format and
``vo_rag.core.embedding.client`` for the async client.
"""

from __future__ import annotations

from .model import BgeM3Worker

__all__ = ["BgeM3Worker"]
