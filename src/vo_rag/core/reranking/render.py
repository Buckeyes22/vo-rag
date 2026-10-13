"""Candidate→text rendering for cross-encoder reranking (VOR-8).

Extracted from ``gte.py`` so the remote ``RerankerClient`` (and any caller on
the client backend) can render candidates WITHOUT importing ``gte`` — which
pulls ``onnxruntime``. Keeping this dependency-free makes the API process on the
``client`` reranker backend genuinely torch/onnx-free.
"""

from __future__ import annotations

from vo_rag.core.retrieval.search import SearchResult


def _render_candidate(candidate: SearchResult) -> str:
    """Render a candidate's payload into a single text string for the cross-encoder.

    Brain passages rank their local section; complete governing context stays in
    the original payload for evidence delivery. Other corpora retain the existing
    precedence:
      1. ``section_title`` (header) is prepended when present.
      2. Body text prefers ``text``, then ``content``, then ``content_text``.
      3. Empty payloads yield an empty string.
    """
    payload = candidate.payload
    local = payload.get("source_section_text")
    if candidate.collection == "vo_rag_brain_v1" and isinstance(local, str) and local.strip():
        headings = [payload.get("title"), payload.get("heading")]
        labels = list(
            dict.fromkeys(label for label in headings if isinstance(label, str) and label)
        )
        return "\n".join([*labels, local]).strip()
    title = payload.get("section_title") or ""
    text = payload.get("text") or payload.get("content") or payload.get("content_text") or ""
    return (title + "\n" + text).strip() if title else text


__all__ = ["_render_candidate"]
