"""contextual: Anthropic-style contextual retrieval — not built.

This package is empty on purpose. The only reusable piece, the shared A/B +
non-inferiority math, lives in ``vo_rag.core.eval.non_inferiority``.
Enrichment and its decision gate do not exist: contextual enrichment
multiplies embedding cost for every chunk, so it needs an explicit owner
decision and a measured A/B result before it is built.
"""

__all__: list[str] = []
