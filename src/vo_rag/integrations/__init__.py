"""Optional third-party integration tiers for vo-rag.

This namespace package is a sibling of ``core/`` and is deliberately kept
outside the hot path: nothing under ``vo_rag.core`` imports anything from
``vo_rag.integrations``. Each integration sub-package guards its own
optional dependency (see ``vo_rag.integrations.llamaindex`` for the
LlamaIndex interop tier), so importing this package — or any of its
sub-packages — is cheap and side-effect-free in environments that have
not installed the corresponding optional extra.
"""
