"""Frozen prompts for LLM-driven synthetic query generation (Story 1.4).

The prompt hash is computed at import time from the concatenation of the system
and user template strings.  Any prompt edit automatically changes
SYNTHETIC_QUERY_PROMPT_HASH, which causes SyntheticQueryConfig (in
synthetic_queries.py) to surface a mismatch with any stored config that
references an earlier hash — ensuring prompt versioning is always auditable.
"""

from __future__ import annotations

from hashlib import sha256

SYNTHETIC_QUERY_PROMPT_VERSION = "v1.0"

SYNTHETIC_QUERY_SYSTEM_PROMPT = (
    "You are a domain expert evaluating retrieval systems for {domain}"
    " knowledge. Generate a realistic question that a {domain} expert would ask,"
    " that can be answered using the provided chunks. Return a single JSON object"
    " containing the question and the 13-field metadata schema exactly as specified"
    " in the task template. Do not include any text outside the JSON object."
)

SYNTHETIC_QUERY_USER_TEMPLATE = """\
## Source Chunks
{chunk_contents}

## Task
Generate 1 realistic evaluation question + metadata (JSON):
1. query_text
2. query_type (one of: factual, conceptual, comparative, procedural, summarization)
3. expected_answer_type (one of: short_fact, paragraph, list, formula, walkthrough)
4. ambiguity_level (low|medium|high)
5. requires_multi_chunk (bool)
6. requires_multi_book (bool)
7. requires_graph (bool)
8. requires_figure (bool)
9. requires_equation (bool)
10. requires_definition (bool)
11. domain (one of {domain} sub-domains)
12. difficulty (1-5)
13. evidence_chunk_ids (list[str], the chunk_ids that support the answer\
 — must be a subset of the chunk_ids supplied above)

Respond with a single JSON object and no surrounding text."""

# Frozen hash — any prompt edit changes this value.
SYNTHETIC_QUERY_PROMPT_HASH: str = sha256(
    (SYNTHETIC_QUERY_SYSTEM_PROMPT + SYNTHETIC_QUERY_USER_TEMPLATE).encode()
).hexdigest()

__all__ = [
    "SYNTHETIC_QUERY_PROMPT_VERSION",
    "SYNTHETIC_QUERY_PROMPT_HASH",
    "SYNTHETIC_QUERY_SYSTEM_PROMPT",
    "SYNTHETIC_QUERY_USER_TEMPLATE",
]
