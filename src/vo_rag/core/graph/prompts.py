# ruff: noqa: E501

from __future__ import annotations

from hashlib import sha256

GRAPH_PROMPT_VERSION = "v2.4"  # v2.4: bias extraction toward canonical directed relationships

GRAPH_EXTRACTION_SYSTEM_PROMPT = (
    "You are a domain expert in technical book analysis. Given a book's metadata "
    "and representative chunks, extract a knowledge graph of entities and their "
    "relationships. Every node and edge must be traceable to specific source chunks "
    "via chunk_id provenance."
)

GRAPH_EXTRACTION_USER_TEMPLATE = """## Book Metadata
Title: {title}
Author: {author}

## Representative Chunks
Each chunk is labeled with a short id in brackets, e.g. [c0], [c1].

{chunk_contents}

## Extraction Task
Extract a knowledge graph from the provided chunks. Respond with ONLY a valid JSON object \
(no commentary, no markdown code fences) containing two top-level arrays:

{{
  "nodes": [
    {{
      "node_type": "<one of: concept, method, equation, person, component>",
      "label": "<unique name for this entity>",
      "aliases": ["<optional alternative names>"],
      "chunk_ids": ["<short chunk label(s) like c0 that mention this entity>"]
    }}
  ],
  "edges": [
    {{
      "source_label": "<label of the source node>",
      "source_type": "<node_type of the source node>",
      "target_label": "<label of the target node>",
      "target_type": "<node_type of the target node>",
      "edge_type": "<one of: mentions, authored_by, relates_to, used_by, part_of, is_a, introduces>",
      "evidence_chunk_ids": ["<short chunk label(s) like c0 that support this relationship>"]
    }}
  ]
}}

## Relationship types
Each edge reads as a directed sentence: SOURCE -> edge_type -> TARGET. Pick the edge_type whose sentence is literally true, and pick the MOST SPECIFIC one (use relates_to only when nothing else fits).
Prefer part_of, is_a, or used_by when a directed relationship is plausible.
Reserve relates_to for genuinely symmetric or unknown relationships.
- is_a: SOURCE is a specific type, instance, or member of the general category TARGET. e.g. "Honda EU2000i Generator -> is_a -> Generator"; "FIA Harness -> is_a -> Harness". Use is_a (NOT part_of) for type/category/instance/membership relationships.
- part_of: SOURCE is a physical or compositional PART of TARGET (source is a piece of target, not a kind of it). A physical part that is installed on, mounted to, bolted to, fitted to, or equipped on a vehicle/engine/assembly is part_of it (the PART is the SOURCE, the whole is the TARGET). e.g. "Intake -> part_of -> Engine"; "Brake Rotor -> part_of -> Brake System"; "Goodyear 3R -> part_of -> SuperMiata" (tire); "XIDA Dampers -> part_of -> SuperMiata"; "Fuel Pump -> part_of -> Engine". If "SOURCE is a kind of TARGET" reads true, use is_a instead.
- used_by: SOURCE MUST be a genuine TOOL / INSTRUMENT (a measuring or working device). TARGET MUST be a PROCESS, TECHNIQUE, ACTIVITY, or MEASUREMENT (an action - often a gerund/-ing, or a method-type node) that employs it. The TOOL is always the SOURCE. e.g. "Pyrometer -> used_by -> Tire Temperature Check". Never put the process/technique as source. If the TARGET is a physical object, vehicle, assembly, or part (not an action), do NOT use used_by. If SOURCE is a part of TARGET, use part_of; otherwise relates_to. WRONG: "XIDA Dampers -> used_by -> SuperMiata" -> CORRECT: "XIDA Dampers -> part_of -> SuperMiata" (a damper is a part of the car). WRONG: "NA8 SM Engine -> used_by -> Intake" -> CORRECT: "Intake -> part_of -> NA8 SM Engine" (the intake is a part of the engine; note the direction - the PART is the source).
- introduces: SOURCE (a section/topic/work) introduces or defines TARGET (a concept/term). e.g. "Suspension Setup -> introduces -> Camber".
- authored_by: SOURCE (a work) is authored by TARGET (a person). e.g. "The Guide -> authored_by -> Author Name".
- mentions: SOURCE references TARGET with no stronger relationship. Use sparingly.
- relates_to: SOURCE is generally associated with TARGET and none of the above fits. Last resort.

Direction check: before emitting each edge, read it as the sentence "SOURCE edge_type TARGET". If the reverse reading "TARGET edge_type SOURCE" is the true one, SWAP source and target. Prefer fewer correct, specific, correctly-directed edges over many vague or mis-directed ones.

Rules:
- Only include edges where BOTH the source node AND the target node are also present
  in the "nodes" array.
- Every chunk reference must be one of the short labels shown above (c0, c1, …) —
  use the label exactly, never the chunk's text.
- Do not include nodes without at least one supporting chunk label.
- Do not include additional fields outside the "nodes" and "edges" arrays.
- The JSON must be valid and parseable.
- Choose the most specific edge_type and verify its direction with the SOURCE edge_type TARGET sentence check above.
"""


GRAPH_EXTRACTION_PROMPT_HASH: str = sha256(
    (GRAPH_EXTRACTION_SYSTEM_PROMPT + GRAPH_EXTRACTION_USER_TEMPLATE).encode()
).hexdigest()

__all__ = [
    "GRAPH_PROMPT_VERSION",
    "GRAPH_EXTRACTION_SYSTEM_PROMPT",
    "GRAPH_EXTRACTION_USER_TEMPLATE",
    "GRAPH_EXTRACTION_PROMPT_HASH",
]
