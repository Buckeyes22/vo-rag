"""RunPod serverless vLLM worker package.

This package ships the Pydantic contracts, the no-network workload→role
mapping, and a tiny OpenAI-compatible ASGI harness used by the test suite
and local dev. Production RunPod endpoints are *not* deployed from this
package — they are provisioned from the JSON templates under
``workers/runpod_vllm/templates/`` using ``runpodctl serverless
create/update`` with the official ``worker-vllm`` hub (see
``workers/runpod_vllm/README.md`` for the deploy-phase checklist).
"""

from __future__ import annotations

from vo_rag.workers.runpod_vllm.contracts import (
    build_chat_payload,
    client_for_workload,
    role_for_workload,
)
from vo_rag.workers.runpod_vllm.schemas import (
    ChatContentPart,
    ChatMessage,
    RunpodWorkload,
    ServerlessRole,
    ServerlessWorkloadRequest,
    ServerlessWorkloadResult,
)

__all__ = [
    "RunpodWorkload",
    "ServerlessRole",
    "ChatContentPart",
    "ChatMessage",
    "ServerlessWorkloadRequest",
    "ServerlessWorkloadResult",
    "role_for_workload",
    "client_for_workload",
    "build_chat_payload",
]
