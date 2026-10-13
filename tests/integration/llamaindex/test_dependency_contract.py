from __future__ import annotations

import inspect


def test_llamaindex_agent_dependency_contract() -> None:
    import llama_index.core.memory as memory_module
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context

    function_agent_params = inspect.signature(FunctionAgent).parameters
    contract = {
        "has_memory": hasattr(memory_module, "Memory"),
        "function_agent_has_output_cls": "output_cls" in function_agent_params,
        "function_agent_has_structured_output_fn": "structured_output_fn" in function_agent_params,
        "context_has_store": hasattr(Context, "store"),
    }

    assert contract == {
        "has_memory": True,
        "function_agent_has_output_cls": True,
        "function_agent_has_structured_output_fn": True,
        "context_has_store": True,
    }
