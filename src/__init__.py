"""
Multi-RAG Agent System

Exports are resolved lazily (PEP 562): importing a light sub-module such as
`src.storage.db` (worker, scripts, migrations) must not load the agent graph or the
embedding / reranker models.
"""

import importlib

_EXPORTS = {
    "RouterAgent": "src.agents",
    "RAGAgent": "src.agents",
    "SupervisorAgent": "src.agents",
    "SynthesizerAgent": "src.agents",
    "run_workflow": "src.graph",
    "app": "src.graph",
    "vector_store_manager": "src.retrieval",
    "reranker": "src.retrieval",
    "query_expander": "src.retrieval",
    "settings": "config.settings",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
