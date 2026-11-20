"""
Retrieval module - Vector stores, reranking, and query expansion

Singletons are created on first access (PEP 562), so `src.retrieval.lang` or
`src.retrieval.onnx_models` can be imported without loading any model.
"""

import importlib

# Eager: the sub-module has the same name as the singleton, so a lazy attribute would
# be shadowed by the module object once `src.retrieval.reranker` is imported. Cheap:
# the reranker loads its model on first use (or at API warm-up).
from .reranker import Reranker, reranker  # noqa: E402,F401

_EXPORTS = {
    "vector_store_manager": ".vector_store",
    "VectorStoreManager": ".vector_store",
    "query_expander": ".query_expansion",
    "QueryExpander": ".query_expansion",
}

__all__ = list(_EXPORTS) + ["reranker", "Reranker"]


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module 'src.retrieval' has no attribute {name!r}")
