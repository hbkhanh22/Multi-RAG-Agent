"""
Optional Langfuse tracing of the agent graph.

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (works with
Langfuse Cloud or a self-hosted instance via LANGFUSE_HOST) and the `langfuse` package
is installed. The LangChain callback handler is attached to the LangGraph run config,
so every node, LLM call (with prompts, tokens, latency) shows up as one trace per
question, tagged with the request id and conversation id.
"""

import logging
from typing import Dict, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_handler_factory = None
_checked = False


def enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _factory():
    global _handler_factory, _checked
    if _checked:
        return _handler_factory
    _checked = True
    if not enabled():
        return None
    try:
        import os

        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        from langfuse.langchain import CallbackHandler  # langfuse >= 3

        _handler_factory = CallbackHandler
        logger.info(f"Langfuse tracing enabled ({settings.langfuse_host})")
    except Exception as e:
        logger.warning(f"Langfuse keys set but tracing unavailable: {e} (pip install langfuse)")
    return _handler_factory


def run_config(request_id: str, conversation_id: Optional[str] = None) -> Dict:
    """Extra LangGraph config (callbacks + metadata) for one run; {} when disabled."""
    factory = _factory()
    if factory is None:
        return {}
    try:
        return {
            "callbacks": [factory()],
            "run_name": "multi-rag-answer",
            "metadata": {
                "langfuse_session_id": conversation_id or request_id,
                "langfuse_tags": ["multi-rag"],
                "request_id": request_id,
            },
        }
    except Exception as e:
        logger.warning(f"Langfuse handler creation failed: {e}")
        return {}
