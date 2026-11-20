"""
Per-request runtime context: request id, cancellation flag and LLM token usage.

The context travels with the LangGraph run (config["configurable"]["run_ctx"]), which
LangGraph exposes inside every node and the threads they spawn with a copied context.
A plain contextvar is the fallback for code running outside a graph.

    ctx = RunContext(request_id)
    register(ctx)            # makes POST /api/query/{id}/cancel work
    ... stream_workflow(query, history, run_ctx=ctx) ...
    ctx.cancel()             # client pressed "stop" or disconnected
"""

import contextvars
import threading
import uuid
from typing import Dict, Optional

_fallback: contextvars.ContextVar[Optional["RunContext"]] = contextvars.ContextVar("run_ctx", default=None)
_active: Dict[str, "RunContext"] = {}
_active_lock = threading.Lock()


class Cancelled(Exception):
    """Raised inside LLM calls once the request has been cancelled."""


class RunContext:
    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or uuid.uuid4().hex[:16]
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self.estimated = False   # True if a provider did not report usage
        self.by_model: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------ cancellation
    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    # ------------------------------------------------------------------ usage
    def add_usage(self, model: str, prompt: int, completion: int, estimated: bool = False) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.llm_calls += 1
            self.estimated = self.estimated or estimated
            m = self.by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            m["prompt_tokens"] += prompt
            m["completion_tokens"] += completion
            m["calls"] += 1

    def usage(self) -> Dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "llm_calls": self.llm_calls,
                "estimated": self.estimated,
                "by_model": {k: dict(v) for k, v in self.by_model.items()},
            }

    def config(self) -> Dict:
        return {"configurable": {"run_ctx": self}}


def current() -> Optional[RunContext]:
    try:
        from langgraph.config import get_config

        ctx = (get_config().get("configurable") or {}).get("run_ctx")
        if ctx is not None:
            return ctx
    except Exception:
        pass
    return _fallback.get()


def bind(ctx: Optional[RunContext]):
    """Set the fallback context for the current thread/context (returns reset token)."""
    return _fallback.set(ctx)


def check_cancelled() -> None:
    ctx = current()
    if ctx is not None and ctx.cancelled:
        raise Cancelled(ctx.request_id)


def register(ctx: RunContext) -> None:
    with _active_lock:
        _active[ctx.request_id] = ctx


def unregister(ctx: RunContext) -> None:
    with _active_lock:
        _active.pop(ctx.request_id, None)


def cancel(request_id: str) -> bool:
    with _active_lock:
        ctx = _active.get(request_id)
    if ctx is None:
        return False
    ctx.cancel()
    return True


def active_count() -> int:
    with _active_lock:
        return len(_active)
