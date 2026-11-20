"""
Prometheus metrics (exposed at GET /metrics).

Everything degrades to no-ops when prometheus_client is missing or metrics are
disabled, so call sites never need guards. With several gunicorn/uvicorn workers set
PROMETHEUS_MULTIPROC_DIR (the Docker image does) so all workers are aggregated.
"""

import logging
import os

from config.settings import settings

logger = logging.getLogger(__name__)

# multi-process mode writes one file per metric/process here; it must exist before
# the first metric is created (the container entrypoint also cleans it at start)
if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _AVAILABLE = settings.metrics_enabled
except ImportError:  # pragma: no cover
    _AVAILABLE = False

_LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60, 120)

if _AVAILABLE:
    HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "path", "status"])
    HTTP_LATENCY = Histogram("rag_http_request_seconds", "HTTP request latency", ["method", "path"],
                             buckets=_LAT_BUCKETS)
    QUERIES = Counter("rag_queries_total", "Answered questions", ["route", "status"])
    QUERY_LATENCY = Histogram("rag_query_seconds", "End-to-end answer latency", ["route"], buckets=_LAT_BUCKETS)
    TTFT = Histogram("rag_time_to_first_token_seconds", "Time to first streamed token", buckets=_LAT_BUCKETS)
    STAGE_LATENCY = Histogram("rag_stage_seconds", "Workflow stage latency", ["stage"], buckets=_LAT_BUCKETS)
    LLM_CALLS = Counter("rag_llm_calls_total", "LLM calls", ["model", "agent"])
    LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["model", "kind"])
    LLM_LATENCY = Histogram("rag_llm_call_seconds", "LLM call latency", ["model"], buckets=_LAT_BUCKETS)
    EVAL_RESULTS = Counter("rag_self_eval_total", "Online judge verdicts", ["result"])
    CANCELLED = Counter("rag_cancelled_total", "Generations cancelled by the user")
    RATE_LIMITED = Counter("rag_rate_limited_total", "Requests rejected by the rate limiter")
    INGEST_JOBS = Counter("rag_ingest_jobs_total", "Ingestion jobs", ["kind", "status"])
    INGESTED_DOCS = Counter("rag_ingested_documents_total", "Documents (re-)indexed", ["status"])
    ACTIVE_STREAMS = Gauge("rag_active_streams", "Answers being generated", multiprocess_mode="livesum")


def _safe(fn):
    def wrapper(*args, **kwargs):
        if not _AVAILABLE:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # metrics must never break a request
            logger.debug(f"metrics error in {fn.__name__}: {e}")
    return wrapper


@_safe
def record_http(method: str, path: str, status: int, seconds: float) -> None:
    HTTP_REQUESTS.labels(method, path, str(status)).inc()
    HTTP_LATENCY.labels(method, path).observe(seconds)


@_safe
def record_llm_call(model: str, agent: str, prompt: int, completion: int, seconds: float) -> None:
    LLM_CALLS.labels(model, agent).inc()
    LLM_TOKENS.labels(model, "prompt").inc(prompt)
    LLM_TOKENS.labels(model, "completion").inc(completion)
    LLM_LATENCY.labels(model).observe(seconds)


@_safe
def record_query(state: dict, ttft: float = None) -> None:
    route = str(state.get("route") or "unknown")
    status = str(state.get("status") or "unknown")
    QUERIES.labels(route, status).inc()
    times = state.get("processing_times") or {}
    if times.get("total"):
        QUERY_LATENCY.labels(route).observe(times["total"])
    for stage, seconds in times.items():
        if stage != "total" and isinstance(seconds, (int, float)):
            STAGE_LATENCY.labels(stage).observe(seconds)
    if ttft is not None:
        TTFT.observe(ttft)
    ev = state.get("evaluation") or {}
    if ev:
        EVAL_RESULTS.labels("refined" if state.get("iteration") else
                            ("pass" if ev.get("passed") else "fail")).inc()


@_safe
def record_cancelled() -> None:
    CANCELLED.inc()


@_safe
def record_rate_limited() -> None:
    RATE_LIMITED.inc()


@_safe
def record_job(kind: str, status: str) -> None:
    INGEST_JOBS.labels(kind, status).inc()


@_safe
def record_ingested(status: str, n: int = 1) -> None:
    INGESTED_DOCS.labels(status).inc(n)


@_safe
def stream_started() -> None:
    ACTIVE_STREAMS.inc()


@_safe
def stream_finished() -> None:
    ACTIVE_STREAMS.dec()


def render():
    """(body, content_type) for the /metrics endpoint."""
    if not _AVAILABLE:
        return b"# metrics disabled\n", "text/plain; version=0.0.4"
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    from prometheus_client import REGISTRY

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
