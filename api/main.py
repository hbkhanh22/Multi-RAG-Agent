"""
FastAPI Main Application - Multi-RAG Agent System API
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from config.settings import settings
from src.observability.logging import configure_logging, request_id_var

configure_logging()

from api.documents import router as documents_router  # noqa: E402
from api.routes import router  # noqa: E402
from src.mcp_client import mcp_manager  # noqa: E402
from src.observability import metrics  # noqa: E402
from src.storage import jobs  # noqa: E402
from src.utils import api_logger  # noqa: E402
from src.worker import Background  # noqa: E402

background = Background()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: queue a document sync, start the inline worker / watcher, MCP."""
    api_logger.info("Starting Multi-RAG Agent System API")
    api_logger.info(
        f"LLM {settings.default_llm_model} | vectors {settings.vector_backend} | "
        f"models {settings.effective_model_backend} | storage {settings.document_storage} | "
        f"worker {settings.ingest_worker} | db {'postgres' if settings.is_postgres else 'sqlite'}"
    )
    try:
        # Indexing happens in the worker, so the API answers immediately
        jobs.enqueue("sync", {"reason": "startup"})
    except Exception as e:
        api_logger.error(f"Could not queue the startup sync: {e}", exc_info=True)
    def _warmup():
        # load embedding + reranker models now, not on the first question;
        # /health/ready reports "not ready" until this is done
        try:
            from src.retrieval import reranker, vector_store_manager  # noqa: F401
            reranker.warmup()
            api_logger.info("Models loaded")
        except Exception as e:
            api_logger.error(f"Model warm-up failed: {e}", exc_info=True)

    import threading
    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    background.start(
        inline_worker=settings.ingest_worker == "inline",
        watch=settings.watch_documents and settings.document_storage == "local"
        and settings.ingest_worker == "inline",
    )
    # Connect to MCP servers in the background too (npx/uvx may download packages on first run)
    mcp_manager.start()

    yield

    mcp_manager.stop()
    background.shutdown()
    api_logger.info("Shutting down Multi-RAG Agent System API")


app = FastAPI(
    title="Multi-RAG Agent System",
    description="Multi-agent Retrieval-Augmented Generation system (LangGraph + pgvector/Chroma + NVIDIA NIM)",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Retry-After"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Request id (logs + X-Request-ID header) and HTTP metrics."""
    rid = request.headers.get("x-request-id", "")
    rid = rid if rid.isalnum() and len(rid) <= 64 else uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    started = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"   # template, not raw path: bounded labels
        metrics.record_http(request.method, path, status, time.time() - started)
        request_id_var.reset(token)


app.include_router(router)
app.include_router(documents_router)

frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=frontend_dist, html=True), name="frontend")


@app.get("/")
async def root():
    return {"message": "Multi-RAG Agent System API", "version": app.version, "docs": "/docs"}


@app.get("/health/live", tags=["Ops"])
async def live():
    """Liveness: the process is up (restart the container if this fails)."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Ops"])
def ready():
    """Readiness: database, vector index, document storage and models are usable
    (the load balancer only sends traffic when this returns 200)."""
    from src.retrieval import reranker, vector_store_manager
    from src.storage.db import ping
    from src.storage.objects import get_storage

    checks = {}
    for name, fn in (
        ("database", ping),
        ("vector_store", vector_store_manager.healthy),
        ("storage", lambda: get_storage().healthy()),
        ("reranker", lambda: reranker.model is not None or not settings.use_reranker),
        ("llm_api_key", lambda: bool(settings.nvidia_api_key)),
    ):
        try:
            checks[name] = bool(fn())
        except Exception as e:
            api_logger.error(f"readiness check {name} failed: {e}")
            checks[name] = False
    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})


@app.get("/metrics", tags=["Ops"], include_in_schema=False)
def metrics_endpoint():
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    api_logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred.",
            "path": request.url.path,
            "request_id": request_id_var.get(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.log_level.lower(), reload=False)
