# syntax=docker/dockerfile:1.7
# Backend image: API, ingestion worker and migrations (one image, three commands).
#   docker build -t multi-rag-agent .
#   docker run ... multi-rag-agent api | worker | migrate
#
# Embeddings / reranker run on ONNX Runtime int8 (no torch => ~1 GB image instead of ~5 GB).
# The ONNX models are NOT baked in: mount data/models/onnx, or let the entrypoint pull
# them from object storage (MODELS_S3_PREFIX), see scripts/sync_models.py.
# Base image from GHCR (Python + uv); override with --build-arg BASE_IMAGE=python:3.11-slim.
ARG BASE_IMAGE=ghcr.io/astral-sh/uv:python3.11-bookworm-slim

FROM ${BASE_IMAGE} AS deps
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY requirements-runtime.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    (command -v uv >/dev/null || pip install --no-cache-dir uv) && \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r requirements-runtime.txt && \
    /opt/venv/bin/python -c "import fastapi, uvicorn, onnxruntime, psycopg, pgvector, langgraph" && \
    test -x /opt/venv/bin/uvicorn && test -x /opt/venv/bin/alembic

FROM ${BASE_IMAGE} AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_BACKEND=onnx \
    VECTOR_BACKEND=pgvector \
    INGEST_WORKER=external \
    MCP_ENABLED=false \
    LOG_JSON=true \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --chown=app:app config ./config
COPY --chown=app:app src ./src
COPY --chown=app:app api ./api
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app scripts/sync_models.py ./scripts/sync_models.py
COPY --chown=app:app docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /app/data /app/logs && chown -R app:app /app/data /app/logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status == 200 else 1)" || exit 1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
