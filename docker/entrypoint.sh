#!/bin/sh
# Container entrypoint: api | worker | migrate | <any command>
set -e

# ONNX models: pull from object storage when they are not mounted
if [ "${MODEL_BACKEND:-onnx}" = "onnx" ] && [ -n "${MODELS_S3_PREFIX:-}" ]; then
    python scripts/sync_models.py pull --if-missing
fi

# Prometheus multi-process mode: one shared dir for all workers, cleaned at start
if [ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
    rm -rf "${PROMETHEUS_MULTIPROC_DIR}" && mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"
fi

case "$1" in
    api)
        # every worker loads the models (~1.3 GB RAM each with int8 ONNX)
        exec uvicorn api.main:app --host 0.0.0.0 --port 8000 \
            --workers "${WEB_CONCURRENCY:-1}" \
            --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}" \
            --timeout-graceful-shutdown 30
        ;;
    worker)
        exec python -m src.worker
        ;;
    migrate)
        exec alembic upgrade head
        ;;
    *)
        exec "$@"
        ;;
esac
