"""
Ingestion worker: consumes `ingest_jobs` (see src/storage/jobs.py).

    python -m src.worker            # standalone process (INGEST_WORKER=external, Docker)

With INGEST_WORKER=inline (dev default) the API runs the same loop in a background
thread instead. Job kinds:
    sync              bring the vector index in line with document storage
    delete {key}      remove a file from storage, then sync (drops its chunks)

The folder watcher (local storage only) turns file-system changes in data/raw into
debounced "sync" jobs, so editing a file there updates the index without a restart.
"""

import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.storage import jobs

logger = logging.getLogger("src.worker")


def handle(job) -> dict:
    from src.ingestion.auto_ingest import sync_documents
    from src.storage.objects import get_storage

    kind = job["kind"]
    payload = job.get("payload") or {}
    if kind == "delete":
        get_storage().delete(payload["key"])
        return sync_documents()
    if kind == "sync":
        return sync_documents()
    raise ValueError(f"unknown job kind {kind!r}")


def run_once() -> bool:
    """Process one queued job. Returns False when the queue was empty."""
    from src.observability import metrics

    job = jobs.claim()
    if job is None:
        return False
    started = time.time()
    logger.info(f"Job #{job['id']} {job['kind']} started (attempt {job['attempts']})")
    try:
        result = handle(job)
        jobs.complete(job["id"], result)
        metrics.record_job(job["kind"], "done")
        logger.info(f"Job #{job['id']} done in {time.time() - started:.1f}s: {result}")
    except Exception as e:
        status = jobs.fail(job, f"{type(e).__name__}: {e}")
        metrics.record_job(job["kind"], status)
        logger.error(f"Job #{job['id']} failed ({status}): {e}", exc_info=True)
    return True


def worker_loop(stop: threading.Event) -> None:
    try:
        n = jobs.requeue_stale()
        if n:
            logger.warning(f"Re-queued {n} stale running job(s)")
    except Exception as e:
        logger.error(f"Could not re-queue stale jobs: {e}")
    while not stop.is_set():
        try:
            if run_once():
                continue
        except Exception as e:   # DB down etc.: back off, keep the loop alive
            logger.error(f"Worker loop error: {e}", exc_info=True)
        jobs.wait_for_work(settings.ingest_poll_interval)


def watch_loop(stop: threading.Event) -> None:
    """Enqueue a sync whenever supported files under data/raw change (debounced)."""
    try:
        from watchfiles import watch
    except ImportError:
        logger.warning("watchfiles not installed: automatic re-indexing disabled")
        return
    root = Path(settings.data_dir) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    suffixes = {s.lower() for s in settings.supported_file_types}

    def relevant(_change, path: str) -> bool:
        return Path(path).suffix.lower() in suffixes

    logger.info(f"Watching {root} for document changes")
    for changes in watch(root, watch_filter=relevant, debounce=settings.watch_debounce_ms,
                         stop_event=stop, raise_interrupt=False):
        names = sorted({Path(p).name for _, p in changes})
        logger.info(f"Document change detected: {', '.join(names[:5])}{' ...' if len(names) > 5 else ''}")
        try:
            jobs.enqueue("sync", {"reason": "watch"})
        except Exception as e:
            logger.error(f"Could not enqueue sync after file change: {e}")


class Background:
    """Threads started by the API in single-node mode (inline worker + watcher)."""

    def __init__(self):
        self.stop = threading.Event()
        self.threads = []

    def start(self, inline_worker: bool, watch: bool) -> None:
        if inline_worker:
            self._spawn("ingest-worker", worker_loop)
        if watch:
            self._spawn("doc-watcher", watch_loop)

    def _spawn(self, name: str, target) -> None:
        t = threading.Thread(target=target, args=(self.stop,), name=name, daemon=True)
        t.start()
        self.threads.append(t)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.stop.set()
        jobs._wakeup.set()
        for t in self.threads:
            t.join(timeout)


def main(argv: Optional[list] = None) -> None:
    from src.observability.logging import configure_logging

    configure_logging()
    stop = threading.Event()

    def _graceful(*_):
        logger.info("Worker stopping...")
        stop.set()
        jobs._wakeup.set()

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    from src.retrieval import vector_store_manager  # load the embedding model before the first job
    logger.info(f"Ingestion worker ready (backend={vector_store_manager.backend.name}, "
                f"storage={settings.document_storage})")
    jobs.enqueue("sync", {"reason": "worker-start"})

    watcher = None
    if settings.watch_documents and settings.document_storage == "local":
        watcher = threading.Thread(target=watch_loop, args=(stop,), name="doc-watcher", daemon=True)
        watcher.start()
    worker_loop(stop)


if __name__ == "__main__":
    main()
