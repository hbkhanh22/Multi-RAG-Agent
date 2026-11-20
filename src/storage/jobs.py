"""
Durable ingestion job queue on the relational database.

No extra broker: jobs are rows in `ingest_jobs`. On PostgreSQL a worker claims the
oldest queued job with `SELECT ... FOR UPDATE SKIP LOCKED`, so any number of workers
can poll the same table without taking the same job twice (the classic Postgres
queue pattern). SQLite (dev, single process) serialises claims with a lock.

Failed jobs are retried up to settings.ingest_max_attempts; jobs left "running" by a
crashed worker are re-queued after STALE_AFTER.
"""

import logging
import threading
from datetime import timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, insert, select, update

from config.settings import settings
from src.storage.db import get_engine, ingest_jobs, utcnow

logger = logging.getLogger(__name__)

STALE_AFTER = timedelta(minutes=30)
_sqlite_claim_lock = threading.Lock()
_wakeup = threading.Event()   # lets an in-process worker react to new jobs immediately


def _row(r) -> Dict[str, Any]:
    d = dict(r._mapping)
    for k in ("created_at", "started_at", "finished_at"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    return d


def enqueue(kind: str, payload: Optional[Dict[str, Any]] = None, dedupe: bool = True) -> int:
    """Add a job; with dedupe an identical job still waiting in the queue is reused."""
    engine = get_engine()
    payload = payload or {}
    with engine.begin() as conn:
        if dedupe:
            for r in conn.execute(
                select(ingest_jobs.c.id, ingest_jobs.c.payload)
                .where(ingest_jobs.c.kind == kind, ingest_jobs.c.status == "queued")
            ).all():
                if (r.payload or {}) == payload:
                    return r.id
        job_id = conn.execute(
            insert(ingest_jobs).values(kind=kind, payload=payload, status="queued", created_at=utcnow())
        ).inserted_primary_key[0]
    _wakeup.set()
    logger.info(f"Enqueued ingest job #{job_id} ({kind} {payload or ''})")
    return job_id


def claim() -> Optional[Dict[str, Any]]:
    """Atomically take the oldest queued job (or None)."""
    engine = get_engine()
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            row = conn.execute(
                select(ingest_jobs).where(ingest_jobs.c.status == "queued")
                .order_by(ingest_jobs.c.id).limit(1).with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            conn.execute(update(ingest_jobs).where(ingest_jobs.c.id == row.id).values(
                status="running", started_at=utcnow(), attempts=row.attempts + 1))
            job = _row(row)
    else:
        with _sqlite_claim_lock, engine.begin() as conn:
            row = conn.execute(
                select(ingest_jobs).where(ingest_jobs.c.status == "queued").order_by(ingest_jobs.c.id).limit(1)
            ).first()
            if row is None:
                return None
            conn.execute(update(ingest_jobs).where(ingest_jobs.c.id == row.id).values(
                status="running", started_at=utcnow(), attempts=row.attempts + 1))
            job = _row(row)
    job["attempts"] += 1
    return job


def complete(job_id: int, result: Optional[Dict[str, Any]] = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(update(ingest_jobs).where(ingest_jobs.c.id == job_id).values(
            status="done", result=result or {}, error=None, finished_at=utcnow()))


def fail(job: Dict[str, Any], error: str) -> str:
    """Record a failure; re-queue while attempts remain. Returns the new status."""
    status = "queued" if job["attempts"] < settings.ingest_max_attempts else "failed"
    with get_engine().begin() as conn:
        conn.execute(update(ingest_jobs).where(ingest_jobs.c.id == job["id"]).values(
            status=status, error=error[:2000], finished_at=utcnow() if status == "failed" else None))
    return status


def requeue_stale() -> int:
    """Jobs stuck in "running" (worker crashed) go back to the queue."""
    cutoff = utcnow() - STALE_AFTER
    with get_engine().begin() as conn:
        return conn.execute(update(ingest_jobs).where(and_(
            ingest_jobs.c.status == "running", ingest_jobs.c.started_at < cutoff)).values(status="queued")).rowcount


def get(job_id: int) -> Optional[Dict[str, Any]]:
    with get_engine().connect() as conn:
        row = conn.execute(select(ingest_jobs).where(ingest_jobs.c.id == job_id)).first()
    return _row(row) if row else None


def recent(limit: int = 20) -> List[Dict[str, Any]]:
    with get_engine().connect() as conn:
        rows = conn.execute(select(ingest_jobs).order_by(ingest_jobs.c.id.desc()).limit(limit)).all()
    return [_row(r) for r in rows]


def wait_for_work(timeout: float) -> None:
    _wakeup.wait(timeout)
    _wakeup.clear()
