"""
Document synchronisation: storage (local data/raw or S3) -> vector index.

`sync_documents()` compares the files in storage with the `documents` registry of the
current index (embedding model + runtime + version):
  * new / modified files   -> load, chunk, embed, replace their chunks, mark "ready"
  * files that disappeared -> their chunks and registry rows are removed
  * unchanged files        -> skipped (size + mtime, or S3 ETag)

It runs at startup, after uploads/deletes (as queued jobs) and whenever the folder
watcher sees a change. A process lock (+ a Postgres advisory lock across processes)
guarantees only one sync runs at a time.

The old JSON manifest (data/vectorstores/.ingested_manifest.<index>.json) is imported
once into the registry so an existing Chroma index is not re-embedded.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sqlalchemy import delete, insert, select, text, update

from config.settings import settings
from src.ingestion.chunker import DocumentChunker
from src.ingestion.contextualizer import PageContextualizer, add_chunk_headers
from src.storage.db import documents, get_engine, utcnow
from src.storage.objects import StoredFile, get_storage

logger = logging.getLogger(__name__)

_sync_lock = threading.Lock()
_ADVISORY_KEY = 727_001   # arbitrary app-wide id for pg_advisory_lock


def _index_name() -> str:
    from src.retrieval.vector_store import embedding_slug
    return embedding_slug()


def _legacy_manifest_file() -> Path:
    return Path(settings.data_dir) / "vectorstores" / f".ingested_manifest.{_index_name()}.json"


def _load_single_document(file_path: Path):
    """Load a single document based on its extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(file_path), encoding="utf-8").load()
    if ext == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(str(file_path)).load()
        except ImportError:
            logger.error(f"docx2txt is not installed - cannot ingest {file_path.name} (pip install docx2txt)")
            return []
    logger.debug(f"Unsupported file format for auto-ingest: {file_path}")
    return []


# ------------------------------------------------------------------ registry

def list_documents(index_name: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(documents).where(documents.c.index_name == (index_name or _index_name()))
            .order_by(documents.c.domain, documents.c.filename)
        ).all()
    out = []
    for r in rows:
        d = dict(r._mapping)
        for k in ("created_at", "updated_at"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        out.append(d)
    return out


def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    with get_engine().connect() as conn:
        row = conn.execute(select(documents).where(documents.c.id == doc_id)).first()
    return dict(row._mapping) if row else None


def mark_pending(domain: str, key: str, filename: str, size: int, user_id: Optional[str] = None) -> int:
    """Register an uploaded file immediately (status "pending") so the UI can show it."""
    index = _index_name()
    now = utcnow()
    with get_engine().begin() as conn:
        row = conn.execute(select(documents.c.id).where(
            documents.c.index_name == index, documents.c.source_key == key)).first()
        if row:
            conn.execute(update(documents).where(documents.c.id == row.id).values(
                status="pending", size=size, error=None, updated_at=now))
            return row.id
        return conn.execute(insert(documents).values(
            index_name=index, source_key=key, domain=domain, filename=filename, size=size,
            status="pending", chunks_count=0, user_id=user_id, created_at=now, updated_at=now,
        )).inserted_primary_key[0]


def _import_legacy_manifest(conn, index: str) -> int:
    """One-time migration from the JSON manifest (Chroma only: its chunks exist)."""
    path = _legacy_manifest_file()
    if settings.vector_backend != "chroma" or not path.exists():
        return 0
    if conn.execute(select(documents.c.id).where(documents.c.index_name == index).limit(1)).first():
        return 0
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Cannot read legacy manifest {path.name}: {e}")
        return 0
    now = utcnow()
    for key, e in manifest.items():
        conn.execute(insert(documents).values(
            index_name=index, source_key=Path(key).as_posix(), domain=e.get("domain", "general"),
            filename=e.get("filename", Path(key).name), size=e.get("size", 0), mtime=e.get("mtime"),
            status="ready", chunks_count=e.get("chunks_count", 0), created_at=now, updated_at=now,
        ))
    logger.info(f"Imported {len(manifest)} entries from legacy manifest {path.name}")
    return len(manifest)


# ------------------------------------------------------------------ sync

@contextmanager
def _exclusive():
    """One sync at a time: in this process (lock) and across processes (Postgres)."""
    with _sync_lock:
        engine = get_engine()
        if engine.dialect.name != "postgresql":
            yield
            return
        with engine.connect() as conn:
            conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _ADVISORY_KEY})
            try:
                yield
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _ADVISORY_KEY})
                conn.commit()


def _ingest_file(f: StoredFile, row, index: str, chunker, storage, state: Dict[str, Any]) -> int:
    from src.retrieval import vector_store_manager

    engine = get_engine()
    now = utcnow()
    with engine.begin() as conn:
        if row is None:
            doc_id = conn.execute(insert(documents).values(
                index_name=index, source_key=f.key, domain=f.domain, filename=f.filename, size=f.size,
                mtime=f.mtime, etag=f.etag, status="indexing", created_at=now, updated_at=now,
            )).inserted_primary_key[0]
        else:
            doc_id = row.id
            conn.execute(update(documents).where(documents.c.id == doc_id).values(
                status="indexing", error=None, updated_at=now))

    try:
        with storage.open_local(f) as path:
            docs = _load_single_document(path)
        if not docs:
            raise ValueError("no text could be extracted")
        for d in docs:
            d.metadata["domain"] = f.domain
            d.metadata["filename"] = f.filename
            d.metadata["source"] = f.key
        if settings.contextual_chunks:
            state["contextualizer"] = state.get("contextualizer") or PageContextualizer()
            state["contextualizer"].contextualize(docs, f.key, f.size, f.mtime or 0)
        chunks = chunker.chunk(docs)
        if settings.contextual_chunks:
            add_chunk_headers(chunks)
        if not chunks:
            raise ValueError("document produced no chunks")

        # the chunks of an older version are replaced (and of a moved file, removed)
        if row is not None and (row.domain != f.domain or row.filename != f.filename):
            vector_store_manager.delete_by_filename(row.domain, row.filename)
        n = vector_store_manager.replace_file(f.domain, f.filename, chunks)

        with engine.begin() as conn:
            conn.execute(update(documents).where(documents.c.id == doc_id).values(
                status="ready", chunks_count=n, size=f.size, mtime=f.mtime, etag=f.etag,
                domain=f.domain, filename=f.filename, error=None, updated_at=utcnow()))
        logger.info(f"Indexed {f.filename} ({n} chunks) into '{f.domain}'")
        return n
    except Exception as e:
        logger.error(f"Error ingesting {f.filename}: {e}", exc_info=True)
        with engine.begin() as conn:
            conn.execute(update(documents).where(documents.c.id == doc_id).values(
                status="failed", error=str(e)[:2000], size=f.size, mtime=f.mtime, etag=f.etag,
                updated_at=utcnow()))
        return -1


def sync_documents() -> Dict[str, int]:
    """Bring the vector index in line with the document storage."""
    from src.observability import metrics
    from src.retrieval import vector_store_manager

    started = time.time()
    storage = get_storage()
    index = _index_name()
    stats = {"sport": 0, "tech": 0, "general": 0, "removed": 0, "failed": 0, "total_chunks": 0, "unchanged": 0}
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    with _exclusive():
        engine = get_engine()
        with engine.begin() as conn:
            _import_legacy_manifest(conn, index)
            rows = {r.source_key: r for r in conn.execute(
                select(documents).where(documents.c.index_name == index)).all()}

        chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        state: Dict[str, Any] = {}
        present = set()
        for f in storage.list_files():
            present.add(f.key)
            row = rows.get(f.key)
            unchanged = (
                row is not None and row.status == "ready"
                and (row.size, row.etag if f.etag else row.mtime) == f.version()
            )
            if unchanged:
                stats["unchanged"] += 1
                continue
            if row is not None and row.status == "failed" and (row.size, row.etag if f.etag else row.mtime) == f.version():
                stats["unchanged"] += 1      # same broken file: do not retry on every sync
                continue
            if f.size > max_bytes:
                logger.warning(f"Skipping {f.filename}: larger than {settings.max_file_size_mb} MB")
                continue
            n = _ingest_file(f, row, index, chunker, storage, state)
            if n < 0:
                stats["failed"] += 1
                metrics.record_ingested("failed")
            else:
                stats[f.domain] += 1
                stats["total_chunks"] += n
                metrics.record_ingested("ready")

        for key, row in rows.items():
            if key in present:
                continue
            removed = vector_store_manager.delete_by_filename(row.domain, row.filename)
            with engine.begin() as conn:
                conn.execute(delete(documents).where(documents.c.id == row.id))
            stats["removed"] += 1
            logger.info(f"Removed {removed} chunks of deleted file: {row.filename}")

    ingested = stats["sport"] + stats["tech"] + stats["general"]
    if ingested or stats["removed"] or stats["failed"]:
        logger.info(
            f"Sync completed in {time.time() - started:.1f}s: {ingested} documents "
            f"({stats['total_chunks']} chunks) indexed, {stats['removed']} removed, {stats['failed']} failed."
        )
    else:
        logger.info("Vector stores are up-to-date.")
    return stats


def auto_ingest_raw_documents() -> Dict[str, int]:
    """Backward-compatible name for sync_documents()."""
    return sync_documents()
