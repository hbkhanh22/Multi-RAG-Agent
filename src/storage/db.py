"""
Relational database layer (SQLAlchemy Core).

One schema serves both deployments:
  * dev:        SQLite file (settings.memory_db_path) - zero setup
  * production: PostgreSQL (settings.database_url), optionally with pgvector holding
                the chunk embeddings in the same database (see pgvector_store.py)

Tables
  conversations / messages  chat history (user_id reserved for Phase 5 auth)
  documents                 registry of ingested files per index (replaces the JSON manifest)
  ingest_jobs               durable job queue consumed by the ingestion worker

Schema changes go through Alembic (alembic/versions). `init_db()` creates missing
tables directly for SQLite / tests; on Postgres the container runs `alembic upgrade head`.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine

from config.settings import settings

logger = logging.getLogger(__name__)

metadata = MetaData()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Chat timestamps are ISO strings (sortable, identical on SQLite and Postgres)."""
    return utcnow().isoformat(timespec="seconds")


conversations = Table(
    "conversations", metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", String(64), nullable=True, index=True),
    Column("title", String(200), nullable=False, default=""),
    Column("created_at", String(40), nullable=False, default=now_iso),
    Column("updated_at", String(40), nullable=False, default=now_iso),
)

messages = Table(
    "messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", String(32), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("meta", JSON, nullable=True),
    Column("prompt_tokens", Integer, nullable=True),
    Column("completion_tokens", Integer, nullable=True),
    Column("created_at", String(40), nullable=False, default=now_iso),
    Index("idx_messages_conv", "conversation_id", "id"),
)

documents = Table(
    "documents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # Embedding index the chunks live in (model + version + runtime), see embedding_slug()
    Column("index_name", String(120), nullable=False),
    # Stable key of the file: project-relative path (local) or object key (s3)
    Column("source_key", String(500), nullable=False),
    Column("domain", String(32), nullable=False),
    Column("filename", String(255), nullable=False),
    Column("size", Integer, nullable=False, default=0),
    Column("mtime", Float, nullable=True),          # local storage change detection
    Column("etag", String(128), nullable=True),     # s3 change detection
    Column("status", String(16), nullable=False, default="pending"),  # pending|indexing|ready|failed
    Column("chunks_count", Integer, nullable=False, default=0),
    Column("error", Text, nullable=True),
    Column("user_id", String(64), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
    Column("updated_at", DateTime(timezone=True), nullable=False, default=utcnow),
    UniqueConstraint("index_name", "source_key", name="uq_documents_index_source"),
)

ingest_jobs = Table(
    "ingest_jobs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kind", String(32), nullable=False),      # sync | ingest | delete
    Column("payload", JSON, nullable=True),
    Column("status", String(16), nullable=False, default="queued"),  # queued|running|done|failed
    Column("attempts", Integer, nullable=False, default=0),
    Column("result", JSON, nullable=True),
    Column("error", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Index("idx_jobs_status", "status", "id"),
)


def _chunks_table(meta: MetaData) -> Table:
    """pgvector chunk table (Postgres only; kept in its own MetaData)."""
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import BigInteger
    from sqlalchemy.dialects.postgresql import JSONB

    return Table(
        "chunks", meta,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("index_name", String(120), nullable=False),
        Column("domain", String(32), nullable=False),
        Column("source_key", String(500), nullable=False),
        Column("filename", String(255), nullable=False),
        Column("content", Text, nullable=False),
        Column("meta", JSONB, nullable=True),
        Column("embedding", Vector(settings.embedding_dimension), nullable=False),
        Index("idx_chunks_index_domain", "index_name", "domain"),
        Index("idx_chunks_file", "index_name", "domain", "filename"),
        Index(
            "idx_chunks_embedding_hnsw", "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


vector_metadata = MetaData()
chunks = _chunks_table(vector_metadata)

_engines: dict = {}
_engine_lock = threading.Lock()


def _sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA busy_timeout = 10000")
    cur.close()


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        path = url.split("sqlite:///", 1)[-1]
        if path and path != ":memory:":
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 10})
        event.listen(engine, "connect", _sqlite_pragmas)
    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return engine


def get_engine(url: Optional[str] = None) -> Engine:
    """Process-wide engine per URL (created lazily, tables ensured on SQLite)."""
    url = url or settings.sqlalchemy_url
    with _engine_lock:
        engine = _engines.get(url)
        if engine is None:
            engine = make_engine(url)
            init_db(engine)
            _engines[url] = engine
    return engine


def init_db(engine: Engine) -> None:
    """Create missing tables. On Postgres the pgvector extension is enabled first."""
    if engine.dialect.name == "postgresql":
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        vector_metadata.create_all(engine)
    metadata.create_all(engine)
    if engine.dialect.name == "sqlite":
        _upgrade_legacy_sqlite(engine)


def _upgrade_legacy_sqlite(engine: Engine) -> None:
    """Add columns introduced after the original hand-written SQLite schema."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    wanted = {
        "conversations": {"user_id": "VARCHAR(64)"},
        "messages": {"prompt_tokens": "INTEGER", "completion_tokens": "INTEGER"},
    }
    with engine.begin() as conn:
        for table, cols in wanted.items():
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    logger.info(f"SQLite schema upgraded: {table}.{name}")


def ping(engine: Optional[Engine] = None) -> bool:
    from sqlalchemy import text
    try:
        with (engine or get_engine()).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database ping failed: {e}")
        return False
