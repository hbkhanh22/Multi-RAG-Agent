"""
SQLite-backed conversation store.

A new connection is opened per call (cheap for SQLite) so it is safe to use from
FastAPI's thread pool; WAL mode lets readers and a writer overlap.
"""
import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    meta TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.memory_db_path)
        self._init_lock = threading.Lock()
        self._initialised = False

    # ------------------------------------------------------------ plumbing

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=10)
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()
            self._initialised = True

    # ------------------------------------------------------------- queries

    def create_conversation(self, title: str = "") -> str:
        cid = uuid.uuid4().hex
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (cid, title[:80], now, now),
            )
        return cid

    def exists(self, cid: Optional[str]) -> bool:
        if not cid:
            return False
        with self._conn() as c:
            return c.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone() is not None

    def add_exchange(self, cid: str, user_text: str, assistant_text: str,
                     meta: Optional[Dict[str, Any]] = None) -> None:
        """Append one user+assistant turn atomically (and title the chat on first turn)."""
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (cid, user_text, now),
            )
            c.execute(
                "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (cid, assistant_text, json.dumps(meta, ensure_ascii=False, default=str) if meta else None, now),
            )
            c.execute(
                "UPDATE conversations SET updated_at = ?, "
                "title = CASE WHEN title = '' THEN ? ELSE title END WHERE id = ?",
                (now, " ".join(user_text.split())[:60], cid),
            )

    def get_recent(self, cid: str, limit: int = 12) -> List[Dict[str, str]]:
        """Last `limit` messages, oldest first: [{role, content}]."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (cid, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_messages(self, cid: str) -> List[Dict[str, Any]]:
        """Full history for the UI, including assistant metadata (sources...)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, meta, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
                (cid,),
            ).fetchall()
        out = []
        for r in rows:
            item = {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
            if r["meta"]:
                try:
                    item["meta"] = json.loads(r["meta"])
                except json.JSONDecodeError:
                    pass
            out.append(item)
        return out

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.id) "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, cid: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM conversations WHERE id = ?", (cid,)).rowcount > 0


conversation_store = ConversationStore()
