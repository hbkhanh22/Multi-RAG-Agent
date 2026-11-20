"""
Conversation store on SQLAlchemy Core: SQLite in dev, PostgreSQL in production
(settings.database_url). Every call runs in its own short transaction, so it is
safe from FastAPI's thread pool and from several API workers.
"""
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import delete, exists, insert, select, update

from src.storage.db import conversations, get_engine, messages, now_iso

logger = logging.getLogger(__name__)


class ConversationStore:
    def __init__(self, db: Optional[Union[str, Path]] = None):
        """`db`: SQLAlchemy URL, or a SQLite file path (tests). None => settings."""
        if db is None:
            self._url = None
        elif isinstance(db, Path) or "://" not in str(db):
            self._url = f"sqlite:///{Path(db).as_posix()}"
        else:
            self._url = str(db)

    @property
    def engine(self):
        return get_engine(self._url)

    # ------------------------------------------------------------- queries

    def create_conversation(self, title: str = "", user_id: Optional[str] = None) -> str:
        cid = uuid.uuid4().hex
        now = now_iso()
        with self.engine.begin() as c:
            c.execute(insert(conversations).values(
                id=cid, title=title[:80], user_id=user_id, created_at=now, updated_at=now))
        return cid

    @staticmethod
    def _owned(cid: str, user_id: Optional[str]):
        """WHERE clause: the conversation, and (when user_id is given) owned by that user.
        user_id=None means no ownership check (authentication disabled)."""
        cond = [conversations.c.id == cid]
        if user_id is not None:
            cond.append(conversations.c.user_id == user_id)
        return cond

    def exists(self, cid: Optional[str], user_id: Optional[str] = None) -> bool:
        if not cid:
            return False
        with self.engine.connect() as c:
            return c.execute(select(conversations.c.id).where(*self._owned(cid, user_id))).first() is not None

    def add_exchange(self, cid: str, user_text: str, assistant_text: str,
                     meta: Optional[Dict[str, Any]] = None,
                     usage: Optional[Dict[str, int]] = None) -> None:
        """Append one user+assistant turn atomically (and title the chat on first turn)."""
        now = now_iso()
        usage = usage or {}
        with self.engine.begin() as c:
            c.execute(insert(messages).values(
                conversation_id=cid, role="user", content=user_text, created_at=now))
            c.execute(insert(messages).values(
                conversation_id=cid, role="assistant", content=assistant_text,
                meta=meta or None, created_at=now,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ))
            title = " ".join(user_text.split())[:60]
            c.execute(update(conversations).where(conversations.c.id == cid).values(updated_at=now))
            c.execute(update(conversations)
                      .where(conversations.c.id == cid, conversations.c.title == "")
                      .values(title=title))

    def get_recent(self, cid: str, limit: int = 12) -> List[Dict[str, str]]:
        """Last `limit` messages, oldest first: [{role, content}]."""
        with self.engine.connect() as c:
            rows = c.execute(
                select(messages.c.role, messages.c.content)
                .where(messages.c.conversation_id == cid)
                .order_by(messages.c.id.desc()).limit(limit)
            ).all()
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    def get_messages(self, cid: str) -> List[Dict[str, Any]]:
        """Full history for the UI, including assistant metadata (sources...)."""
        with self.engine.connect() as c:
            rows = c.execute(
                select(messages.c.role, messages.c.content, messages.c.meta, messages.c.created_at)
                .where(messages.c.conversation_id == cid).order_by(messages.c.id)
            ).all()
        out = []
        for r in rows:
            item = {"role": r.role, "content": r.content, "created_at": r.created_at}
            if isinstance(r.meta, dict):
                item["meta"] = r.meta
            out.append(item)
        return out

    def list_conversations(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        has_messages = exists().where(messages.c.conversation_id == conversations.c.id)
        cond = [has_messages] + ([conversations.c.user_id == user_id] if user_id is not None else [])
        with self.engine.connect() as c:
            rows = c.execute(
                select(conversations.c.id, conversations.c.title,
                       conversations.c.created_at, conversations.c.updated_at)
                .where(*cond)
                .order_by(conversations.c.updated_at.desc()).limit(limit)
            ).all()
        return [dict(r._mapping) for r in rows]

    def delete_conversation(self, cid: str, user_id: Optional[str] = None) -> bool:
        with self.engine.begin() as c:
            if c.execute(select(conversations.c.id).where(*self._owned(cid, user_id))).first() is None:
                return False
            # explicit child delete: SQLite only cascades with PRAGMA foreign_keys
            c.execute(delete(messages).where(messages.c.conversation_id == cid))
            return c.execute(delete(conversations).where(conversations.c.id == cid)).rowcount > 0

conversation_store = ConversationStore()
