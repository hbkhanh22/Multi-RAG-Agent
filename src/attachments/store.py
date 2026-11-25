"""
Attachments in the application database, so every API worker / replica sees them.

Only the extracted text (and a downscaled copy of images) is kept, not the original file.
An attachment belongs to the user who uploaded it; it is bound to a conversation once the
turn that uses it succeeds and is deleted with that conversation. Uploads that were never
used are purged after a day.
"""

import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, insert, select, update

from config.settings import settings
from src.attachments.reader import ReadResult
from src.storage.db import attachments, get_engine, utcnow


class AttachmentStore:
    def __init__(self, url: Optional[str] = None):
        self._url = url

    @property
    def engine(self):
        return get_engine(self._url)

    def add(self, filename: str, size: int, result: ReadResult, user_id: Optional[str] = None) -> Dict[str, Any]:
        aid = uuid.uuid4().hex
        with self.engine.begin() as c:
            c.execute(delete(attachments).where(
                attachments.c.conversation_id.is_(None),
                attachments.c.created_at < utcnow() - timedelta(hours=settings.attachment_unused_hours)))
            c.execute(insert(attachments).values(
                id=aid, user_id=user_id, filename=filename, kind=result.kind, size=size, pages=result.pages,
                chars=result.chars, segments=result.segments, image=result.image, created_at=utcnow()))
        return {"id": aid, "filename": filename, "kind": result.kind, "size": size, "pages": result.pages,
                "chars": result.chars, "truncated": result.truncated, "warning": result.warning}

    @staticmethod
    def _mine(user_id: Optional[str]) -> list:
        """user_id=None: no ownership check (authentication disabled)."""
        return [attachments.c.user_id == user_id] if user_id is not None else []

    def describe(self, ids: List[str], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Metadata (no content) of the caller's attachments, in the order of `ids`."""
        return self._meta(attachments.c.id.in_(ids), *self._mine(user_id), order=ids) if ids else []

    def in_conversation(self, cid: str) -> List[Dict[str, Any]]:
        """Metadata of the files already used in a conversation, oldest first."""
        return self._meta(attachments.c.conversation_id == cid)

    def _meta(self, *cond, order: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        cols = [attachments.c[n] for n in ("id", "filename", "kind", "size", "pages")]
        with self.engine.connect() as c:
            rows = [dict(r._mapping) for r in
                    c.execute(select(*cols).where(*cond).order_by(attachments.c.created_at))]
        return sorted(rows, key=lambda r: order.index(r["id"])) if order else rows

    def load(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Full content. Callers pass ids already checked with describe() / in_conversation()."""
        if not ids:
            return []
        with self.engine.connect() as c:
            rows = [dict(r._mapping) for r in c.execute(select(attachments).where(attachments.c.id.in_(ids)))]
        return sorted(rows, key=lambda r: ids.index(r["id"]))

    def bind(self, ids: List[str], cid: str) -> None:
        if ids:
            with self.engine.begin() as c:
                c.execute(update(attachments).values(conversation_id=cid)
                          .where(attachments.c.id.in_(ids), attachments.c.conversation_id.is_(None)))

    def delete_unused(self, aid: str, user_id: Optional[str] = None) -> bool:
        with self.engine.begin() as c:
            return c.execute(delete(attachments).where(
                attachments.c.id == aid, attachments.c.conversation_id.is_(None),
                *self._mine(user_id))).rowcount > 0


attachment_store = AttachmentStore()
