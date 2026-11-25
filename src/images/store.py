"""
Generated images in the application database (every API worker sees them). An image belongs
to the user who asked for it, is bound to the conversation once the turn is saved and is
deleted with it; images of turns that were never saved are purged after a day.
"""

import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, insert, select, update

from config.settings import settings
from src.images.generator import GeneratedImage
from src.storage.db import generated_images as t
from src.storage.db import get_engine, utcnow


class ImageStore:
    def add(self, image: GeneratedImage, prompt: str, user_id: Optional[str]) -> str:
        iid = uuid.uuid4().hex
        with get_engine().begin() as c:
            c.execute(delete(t).where(t.c.conversation_id.is_(None), t.c.created_at < utcnow()
                                      - timedelta(hours=settings.attachment_unused_hours)))
            c.execute(insert(t).values(id=iid, user_id=user_id, prompt=prompt, model=image.model, data=image.data,
                                       media_type=image.media_type, created_at=utcnow()))
        return iid

    def recent_count(self, user_id: Optional[str]) -> int:
        """Images the user made during the last hour (shared by all workers)."""
        owner = t.c.user_id.is_(None) if user_id is None else t.c.user_id == user_id
        with get_engine().connect() as c:
            return c.execute(select(func.count()).select_from(t).where(
                owner, t.c.created_at > utcnow() - timedelta(hours=1))).scalar_one()

    def load(self, iid: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """user_id=None: no ownership check (authentication disabled)."""
        cond = [t.c.id == iid] + ([t.c.user_id == user_id] if user_id is not None else [])
        with get_engine().connect() as c:
            row = c.execute(select(t.c.data, t.c.media_type).where(*cond)).first()
        return dict(row._mapping) if row else None

    def bind(self, ids: List[str], cid: str) -> None:
        if ids:
            with get_engine().begin() as c:
                c.execute(update(t).values(conversation_id=cid).where(t.c.id.in_(ids), t.c.conversation_id.is_(None)))


image_store = ImageStore()
