"""
Files attached to a chat message: POST /api/attachments reads one file and returns its id,
which the client then sends as `attachment_ids` with the question.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.deps import current_user, rate_limit
from config.settings import settings
from src.attachments import (
    UnreadableFile,
    UnsupportedFile,
    VisionUnavailable,
    accepted_types,
    attachment_store,
    read_file,
)
from src.auth.users import User
from src.guardrails.injection import neutralise

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Attachments"])


def _owner(user: User) -> Optional[str]:
    return user.id if settings.auth_enabled else None


def _clean_name(name: str) -> str:
    """Display name: no path, no control / markup characters (the LLM sees it too)."""
    path = Path((name or "").replace("\\", "/").rsplit("/", 1)[-1])
    stem = "".join(ch for ch in path.stem if ch.isalnum() or ch in " ._-()").strip()
    stem, _ = neutralise(stem)
    return (stem[:120] or "tep") + path.suffix.lower()


@router.get("/attachments/types")
def types_endpoint():
    return {"types": accepted_types(), "max_mb": settings.attachment_max_mb,
            "max_files": settings.attachment_max_files}


@router.post("/attachments", status_code=201, dependencies=[Depends(rate_limit)])
def upload_attachment(file: UploadFile = File(...), user: User = Depends(current_user)):
    """Read a file (a scan or an image takes a few seconds) and keep its content."""
    name = _clean_name(file.filename or "")
    if Path(name).suffix not in accepted_types():
        raise HTTPException(status_code=415, detail="Định dạng tệp chưa được hỗ trợ.")
    limit = settings.attachment_max_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Tệp lớn hơn {settings.attachment_max_mb} MB.")
    if not data:
        raise HTTPException(status_code=422, detail="Tệp rỗng.")
    try:
        result = read_file(name, data)
    except UnsupportedFile:
        raise HTTPException(status_code=415, detail="Định dạng tệp chưa được hỗ trợ.")
    except UnreadableFile as e:
        raise HTTPException(status_code=422, detail=str(e))
    except VisionUnavailable:
        raise HTTPException(status_code=503, detail="Dịch vụ đọc ảnh tạm thời không khả dụng, vui lòng thử lại.")
    except Exception as e:
        logger.error(f"Reading attachment failed: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail="Không đọc được tệp này.")
    logger.info(f"Attachment read: {result.kind}, {len(data)} bytes -> {result.chars} chars")
    return attachment_store.add(name, len(data), result, _owner(user))


@router.delete("/attachments/{attachment_id}")
def delete_attachment(attachment_id: str, user: User = Depends(current_user)):
    """Discard a file that has not been used in a message yet."""
    if not attachment_store.delete_unused(attachment_id, _owner(user)):
        raise HTTPException(status_code=404, detail="Attachment not found")
    return {"deleted": attachment_id}
