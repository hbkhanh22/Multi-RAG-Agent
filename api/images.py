"""Images drawn by the Image Agent: GET /api/images/{id} (only for the user who asked for it)."""

from fastapi import APIRouter, Depends, HTTPException, Response

from api.deps import current_user
from config.settings import settings
from src.auth.users import User
from src.images import image_store

router = APIRouter(prefix="/api", tags=["Images"])


@router.get("/images/{image_id}")
def get_image(image_id: str, user: User = Depends(current_user)):
    image = image_store.load(image_id, user.id if settings.auth_enabled else None)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(image["data"], media_type=image["media_type"], headers={
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
