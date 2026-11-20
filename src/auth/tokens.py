"""
Session tokens.

  access token   short-lived JWT (HS256, ACCESS_TOKEN_MINUTES) sent as
                 "Authorization: Bearer ..." - stateless, checked on every request
  refresh token  opaque random string in an httpOnly cookie (JS cannot read it),
                 stored only as a SHA-256 hash; rotated on every use. Presenting an
                 already-rotated token revokes the whole family (token theft signal).
"""

import hashlib
import logging
import secrets
import uuid
from datetime import timedelta
from typing import Optional, Tuple

import jwt
from sqlalchemy import insert, select, update

from config.settings import settings
from src.storage.db import get_engine, refresh_tokens, utcnow

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ISSUER = "multi-rag-agent"
REFRESH_COOKIE = "rag_refresh"

_ephemeral_secret = None


def _secret() -> str:
    global _ephemeral_secret
    if settings.jwt_secret:
        return settings.jwt_secret
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_urlsafe(48)
        if settings.auth_enabled:
            logger.warning("JWT_SECRET is not set: using a random per-process key "
                           "(sessions end on restart and differ between workers)")
    return _ephemeral_secret


def create_access_token(user) -> str:
    now = utcnow()
    payload = {
        "sub": user.id, "email": user.email, "role": user.role, "name": user.name,
        "iat": now, "nbf": now, "exp": now + timedelta(minutes=settings.access_token_minutes),
        "iss": ISSUER, "typ": "access", "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        claims = jwt.decode(token, _secret(), algorithms=[ALGORITHM], issuer=ISSUER,
                            options={"require": ["exp", "sub", "iss"]})
    except jwt.PyJWTError:
        return None
    return claims if claims.get("typ") == "access" else None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def issue_refresh_token(user_id: str, family: Optional[str] = None, user_agent: str = "") -> str:
    token = secrets.token_urlsafe(48)
    with get_engine().begin() as conn:
        conn.execute(insert(refresh_tokens).values(
            user_id=user_id, token_hash=_hash(token), family=family or uuid.uuid4().hex,
            expires_at=utcnow() + timedelta(days=settings.refresh_token_days),
            created_at=utcnow(), user_agent=(user_agent or "")[:300],
        ))
    return token


def _aware(dt):
    from datetime import timezone
    return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=timezone.utc)   # SQLite drops tz


def rotate_refresh_token(token: str, user_agent: str = "") -> Optional[Tuple[str, str]]:
    """Exchange a valid refresh token for (user_id, new_refresh_token). None if invalid.
    Re-use of a rotated token revokes every token of that login (family)."""
    if not token:
        return None
    now = utcnow()
    with get_engine().begin() as conn:
        row = conn.execute(select(refresh_tokens).where(refresh_tokens.c.token_hash == _hash(token))).first()
        if row is None:
            return None
        if row.revoked_at is not None:
            conn.execute(update(refresh_tokens)
                         .where(refresh_tokens.c.family == row.family, refresh_tokens.c.revoked_at.is_(None))
                         .values(revoked_at=now))
            logger.warning(f"Refresh token reuse detected for user {row.user_id}: session family revoked")
            return None
        if _aware(row.expires_at) <= now:
            return None
        conn.execute(update(refresh_tokens).where(refresh_tokens.c.id == row.id).values(revoked_at=now))
    return row.user_id, issue_refresh_token(row.user_id, family=row.family, user_agent=user_agent)


def revoke_refresh_token(token: str) -> None:
    if not token:
        return
    with get_engine().begin() as conn:
        row = conn.execute(select(refresh_tokens.c.family).where(refresh_tokens.c.token_hash == _hash(token))).first()
        if row:
            conn.execute(update(refresh_tokens)
                         .where(refresh_tokens.c.family == row.family, refresh_tokens.c.revoked_at.is_(None))
                         .values(revoked_at=utcnow()))


def revoke_all(user_id: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(update(refresh_tokens)
                     .where(refresh_tokens.c.user_id == user_id, refresh_tokens.c.revoked_at.is_(None))
                     .values(revoked_at=utcnow()))
