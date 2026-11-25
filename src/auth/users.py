"""User accounts (email + password and/or Google), roles and account lookup."""

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import insert, or_, select, update

from config.settings import settings
from src.auth.passwords import DUMMY_HASH, MAX_PASSWORD_BYTES, hash_password, needs_rehash, verify_password
from src.auth.tokens import revoke_all
from src.storage.db import get_engine, users, utcnow

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str
    role: str = "user"
    is_active: bool = True
    has_password: bool = False
    google_linked: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name, "role": self.role,
                "has_password": self.has_password, "google_linked": self.google_linked}


# Principal used when AUTH_ENABLED=false: everything behaves as before Phase 5
ANONYMOUS = User(id="anonymous", email="", name="Anonymous", role="admin")


class AuthError(Exception):
    """Login / registration failure with a user-facing message (never leaks which part failed)."""


def _row_to_user(row) -> User:
    return User(id=row.id, email=row.email, name=row.name or "", role=row.role,
                is_active=bool(row.is_active), has_password=bool(row.password_hash),
                google_linked=bool(row.google_sub))


def _admin_emails() -> set:
    return {e.strip().lower() for e in settings.admin_emails.split(",") if e.strip()}


def _role_for_verified(email: str, current: str = "user") -> str:
    """ADMIN_EMAILS only counts for an address whose ownership was proven (Google).
    Password sign-up proves nothing, so it must never grant a role."""
    return "admin" if email in _admin_emails() else current


def _check_password(password: str) -> None:
    if len(password) < settings.min_password_length:
        raise AuthError(f"Mật khẩu cần ít nhất {settings.min_password_length} ký tự")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise AuthError("Mật khẩu quá dài")


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 320:
        raise AuthError("Email không hợp lệ")
    return email


def get_user(user_id: str) -> Optional[User]:
    with get_engine().connect() as conn:
        row = conn.execute(select(users).where(users.c.id == user_id)).first()
    return _row_to_user(row) if row else None


def find_by_email(email: str) -> Optional[User]:
    with get_engine().connect() as conn:
        row = conn.execute(select(users).where(users.c.email == normalize_email(email))).first()
    return _row_to_user(row) if row else None


def create_user(email: str, password: str, name: str = "", self_service: bool = True) -> User:
    """Always a normal user. Self sign-up cannot take an address reserved in ADMIN_EMAILS
    (same message as a taken address: it does not reveal who the admins are)."""
    email = normalize_email(email)
    _check_password(password)
    refused = AuthError("Không thể đăng ký bằng email này. Nếu đã có tài khoản, hãy đăng nhập.")
    if self_service and email in _admin_emails():
        logger.warning("Sign-up refused: address reserved in ADMIN_EMAILS (use src.auth.cli)")
        raise refused
    with get_engine().begin() as conn:
        if conn.execute(select(users.c.id).where(users.c.email == email)).first():
            raise refused
        uid = uuid.uuid4().hex
        conn.execute(insert(users).values(
            id=uid, email=email, name=(name or email.split("@")[0])[:200],
            password_hash=hash_password(password), role="user",
            is_active=True, created_at=utcnow(),
        ))
    logger.info(f"User registered: {uid}")
    return get_user(uid)


def authenticate(email: str, password: str) -> User:
    """Email + password login. Same error and similar timing for every failure."""
    fail = AuthError("Email hoặc mật khẩu không đúng")
    try:
        email = normalize_email(email)
    except AuthError:
        verify_password(password, DUMMY_HASH)
        raise fail
    with get_engine().connect() as conn:
        row = conn.execute(select(users).where(users.c.email == email)).first()
    if row is None or not row.password_hash:
        verify_password(password, DUMMY_HASH)
        raise fail
    if not verify_password(password, row.password_hash):
        raise fail
    if not row.is_active:
        raise AuthError("Tài khoản đã bị khóa")
    values = {"last_login_at": utcnow()}
    if needs_rehash(row.password_hash):
        values["password_hash"] = hash_password(password)
    with get_engine().begin() as conn:
        conn.execute(update(users).where(users.c.id == row.id).values(**values))
    return get_user(row.id)


def upsert_google_user(sub: str, email: str, name: str, email_verified: bool) -> User:
    """Sign in with Google: match by Google id, else link an existing account with the
    same *verified* email, else create a Google-only account."""
    if not email_verified:
        raise AuthError("Email Google chưa được xác minh")
    email = normalize_email(email)
    with get_engine().begin() as conn:
        row = conn.execute(select(users).where(or_(users.c.google_sub == sub, users.c.email == email))
                           .order_by(users.c.google_sub.is_(None))).first()
        if row is None:
            uid = uuid.uuid4().hex
            conn.execute(insert(users).values(
                id=uid, email=email, name=(name or email.split("@")[0])[:200], google_sub=sub,
                role=_role_for_verified(email), is_active=True, created_at=utcnow(), last_login_at=utcnow(),
            ))
            logger.info(f"User registered via Google: {uid}")
        else:
            if not row.is_active:
                raise AuthError("Tài khoản đã bị khóa")
            uid = row.id
            values = {"google_sub": sub, "last_login_at": utcnow(), "role": _role_for_verified(email, row.role)}
            if row.google_sub is None and row.password_hash and row.role != "admin":
                # The password was set by someone who never proved owning this address
                # (it may be an attacker waiting for the owner): the verified owner takes over.
                values["password_hash"] = None
                revoke_all(uid, conn)
                logger.warning(f"Google linked to {uid}: unverified password removed, sessions revoked")
            conn.execute(update(users).where(users.c.id == uid).values(**values))
    return get_user(uid)        # read after commit (another connection)


# ------------------------------------------------------------ operator actions (src.auth.cli)

def set_password(user_id: str, password: str, **extra) -> None:
    """New password and sign out everywhere, so whoever held the old one is out."""
    _check_password(password)
    with get_engine().begin() as conn:
        conn.execute(update(users).where(users.c.id == user_id).values(
            password_hash=hash_password(password), **extra))
        revoke_all(user_id, conn)


def set_admin(email: str, password: str) -> User:
    """Create the admin, or take the account over with a new password: promoting an account
    without changing its password could promote a stranger who registered the address."""
    user = find_by_email(email) or create_user(email, password, self_service=False)
    set_password(user.id, password, role="admin", is_active=True)
    return get_user(user.id)


def demote(user_id: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(update(users).where(users.c.id == user_id).values(role="user"))
