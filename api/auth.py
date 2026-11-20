"""
Authentication API.

  GET  /api/auth/config            which sign-in methods are enabled (for the login page)
  POST /api/auth/register          email + password sign-up   -> access token (+ refresh cookie)
  POST /api/auth/login             email + password           -> access token (+ refresh cookie)
  POST /api/auth/refresh           refresh cookie             -> new access token (cookie rotated)
  POST /api/auth/logout            revoke the refresh cookie's session
  GET  /api/auth/me                current user
  GET  /api/auth/google/login      redirect to Google
  GET  /api/auth/google/callback   back from Google -> refresh cookie -> redirect to the app

The access token lives in the SPA's memory only; the refresh token is an httpOnly,
SameSite=Lax cookie scoped to /api/auth, so XSS cannot read it and other sites cannot
use it in cross-site POSTs.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from api.deps import current_user, login_limiter
from config.settings import settings
from src.auth import google
from src.auth.tokens import (REFRESH_COOKIE, create_access_token, issue_refresh_token,
                             revoke_all, revoke_refresh_token, rotate_refresh_token)
from src.auth.users import AuthError, User, authenticate, create_user, get_user, upsert_google_user
from src.observability import metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Auth"])


class Credentials(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., min_length=1, max_length=256)


class Registration(Credentials):
    name: str = Field("", max_length=200)


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        REFRESH_COOKIE, token,
        max_age=settings.refresh_token_days * 86400,
        httponly=True, secure=settings.cookie_secure, samesite="lax", path="/api/auth",
    )


def _session(response: Response, user: User, request: Request) -> dict:
    _set_refresh_cookie(response, issue_refresh_token(user.id, user_agent=request.headers.get("user-agent", "")))
    return {"access_token": create_access_token(user), "token_type": "bearer",
            "expires_in": settings.access_token_minutes * 60, "user": user.public()}


def _require_auth_enabled() -> None:
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Authentication is disabled (AUTH_ENABLED=false)")


@router.get("/config")
def auth_config():
    return {
        "enabled": settings.auth_enabled,
        "password": settings.auth_enabled,
        "registration": settings.auth_enabled and settings.allow_registration,
        "google": settings.auth_enabled and google.enabled(),
        "min_password_length": settings.min_password_length,
    }


@router.post("/register")
def register(body: Registration, request: Request, response: Response):
    _require_auth_enabled()
    if not settings.allow_registration:
        raise HTTPException(status_code=403, detail="Registration is disabled")
    try:
        user = create_user(body.email, body.password, body.name)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    metrics.record_auth("register", "ok")
    return _session(response, user, request)


@router.post("/login")
def login(body: Credentials, request: Request, response: Response):
    _require_auth_enabled()
    ip = request.client.host if request.client else "unknown"
    retry = login_limiter.check(f"{ip}|{body.email.strip().lower()}")
    if retry:
        metrics.record_auth("login", "throttled")
        raise HTTPException(status_code=429, detail="Too many login attempts, try again later",
                            headers={"Retry-After": str(int(retry) + 1)})
    try:
        user = authenticate(body.email, body.password)
    except AuthError as e:
        metrics.record_auth("login", "failed")
        raise HTTPException(status_code=401, detail=str(e))
    metrics.record_auth("login", "ok")
    return _session(response, user, request)


@router.post("/refresh")
def refresh(request: Request, response: Response,
            rag_refresh: Optional[str] = Cookie(default=None)):
    _require_auth_enabled()
    rotated = rotate_refresh_token(rag_refresh or "", request.headers.get("user-agent", ""))
    user = get_user(rotated[0]) if rotated else None
    if not rotated or user is None or not user.is_active:
        response.delete_cookie(REFRESH_COOKIE, path="/api/auth")
        raise HTTPException(status_code=401, detail="Session expired, please sign in again")
    _set_refresh_cookie(response, rotated[1])
    return {"access_token": create_access_token(user), "token_type": "bearer",
            "expires_in": settings.access_token_minutes * 60, "user": user.public()}


@router.post("/logout")
def logout(response: Response, rag_refresh: Optional[str] = Cookie(default=None)):
    revoke_refresh_token(rag_refresh or "")
    response.delete_cookie(REFRESH_COOKIE, path="/api/auth")
    return {"ok": True}


@router.post("/logout-all")
def logout_all(response: Response, user: User = Depends(current_user)):
    """Sign out on every device (revokes all refresh tokens of the user)."""
    _require_auth_enabled()
    revoke_all(user.id)
    response.delete_cookie(REFRESH_COOKIE, path="/api/auth")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    if not settings.auth_enabled:
        return {"user": user.public(), "auth_enabled": False}
    full = get_user(user.id)
    if full is None or not full.is_active:
        raise HTTPException(status_code=401, detail="Account not found or disabled")
    return {"user": full.public(), "auth_enabled": True}


# ------------------------------------------------------------------ Google

@router.get("/google/login")
def google_login():
    _require_auth_enabled()
    if not google.enabled():
        raise HTTPException(status_code=404, detail="Google sign-in is not configured")
    flow = google.start()
    resp = RedirectResponse(flow["url"], status_code=302)
    resp.set_cookie(google.FLOW_COOKIE, flow["flow"], max_age=google.FLOW_TTL, httponly=True,
                    secure=settings.cookie_secure, samesite="lax", path="/api/auth/google")
    return resp


@router.get("/google/callback")
def google_callback(request: Request, code: str = "", state: str = "", error: str = "",
                    rag_oauth: Optional[str] = Cookie(default=None)):
    _require_auth_enabled()
    target = settings.frontend_url.rstrip("/")
    if error or not code or not rag_oauth:
        metrics.record_auth("google", "failed")
        return RedirectResponse(f"{target}/#auth_error=google_cancelled", status_code=302)
    try:
        claims = google.finish(code, state, rag_oauth)
        user = upsert_google_user(claims["sub"], claims.get("email", ""), claims.get("name", ""),
                                  bool(claims.get("email_verified")))
    except AuthError as e:
        logger.warning(f"Google sign-in refused: {e}")
        metrics.record_auth("google", "failed")
        return RedirectResponse(f"{target}/#auth_error=account", status_code=302)
    except Exception as e:
        logger.error(f"Google sign-in failed: {e}", exc_info=True)
        metrics.record_auth("google", "failed")
        return RedirectResponse(f"{target}/#auth_error=google", status_code=302)

    metrics.record_auth("google", "ok")
    # The SPA calls POST /api/auth/refresh on load and receives its access token
    resp = RedirectResponse(f"{target}/#signed_in", status_code=302)
    _set_refresh_cookie(resp, issue_refresh_token(user.id, user_agent=request.headers.get("user-agent", "")))
    resp.delete_cookie(google.FLOW_COOKIE, path="/api/auth/google")
    return resp
