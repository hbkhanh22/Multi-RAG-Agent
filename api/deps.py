"""
Request guards: authentication (current user / admin), rate limiting.

The limiter is a sliding window per client (IP, or the Bearer token when present),
kept in process memory. With N API workers the effective limit is up to N x the
configured value; a shared store (Redis) is the upgrade path when that matters.
uvicorn runs with --proxy-headers behind nginx, so request.client.host is the real
client address.
"""

import hmac
import threading
import time
from collections import deque
from typing import Deque, Dict

from fastapi import Depends, HTTPException, Request

from config.settings import settings
from src.auth.tokens import decode_access_token
from src.auth.users import ANONYMOUS, User
from src.observability import metrics


class SlidingWindowLimiter:
    def __init__(self, limit: int, window: float = 60.0, max_keys: int = 10_000):
        self.limit = limit
        self.window = window
        self.max_keys = max_keys
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """0 if allowed, else seconds until the next request would be allowed."""
        if self.limit <= 0:
            return 0.0
        now = time.monotonic()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self.max_keys:   # bound memory: drop idle clients
                    for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                        del self._hits[k]
                hits = self._hits[key] = deque()
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.0, self.window - (now - hits[0]))
            hits.append(now)
            return 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


query_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
login_limiter = SlidingWindowLimiter(settings.login_attempts_per_minute)


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def client_key(request: Request) -> str:
    """Rate-limit identity: the signed-in user, else the client IP."""
    user = getattr(request.state, "user", None)
    if user is not None and user.id != ANONYMOUS.id:
        return "user:" + user.id
    return "ip:" + (request.client.host if request.client else "unknown")


def current_user(request: Request) -> User:
    """The caller. With AUTH_ENABLED=false everyone is the anonymous admin (dev mode).
    With auth on: a valid access token, or 401."""
    if not settings.auth_enabled:
        request.state.user = ANONYMOUS
        return ANONYMOUS
    claims = decode_access_token(_bearer(request))
    if claims is None:
        raise HTTPException(status_code=401, detail="Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    user = User(id=claims["sub"], email=claims.get("email", ""), name=claims.get("name", ""),
                role=claims.get("role", "user"))
    request.state.user = user
    return user


def rate_limit(request: Request, user: User = Depends(current_user)) -> None:
    """FastAPI dependency for the expensive (LLM) endpoints (runs after authentication)."""
    retry = query_limiter.check(client_key(request))
    if retry:
        metrics.record_rate_limited()
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests: limit is {query_limiter.limit} per minute",
            headers={"Retry-After": str(int(retry) + 1)},
        )


def require_admin(request: Request) -> User:
    """Document management: an admin user, or the ADMIN_TOKEN service token (automation).
    With auth disabled and no ADMIN_TOKEN the endpoints are open (dev only)."""
    supplied = request.headers.get("x-admin-token", "") or _bearer(request)
    if settings.admin_token and supplied and hmac.compare_digest(supplied, settings.admin_token):
        return ANONYMOUS
    if not settings.auth_enabled:
        if settings.admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        return ANONYMOUS
    user = current_user(request)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin role required")
    return user
