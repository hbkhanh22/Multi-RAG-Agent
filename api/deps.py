"""
Request guards: rate limiting and the admin token for document management.

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

from fastapi import Header, HTTPException, Request

from config.settings import settings
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


def client_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return "tok:" + auth[7:][:64]
    return "ip:" + (request.client.host if request.client else "unknown")


def rate_limit(request: Request) -> None:
    """FastAPI dependency for the expensive (LLM) endpoints."""
    retry = query_limiter.check(client_key(request))
    if retry:
        metrics.record_rate_limited()
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests: limit is {query_limiter.limit} per minute",
            headers={"Retry-After": str(int(retry) + 1)},
        )


def require_admin(authorization: str = Header(default=""), x_admin_token: str = Header(default="")) -> None:
    """Guards document management. Open when ADMIN_TOKEN is unset (dev only)."""
    expected = settings.admin_token
    if not expected:
        return
    supplied = x_admin_token or (authorization[7:] if authorization.lower().startswith("bearer ") else "")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Admin token required")
