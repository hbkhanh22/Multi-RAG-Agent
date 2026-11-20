"""
Sign in with Google (OpenID Connect, authorization-code flow + PKCE).

1. /api/auth/google/login     -> redirect to Google with state, nonce, PKCE challenge
                                 (the three secrets ride in a short-lived signed httpOnly cookie)
2. /api/auth/google/callback  -> check state, exchange the code (+ PKCE verifier) for tokens,
                                 verify the ID token signature (Google JWKS), audience, issuer,
                                 expiry and nonce; then log the user in.
"""

import base64
import hashlib
import secrets
import time
from typing import Dict
from urllib.parse import urlencode

import httpx
import jwt

from config.settings import settings

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = ("https://accounts.google.com", "accounts.google.com")
FLOW_COOKIE = "rag_oauth"
FLOW_TTL = 600

_jwks_client = None


def enabled() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def redirect_uri() -> str:
    return settings.public_base_url.rstrip("/") + "/api/auth/google/callback"


def start() -> Dict[str, str]:
    """Returns {"url": <Google consent URL>, "flow": <signed cookie value>}."""
    from src.auth.tokens import ALGORITHM, _secret

    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    url = AUTH_URL + "?" + urlencode({
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    })
    flow = jwt.encode({"state": state, "nonce": nonce, "verifier": verifier, "exp": int(time.time()) + FLOW_TTL,
                       "typ": "oauth_flow"}, _secret(), algorithm=ALGORITHM)
    return {"url": url, "flow": flow}


def _read_flow(flow_cookie: str) -> Dict[str, str]:
    from src.auth.tokens import ALGORITHM, _secret

    claims = jwt.decode(flow_cookie, _secret(), algorithms=[ALGORITHM])
    if claims.get("typ") != "oauth_flow":
        raise ValueError("bad flow cookie")
    return claims


def _verify_id_token(id_token: str, nonce: str) -> Dict:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(JWKS_URL, cache_keys=True)
    key = _jwks_client.get_signing_key_from_jwt(id_token)
    claims = jwt.decode(id_token, key.key, algorithms=["RS256"], audience=settings.google_client_id,
                        options={"require": ["exp", "iat", "iss", "sub", "aud"]})
    if claims.get("iss") not in ISSUERS:
        raise ValueError("wrong issuer")
    if claims.get("nonce") != nonce:
        raise ValueError("nonce mismatch")
    return claims


def finish(code: str, state: str, flow_cookie: str) -> Dict:
    """Validate the callback and return the verified ID-token claims."""
    flow = _read_flow(flow_cookie)
    if not state or not secrets.compare_digest(state, flow["state"]):
        raise ValueError("state mismatch")
    resp = httpx.post(TOKEN_URL, timeout=15, data={
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": redirect_uri(),
        "grant_type": "authorization_code",
        "code_verifier": flow["verifier"],
    })
    resp.raise_for_status()
    return _verify_id_token(resp.json()["id_token"], flow["nonce"])
