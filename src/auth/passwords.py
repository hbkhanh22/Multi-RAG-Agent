"""
Password hashing with scrypt (memory-hard KDF from the standard library).

Stored format:  scrypt$<n>$<r>$<p>$<salt b64>$<hash b64>
Parameters are stored with the hash, so they can be raised later; `needs_rehash`
tells the login path to upgrade old hashes transparently.
"""

import base64
import hashlib
import hmac
import secrets

N, R, P = 2 ** 14, 8, 1          # ~16 MB, ~50 ms per hash on a laptop CPU
DKLEN = 32
MAX_PASSWORD_BYTES = 1024        # bound the work an attacker can make us do


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    if len(pw) > MAX_PASSWORD_BYTES:
        raise ValueError("password too long")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(pw, salt=salt, n=N, r=R, p=P, dklen=DKLEN, maxmem=64 * 1024 * 1024)
    return f"scrypt${N}${R}${P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        pw = password.encode("utf-8")
        if len(pw) > MAX_PASSWORD_BYTES:
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(pw, salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                                dklen=len(expected), maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def needs_rehash(stored: str) -> bool:
    try:
        _, n, r, p, *_ = stored.split("$")
        return (int(n), int(r), int(p)) != (N, R, P)
    except ValueError:
        return True


# Used to equalise timing when the account does not exist (no user enumeration)
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
