"""
PII redaction for logs and traces (e-mails, phone numbers, ID / card numbers, IPs).

Patterns target Vietnamese and international formats:
  e-mail                      a.nguyen@example.com           -> [EMAIL]
  phone (VN mobile/landline)  0912 345 678, +84 912345678    -> [PHONE]
  CCCD / CMND                 079123456789 / 123456789       -> [ID_NUMBER]
  payment card (Luhn-valid)   4111 1111 1111 1111            -> [CARD]
  IPv4                        192.168.1.10                   -> [IP]

Redaction is applied to what leaves the process for observability (log lines,
Langfuse traces). The LLM itself still receives the user's text: answering about
"my order 0912..." needs it, and the model provider is already trusted with the
conversation.
"""

import re
from typing import Any

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<![\w.])(?:\+?84|0)(?:[ .-]?\d){8,10}(?![\w])")
_ID = re.compile(r"(?<!\d)(?:\d{12}|\d{9})(?!\d)")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)


def redact(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _CARD.sub(_card, text)          # before phone/ID: a card contains such digit runs
    text = _PHONE.sub("[PHONE]", text)
    text = _ID.sub("[ID_NUMBER]", text)
    text = _IPV4.sub("[IP]", text)
    return text


def redact_any(data: Any) -> Any:
    """Recursively redact strings inside dicts / lists (trace payloads)."""
    if isinstance(data, str):
        return redact(data)
    if isinstance(data, dict):
        return {k: redact_any(v) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(redact_any(v) for v in data)
    if isinstance(data, list):
        return [redact_any(v) for v in data]
    if hasattr(data, "model_dump"):          # LangChain messages, pydantic models
        return redact_any(data.model_dump())
    if data is None or isinstance(data, (bool, int, float)):
        return data
    return redact(str(data))                 # unknown object: never pass it on unmasked
