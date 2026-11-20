"""
Input guardrail: is the user's request safe to answer?

Primary: NVIDIA NemoGuard safety classifier (llama-3.1-nemotron-safety-guard-8b-v3 on
NIM, ~1 s; multilingual incl. Vietnamese) -> {"User Safety": "safe"|"unsafe",
"Safety Categories": "..."}. It runs in parallel with the router, so it adds no
latency to safe requests.

Allowed categories (GUARD_ALLOWED_CATEGORIES, default "PII/Privacy") do not block:
a user sharing their own phone number is not an attack.

Fallback when the classifier is unavailable or times out: the prompt-injection
patterns (fail-open for everything else, so an outage of the guard model does not
take the assistant down - the decision is logged and counted).
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import httpx

from config.settings import settings
from src.guardrails import injection
from src.observability import metrics

logger = logging.getLogger(__name__)

REFUSAL = (
    "Xin lỗi, tôi không thể hỗ trợ yêu cầu này vì nó vi phạm chính sách an toàn của hệ thống "
    "({reason}). Nếu bạn nghĩ đây là nhầm lẫn, hãy diễn đạt lại câu hỏi.\n\n"
    "Sorry, I can't help with this request because it violates the safety policy."
)


@dataclass
class Verdict:
    blocked: bool
    categories: List[str] = field(default_factory=list)
    source: str = "model"          # model | heuristic | disabled | skipped
    latency_s: float = 0.0
    error: Optional[str] = None

    @property
    def reason(self) -> str:
        return ", ".join(self.categories) or "unsafe content"

    def public(self) -> dict:
        d = asdict(self)
        d["reason"] = self.reason
        return d


def _allowed() -> set:
    return {c.strip().lower() for c in settings.guard_allowed_categories.split(",") if c.strip()}


def _parse(raw: str) -> tuple:
    """-> (unsafe: bool, categories: list). Tolerates text around the JSON object."""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    data = json.loads(m.group(0)) if m else {}
    unsafe = str(data.get("User Safety", "safe")).strip().lower() == "unsafe"
    cats = [c.strip() for c in str(data.get("Safety Categories", "")).split(",") if c.strip()]
    return unsafe, cats


def _classify(text: str) -> tuple:
    resp = httpx.post(
        f"{settings.nvidia_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.nvidia_api_key}", "Accept": "application/json"},
        json={"model": settings.guard_model, "messages": [{"role": "user", "content": text[:4000]}],
              "max_tokens": 60, "temperature": 0},
        timeout=settings.guard_timeout,
    )
    resp.raise_for_status()
    return _parse(resp.json()["choices"][0]["message"]["content"])


def check(text: str) -> Verdict:
    if not settings.guardrails_enabled:
        return Verdict(blocked=False, source="disabled")
    started = time.time()
    # Explicit jailbreak commands ("ignore all previous instructions...", "bỏ qua mọi hướng
    # dẫn...") are blocked outright: the classifier was inconsistent on them in testing
    # (same Vietnamese jailbreak judged unsafe once, safe once). Questions *about*
    # prompt injection do not match these imperative patterns.
    if injection.detect(text):
        v = Verdict(blocked=True, categories=["Prompt Injection"], source="heuristic",
                    latency_s=round(time.time() - started, 3))
        metrics.record_guard("blocked", "Prompt Injection")
        logger.warning("Guardrail blocked request: prompt-injection pattern")
        return v
    try:
        unsafe, cats = _classify(text)
        allowed = _allowed()
        blocking = [c for c in cats if c.lower() not in allowed]
        blocked = unsafe and (bool(blocking) or not cats)
        v = Verdict(blocked=blocked, categories=blocking if blocked else cats, source="model",
                    latency_s=round(time.time() - started, 3))
    except Exception as e:  # guard model down / slow: fall back to the injection patterns
        hit = injection.detect(text)
        v = Verdict(blocked=hit, categories=["Prompt Injection"] if hit else [], source="heuristic",
                    latency_s=round(time.time() - started, 3), error=f"{type(e).__name__}: {e}"[:200])
        logger.warning(f"Guard model unavailable ({v.error}); heuristic verdict blocked={hit}")
    metrics.record_guard("blocked" if v.blocked else "allowed",
                         v.categories[0] if v.categories else "none")
    if v.blocked:
        logger.warning(f"Guardrail blocked request: {v.reason} (source={v.source})")
    return v


def refusal(verdict: Verdict) -> str:
    return REFUSAL.format(reason=verdict.reason)
