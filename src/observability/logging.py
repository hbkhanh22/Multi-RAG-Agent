"""
Request-scoped logging: every log line carries the request id of the HTTP request
that produced it (also returned to the client as X-Request-ID), and can be emitted as
one JSON object per line (LOG_JSON=true) for Loki / ELK / Cloud Logging.
"""

import contextvars
import json
import logging
from datetime import datetime, timezone

from config.settings import settings

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


class PiiFilter(logging.Filter):
    """Masks e-mails, phone / ID / card numbers in the final log message."""

    def filter(self, record: logging.LogRecord) -> bool:
        if settings.pii_redaction and not getattr(record, "_pii_done", False):
            from src.guardrails.pii import redact
            try:
                record.msg = redact(record.getMessage())
                record.args = ()
            except Exception:
                pass
            record._pii_done = True
        return True


def add_filters(handler: logging.Handler) -> None:
    for cls in (RequestIdFilter, PiiFilter):
        if not any(isinstance(f, cls) for f in handler.filters):
            handler.addFilter(cls())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


TEXT_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"


def configure_logging() -> None:
    """Install the request-id filter (+ JSON formatter if enabled) on the root logger."""
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    formatter = JsonFormatter() if settings.log_json else logging.Formatter(TEXT_FORMAT)
    for handler in root.handlers:
        add_filters(handler)
        handler.setFormatter(formatter)
