"""
Logger configuration.

Console output goes through the root logger (configured by
src.observability.logging.configure_logging: text or JSON, request id, PII masking),
so every record is printed once. `setup_logger` only adds a rotating log file per
logger (logs/<name>.log), with PII masking too.
"""

import logging
import logging.handlers
import sys

from config.settings import settings

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")

settings.log_dir.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = getattr(logging, settings.log_level.upper(), logging.INFO)

# Scripts that never call configure_logging() still get console output
if not logging.getLogger().handlers:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def setup_logger(name: str) -> logging.Logger:
    """Logger `name` writing to logs/<name>.log (rotating) and, via the root logger, to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    if logger.handlers:
        return logger

    from src.observability.logging import PiiFilter

    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_dir / f"{name.replace('.', '_')}.log",
        maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    file_handler.addFilter(PiiFilter())   # log files must not keep e-mails / phone numbers
    logger.addHandler(file_handler)
    return logger


app_logger = setup_logger("multi_rag_agent")
api_logger = setup_logger("multi_rag_agent.api")
