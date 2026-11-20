"""
Logger Configuration - Setup logging for the entire application
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from config.settings import settings

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")

# Create logs directory
settings.log_dir.mkdir(parents=True, exist_ok=True)

# Define log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Get log level from settings
LOG_LEVEL = getattr(logging, settings.log_level.upper(), logging.INFO)


def setup_logger(name: str) -> logging.Logger:
    """
    Setup logger for a module
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Skip if handlers already configured
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    log_file = settings.log_dir / f"{name.replace('.', '_')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    from src.observability.logging import PiiFilter
    for h in (console_handler, file_handler):  # log files must not keep e-mails / phone numbers
        h.addFilter(PiiFilter())
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Main application logger
app_logger = setup_logger("multi_rag_agent")

# Module loggers
ingestion_logger = setup_logger("multi_rag_agent.ingestion")
retrieval_logger = setup_logger("multi_rag_agent.retrieval")
agents_logger = setup_logger("multi_rag_agent.agents")
api_logger = setup_logger("multi_rag_agent.api")
