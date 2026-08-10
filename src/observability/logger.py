"""
Structured logging with JSON format and multiple outputs.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler

from src.config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    """Setup structured logging."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (JSON for production, readable for dev)
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.is_production:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    root_logger.addHandler(console_handler)

    # File handler (always JSON)
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = RotatingFileHandler(
        "logs/error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    logging.info("✅ Logging configured")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with extra fields support."""
    return logging.getLogger(name)