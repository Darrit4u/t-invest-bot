"""Application logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """Configure root logger with console and rotating file handlers."""

    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    app_handler = RotatingFileHandler(
        log_dir / "application.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)

    error_handler = RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    telegram_logger = logging.getLogger("telegram")
    telegram_logger.handlers.clear()
    telegram_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    telegram_handler = RotatingFileHandler(
        log_dir / "telegram.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    telegram_handler.setFormatter(formatter)
    telegram_logger.addHandler(telegram_handler)
    telegram_logger.propagate = True
