"""Logging compatibility helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


class StandardLoggerAdapter:
    """Small Loguru-compatible adapter backed by the standard library."""

    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
        self._logger = logging.getLogger("recruitai")

    def add(self, sink: Path | str, rotation: str | None = None, retention: str | None = None) -> None:
        handler = logging.FileHandler(sink, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self._logger.addHandler(handler)

    def debug(self, message: str, *args: Any) -> None:
        self._logger.debug(format_message(message, args))

    def info(self, message: str, *args: Any) -> None:
        self._logger.info(format_message(message, args))

    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(format_message(message, args))

    def error(self, message: str, *args: Any) -> None:
        self._logger.error(format_message(message, args))


def format_message(message: str, args: tuple[Any, ...]) -> str:
    """Format Loguru-style braces for the fallback logger."""
    if not args:
        return message
    try:
        return message.format(*args)
    except (IndexError, KeyError, ValueError):
        return f"{message} {' '.join(str(arg) for arg in args)}"


try:
    from loguru import logger as logger
except ImportError:
    logger = StandardLoggerAdapter()
