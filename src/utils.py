"""Shared utility helpers for RecruitAI."""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(log_dir: Path | str = "logs") -> None:
    """Configure file-based Loguru logging for local execution."""
    directory = ensure_directory(log_dir)
    logger.add(directory / "recruitai.log", rotation="10 MB", retention="14 days")

