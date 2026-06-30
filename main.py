"""Command-line entry point for RecruitAI."""

from __future__ import annotations

import typer
from loguru import logger

from src.config import get_settings

cli = typer.Typer(help="RecruitAI offline candidate ranking system.")


@cli.command()
def rank() -> None:
    """Placeholder command for future offline candidate ranking."""
    settings = get_settings()
    logger.info("RecruitAI ranking pipeline is not implemented yet.")
    logger.debug("Loaded settings: {}", settings.model_dump())


@cli.command()
def health() -> None:
    """Verify that the CLI can load project configuration."""
    settings = get_settings()
    typer.echo(f"RecruitAI is configured for project: {settings.project_name}")


if __name__ == "__main__":
    cli()

