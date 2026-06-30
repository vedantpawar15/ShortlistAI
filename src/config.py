"""Configuration primitives for the RecruitAI system."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    BaseSettings = BaseModel

    def SettingsConfigDict(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)


class Settings(BaseSettings):
    """Application settings for offline candidate ranking."""

    project_name: str = "RecruitAI"
    environment: str = "local"
    data_dir: Path = Field(default=Path("data"))
    outputs_dir: Path = Field(default=Path("outputs"))
    logs_dir: Path = Field(default=Path("logs"))
    models_dir: Path = Field(default=Path("models"))
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    faiss_index_name: str = "candidate_index.faiss"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(env_prefix="RECRUITAI_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


