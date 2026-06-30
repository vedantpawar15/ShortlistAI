"""FastAPI application shell for RecruitAI."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.project_name, version="0.1.0")


class HealthResponse(BaseModel):
    """Health-check response model."""

    status: str = Field(default="ok")
    project: str


class RankRequest(BaseModel):
    """Request model for future candidate ranking."""

    job_id: str
    top_k: int = Field(default=10, ge=1)


class RankResponse(BaseModel):
    """Response model for future candidate ranking."""

    job_id: str
    results: list[dict[str, object]] = Field(default_factory=list)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(project=settings.project_name)


@app.post("/rank", response_model=RankResponse)
def rank_candidates(request: RankRequest) -> RankResponse:
    """Placeholder endpoint for future candidate ranking."""
    return RankResponse(job_id=request.job_id)

