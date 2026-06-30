"""FastAPI application for RecruitAI offline ranking."""

from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError:
    class FastAPI:
        """Minimal decorator-compatible fallback when FastAPI is not installed."""

        def __init__(self, title: str, version: str) -> None:
            self.title = title
            self.version = version

        def get(self, path: str, response_model: object | None = None) -> object:
            return lambda function: function

        def post(self, path: str, response_model: object | None = None) -> object:
            return lambda function: function

from pydantic import BaseModel, Field

from src.config import get_settings
from src.data_loader import DataLoader
from src.ranking_engine import RankingEngine

settings = get_settings()
app = FastAPI(title=settings.project_name, version="0.1.0")


class HealthResponse(BaseModel):
    """Health-check response model."""

    status: str = Field(default="ok")
    project: str


class RankRequest(BaseModel):
    """Request model for candidate ranking."""

    job_id: str = Field(default="job_description")
    top_k: int = Field(default=10, ge=1)


class RankResponse(BaseModel):
    """Response model for candidate ranking."""

    job_id: str
    results: list[dict[str, object]] = Field(default_factory=list)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(project=settings.project_name)


@app.post("/rank", response_model=RankResponse)
def rank_candidates(request: RankRequest) -> RankResponse:
    """Rank candidates using local datasets and the offline scoring pipeline."""
    loader = DataLoader(settings.data_dir)
    candidates = loader.load_candidates("sample_candidates.json")
    jobs = loader.load_jobs("job_description.docx")
    ranked = RankingEngine().rank_candidates(candidates, jobs.iloc[0]).head(request.top_k)
    return RankResponse(job_id=request.job_id, results=ranked.to_dict(orient="records"))

