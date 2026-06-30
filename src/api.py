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


class HealthResponse(BaseModel):
    """Health-check response model."""

    status: str = Field(default="ok")
    project: str


class RankRequest(BaseModel):
    """Request model for candidate ranking."""

    job_id: str = Field(default="job_description")
    top_k: int = Field(default=10, ge=1)
    candidate_file: str = Field(default="sample_candidates.json")
    job_file: str = Field(default="job_description.docx")


class RankResponse(BaseModel):
    """Response model for candidate ranking."""

    job_id: str
    results: list[dict[str, object]] = Field(default_factory=list)


class DatasetSummaryResponse(BaseModel):
    """Response model for available local datasets."""

    candidates: int
    jobs: list[str] = Field(default_factory=list)


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title=settings.project_name, version="0.1.0")

    @app.get("/", response_model=HealthResponse)
    def root() -> HealthResponse:
        return health()

    @app.get("/health", response_model=HealthResponse)
    def health_endpoint() -> HealthResponse:
        return health()

    @app.get("/datasets", response_model=DatasetSummaryResponse)
    def list_datasets() -> DatasetSummaryResponse:
        return dataset_summary()

    @app.post("/rank", response_model=RankResponse)
    def rank_candidates_endpoint(request: RankRequest) -> RankResponse:
        return rank_candidates(request)

    return app


app = create_app()


def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(project=settings.project_name)


def rank_candidates(request: RankRequest) -> RankResponse:
    """Rank candidates using local datasets and the offline scoring pipeline."""
    loader = DataLoader(settings.data_dir)
    candidates = loader.load_candidates(request.candidate_file)
    jobs = loader.load_jobs(request.job_file)
    matching_jobs = jobs[jobs["job_id"].astype(str) == request.job_id]
    if matching_jobs.empty:
        matching_jobs = jobs[jobs["title"].astype(str) == request.job_id]
    if matching_jobs.empty:
        raise ValueError(f"Job '{request.job_id}' was not found in {request.job_file}")
    ranked = RankingEngine().rank_candidates(candidates, matching_jobs.iloc[0]).head(request.top_k)
    return RankResponse(job_id=str(matching_jobs.iloc[0]["job_id"]), results=ranked.to_dict(orient="records"))


def dataset_summary() -> DatasetSummaryResponse:
    """Summarize the configured local candidate and job datasets."""
    loader = DataLoader(settings.data_dir)
    candidates = loader.load_candidates("sample_candidates.json")
    jobs = loader.load_jobs("job_description.docx")
    return DatasetSummaryResponse(
        candidates=len(candidates),
        jobs=[str(job_id) for job_id in jobs["job_id"].astype(str).tolist()],
    )

