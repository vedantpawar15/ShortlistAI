"""FastAPI application for ShortlistAI offline ranking.

Endpoints
---------
GET  /                          — Health check
GET  /health                    — Health check
GET  /datasets                  — Summarise local candidate and job datasets
POST /rank                      — Rank candidates for a given job
GET  /rank/{job_id}             — Rank candidates (convenience GET form)
POST /explain/{candidate_id}    — Generate a structured explanation for one candidate
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

    class FastAPI:  # type: ignore[no-redef]
        """Minimal decorator-compatible fallback when FastAPI is not installed."""

        def __init__(self, title: str, version: str, description: str = "") -> None:
            self.title = title
            self.version = version

        def get(self, path: str, response_model: object | None = None, **kw: object) -> object:
            return lambda function: function

        def post(self, path: str, response_model: object | None = None, **kw: object) -> object:
            return lambda function: function

        def add_middleware(self, middleware_class: object, **kwargs: object) -> None:
            pass

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int = 400, detail: str = "") -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class CORSMiddleware:  # type: ignore[no-redef]
        pass

    def Query(default: object = None, **kw: object) -> object:  # type: ignore[misc]
        return default

import pandas as pd
from pydantic import BaseModel, Field

from src.config import get_settings
from src.data_loader import DataLoader
from src.explainer import CandidateExplanation, RankingExplainer
from src.ranking_engine import RankingEngine

settings = get_settings()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health-check response model."""

    status: str = Field(default="ok")
    project: str
    version: str = Field(default="1.0.0")


class RankRequest(BaseModel):
    """Request model for candidate ranking."""

    job_id: str = Field(default="job_description")
    top_k: int = Field(default=10, ge=1, le=200)
    candidate_file: str = Field(default="sample_candidates.json")
    job_file: str = Field(default="job_description.docx")
    page: int = Field(default=1, ge=1, description="1-based page index for paginated results.")
    page_size: int = Field(default=10, ge=1, le=100, description="Results per page.")


class RankResponse(BaseModel):
    """Response model for candidate ranking."""

    job_id: str
    results: list[dict[str, object]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=10)


class DatasetSummaryResponse(BaseModel):
    """Response model for available local datasets."""

    candidates: int
    jobs: list[str] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    """Request body for /explain/{candidate_id}."""

    job_id: str = Field(default="job_description")
    job_file: str = Field(default="job_description.docx")
    candidate_file: str = Field(default="sample_candidates.json")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.project_name,
        version="1.0.0",
        description=(
            "Offline-first AI candidate ranking system. "
            "Combines BGE embeddings, FAISS retrieval, and a transparent "
            "weighted-feature scoring pipeline."
        ),
    )

    if _FASTAPI:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/", response_model=HealthResponse, tags=["health"])
    def root() -> HealthResponse:
        """Root health check."""
        return health()

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health_endpoint() -> HealthResponse:
        """Service health check."""
        return health()

    @app.get("/datasets", response_model=DatasetSummaryResponse, tags=["data"])
    def list_datasets() -> DatasetSummaryResponse:
        """Summarise the configured local candidate and job datasets."""
        return dataset_summary()

    @app.post("/rank", response_model=RankResponse, tags=["ranking"])
    def rank_candidates_endpoint(request: RankRequest) -> RankResponse:
        """Rank candidates for a given job posting."""
        return rank_candidates(request)

    @app.get("/rank/{job_id}", response_model=RankResponse, tags=["ranking"])
    def rank_by_job_id(
        job_id: str,
        top_k: int = Query(default=10, ge=1, le=200),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=10, ge=1, le=100),
    ) -> RankResponse:
        """Rank candidates for a given job ID (convenience GET form)."""
        return rank_candidates(
            RankRequest(job_id=job_id, top_k=top_k, page=page, page_size=page_size)
        )

    @app.post("/explain/{candidate_id}", response_model=CandidateExplanation, tags=["explainability"])
    def explain_candidate(candidate_id: str, request: ExplainRequest) -> CandidateExplanation:
        """Generate a structured explanation for a single candidate."""
        return explain(candidate_id, request)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Business logic (pure functions — testable without the HTTP layer)
# ---------------------------------------------------------------------------


def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(project=settings.project_name)


def rank_candidates(request: RankRequest) -> RankResponse:
    """Rank candidates using local datasets and the offline scoring pipeline."""
    loader = DataLoader(settings.data_dir)
    try:
        candidates = loader.load_candidates(request.candidate_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        jobs = loader.load_jobs(request.job_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Resolve job by job_id, then by title as a fallback.
    matching_jobs = jobs[jobs["job_id"].astype(str) == request.job_id]
    if matching_jobs.empty:
        matching_jobs = jobs[jobs["title"].astype(str) == request.job_id]
    if matching_jobs.empty:
        raise ValueError(f"Job '{request.job_id}' was not found in {request.job_file}")

    job_row = matching_jobs.iloc[0]
    all_ranked = RankingEngine().rank_candidates(candidates, job_row)
    total = len(all_ranked)

    # Apply pagination.
    start = (request.page - 1) * request.page_size
    end = start + request.page_size
    paged = all_ranked.iloc[start:end].head(request.top_k)

    return RankResponse(
        job_id=str(job_row["job_id"]),
        results=paged.to_dict(orient="records"),
        total=total,
        page=request.page,
        page_size=request.page_size,
    )


def dataset_summary() -> DatasetSummaryResponse:
    """Summarise the configured local candidate and job datasets."""
    loader = DataLoader(settings.data_dir)
    candidates = loader.load_candidates("sample_candidates.json")
    jobs = loader.load_jobs("job_description.docx")
    return DatasetSummaryResponse(
        candidates=len(candidates),
        jobs=[str(job_id) for job_id in jobs["job_id"].astype(str).tolist()],
    )


def explain(candidate_id: str, request: ExplainRequest) -> CandidateExplanation:
    """Generate a structured explanation for one candidate against one job."""
    loader = DataLoader(settings.data_dir)
    try:
        candidates = loader.load_candidates(request.candidate_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        jobs = loader.load_jobs(request.job_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    matching_jobs = jobs[jobs["job_id"].astype(str) == request.job_id]
    if matching_jobs.empty:
        raise HTTPException(status_code=404, detail=f"Job '{request.job_id}' not found.")

    matching_candidates = candidates[candidates["candidate_id"].astype(str) == candidate_id]
    if matching_candidates.empty:
        raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found.")

    job_row = matching_jobs.iloc[0]
    all_ranked = RankingEngine().rank_candidates(candidates, job_row)

    # Find the candidate row in the ranked output.
    candidate_ranked_row = all_ranked[all_ranked["candidate_id"].astype(str) == candidate_id]
    if candidate_ranked_row.empty:
        raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' did not appear in ranking results.")

    row_dict = candidate_ranked_row.iloc[0].to_dict()
    explainer = RankingExplainer()
    return explainer.explain(candidate_id, row_dict)
